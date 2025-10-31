import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import copy

from opencood.utils.box_utils import project_points_by_matrix_torch, project_points_by_matrix
from opencood.utils.transformation_utils import x1_to_x2
from opencood.visualization.cppc_vis import VisUtil
from opencood.utils.fsd_metric import MetricUtil
from opencood.models.sub_modules.cluster_align import cluster_alignment_relative_sample_np
from opencood.models.sub_modules.cluster_latency import cluster_latency_align

# Use standalone ops instead of mmdet3d (for L40S/CUDA 11.6 compatibility)
from opencood.models.fuse_modules.mmdet3d_ops_standalone import (
    find_connected_componets, scatter_v2, furthest_point_sample,
    get_inner_win_inds, ball_query, Voxelization
)


def fps(points, rate):
    N = int(points.shape[0] * rate)
    idx = furthest_point_sample(points.unsqueeze(0), N)
    idx = idx.squeeze(0).long()
    return idx

class ClusterFusion(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.seg_score_th = cfgs.get('seg_score_th', 0.5)
        self.cluster_topk = cfgs.get('cluster_topk', 100)
        self.connect_center_dist = cfgs.get('connect_center_dist', 1)
        self.use_req = False
        self.use_merge = cfgs.get('use_merge', True)
        # consider attributes(xyz and feats) only from fg points for two-stage refine
        # cluster_center cluster_mean or cluster_pts
        self.two_stage_xyz_mode = cfgs.get('two_stage_xyz', 'cluster_pts')
        # cluster_pts_mean cluster_pts_max cluster_feat
        self.two_stage_feat_mode = cfgs.get('two_stage_feat', 'cluster_pts_mean')
        if self.two_stage_feat_mode == 'cluster_feat':
            self.point_feat_proj = nn.Linear(768, 128)
        
        self.points_wise_keys = ['batch_idx', 'seg_feats', 'seg_points']
        self.cluster_pts_wise_keys = ['cluster_pts_inds', 'cluster_pts_feats', 'cluster_pts_xyz']
        self.cluster_wise_keys = ['cluster_inds', 'cluster_feats', 'cluster_xyz', 'cluster_seg_score', 'cluster_mean_xyz', 'cluster_seg_feat']
        self.no_coop = False    # bug with training

        rate = 0
        self.rate = (0.5)**rate
        self.is_refine_pose = True
        self.refine_use = "cluster" # ["box", "cluster", "point"]

        self.use_both_sample = False
        self.use_score_sample = False
        if self.use_score_sample:
            self.rate = self.rate * 2
        self.use_density_sample = False
        if self.use_density_sample or self.use_both_sample:
            from opencood.models.sub_modules.kde_util import GaussianKernelDensityEstimation
            # bandwidth(kernel size) should be larger for larger rate(less points with larger distance)
            bandwidth = 0.25 * rate
            self.density_estimator = GaussianKernelDensityEstimation(bandwidth=bandwidth)
            self.rate = self.rate * 2
        self.use_rand_sample = False

        random.seed(42)
        self.is_time_latency = False
        self.is_refine_time_latency = False
        self.time_latency = 5
        self.now_time_latency = 0

    def refine_pose(self, dict_to_sample, sampled_out, extracted_outs, img_metas):
        """Refine pose from noego cars

        Args:
            dict_to_sample (dict): N is point number of all scenes
                - "seg_points" (Tensor): shape(Nx4) origin coordinates for per point
                - "seg_logits" (Tensor): shape(Nx1) segment confidence logit for per points
                - "seg_vote_preds" (Tensor): shape(Nx3)
                - "seg_feats" (Tensor): shape(Nx67)
                - "vote_offsets" (Tensor): shape(Nx3)   TODO: the difference between vote_offsets and seg_vote_preds
                - "batch_idx" (Tensor): shape(N) batch idx for per point

            sampled_out (dict): M is point number of foreground
                - "seg_points" (list[Tensor]): shape(Mx4) 
                - "seg_logits" (list[Tensor)]): shape(Mx1) 
                - "seg_vote_preds" (list[Tensor)]): shape(Mx3)
                - "seg_feats" (list[Tensor)]): shape(Mx67)
                - "vote_offsets" (list[Tensor)]): shape(Mx3)   
                - "batch_idx" (list[Tensor)]): shape(M) 

                - "fg_mask_list" (list[Tensor)]): shape(N)
                - "center_preds" (list[Tensor)]): shape(Mx3) 

            extracted_outs (dict): P is cluster number of foreground
                - 'cluster_feats' (Tensor): shape(Px768)
                    - 768 = 128(point feature cannel) * 6(block number of SIR)
                - 'cluster_xyz' (Tensor): shape(Px3)
                - 'cluster_inds' (Tensor): shape(Px3)
                - 'cluster_seg_score' (Tensor): shape(Px1)
                - 'cluster_mean_xyz' (Tensor): shape(Px4)
                - 'cluster_seg_feat' (Tensor): shape(Px67)

                - 'cluster_pts_feats' (Tensor): shape(Mx128)
                    - 128: point feature cannel
                - 'cluster_pts_xyz' (Tensor): shape(Mx4)
                    - equal to sampled_out['seg_points']
                - 'cluster_pts_inds' (Tensor): shape(Mx3)
                    - [class_id, batch_id, cluster_id] 
                    - batch_id is car level and not scene level

            img_metas (list[dict]):
                - length: car num
                - object (dict):
                    - 'box_type_3d': class type (default LiDARInstance3DBoxes)
                    - 'record_len' (list): length is scene level batch, include car number per scene
                    - 'model_dir' (str)
                    - 'vis_dir' (str)
                    - 'batch_idx' (int): scene level batch for car level data 
                    - 'vis_n' (int)
                    - 'vis_type' (str): 'bev' / '3d'
                    - 'proj_first' (bool)
                    - 'proj2ego_matrix' (Tensor): shape(4x4) pose transfer matrix

                    - noisy_lidar_pose (list)

        """

        if VisUtil.is_vis_now():
            VisUtil.get_scene_points(
                dict_to_sample['seg_points'], 
                dict_to_sample['vote_offsets'],
                dict_to_sample['batch_idx'],
                sampled_out['fg_mask_list'][0], 
                img_metas)
            VisUtil.vis(img_metas[0]['vis_type'], False, False, False, 'one')
        
        if self.refine_use == "cluster":
            cluster_anchor = extracted_outs["cluster_xyz"]
        elif self.refine_use == "point":
            cluster_anchor, _ = scatter_v2(extracted_outs['cluster_pts_xyz'], extracted_outs['cluster_pts_inds'], mode='avg', return_inv=False)
            cluster_anchor = cluster_anchor[:, :3]
        elif self.refine_use == "box":
            cluster_anchor = extracted_outs['box_center']
        else:
            raise NotImplementedError()

        bs = len(img_metas)  
        pred_clusters_list = [
            cluster_anchor[extracted_outs["cluster_inds"][:, 1].long() == i].cpu() for i in range(bs)
        ]
        noisy_lidar_pose = torch.cat(
            [img_metas[i]['lidar_pose'].reshape(1, -1) for i in range(bs)], dim=0
        ).cpu().float()
        clean_lidar_pose = torch.cat(
            [img_metas[i]['lidar_pose_clean'].reshape(1, -1) for i in range(bs)], dim=0
        ).cpu().float()

        proj2local_matrix = [
            torch.tensor(x1_to_x2(img_metas[0]['lidar_pose'].cpu(), img_metas[i]['lidar_pose'].cpu())).float() for i in range(bs)
        ]
        refined_lidar_poses = cluster_alignment_relative_sample_np(
            pred_clusters_list=pred_clusters_list,
            noisy_lidar_pose=noisy_lidar_pose,
            proj2local_matrix=proj2local_matrix,
            landmark_SE2=False,
            abandon_hard_cases=False,
            max_iterations=1000,
            thres=1.5,
            proj_first=img_metas[0]['proj_first']
        )
        unrefined_lidar_pose = copy.deepcopy(noisy_lidar_pose)
        refined_lidar_pose = noisy_lidar_pose
        refined_lidar_pose[:, [0,1,4]] = torch.tensor(refined_lidar_poses).float()

        trans_matrix = self.get_real_trans(unrefined_lidar_pose, refined_lidar_pose)
        self.project_xyz(trans_matrix, dict_to_sample, sampled_out, extracted_outs)

        VisUtil.get_scene_points(
            dict_to_sample['seg_points'], 
            dict_to_sample['vote_offsets'],
            dict_to_sample['batch_idx'],
            sampled_out['fg_mask_list'][0], 
            img_metas)

    def get_real_trans(self, cav_lidar_poses_error, cav_lidar_poses_clean):
        real_trans_list = []

        bs = cav_lidar_poses_clean.shape[0]
        ego_lidar_pose = cav_lidar_poses_clean[0]
        for i in range(bs):
            cav_lidar_pose_error = cav_lidar_poses_error[i]
            cav_lidar_pose_clean = cav_lidar_poses_clean[i]
            matrix1 = x1_to_x2(ego_lidar_pose, cav_lidar_pose_error)    # world_to_error @ ego_to_world
            matrix2 = x1_to_x2(cav_lidar_pose_clean, ego_lidar_pose)    # world_to_ego @ cav_to_world
            real_trans_list.append(torch.from_numpy(matrix2 @ matrix1).float().cuda())

        return torch.stack(real_trans_list, dim=0)
    
    def cache_time_latency(self, dict_to_sample, sampled_out, extracted_outs, img_metas, dataset='DAIR-V2X'):
        frame_id = img_metas[0].get('frame_id', '------')
        file_path = f'./dataset/time_latency/{dataset}/{frame_id}.pth'

        cache_data = {
            'dict_to_sample': dict_to_sample,
            'sampled_out': sampled_out,
            'extracted_outs': extracted_outs,
            'img_metas': img_metas,
        }
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        if not os.path.exists(file_path):
            torch.save(cache_data, file_path)

    def get_time_latency_data(self, dict_to_sample, sampled_out, extracted_outs, img_metas, dataset='DAIR-V2X'):
        if self.time_latency < 0:
            return 
        else:
            time_latency = self.time_latency
            frame_id = img_metas[0].get('frame_id', '------') 

            all_latency_data = []
            time_step = 2

            for i in range(time_step):
                new_frame_id = int(frame_id) - time_latency - i
                new_frame_id = f"{new_frame_id:06}"
                
                file_path = f'./dataset/time_latency/{dataset}/{new_frame_id}.pth'

                try:
                    data = torch.load(file_path)
                except:
                    time_latency = 0
                    data = {
                        'time': int(new_frame_id),
                        'dict_to_sample': copy.deepcopy(dict_to_sample),
                        'sampled_out': copy.deepcopy(sampled_out),
                        'extracted_outs': copy.deepcopy(extracted_outs),
                        'img_metas': copy.deepcopy(img_metas),
                    }
                    print(f"{new_frame_id}\n")
                all_latency_data.append(data)

                # reset proj2ego_matrix
                old_img_metas = data['img_metas']
                noego_lidar_pose_old = old_img_metas[1]['lidar_pose']
                ego_lidar_pose_new = img_metas[0]['lidar_pose']
                
                proj2ego_matrix = torch.tensor(x1_to_x2(noego_lidar_pose_old.cpu(), ego_lidar_pose_new.cpu())).float()    # exactly it's x2_to_x1
                old_img_metas[1]['proj2ego_matrix'] = proj2ego_matrix.to(old_img_metas[1]['proj2ego_matrix'].device)
            
            self.merge_time_latency_data(dict_to_sample, sampled_out, extracted_outs, img_metas, all_latency_data, dataset)

            self.now_time_latency = time_latency
            # merge not-now data to now data batch

    def merge_time_latency_data(self, dict_to_sample, sampled_out, extracted_outs, img_metas, data, dataset='DAIR-V2X'):
        for i, _data in enumerate(data):
            old_dict_to_sample = _data['dict_to_sample']
            old_sampled_out = _data['sampled_out']
            old_extracted_outs = _data['extracted_outs']
            old_img_metas = _data['img_metas']

            # merge dict_to_sample
            old_dict_to_sample['batch_idx'] += (i + 1) * 2
            for k in self.points_wise_keys:
                dict_to_sample[k] = torch.cat([dict_to_sample[k], old_dict_to_sample[k]], dim=0)
            _, bs_sorted_inds = torch.sort(dict_to_sample['batch_idx'], stable=True)
            for k in self.points_wise_keys:
                dict_to_sample[k] = dict_to_sample[k][bs_sorted_inds]

            # merge extracted_outs
            old_extracted_outs['cluster_pts_inds'][:, 1] += (i + 1) * 2
            for k in self.cluster_pts_wise_keys:
                extracted_outs[k] = torch.cat([extracted_outs[k], old_extracted_outs[k]], dim=0)
            _, cluster_sorted_inds = torch.sort(extracted_outs['cluster_pts_inds'][:, 1], stable=True)
            for k in self.cluster_pts_wise_keys:
                extracted_outs[k] = extracted_outs[k][cluster_sorted_inds]

            old_extracted_outs['cluster_inds'][:, 1] += (i + 1) * 2
            for k in self.cluster_wise_keys:
                extracted_outs[k] = torch.cat([extracted_outs[k], old_extracted_outs[k]], dim=0)
            _, cluster_sorted_inds = torch.sort(extracted_outs['cluster_inds'][:, 1], stable=True)
            for k in self.cluster_wise_keys:
                extracted_outs[k] = extracted_outs[k][cluster_sorted_inds]
                
            sampled_out['fg_mask_list'][0] = torch.cat([sampled_out['fg_mask_list'][0], old_sampled_out['fg_mask_list'][0]], dim=0)
            sampled_out['fg_mask_list'][0] = sampled_out['fg_mask_list'][0][bs_sorted_inds] 

            img_metas += old_img_metas

    def refine_time_latency(self, dict_to_sample, sampled_out, extracted_outs, img_metas, latency_time=0.5):
        noego_bs_list = [1, 2]

        pred_clusters_list = [
            extracted_outs["cluster_xyz"][extracted_outs["cluster_inds"][:, 1].long() == i] for i in noego_bs_list
        ]   # [bs, cluster_num, 3]
        
        latest_cluster_mask = extracted_outs['cluster_inds'][:, 1].long() == 1
        latest_cluster_points_mask = extracted_outs['cluster_pts_inds'][:, 1].long() == 1
        latest_points_mask = dict_to_sample['batch_idx'].long() == 1

        up_thres = 2
        down_thres = 0.5
        dt = img_metas[2]['time_delay'] - img_metas[1]['time_delay']
        latency_time = img_metas[1]['time_delay']

        fore_points, cluster_refine = cluster_latency_align(
            pred_clusters_list,
            extracted_outs["cluster_inds"][latest_cluster_mask],
            extracted_outs["cluster_pts_xyz"][latest_cluster_points_mask],
            extracted_outs["cluster_pts_inds"][latest_cluster_points_mask],
            up_thres = up_thres,
            down_thres = down_thres,
            dt = dt,       # delta t between two frames
            latency_time = latency_time,   # latency time
        )
        
        extracted_outs["cluster_xyz"][latest_cluster_mask] = cluster_refine
        extracted_outs["cluster_pts_xyz"][latest_cluster_points_mask] = fore_points
        dict_to_sample['seg_points'][torch.all(torch.stack([latest_points_mask, sampled_out['fg_mask_list'][0]], dim=1), dim=1)] = fore_points
        
    def reshape_muti_time_data_to_one(self, dict_to_sample, sampled_out, extracted_outs, img_metas):
        device = img_metas[0]['proj2ego_matrix'].device
        save_bs = torch.tensor([0, 1], device=device)
        cluster_save_mask = self.get_ego_mask(extracted_outs['cluster_inds'][:, 1], save_bs)
        cluster_poins_save_mask = self.get_ego_mask(extracted_outs['cluster_pts_inds'][:, 1], save_bs)
        points_save_mask = self.get_ego_mask(dict_to_sample['batch_idx'], save_bs)

        for key in self.cluster_wise_keys:
            extracted_outs[key] = extracted_outs[key][cluster_save_mask]
        for key in self.cluster_pts_wise_keys:
            extracted_outs[key] = extracted_outs[key][cluster_poins_save_mask]
        for key in self.points_wise_keys:
            dict_to_sample[key] = dict_to_sample[key][points_save_mask]
        sampled_out['fg_mask_list'][0] = sampled_out['fg_mask_list'][0][points_save_mask]
        
        img_metas = [img_metas[0], img_metas[1]]

        # extracted_outs['cluster_inds'][:, 1][extracted_outs['cluster_inds'][:, 1] == 1] = 1
        # extracted_outs['cluster_pts_inds'][:, 1][extracted_outs['cluster_pts_inds'][:, 1] == 1] = 1
        # dict_to_sample['batch_idx'][dict_to_sample['batch_idx'] == 1] = 1

        # fix noego cluster idx
        # max_ego_cluster_id = extracted_outs['cluster_inds'][:, 2][extracted_outs['cluster_inds'][:, 1] == 0].max()
        # min_noego_cluster_id = extracted_outs['cluster_inds'][:, 2][extracted_outs['cluster_inds'][:, 1] == 1].min()
        # extracted_outs['cluster_inds'][:, 2][extracted_outs['cluster_inds'][:, 1] == 1] += -min_noego_cluster_id + max_ego_cluster_id + 1
        # extracted_outs['cluster_pts_inds'][:, 2][extracted_outs['cluster_pts_inds'][:, 1] == 1] += -min_noego_cluster_id + max_ego_cluster_id + 1

    def forward(self, dict_to_sample, sampled_out, extracted_outs, img_metas):
        # self.cache_time_late ncy(dict_to_sample, sampled_out, extracted_outs, img_metas)
        bs = len(img_metas)         # car-wise batch size
        pos_cluster_num = (extracted_outs['cluster_seg_score'] > self.seg_score_th).sum()

        # when project late, project here
        if not img_metas[0]['proj_first']:
            # for ego trans_matrix is eyes matrix, others are project to ego
            trans_matrix = torch.stack([img_metas[i]['proj2ego_matrix'] for i in range(bs)], dim=0).float()    # [BS, 4, 4]
            self.project_xyz(trans_matrix, dict_to_sample, sampled_out, extracted_outs)
            
        if VisUtil.is_vis_now():
            # vis cluster center before fusion
            VisUtil.get_scene_points(
                    dict_to_sample['seg_points'], 
                    dict_to_sample['vote_offsets'],
                    dict_to_sample['batch_idx'],
                    sampled_out['fg_mask_list'][0], 
                    img_metas,
                    ignore_proj=True)
            VisUtil.get_cluster_points(extracted_outs['cluster_xyz'], extracted_outs['cluster_inds'], agent_num=3)
            VisUtil.vis(img_metas[0]['vis_type'], False, False, True, 'one', 'before_refine')

        if self.is_refine_pose:
            self.refine_pose(dict_to_sample, sampled_out, extracted_outs, img_metas)
        
        if self.is_refine_time_latency:
            self.refine_time_latency(dict_to_sample, sampled_out, extracted_outs, img_metas, self.now_time_latency * 0.1)
        
        if self.is_time_latency:
            self.reshape_muti_time_data_to_one(dict_to_sample, sampled_out, extracted_outs, img_metas)
            
        if VisUtil.is_vis_now():
            VisUtil.get_scene_points(
                    dict_to_sample['seg_points'], 
                    dict_to_sample['vote_offsets'],
                    dict_to_sample['batch_idx'],
                    sampled_out['fg_mask_list'][0], 
                    img_metas,
                    ignore_proj=True)
            VisUtil.get_cluster_points(extracted_outs['cluster_xyz'], extracted_outs['cluster_inds'], agent_num=2)
            VisUtil.vis(img_metas[0]['vis_type'], False, False, True, 'one', 'after_refine')
        
        if self.use_req:
            request_msg = self.request_message(extracted_outs, img_metas[0]['record_len'])
        else:
            # trans all cluster to ego
            # merge clusters from car-wise batch to scene-wise batch
            new_img_metas = self.cluster_trans(dict_to_sample, sampled_out, extracted_outs, img_metas)
            # merge clsuter attributes based on connection
            if self.use_merge:
                new_num_clusters = self.cluster_connect(extracted_outs) 
            else:
                new_num_clusters = extracted_outs['cluster_inds'][-1].max().item() + 1
        
        new_pos_cluster_num = (extracted_outs['cluster_seg_score'] > self.seg_score_th).sum()
        return new_img_metas, new_num_clusters, (new_pos_cluster_num / pos_cluster_num)
        
    def request_message(self, extracted_outs, record_len):
        """_summary_
        """
        # get batch idx of ego car
        ego_bs_idx = record_len.cumsum(dim=0) - record_len[0]
        # get cluster mask of ego car
        ego_cluster_mask = (extracted_outs['cluster_inds'][:, 1].unsqueeze(1).eq(ego_bs_idx)).any(dim=1)
        cluster_score = extracted_outs['cluster_seg_score'][ego_cluster_mask]     # [M, 1] in (0,1)
        cluster_xyz = extracted_outs['cluster_xyz'][ego_cluster_mask]             # [M, 3]
        if self.training:
            # use top k
            _, topk_indices = torch.topk(cluster_score, k=self.cluster_topk, dim=0)
        else:
            # use score threshold
            topk_indices = (cluster_score > self.seg_score_th).nonzero(as_tuple=False)[:, 0]
        request_flag = torch.zeros_like(cluster_score, dtype=torch.bool)  # [M, 1]
        request_flag[topk_indices] = 1
        request_msg = torch.cat([cluster_xyz, request_flag], dim=1)  # [M, 4]
        return request_msg
    
    def response_message(self, request_msg, extracted_outs):
        pass
    
    def project_xyz(self, trans_matrix, dict_to_sample, sampled_out, extracted_outs): 
        """ porject points to ego coordinate system when set to project late
        """
        p_trans_matrix = trans_matrix[dict_to_sample['batch_idx']]  # [points_num, 4, 4]
        dict_to_sample['seg_points'][:, :3] = project_points_by_matrix(dict_to_sample['seg_points'][:, :3], p_trans_matrix)
        c_trans_matrix = trans_matrix[extracted_outs['cluster_inds'][:, 1].long()]  # [cluster_num, 4, 4]
        extracted_outs['cluster_xyz'] = project_points_by_matrix(extracted_outs['cluster_xyz'], c_trans_matrix)
        cp_trans_matrix = trans_matrix[extracted_outs['cluster_pts_inds'][:, 1].long()]  # [cluster_pts_num, 4, 4]
        extracted_outs['cluster_pts_xyz'][:, :3] = project_points_by_matrix(extracted_outs['cluster_pts_xyz'][:, :3], cp_trans_matrix)
    
    def get_ego_mask(self, batch_idx, ego_idx):
        """
        args:
        - batch_idx: [N], input batch idx with clusters/points 
        - ego_idx: [M], ego idx in batch idx
        """
        ego_mask = batch_idx.unsqueeze(1).expand(
            batch_idx.shape[0], ego_idx.shape[0])
        ego_mask = torch.any((ego_mask == ego_idx), dim=-1)
        return ego_mask
    
    def remove_noego_pts_values(self, dict_to_sample, sampled_out, extracted_outs, ego_bs_idx):
        """ 
        """
        ego_cluster_pts_mask = self.get_ego_mask(
            extracted_outs['cluster_pts_inds'][:, 1], ego_bs_idx)
        for key in self.cluster_pts_wise_keys:
            extracted_outs[key] = extracted_outs[key][ego_cluster_pts_mask]
        
        ego_pts_mask = self.get_ego_mask(
            dict_to_sample['batch_idx'], ego_bs_idx)
        for key in self.points_wise_keys:
            dict_to_sample[key] = dict_to_sample[key][ego_pts_mask]
        sampled_out['fg_mask_list'][0] = sampled_out['fg_mask_list'][0][ego_pts_mask] 
        
    def remove_noego_cluster_values(self, dict_to_sample, sampled_out, extracted_outs, ego_bs_idx):
        ego_cluster_mask = self.get_ego_mask(extracted_outs['cluster_inds'][:, 1], ego_bs_idx)
        for key in self.cluster_wise_keys:
            extracted_outs[key] = extracted_outs[key][ego_cluster_mask]
    
    def get_noego_values(self, dict_to_sample, sampled_out, extracted_outs, ego_bs_idx, record_len):
        """
        """
        noego_feats = {}
        ego_cluster_mask = self.get_ego_mask(
            extracted_outs['cluster_inds'][:, 1], ego_bs_idx)
        for key in self.cluster_wise_keys:
            noego_feats[key] = extracted_outs[key][~ego_cluster_mask]
        
        ego_cluster_pts_mask = self.get_ego_mask(
            extracted_outs['cluster_pts_inds'][:, 1], ego_bs_idx)
        for key in self.cluster_pts_wise_keys:
            noego_feats[key] = extracted_outs[key][~ego_cluster_pts_mask]
        if self.use_score_sample or self.use_both_sample:
            noego_feats['cluster_pts_score'] = sampled_out['seg_logits'][0][~ego_cluster_pts_mask]
        
        ego_pts_mask = self.get_ego_mask(
            dict_to_sample['batch_idx'], ego_bs_idx)
        for key in self.points_wise_keys:
            noego_feats[key] = dict_to_sample[key][~ego_pts_mask]
        noego_feats['fg_mask_list'] = sampled_out['fg_mask_list'][0][~ego_pts_mask]
        
        # trans noego batch ids to belong ego id
        mapping2ego_bs = torch.repeat_interleave(ego_bs_idx, record_len)
        noego_feats['cluster_inds'][:, 1] = mapping2ego_bs[noego_feats['cluster_inds'][:, 1].long()].int()
        noego_feats['batch_idx'] = mapping2ego_bs[noego_feats['batch_idx']]
        noego_feats['cluster_pts_inds'][:, 1] = mapping2ego_bs[noego_feats['cluster_pts_inds'][:, 1].long()].int()
        
        # trans noego 
                                                               
        return noego_feats
    
    def trans2ego_feat(self, dict_to_sample, sampled_out, extracted_outs, trans_feats):
        """ 
        concat all trans_feats to dict_to_sample, extracted_outs and sampled_out
        """
        if len(trans_feats) == 0:
            return
        
        for k in self.points_wise_keys:
            dict_to_sample[k] = torch.cat([dict_to_sample[k], trans_feats[k]], dim=0)
        _, bs_sorted_inds = torch.sort(dict_to_sample['batch_idx'], stable=True)
        for k in self.points_wise_keys:
            dict_to_sample[k] = dict_to_sample[k][bs_sorted_inds]
        
        for k in self.cluster_pts_wise_keys:
            extracted_outs[k] = torch.cat([extracted_outs[k], trans_feats[k]], dim=0)
        _, cluster_sorted_inds = torch.sort(extracted_outs['cluster_pts_inds'][:, 1], stable=True)
        for k in self.cluster_pts_wise_keys:
            extracted_outs[k] = extracted_outs[k][cluster_sorted_inds]
            
        sampled_out['fg_mask_list'][0] = torch.cat([sampled_out['fg_mask_list'][0], trans_feats['fg_mask']], dim=0)
        sampled_out['fg_mask_list'][0] = sampled_out['fg_mask_list'][0][bs_sorted_inds] 
        
    def reduce_points(self, noego_feats):
        if self.use_rand_sample:
            # random sample for points
            num_points = noego_feats['cluster_pts_inds'].shape[0]
            topk_count = int(num_points * self.rate)
            idx = torch.randperm(num_points)[:topk_count]
            noego_feats['cluster_pts_inds'] = noego_feats['cluster_pts_inds'][idx]
            noego_feats['cluster_pts_xyz'] = noego_feats['cluster_pts_xyz'][idx]
            noego_feats['cluster_pts_feats'] = noego_feats['cluster_pts_feats'][idx] 
            return

        cluster_pts_xyz = noego_feats['cluster_pts_xyz'][:, :-1].contiguous()
        idx = fps(cluster_pts_xyz, self.rate)

        noego_feats['cluster_pts_inds'] = noego_feats['cluster_pts_inds'][idx]
        noego_feats['cluster_pts_xyz'] = noego_feats['cluster_pts_xyz'][idx]
        noego_feats['cluster_pts_feats'] = noego_feats['cluster_pts_feats'][idx]


        if self.use_score_sample:
            cluster_pts_score = noego_feats['cluster_pts_score'][idx]
            topk_count = int(cluster_pts_score.shape[0] * 0.5)
            _, topk_indices = torch.topk(cluster_pts_score, topk_count, dim=0)
            topk_indices = topk_indices.squeeze(dim=1)
            noego_feats['cluster_pts_inds'] = noego_feats['cluster_pts_inds'][topk_indices]
            noego_feats['cluster_pts_xyz'] = noego_feats['cluster_pts_xyz'][topk_indices]
            noego_feats['cluster_pts_feats'] = noego_feats['cluster_pts_feats'][topk_indices]
            del noego_feats['cluster_pts_score']
        if self.use_density_sample:
            point_xyz = noego_feats['cluster_pts_xyz'][:, :-1]
            den_mask = torch.ones(1, point_xyz.shape[0]).to(point_xyz.device)
            point_density = self.density_estimator.score_samples(
                point_xyz.unsqueeze(0), den_mask, point_xyz.unsqueeze(0) 
            )[0]
            mask_value = point_density.max() + 1
            # mask 10% min density index in case they are irregular point
            _, min_idx = torch.topk(-point_density, int(point_xyz.shape[0] * 0.1), dim=0) 
            point_density[min_idx] = mask_value
            # select points with small density
            topk_count = int(point_xyz.shape[0] * 0.5)
            _, topk_indices = torch.topk(-point_density, topk_count, dim=0)
            noego_feats['cluster_pts_inds'] = noego_feats['cluster_pts_inds'][topk_indices]
            noego_feats['cluster_pts_xyz'] = noego_feats['cluster_pts_xyz'][topk_indices]
            noego_feats['cluster_pts_feats'] = noego_feats['cluster_pts_feats'][topk_indices]
        if self.use_both_sample:
            cluster_pts_score = noego_feats['cluster_pts_score'][idx]
            topk_count = int(cluster_pts_score.shape[0] * 0.5)
            point_xyz = noego_feats['cluster_pts_xyz'][:, :-1]
            den_mask = torch.ones(1, point_xyz.shape[0]).to(point_xyz.device)
            point_density = self.density_estimator.score_samples(
                point_xyz.unsqueeze(0), den_mask, point_xyz.unsqueeze(0) 
            )[0]
            # combine semtanic score and point density
            # min_max normalize for score
            def max_min_norm(A):
                A -= A.min()
                A /= A.max()
                return A
            norm_semantic_score = max_min_norm(cluster_pts_score[:, 0])
            norm_density_score = max_min_norm(-point_density)
            weight = 0.8
            norm_score  = weight * norm_semantic_score + (1 - weight) * norm_density_score
            _, topk_indices = torch.topk(norm_score, topk_count, dim=0)
            noego_feats['cluster_pts_inds'] = noego_feats['cluster_pts_inds'][topk_indices]
            noego_feats['cluster_pts_xyz'] = noego_feats['cluster_pts_xyz'][topk_indices]
            noego_feats['cluster_pts_feats'] = noego_feats['cluster_pts_feats'][topk_indices]
    
    def cluster_trans(self, dict_to_sample, sampled_out, extracted_outs, img_metas):
        # get mapping from car-wise batch to scene-wise batch
        record_len = img_metas[0]['record_len']
        new_img_metas = [img_metas[i] for i in (record_len.cumsum(dim=0) - record_len[0])]
        record_sum = torch.cumsum(record_len, dim=0)
        ego_bs_idx = torch.zeros((record_len.shape[0]), device=record_len.device, dtype=torch.long)
        ego_bs_idx[1:record_sum.shape[0]] = record_sum[:-1]     # get ego idex of car-wise batch
        
        # get noego features and mapping batch-wise ids to belong ego
        noego_feats = self.get_noego_values(dict_to_sample, sampled_out, extracted_outs, ego_bs_idx, record_len) 
        
        # get two-stage trans feature (only cluster global feat, no pts-wise feat)
        if self.two_stage_feat_mode == 'cluster_pts_mean':
            trans_cluster_feat, _ = scatter_v2(noego_feats['cluster_pts_feats'], noego_feats['cluster_pts_inds'], mode='avg', return_inv=False)
        elif self.two_stage_feat_mode == 'cluster_pts_max':
            trans_cluster_feat, _ = scatter_v2(noego_feats['cluster_pts_feats'], noego_feats['cluster_pts_inds'], mode='max', return_inv=False)
        elif self.two_stage_feat_mode == 'cluster_feat':
            trans_cluster_feat = self.point_feat_proj(noego_feats['cluster_feats'])
        else:
            raise NotImplementedError(f'no such two-stage feature mode {self.two_stage_feat_mode}')
        
        # reduce noego point
        # DISABLED: FPS with rate=1.0 samples 100% of points (pointless and slow)
        # Disabling FPS speeds up validation/inference by 4× with zero accuracy impact
        # if not self.training:
        #     self.reduce_points(noego_feats)
        
        # get two-stage trans coors (cluster coors or pts coors)
        if self.two_stage_xyz_mode == 'cluster_mean':
            trans_feats = {
                'seg_feats': noego_feats['cluster_seg_feat'], 
                'seg_points': noego_feats['cluster_mean_xyz'],
                'batch_idx': noego_feats['cluster_inds'][:, 1],
                'cluster_pts_feats': trans_cluster_feat, 
                'cluster_pts_xyz': noego_feats['cluster_mean_xyz'], 
                'cluster_pts_inds': noego_feats['cluster_inds'],
                'fg_mask': torch.ones((noego_feats['cluster_mean_xyz'].shape[0]), 
                    dtype=torch.bool, device=noego_feats['cluster_mean_xyz'].device)
            } 
        elif self.two_stage_xyz_mode == 'cluster_center':
            cluster_xyz_feat = torch.cat([noego_feats['cluster_xyz'], torch.zeros_like(noego_feats['cluster_xyz'][:, 0:1])], dim=-1)
            trans_feats = {
                'seg_feats': noego_feats['cluster_seg_feat'], 
                'seg_points': cluster_xyz_feat, 
                'batch_idx': noego_feats['cluster_inds'][:, 1],
                'cluster_pts_feats': trans_cluster_feat, 
                'cluster_pts_xyz': cluster_xyz_feat, 
                'cluster_pts_inds': noego_feats['cluster_inds'],
                'fg_mask': torch.ones((noego_feats['cluster_xyz'].shape[0]), 
                    dtype=torch.bool, device=noego_feats['cluster_xyz'].device)
            }
        elif self.two_stage_xyz_mode == 'cluster_pts': 
            # when noego cluster num is 0, need special process
            try:
                cluster_num = noego_feats['cluster_inds'][:, 2].max()
            except:
                cluster_num = 0
            # map no ego clusters' batch idx to ego clusters' batch idx
            map_idx = torch.zeros(cluster_num + 1, 
                                  device=noego_feats['cluster_inds'].device, dtype=torch.long)
            map_idx[noego_feats['cluster_inds'][:, 2].long()] = torch.arange(noego_feats['cluster_inds'][:, 2].shape[0], 
                                                                             device=noego_feats['cluster_inds'].device, dtype=torch.long)  

            trans_feats = {
                # use clusters' seg feature replace points' seg feature for noego fore-points
                'seg_feats': noego_feats['cluster_seg_feat'][map_idx[noego_feats['cluster_pts_inds'][:, 2].long()]],    
                # norm fore-points xyz
                'seg_points': noego_feats['cluster_pts_xyz'],
                # batch_idx after map to ego
                'batch_idx': noego_feats['cluster_pts_inds'][:, 1],
                # use clusters' SIR feature replace points' SIR feature for noego fore-points
                'cluster_pts_feats': trans_cluster_feat[map_idx[noego_feats['cluster_pts_inds'][:, 2].long()]],
                # norm fore-points xyz
                'cluster_pts_xyz': noego_feats['cluster_pts_xyz'],
                # batch_idx after map to ego
                'cluster_pts_inds': noego_feats['cluster_pts_inds'],
                # all points are fore-points
                'fg_mask': torch.ones((noego_feats['cluster_pts_xyz'].shape[0]),
                    dtype=torch.bool, device=noego_feats['cluster_pts_xyz'].device) 
            }
        else:
            raise NotImplementedError(f'no such {self.two_stage_trans_mode} two stage trans mode')
        
        if not self.training:
            MetricUtil.record_volume(trans_feats['cluster_pts_xyz'].shape[0], 
                                     trans_cluster_feat.shape[0],
                                     3,
                                     trans_cluster_feat.shape[1])
        
        if not self.no_coop:
            self.trans2ego_feat(dict_to_sample, sampled_out, extracted_outs, trans_feats)
        self.remove_noego_pts_values(dict_to_sample, sampled_out, extracted_outs, ego_bs_idx)
        if VisUtil.is_vis_now() and VisUtil.fusion_debug:
            VisUtil.set_args(
                scene_points=[
                    dict_to_sample['seg_points'],
                    noego_feats['cluster_pts_xyz']
                ],
            )
            VisUtil.vis('bev', False, False, False)
       
        # change batch idx from car-wise to scene-wise
        if not self.no_coop:
            extracted_outs['cluster_inds'][:, 1] = torch.searchsorted(record_sum, 
                extracted_outs['cluster_inds'][:, 1].contiguous(), right=True) 
            extracted_outs['cluster_pts_inds'][:, 1] = torch.searchsorted(record_sum, 
                extracted_outs['cluster_pts_inds'][:, 1].contiguous(), right=True) 
            dict_to_sample['batch_idx'] = torch.searchsorted(record_sum, dict_to_sample['batch_idx'].contiguous(), right=True)
        else:
            self.remove_noego_cluster_values(dict_to_sample, sampled_out, extracted_outs, ego_bs_idx)
                    
        return new_img_metas
    
    def cluster_connect(self, extracted_outs):
        
        cluster_attrs_key = ['cluster_feats', 'cluster_xyz', 'cluster_seg_score']
        cluster_xyz = extracted_outs['cluster_xyz']
        cluster_inds = extracted_outs['cluster_inds'] # [class, batch, groups]
        
        # find connection based on center distance
        cluster_c_inds = find_connected_componets(cluster_xyz, cluster_inds[:, 1], self.connect_center_dist)
        # aggregate clusters attributes based on mean pooling
        attris_ch_num = [0] + [extracted_outs[k].shape[1] for k in cluster_attrs_key]
        attris_ch_num = np.cumsum(attris_ch_num)
        cluster_attrs = torch.cat([extracted_outs[k] for k in cluster_attrs_key], dim=1)
        cluster_c_attris,  _, inv_c_inds = scatter_v2(cluster_attrs, cluster_c_inds, mode='mean', return_inv=True)
        # update cluster attributes
        for i, k in enumerate(cluster_attrs_key):
            extracted_outs[k] = cluster_c_attris[:, attris_ch_num[i]:attris_ch_num[i+1]]
            
        # update cluster_inds by first apperaed batch idx
        new_cluster_num = cluster_c_inds.max().item() + 1
        one_hot = F.one_hot(cluster_c_inds.long(), num_classes=new_cluster_num)
        # [class, batch, groups]
        cluster_c_inds = torch.stack([torch.zeros((new_cluster_num), device=cluster_inds.device),
            cluster_inds[torch.argmax(one_hot, dim=0), 1], 
            torch.arange(new_cluster_num, device=cluster_inds.device)], dim=1)
        extracted_outs['cluster_inds'] = cluster_c_inds
        
        # update point-wise clsuter ids
        # TODO: some bug here
        new_cluster_pts_inds = inv_c_inds[extracted_outs['cluster_pts_inds'][:, 2].long()]
        extracted_outs['cluster_pts_inds'][:, 2] = new_cluster_pts_inds
        return new_cluster_num