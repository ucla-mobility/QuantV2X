"""
Dataset class for intermediate fusion (DAIR-V2X)
"""
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

import opencood.data_utils.post_processor as post_processor
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils import box_utils
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, muilt_coord, tfm_to_pose


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data


def veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file, novatel_to_world_json_file):
    matrix = np.empty([4, 4])
    rotationA2B = lidar_to_novatel_json_file["transform"]["rotation"]
    translationA2B = lidar_to_novatel_json_file["transform"]["translation"]
    rotationB2C = novatel_to_world_json_file["rotation"]
    translationB2C = novatel_to_world_json_file["translation"]
    rotation, translation = muilt_coord(rotationA2B, translationA2B, rotationB2C, translationB2C)
    matrix[0:3, 0:3] = rotation
    matrix[:, 3][0:3] = np.array(translation)[:, 0]
    matrix[3, 0:3] = 0
    matrix[3, 3] = 1

    return matrix


def inf_side_rot_and_trans_to_trasnformation_matrix(json_file, system_error_offset):
    matrix = np.empty([4,4])
    matrix[0:3, 0:3] = json_file["rotation"]
    translation = np.array(json_file["translation"])
    translation[0][0] = translation[0][0] + system_error_offset["delta_x"]
    translation[1][0] = translation[1][0] + system_error_offset["delta_y"]  # translation shape (3,1)
    matrix[:, 3][0:3] = translation[:, 0]
    matrix[3, 0:3] = 0
    matrix[3, 3] = 1

    return matrix


class IntermediateFusionDatasetDAIR(Dataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        if 'dair_params' in params:
            self.car_only = params['dair_params']['car_only']
            self.copy_v_to_x = params['dair_params']['copy_v_to_x']
        else:
            self.car_only = False
            self.copy_v_to_x = True

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 2
        else:
            self.max_cav = params['train_params']['max_cav']
        assert self.max_cav >= 2

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        self.proj_first =  params['fusion']['args'].get('proj_first', False)
        
        # if use_single_label, then load car-level label from ego own json
        # else, project scene-level coop label to each ego
        self.use_single_label = params['fusion']['args'].get('use_single_label', False)

        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)
        
        self.no_collaboration = params['fusion']['args'].get('no_collaboration', False)

        self.root_dir = params['dair_data_dir']
        split = 'train' if train or 'pseudo_label_generation' in params else 'val'
        self.split_info = load_json(os.path.join(self.root_dir, f'{split}.json'))
        co_datainfo = load_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        self.co_data = OrderedDict()
        for frame_info in co_datainfo:
            veh_frame_id = frame_info['vehicle_pointcloud_path'].split("/")[-1].replace(".pcd", "")
            self.co_data[veh_frame_id] = frame_info

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        def _ensure_points_array(points_like):
            # Accept (points, meta) or [points, ...] or points
            if isinstance(points_like, (tuple, list)):
                for x in points_like:
                    if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] >= 3:
                        return x
                # fallback: try to coerce
                return np.asarray(points_like)
            return points_like
        
        veh_frame_id = self.split_info[idx]
        frame_info = self.co_data[veh_frame_id]
        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()
        data[0] = OrderedDict() # veh-side
        data[0]['ego'] = True
        data[1] = OrderedDict() # inf-side
        data[1]['ego'] = False
 
        data[0]['params'] = OrderedDict()
        data[0]['params']['vehicles_coop'] = load_json(os.path.join(self.root_dir, frame_info['cooperative_label_path']))

        if self.car_only:
            data[0]['params']['vehicles_coop'] = [v for v in data[0]['params']['vehicles_coop'] if v['type'].lower() == 'car']

        lidar_to_novatel_json_file = load_json(os.path.join(self.root_dir,f'vehicle-side/calib/lidar_to_novatel/{veh_frame_id}.json'))
        novatel_to_world_json_file = load_json(os.path.join(self.root_dir,f'vehicle-side/calib/novatel_to_world/{veh_frame_id}.json'))
        transformation_matrix_v = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file,
                                                                                novatel_to_world_json_file)
        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix_v)

        ######################## Single View GT ########################
        vehicle_side_path = os.path.join(self.root_dir, f'vehicle-side/label/lidar/{veh_frame_id}.json')
        
        data[0]['params']['vehicles'] = load_json(vehicle_side_path)

        if self.car_only:
            data[0]['params']['vehicles'] = [v for v in data[0]['params']['vehicles'] if v['type'].lower() == 'car']
        ######################## Single View GT ########################

        raw_v = pcd_utils.read_pcd(os.path.join(self.root_dir, frame_info["vehicle_pointcloud_path"]))
        data[0]['lidar_np'] = _ensure_points_array(raw_v).astype(np.float32)
        assert isinstance(data[0]['lidar_np'], np.ndarray) and data[0]['lidar_np'].ndim == 2

        data[1]['params'] = OrderedDict()
        inf_frame_id = frame_info['infrastructure_pointcloud_path'].split("/")[-1].replace(".pcd", "")
        data[1]['params']['vehicles_coop'] = [] # we only load cooperative once in veh-side
        virtuallidar_to_world_json_file = load_json(os.path.join(self.root_dir,
                                                                 f'infrastructure-side/calib/virtuallidar_to_world/{inf_frame_id}.json'))
        transformation_matrix_i = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world_json_file, system_error_offset)
        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix_i)

        ######################## Single View GT ########################
        infra_side_path = os.path.join(self.root_dir, f'infrastructure-side/label/virtuallidar/{inf_frame_id}.json')
        
        data[1]['params']['vehicles'] = load_json(infra_side_path)

        if self.car_only:
            data[1]['params']['vehicles'] = [v for v in data[1]['params']['vehicles'] if v['type'].lower() == 'car']
        ######################## Single View GT ########################
        # TODO: v use real lidar, i use time delay lidar
        # i: gt:    real lidar
        #    input: time delay lidar
        raw_i = pcd_utils.read_pcd(os.path.join(self.root_dir, frame_info["infrastructure_pointcloud_path"]))
        data[1]['lidar_np'] = _ensure_points_array(raw_i).astype(np.float32)
        assert isinstance(data[1]['lidar_np'], np.ndarray) and data[1]['lidar_np'].ndim == 2

        return data

    def __len__(self):
        return len(self.split_info)

    def reinitialize(self):
        """
        Reinitialize the dataset between epochs.
        This is called by the training loop but is not needed for DAIR-V2X.
        """
        pass

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask

    def get_unique_label(self, object_stack, object_id_stack):
        # IoU
        object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])

        if len(object_stack) > 0:
            # exclude all repetitive objects    
            unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack) if len(object_stack) > 1 else object_stack[0]
            object_stack = object_stack[unique_indices]
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1
            updated_object_id_stack = [object_id_stack[i] for i in unique_indices]
        else:
            updated_object_id_stack = object_id_stack

        return object_bbx_center, mask, updated_object_id_stack

    @staticmethod
    def add_noise_data_dict(data_dict, noise_setting):
        """ Update the base data dict.
            We retrieve lidar_pose and add_noise to it.
            And set a clean pose.
        """
        if noise_setting['loc_err']:
            for cav_id, cav_content in data_dict.items():
                cav_content['params']['lidar_pose_clean'] = cav_content['params']['lidar_pose'] # 6 dof pose
                xy_noise = np.random.normal(0, noise_setting['xyz_std'], 2)
                yaw_noise = np.random.normal(0, noise_setting['ryp_std'], 1)
                cav_content['params']['lidar_pose'][:2] += xy_noise
                cav_content['params']['lidar_pose'][5] += yaw_noise
        else:
            for cav_id, cav_content in data_dict.items():
                cav_content['params']['lidar_pose_clean'] = cav_content['params']['lidar_pose'] # 6 dof pose

        return data_dict

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)

        base_data_dict = self.add_noise_data_dict(base_data_dict,self.params['wild_setting'])

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {'frame_id': self.split_info[idx]}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                break
            
        assert cav_id == 0, "0 must be ego"
        assert ego_id == 0
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix = self.get_pairwise_transformation(base_data_dict, self.max_cav)

        processed_features = []
        object_stack = []
        object_id_stack = []
        object_stack_single_v = []
        object_id_stack_single_v = []
        object_stack_single_i = []
        object_id_stack_single_i = []
        lidar_pose_list = []
        lidar_pose_clean_list = []
        cav_id_list = []

        # For pseudo label
        object_stack_low_thresh = []
        object_id_stack_low_thresh = []

        # prior knowledge for time delay correction and indicating data type
        # (V2V vs V2i)
        velocity = []
        time_delay = []
        infra = []
        spatial_correction_matrix = []            

        if self.visualize:
            projected_lidar_stack = []

        if self.copy_v_to_x:
            base_data_dict[1]['params']['vehicles_coop'] = base_data_dict[0]['params']['vehicles_coop']
        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
            lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
            cav_id_list.append(cav_id)

        for cav_id in cav_id_list:
            selected_cav_base = base_data_dict[cav_id]
            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose, 
                ego_lidar_pose_clean)
                
            object_stack.append(selected_cav_processed['object_bbx_center_coop'])
            object_id_stack += selected_cav_processed['object_ids_coop']

            ######################## Single View GT ########################
            if cav_id == 0:
                object_stack_single_v.append(selected_cav_processed['object_bbx_center'])
                object_id_stack_single_v += selected_cav_processed['object_ids']
            else:
                object_stack_single_i.append(selected_cav_processed['object_bbx_center'])
                object_id_stack_single_i += selected_cav_processed['object_ids']
            ######################## Single View GT ########################
            # convert i to v pose
            if self.proj_first and cav_id == 1:
                all_car_i = object_stack_single_i[0]
                
                lidar_to_world = x_to_world(ego_lidar_pose_clean)
                world_to_lidar = np.linalg.inv(lidar_to_world)
                pose_v = lidar_pose_clean_list[0]
                pose_i = lidar_pose_clean_list[1]
                transforms = x1_to_x2(pose_i, pose_v)
                
                all_car_i_corners = box_utils.boxes_to_corners_3d(all_car_i, "lwh") # TODO check
                n, _, _ = all_car_i_corners.shape
                all_car_i_corners = all_car_i_corners.reshape(n*8, 3)
                all_car_v_corners = box_utils.project_points_by_matrix_torch(all_car_i_corners, transforms)
                all_car_v_corners = all_car_v_corners.reshape(n, 8, 3)
                all_car_v = box_utils.corner_to_center(all_car_v_corners)
                object_stack_single_i = [all_car_v]

            processed_features.append(selected_cav_processed['processed_features'])

            velocity.append(0)
            time_delay.append(0)

            spatial_correction_matrix.append(np.eye(4))
            infra.append(1 if int(cav_id) == 1 else 0)

            if self.visualize:
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])

        lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
        lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
        
        # TODO new added for objects idex (check twice)
        object_ids_set = set(object_id_stack)
        object_cum_nums = np.array([object_stack.shape[0] for object_stack in object_stack])
        object_cum_nums = np.cumsum(object_cum_nums)
        
        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in object_ids_set]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]
        
        # record the object id mapping to car id
        object_bbx_idx = np.zeros((self.params['postprocess']['max_num'],
                                   self.params['train_params']['max_cav']))
        object_ids_list = np.array(object_id_stack)
        for i, obj_id in enumerate(object_ids_set):
            scene_idxs = np.where(object_ids_list == obj_id)[0]
            scene_idxs = np.searchsorted(object_cum_nums, scene_idxs, side='left')
            object_bbx_idx[i][scene_idxs] = 1
        

        object_bbx_center, mask, object_id_stack = self.get_unique_label(object_stack, object_id_stack)
        
        ######################## Single View GT ########################
        object_bbx_center_single_v, mask_single_v, object_id_stack_single_v = self.get_unique_label(object_stack_single_v, object_id_stack_single_v)
        object_bbx_center_single_i, mask_single_i, object_id_stack_single_i = self.get_unique_label(object_stack_single_i, object_id_stack_single_i)
        ######################## Single View GT ########################

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(processed_features)
        merged_feature_dict = self.merge_features_to_dict(processed_features)

        # generate the anchor boxes
        if self.params['postprocess'].get('use_anchor_box', True):
            anchor_box = self.post_processor.generate_anchor_box()
            processed_data_dict['ego'].update({'anchor_box': anchor_box})

        # generate targets label
        if self.params['postprocess'].get('use_target_label', True):
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=anchor_box,
                    mask=mask)
            processed_data_dict['ego'].update({'label_dict': label_dict})
            
            label_dict_single_v = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center_single_v,
                    anchors=anchor_box,
                    mask=mask_single_v)
            processed_data_dict['ego'].update({'label_dict_single_v': label_dict_single_v})
            
            label_dict_single_i = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center_single_i,
                    anchors=anchor_box,
                    mask=mask_single_i)
            processed_data_dict['ego'].update({'label_dict_single_i': label_dict_single_i})
                

        # pad dv, dt, infra to max_cav
        velocity = velocity + (self.max_cav - len(velocity)) * [0.]
        time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
        infra = infra + (self.max_cav - len(infra)) * [0.]
        spatial_correction_matrix = np.stack(spatial_correction_matrix)
        padding_eye = np.tile(np.eye(4)[None],(self.max_cav - len(
                                               spatial_correction_matrix),1,1))
        spatial_correction_matrix = np.concatenate([spatial_correction_matrix,
                                                   padding_eye], axis=0)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_bbx_idx': object_bbx_idx,
             'object_ids': object_id_stack,
             # TODO new added property
             'object_bbx_center_single_v': object_bbx_center_single_v,
             'object_bbx_mask_single_v': mask_single_v,
             'object_ids_single_v': object_id_stack_single_v,
             'object_bbx_center_single_i': object_bbx_center_single_i,
             'object_bbx_mask_single_i': mask_single_i,
             'object_ids_single_i': object_id_stack_single_i,
             
             'processed_lidar': merged_feature_dict,
             'cav_num': cav_num,
             'velocity': velocity,
             'time_delay': time_delay,
             'infra': infra,
             'spatial_correction_matrix': spatial_correction_matrix,
             'pairwise_t_matrix': pairwise_t_matrix,
             # TODO new added property
             'lidar_poses_clean': lidar_poses_clean,
             'lidar_poses': lidar_poses})

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(
                    projected_lidar_stack)})

            processed_data_dict['ego'].update({'origin_lidar_v':
                    projected_lidar_stack[0]})
            processed_data_dict['ego'].update({'origin_lidar_i':
                    projected_lidar_stack[1]})
            
        if self.params['postprocess'].get('coord_type', None) == 'right':
            # mmdet3d use right hand coordinate, need to -yaw
            object_bbx_center[:, -1] = -object_bbx_center[:, -1]
            object_bbx_center_single_v[:, -1] = -object_bbx_center_single_v[:, -1]
            object_bbx_center_single_i[:, -1] = -object_bbx_center_single_i[:, -1]

        return processed_data_dict
    
    def get_item_single_car(self, selected_cav_base, ego_pose, ego_pose_clean):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list, length 6
            The ego vehicle lidar pose under world coordinate with noise.
        ego_pose_clean : list, length 6
            The ego vehicle lidar pose under world coordinate without noise. Only used for gt box generation.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = \
            x1_to_x2(selected_cav_base['params']['lidar_pose'],
                     ego_pose) # T_ego_cav
        transformation_matrix_clean = \
            x1_to_x2(selected_cav_base['params']['lidar_pose_clean'], ego_pose_clean)

        # retrieve objects under ego coordinates
        # this is used to generate accurate GT bounding box.
        object_bbx_center_coop, object_bbx_mask_coop, object_ids_coop = \
            self.post_processor.generate_object_center_dairv2x([selected_cav_base], ego_pose_clean)

        if self.use_single_label:
            # load from own label json
            object_bbx_center, object_bbx_mask, object_ids = \
                self.post_processor.generate_object_center_dairv2x_late_fusion([selected_cav_base])
        else:
            # project cop label to own pose
            object_bbx_center, object_bbx_mask, object_ids = \
                self.post_processor.generate_object_center_dairv2x([selected_cav_base], selected_cav_base['params']['lidar_pose'])

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        if self.visualize:
            # project first to vis on ego
            vis_lidar = box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                            transformation_matrix)
            vis_lidar = np.concatenate([vis_lidar, lidar_np[:, -1:]], axis=-1)
            selected_cav_processed.update({'projected_lidar': vis_lidar})
        
        if self.proj_first:
            lidar_np[:, :3] = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                         transformation_matrix)
            
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess']['cav_lidar_range'])
        processed_lidar = self.pre_processor.preprocess(lidar_np)


        selected_cav_processed.update(
            {'object_bbx_center_coop': object_bbx_center_coop[object_bbx_mask_coop == 1],
             'object_ids_coop': object_ids_coop,
             'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'processed_features': processed_lidar,
             'transformation_matrix': transformation_matrix,
             'transformation_matrix_clean': transformation_matrix_clean})
        
        return selected_cav_processed

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    # merged_feature_dict['coords'] = [f1,f2,f3,f4]
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict

    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_bbx_idx = []
        object_ids = []
        label_dict_list = []

        ######################## Single View GT ########################
        object_bbx_center_single_v = []
        object_bbx_mask_single_v = []
        object_ids_single_v = []
        label_dict_list_single_v = []

        object_bbx_center_single_i = []
        object_bbx_mask_single_i = []
        object_ids_single_i = []
        label_dict_list_single_i = []
        ######################## Single View GT ########################

        # used for PriorEncoding for models
        velocity = []
        time_delay = []
        infra = []
        
        processed_lidar_list = []
        # used to record different scenario
        record_len = []
        lidar_pose_list = []
        lidar_pose_clean_list = []
        
        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        spatial_correction_matrix_list = []

        frame_ids = []

        if self.visualize:
            origin_lidar = []
            origin_lidar_v = []
            origin_lidar_i = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            frame_ids.append(ego_dict['frame_id'])
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_bbx_idx.append(ego_dict['object_bbx_idx'])
            object_ids.append(ego_dict['object_ids'])
            if ego_dict.get('label_dict', None) is not None:
                label_dict_list.append(ego_dict['label_dict'])

            ######################## Single View GT ########################
            object_bbx_center_single_v.append(ego_dict['object_bbx_center_single_v'])
            object_bbx_mask_single_v.append(ego_dict['object_bbx_mask_single_v'])
            object_ids_single_v.append(ego_dict['object_ids_single_v'])
            if ego_dict.get('label_dict_single_v', None) is not None:
                label_dict_list_single_v.append(ego_dict['label_dict_single_v'])

            object_bbx_center_single_i.append(ego_dict['object_bbx_center_single_i'])
            object_bbx_mask_single_i.append(ego_dict['object_bbx_mask_single_i'])
            object_ids_single_i.append(ego_dict['object_ids_single_i'])
            if ego_dict.get('label_dict_single_i', None) is not None:
                label_dict_list_single_i.append(ego_dict['label_dict_single_i'])
            ######################## Single View GT ########################
            
            lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
            lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])

            # ego_dict['processed_lidar'] is a dict with keys like 'voxel_features', 'voxel_coords', 'points', etc.
            # The dict contains lists (one entry per CAV in the scene)
            processed_lidar_list.append(ego_dict['processed_lidar'])
            record_len.append(ego_dict['cav_num'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
            velocity.append(ego_dict['velocity'])
            time_delay.append(ego_dict['time_delay'])
            infra.append(ego_dict['infra'])
            spatial_correction_matrix_list.append(ego_dict['spatial_correction_matrix'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])
                origin_lidar_v.append(ego_dict['origin_lidar_v'])
                origin_lidar_i.append(ego_dict['origin_lidar_i'])
        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
        object_bbx_idx = torch.from_numpy(np.array(object_bbx_idx))

        ######################## Single View GT ########################
        object_bbx_center_single_v = torch.from_numpy(np.array(object_bbx_center_single_v))
        object_bbx_mask_single_v = torch.from_numpy(np.array(object_bbx_mask_single_v))

        object_bbx_center_single_i = torch.from_numpy(np.array(object_bbx_center_single_i))
        object_bbx_mask_single_i = torch.from_numpy(np.array(object_bbx_mask_single_i))
        ######################## Single View GT ########################

        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)

        # [sum(record_len), C, H, W]
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)
        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        # [[N1, 6], [N2, 6]...] -> [[N1+N2+...], 6]
        lidar_pose = torch.from_numpy(np.stack(lidar_pose_list, axis=0))
        # TODO: check this unalign with normal dataset
        # change concat to stack
        lidar_pose_clean = torch.from_numpy(np.stack(lidar_pose_clean_list, axis=0))
        
        # (B, max_cav)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))
        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        spatial_correction_matrix = torch.from_numpy(np.array(spatial_correction_matrix_list))

        # (B, max_cav, 3)
        prior_encoding = \
            torch.stack([velocity, time_delay, infra], dim=-1).float()
        if len(label_dict_list) > 0:
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)
            # add pairwise_t_matrix to label dict
            label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            label_torch_dict['record_len'] = record_len
            output_dict['ego'].update({'label_dict': label_torch_dict})
        if len(label_dict_list_single_v) > 0:
            label_torch_dict_single_v = \
                self.post_processor.collate_batch(label_dict_list_single_v)
            label_torch_dict_single_v['pairwise_t_matrix'] = pairwise_t_matrix
            label_torch_dict_single_v['record_len'] = record_len
            output_dict['ego'].update({'label_dict_single_v': label_torch_dict_single_v})
        if len(label_dict_list_single_i) > 0:
            label_torch_dict_single_i = \
                self.post_processor.collate_batch(label_dict_list_single_i)
            label_torch_dict_single_i['pairwise_t_matrix'] = pairwise_t_matrix
            label_torch_dict_single_i['record_len'] = record_len
            output_dict['ego'].update({'label_dict_single_i': label_torch_dict_single_i})
        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'object_bbx_idx': object_bbx_idx,
                                   
                                   'object_bbx_center_single_v': object_bbx_center_single_v,
                                   'object_bbx_mask_single_v': object_bbx_mask_single_v,
                                   'object_ids_single_v': object_ids_single_v[0],
                                   'object_bbx_center_single_i': object_bbx_center_single_i,
                                   'object_bbx_mask_single_i': object_bbx_mask_single_i,
                                   'object_ids_single_i': object_ids_single_i[0],
                                   
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'record_len': record_len,
                                   
                                   'lidar_pose_clean': lidar_pose_clean,
                                   'lidar_pose': lidar_pose,
                                   
                                   'object_ids': object_ids[0],
                                   'prior_encoding': prior_encoding,
                                   'spatial_correction_matrix': spatial_correction_matrix,
                                   'pairwise_t_matrix': pairwise_t_matrix,
                                   'frame_ids': frame_ids})

        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

            origin_lidar_v = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar_v))
            origin_lidar_v = torch.from_numpy(origin_lidar_v)
            output_dict['ego'].update({'origin_lidar_v': origin_lidar_v})
        
            origin_lidar_i = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar_i))
            origin_lidar_i = torch.from_numpy(origin_lidar_i)
            output_dict['ego'].update({'origin_lidar_i': origin_lidar_i})

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)
        if output_dict is None:
            return None

        # check if anchor box in the batch
        if self.params['postprocess'].get('use_anchor_box', True):
            if batch[0]['ego']['anchor_box'] is not None:
                output_dict['ego'].update({'anchor_box':
                    torch.from_numpy(np.array(batch[0]['ego']['anchor_box']))})

        # save the transformation matrix (4, 4) to ego vehicle
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        transformation_matrix_clean_torch = \
            torch.from_numpy(np.identity(4)).float()

        output_dict['ego'].update({'transformation_matrix':
                                       transformation_matrix_torch,
                                   'transformation_matrix_clean':
                                       transformation_matrix_clean_torch,
                                   'frame_id': batch[0]['ego']['frame_id']})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4), L is the max cav number in a scene
            pairwise_t_matrix[i, j] is Tji, i_to_j
        """
        pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1)) # (L, L, 4, 4)

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            # no need to warp again in fusion time.

            # pairwise_t_matrix[:, :] = np.identity(4)
            return pairwise_t_matrix
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                lidar_pose = cav_content['params']['lidar_pose']
                t_list.append(x_to_world(lidar_pose))  # Twx

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i != j:
                        # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                        # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                        t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji   
                        pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix
    
   
    
    
    # TODO new added func
    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points
            with shape (img_row, img_col).

        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)

    def visualize_result(self, pred_box_tensor,
                         pred_score_tensor,
                         gt_tensor,
                         pcd,
                         show_vis,
                         save_path,
                         dataset=None):
        # visualize the model output
        self.post_processor.visualize(pred_box_tensor,
                                      pred_score_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis,
                                      save_path,
                                      dataset=dataset)
