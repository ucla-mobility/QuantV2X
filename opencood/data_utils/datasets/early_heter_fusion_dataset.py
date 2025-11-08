# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# heter functionality added by Aiden Wong <aidenwong@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import numpy as np
from opencood.utils.pcd_utils import downsample_lidar_minimum
import math
from collections import OrderedDict

from opencood.utils import box_utils
from opencood.utils.common_utils import merge_features_to_dict, compute_iou, convert_format
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2, get_pairwise_transformation
from opencood.utils.common_utils import read_json
from opencood.utils.common_utils import merge_features_to_dict
from opencood.utils.heter_utils import Adaptor

def getEarlyheterFusionDataset(cls):
    class EarlyheterFusionDataset(cls):
        """
        This dataset is used for early fusion, where each CAV transmit the raw
        point cloud to the ego vehicle.
        """
        def __init__(self, params, visualize, train=True, calibrate=False):
            super().__init__(params, visualize, train, calibrate)
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
            assert self.supervise_single is False
            self.proj_first = False if 'proj_first' not in params['fusion']['args']\
                                         else params['fusion']['args']['proj_first']

            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.heterogeneous = True
            self.modality_assignment = None if ('assignment_path' not in params['heter'] or params['heter']['assignment_path'] is None) \
                                            else read_json(params['heter']['assignment_path'])
            self.ego_modality = params['heter']['ego_modality'] # "m1" or "m1&m2" or "m3"

            self.modality_name_list = list(params['heter']['modality_setting'].keys())
            self.sensor_type_dict = OrderedDict()
            
            lidar_channels_dict = params['heter'].get('lidar_channels_dict', OrderedDict())
            mapping_dict = params['heter']['mapping_dict']
            cav_preference = params['heter'].get("cav_preference", None)

            self.adaptor = Adaptor(self.ego_modality, 
                                   self.modality_name_list,
                                   self.modality_assignment,
                                   lidar_channels_dict,
                                   mapping_dict,
                                   cav_preference,
                                   train,
                                   calibrate)

            for modality_name, modal_setting in params['heter']['modality_setting'].items():
                self.sensor_type_dict[modality_name] = modal_setting['sensor_type']
                if modal_setting['sensor_type'] == 'lidar':
                    setattr(self, f"pre_processor_{modality_name}", build_preprocessor(modal_setting['preprocess'], train))

                else:
                    raise("Don't support this type of sensor, early fusion only supports lidar")

            self.reinitialize()


        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)

            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_cav_base = cav_content
                    break

            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            projected_lidar_stack = []
            object_stack = []
            object_id_stack = []
            exclude_agent = []
            selected_cavs = OrderedDict()

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)
                if distance > self.params['comm_range']:
                    exclude_agent.append(cav_id)
                    continue

                # if modality not match
                if self.adaptor.unmatched_modality(selected_cav_base['modality_name']):
                    exclude_agent.append(cav_id)
                    continue
                selected_cavs[cav_id] = selected_cav_base
                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base,
                    ego_cav_base)

                # all these lidar and object coordinates are projected to ego
                # already.
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']

            if len(selected_cavs) == 0:
                return None

            # exclude all repetitive objects
            unique_indices = \
                [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1

            # convert list to numpy array, (N, 4)
            projected_lidar_stack = np.vstack(projected_lidar_stack)

            # data augmentation
            projected_lidar_stack, object_bbx_center, mask = \
                self.augment(projected_lidar_stack, object_bbx_center, mask)

            # we do lidar filtering in the stacked lidar
            projected_lidar_stack = mask_points_by_range(projected_lidar_stack,
                                                        self.params['preprocess'][
                                                            'cav_lidar_range'])
            # augmentation may remove some of the bbx out of range
            object_bbx_center_valid = object_bbx_center[mask == 1]
            object_bbx_center_valid, range_mask = \
                box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid,
                                                        self.params['preprocess'][
                                                            'cav_lidar_range'],
                                                        self.params['postprocess'][
                                                            'order'],
                                                        return_mask=True
                                                        )
            mask[object_bbx_center_valid.shape[0]:] = 0
            object_bbx_center[:object_bbx_center_valid.shape[0]] = \
                object_bbx_center_valid
            object_bbx_center[object_bbx_center_valid.shape[0]:] = 0
            unique_indices = list(np.array(unique_indices)[range_mask])

            # pre-process the lidar to voxel/bev/downsampled lidar
            lidar_dict = self.pre_processor.preprocess(projected_lidar_stack)

            # generate the anchor boxes
            anchor_box = self.post_processor.generate_anchor_box()

            # generate targets label
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=anchor_box,
                    mask=mask)

            pairwise_t_matrix = get_pairwise_transformation(selected_cavs, self.max_cav, self.proj_first)
            lidar_agent = np.ones((len(selected_cavs),), dtype=np.float32)
            lidar_pose = np.array([cav['params']['lidar_pose'] for cav in selected_cavs.values()])
            lidar_pose_clean = np.array([cav['params'].get('lidar_pose_clean',
                                                           cav['params']['lidar_pose']) for cav in selected_cavs.values()])
            agent_modality_list = [cav['modality_name'] for cav in selected_cavs.values()]

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': anchor_box,
                'processed_lidar': lidar_dict,
                'label_dict': label_dict,
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_agent': lidar_agent,
                'lidar_pose': lidar_pose,
                'lidar_pose_clean': lidar_pose_clean,
                'agent_modality_list': agent_modality_list,
                'cav_num': len(selected_cavs)})

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar':
                                                    projected_lidar_stack})

            return processed_data_dict

        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            Project the lidar and bbx to ego space first, and then do clipping.

            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
            ego_pose : list
                The ego vehicle lidar pose under world coordinate.

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}
            ego_pose = ego_cav_base['params']['lidar_pose']
            ego_pose_clean = ego_cav_base['params'].get('lidar_pose_clean', ego_pose)

            # calculate the transformation matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['params']['lidar_pose'],
                        ego_pose)

            cav_pose_clean = selected_cav_base['params'].get('lidar_pose_clean',
                                                             selected_cav_base['params']['lidar_pose'])
            transformation_matrix_clean = \
                x1_to_x2(cav_pose_clean, ego_pose_clean)

            # retrieve objects under ego coordinates
            object_bbx_center, object_bbx_mask, object_ids = \
                self.generate_object_center([selected_cav_base],
                                                        ego_pose)
            modality_name = selected_cav_base['modality_name']
            sensor_type = self.sensor_type_dict[modality_name]

            if sensor_type == "lidar" or self.visualize:
                # filter lidar
                lidar_np = selected_cav_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                # remove points that hit itself
                lidar_np = mask_ego_points(lidar_np)
                # project the lidar to ego space
                lidar_np[:, :3] = \
                    box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                            transformation_matrix)

            selected_cav_processed.update(
                {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
                'object_ids': object_ids,
                'projected_lidar': lidar_np,
                'transformation_matrix': transformation_matrix,
                'transformation_matrix_clean': transformation_matrix_clean})

            return selected_cav_processed

        def collate_batch_test(self, batch):
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            return self.collate_batch_train(batch)
        
        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
            image_inputs_list = []
            label_dict_list = []
            origin_lidar = []
            lidar_agent_list = []
            pairwise_t_matrix_list = []
            record_len_list = []
            agent_modality_total = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            record_len_list = []
            
            ### 2022.10.10 single gt ####
            if self.supervise_single:
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                if self.load_lidar_file:
                    processed_lidar_list.append(ego_dict['processed_lidar'])
                if self.load_camera_file:
                    image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.
                
                label_dict_list.append(ego_dict['label_dict'])
                if self.heterogeneous:
                    lidar_agent_list.append(ego_dict.get('lidar_agent', np.ones(1, dtype=np.float32)))
                pairwise_t_matrix_list.append(ego_dict.get('pairwise_t_matrix', np.tile(np.eye(4), (self.max_cav, self.max_cav, 1, 1))))
                record_len_list.append(ego_dict.get('cav_num', 1))
                agent_modality_total.extend(ego_dict.get('agent_modality_list', ['m1'] * ego_dict.get('cav_num', 1)))
                lidar_pose_list.append(ego_dict.get('lidar_pose', np.zeros((ego_dict.get('cav_num', 1), 6))))
                lidar_pose_clean_list.append(ego_dict.get('lidar_pose_clean', np.zeros((ego_dict.get('cav_num', 1), 6))))

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

                ### 2022.10.10 single gt ####
                if self.supervise_single:
                    pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                    neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                    targets_single.append(ego_dict['single_label_dict_torch']['targets'])

                # heterogeneous
            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_lidar_list)

                processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(merged_feature_dict)
                output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})

            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')

                output_dict['ego'].update({'image_inputs': merged_image_inputs_dict})
            
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask})

            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))
            record_len_tensor = torch.from_numpy(np.array(record_len_list, dtype=int))
            lidar_pose_tensor = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
            lidar_pose_clean_tensor = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
            label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            label_torch_dict['record_len'] = record_len_tensor

            # add pairwise_t_matrix to label dict

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            if len(agent_modality_total) == 0:
                agent_modality_total = ['m1'] * int(record_len_tensor.sum().item())

            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'label_dict': label_torch_dict,
                                    'object_ids': object_ids[0],
                                    'pairwise_t_matrix': pairwise_t_matrix,
                                    'record_len': record_len_tensor,
                                    'lidar_pose': lidar_pose_tensor,
                                    'lidar_pose_clean': lidar_pose_clean_tensor,
                                    'agent_modality_list': agent_modality_total,
                                    'anchor_box': self.anchor_box_torch})

            transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
            transformation_matrix_clean_torch = torch.from_numpy(np.identity(4)).float()
            output_dict['ego'].update({
                'transformation_matrix': transformation_matrix_torch,
                'transformation_matrix_clean': transformation_matrix_clean_torch
            })


            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.supervise_single:
                output_dict['ego'].update({
                    "label_dict_single" : 
                        {"pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                        "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                        "targets": torch.cat(targets_single, dim=0)}
                })

            if self.heterogeneous:
                output_dict['ego'].update({
                    "lidar_agent_record": torch.from_numpy(np.concatenate(lidar_agent_list)) # [0,1,1,0,1...]
                })

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

    return EarlyheterFusionDataset
