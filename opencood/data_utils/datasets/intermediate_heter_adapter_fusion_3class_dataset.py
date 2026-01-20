# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License
# This is used for STAMP

"""
-*- coding: utf-8 -*-
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
License: TDG-Attribution-NonCommercial-NoDistrib

intermediate heter fusion dataset

Note that for DAIR-V2X dataset,
Each agent should retrieve the objects itself, and merge them by iou, 
instead of using the cooperative label.
"""

import random
import math
from collections import OrderedDict
import numpy as np
import torch
import copy
from icecream import ic
from PIL import Image
import pickle as pkl
from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.common_utils import merge_features_to_dict, compute_iou, convert_format
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.common_utils import read_json
from opencood.utils.heter_utils import Adaptor
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt


def _split_anchor_box(anchor_box):
    if isinstance(anchor_box, tuple):
        return anchor_box[0], anchor_box[1]
    return anchor_box, None


def _anchor_box_to_torch(anchor_box, num_anchors_per_location):
    if anchor_box is None:
        return None
    if num_anchors_per_location is None:
        return torch.from_numpy(anchor_box)
    return torch.from_numpy(np.array(anchor_box)).float()


def getIntermediateheteradapter3classFusionDataset(cls):
    """
    cls: the Basedataset.
    """

    class Intermediateheteradapter3classFusionDataset(cls):
        def __init__(self, params, visualize, train=True, calibrate=False):
            super().__init__(params, visualize, train, calibrate)
            # intermediate and supervise single
            self.supervise_single = (
                True
                if ("supervise_single" in params["model"]["args"] and params["model"]["args"]["supervise_single"])
                else False
            )
            self.proj_first = (
                False if "proj_first" not in params["fusion"]["args"] else params["fusion"]["args"]["proj_first"]
            )
            self.ego_modality = params["heter"]["ego_modality"]  # "m1" or "m1&m2" or "m3"

            if len(self.ego_modality.split("&")) == 1 and not hasattr(self, "post_processor"):
                # define post_processor for the ego modality and protocol modality only.
                self.post_processor = build_postprocessor(
                    params["heter"]["modality_setting"][self.ego_modality]["postprocess"], train
                )
                self.post_process_params = params["heter"]["modality_setting"][self.ego_modality]["postprocess"]
                self.post_processor_protocol = build_postprocessor(
                    params["heter"]["modality_setting"]["m0"]["postprocess"], train
                )
                self.post_process_params_protocol = params["heter"]["modality_setting"]["m0"]["postprocess"]
            else:
                assert hasattr(
                    self, "post_processor"
                ), "post_processor must be defined globally if ego_modality contains multiple modalities"
                self.post_process_params = params["postprocess"]
                if "heter" in params and "modality_setting" in params["heter"] and "m0" in params["heter"]["modality_setting"]:
                    self.post_processor_protocol = build_postprocessor(
                        params["heter"]["modality_setting"]["m0"]["postprocess"], train
                    )
                    self.post_process_params_protocol = params["heter"]["modality_setting"]["m0"]["postprocess"]
                else:
                    self.post_processor_protocol = self.post_processor
                    self.post_process_params_protocol = self.post_process_params

            self.modality_name_list = list(params["heter"]["modality_setting"].keys())

            self.post_processor_dict = dict()
            self.num_anchors_per_location_dict = dict()

            for modality_name in self.modality_name_list:
                self.post_processor_dict[modality_name] = build_postprocessor(
                    params["heter"]["modality_setting"][modality_name]["postprocess"], train
                )
                anchor_box, num_anchors_per_location = _split_anchor_box(
                    self.post_processor_dict[modality_name].generate_anchor_box()
                )
                setattr(
                    self, f"anchor_box_{modality_name}", anchor_box
                )
                self.num_anchors_per_location_dict[modality_name] = num_anchors_per_location

            self.anchor_box, self.num_anchors_per_location = _split_anchor_box(
                self.post_processor.generate_anchor_box()
            )
            self.anchor_box_torch = _anchor_box_to_torch(self.anchor_box, self.num_anchors_per_location)

            self.anchor_box_protocol, self.num_anchors_per_location_protocol = _split_anchor_box(
                self.post_processor_protocol.generate_anchor_box()
            )
            self.anchor_box_torch_protocol = _anchor_box_to_torch(
                self.anchor_box_protocol, self.num_anchors_per_location_protocol
            )

            self.anchor_box_dict = dict()
            self.anchor_box_torch_dict = dict()
            for modality_name in self.modality_name_list + ["m0"]:
                anchor_box = getattr(self, f"anchor_box_{modality_name}", None)
                num_anchors_per_location = self.num_anchors_per_location_dict.get(modality_name)
                if anchor_box is None and modality_name == "m0":
                    anchor_box = self.anchor_box_protocol
                    num_anchors_per_location = self.num_anchors_per_location_protocol
                self.anchor_box_dict[modality_name] = anchor_box
                self.anchor_box_torch_dict[modality_name] = _anchor_box_to_torch(
                    anchor_box, num_anchors_per_location
                )

            # only load camera/depth if any modality uses camera
            self.load_lidar_file = True
            self.load_camera_file = any(
                "camera" in modal_setting["sensor_type"]
                for modal_setting in params["heter"]["modality_setting"].values()
            )
            self.load_depth_file = self.load_camera_file

            # # Here we set heterogeneous to True if heterogeneous is not set to False (including not provided)
            # # This is for keeping the compatibility with the old version of the configuration file.
            # self.heterogeneous = False if ('heterogeneous' in params['model']['args'] and (not params['model']['args']['heterogeneous'])) \
            #                             else True
            self.heterogeneous = True

            if "m0" in self.modality_name_list and len(self.modality_name_list) > 1:
                self.modality_name_list.remove("m0")

            self.sensor_type_dict = OrderedDict()

            lidar_channels_dict = params["heter"].get("lidar_channels_dict", OrderedDict())
            mapping_dict = params["heter"]["mapping_dict"]
            cav_preference = params["heter"].get("cav_preference", None)

            self.modality_assignment = (
                None
                if ("assignment_path" not in params["heter"] or params["heter"]["assignment_path"] is None)
                else read_json(params["heter"]["assignment_path"])
            )
            if self.params['fusion']['dataset'] == 'dairv2x':
                self.modality_assignment = None
                
            self.adaptor = Adaptor(
                self.ego_modality,
                self.modality_name_list,
                self.modality_assignment,
                lidar_channels_dict,
                mapping_dict,
                cav_preference,
                train,
                calibrate,
            )

            for modality_name, modal_setting in params["heter"]["modality_setting"].items():
                self.sensor_type_dict[modality_name] = modal_setting["sensor_type"]
                if "lidar" in modal_setting["sensor_type"]:
                    setattr(
                        self, f"pre_processor_{modality_name}", build_preprocessor(modal_setting["preprocess"], train)
                    )

                if "camera" in modal_setting["sensor_type"]:
                    setattr(self, f"data_aug_conf_{modality_name}", modal_setting["data_aug_conf"])

            self.reinitialize()

            self.kd_flag = params.get("kd_flag", False)

            self.box_align = False
            if "box_align" in params:
                self.box_align = True
                self.stage1_result_path = (
                    params["box_align"]["train_result"] if train else params["box_align"]["val_result"]
                )
                self.stage1_result = read_json(self.stage1_result_path)
                self.box_align_args = params["box_align"]["args"]

            self.visible = params["train_params"].get("visible", False)

        def _collate_label_dict(self, post_processor, label_list):
            if hasattr(post_processor, "collate_batch_stamp"):
                return post_processor.collate_batch_stamp(label_list)
            return post_processor.collate_batch(label_list)

        def _build_label_kwargs(
            self,
            gt_box_center,
            anchors,
            mask,
            dynamic_bev,
            road_bev,
            lane_bev,
            num_anchors_per_location,
        ):
            label_kwargs = {
                "gt_box_center": gt_box_center,
                "anchors": anchors,
                "mask": mask,
                "dynamic_bev": dynamic_bev,
                "road_bev": road_bev,
                "lane_bev": lane_bev,
            }
            if num_anchors_per_location is not None:
                label_kwargs["num_anchors_per_location"] = num_anchors_per_location
            return label_kwargs

        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            base_data_dict = add_noise_data_dict(base_data_dict, self.params["noise_setting"])

            processed_data_dict = OrderedDict()
            processed_data_dict["ego"] = {}

            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content["ego"]:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content["params"]["lidar_pose"]
                    ego_cav_base = cav_content
                    break

            assert cav_id == list(base_data_dict.keys())[0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            agent_modality_list = []
            object_stack = []
            object_id_stack = []
            single_label_list = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            exclude_agent = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            cav_id_list = []
            projected_lidar_clean_list = []  # disconet
            transformation_matrix_list = []
            transformation_matrix_clean_list = []

            if self.visualize or self.kd_flag:
                projected_lidar_stack = []
                input_list_m0_proj = []
                input_list_m1_proj = []  # 2023.8.31 to correct discretization errors with kd flag
                input_list_m2_proj = []
                input_list_m3_proj = []
                input_list_m4_proj = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = math.sqrt(
                    (selected_cav_base["params"]["lidar_pose"][0] - ego_lidar_pose[0]) ** 2
                    + (selected_cav_base["params"]["lidar_pose"][1] - ego_lidar_pose[1]) ** 2
                )

                # if distance is too far, we will just skip this agent
                if distance > self.params["comm_range"]:
                    exclude_agent.append(cav_id)
                    continue

                # if modality not match
                if self.adaptor.unmatched_modality(selected_cav_base["modality_name"]):
                    exclude_agent.append(cav_id)
                    continue

                lidar_pose_clean_list.append(selected_cav_base["params"]["lidar_pose_clean"])
                lidar_pose_list.append(selected_cav_base["params"]["lidar_pose"])  # 6dof pose
                cav_id_list.append(cav_id)

            if len(cav_id_list) == 0:
                return None

            for cav_id in exclude_agent:
                base_data_dict.pop(cav_id)

            ########## Updated by Yifan Lu 2022.1.26 ############
            # box align to correct pose.
            # stage1_content contains all agent. Even out of comm range.
            if self.box_align and str(idx) in self.stage1_result.keys():
                from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np

                stage1_content = self.stage1_result[str(idx)]
                if stage1_content is not None:
                    all_agent_id_list = stage1_content["cav_id_list"]  # include those out of range
                    all_agent_corners_list = stage1_content["pred_corner3d_np_list"]
                    all_agent_uncertainty_list = stage1_content["uncertainty_np_list"]

                    cur_agent_id_list = cav_id_list
                    cur_agent_pose = [base_data_dict[cav_id]["params"]["lidar_pose"] for cav_id in cav_id_list]
                    cur_agnet_pose = np.array(cur_agent_pose)
                    cur_agent_in_all_agent = [
                        all_agent_id_list.index(cur_agent) for cur_agent in cur_agent_id_list
                    ]  # indexing current agent in `all_agent_id_list`

                    pred_corners_list = [
                        np.array(all_agent_corners_list[cur_in_all_ind], dtype=np.float64)
                        for cur_in_all_ind in cur_agent_in_all_agent
                    ]
                    uncertainty_list = [
                        np.array(all_agent_uncertainty_list[cur_in_all_ind], dtype=np.float64)
                        for cur_in_all_ind in cur_agent_in_all_agent
                    ]

                    if sum([len(pred_corners) for pred_corners in pred_corners_list]) != 0:
                        refined_pose = box_alignment_relative_sample_np(
                            pred_corners_list, cur_agnet_pose, uncertainty_list=uncertainty_list, **self.box_align_args
                        )
                        cur_agnet_pose[:, [0, 1, 4]] = refined_pose

                        for i, cav_id in enumerate(cav_id_list):
                            lidar_pose_list[i] = cur_agnet_pose[i].tolist()
                            base_data_dict[cav_id]["params"]["lidar_pose"] = cur_agnet_pose[i].tolist()

            pairwise_t_matrix = get_pairwise_transformation(base_data_dict, self.max_cav, self.proj_first)

            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]

            # merge preprocessed features from different cavs into the same dict
            cav_num = len(cav_id_list)
            
            ##### STAMP #####
            protocol_sensor_type = self.sensor_type_dict["m0"]
            for modality_name in self.modality_name_list:
                exec(f"label_dict_list_{modality_name} = []")

                sensor_type = self.sensor_type_dict[modality_name]
                if sensor_type == "lidar" or sensor_type == "camera":
                    exec(f"input_list_{modality_name} = []")
                elif "camera" in sensor_type and "lidar" in sensor_type:
                    exec(f"input_list_{modality_name} = dict()")
                    eval(f"input_list_{modality_name}")["lidar"] = []
                    eval(f"input_list_{modality_name}")["camera"] = []
                else:
                    raise ValueError("Not support this type of sensor")

            if protocol_sensor_type == "lidar" or protocol_sensor_type == "camera":
                exec(f"input_list_m0 = []")
            elif "camera" in protocol_sensor_type and "lidar" in protocol_sensor_type:
                exec(f"input_list_m0 = dict()")
                eval(f"input_list_m0")["lidar"] = []
                eval(f"input_list_m0")["camera"] = []
            else:
                raise ValueError("Not support this type of sensor")
            ##### STAMP #####

            for _i, cav_id in enumerate(cav_id_list):
                selected_cav_base = base_data_dict[cav_id]
                modality_name = selected_cav_base["modality_name"]
                sensor_type = self.sensor_type_dict[selected_cav_base["modality_name"]]

                # dynamic object center generator! for heterogeneous input
                if not self.visualize:
                    if "lidar" in sensor_type:
                        self.generate_object_center = eval(f"self.generate_object_center_lidar")
                    else:
                        self.generate_object_center = eval(f"self.generate_object_center_camera")
                # need discussion. In test phase, use lidar label.
                else:
                    self.generate_object_center = self.generate_object_center_lidar

                selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_cav_base)
                ##### STAMP #####
                transformation_matrix_list.append(selected_cav_processed["transformation_matrix"])
                transformation_matrix_clean_list.append(selected_cav_processed["transformation_matrix_clean"])
                ##### STAMP #####
                object_stack.append(selected_cav_processed["object_bbx_center"])
                object_id_stack += selected_cav_processed["object_ids"]
                ##### STAMP #####
                eval(f"label_dict_list_{modality_name}").append(selected_cav_processed["single_label_dict"])
                ##### STAMP #####
                
                if sensor_type == "lidar":
                    eval(f"input_list_{modality_name}").append(
                        selected_cav_processed[f"processed_features_{modality_name}"]
                    )
                elif sensor_type == "camera":
                    eval(f"input_list_{modality_name}").append(selected_cav_processed[f"image_inputs_{modality_name}"])
                elif "camera" in sensor_type and "lidar" in sensor_type:
                    eval(f"input_list_{modality_name}")["lidar"].append(
                        selected_cav_processed[f"processed_features_{modality_name}"]
                    )
                    eval(f"input_list_{modality_name}")["camera"].append(
                        selected_cav_processed[f"image_inputs_{modality_name}"]
                    )
                else:
                    raise ValueError("Not support this type of sensor")
                ##### STAMP #####
                sensor_type_protocol = self.sensor_type_dict["m0"]
                if sensor_type_protocol == "lidar":
                    eval(f"input_list_m0").append(selected_cav_processed[f"processed_features_m0"])
                elif sensor_type_protocol == "camera":
                    eval(f"input_list_m0").append(selected_cav_processed[f"image_inputs_m0"])
                elif "camera" in sensor_type_protocol and "lidar" in sensor_type_protocol:
                    eval(f"input_list_m0")["lidar"].append(selected_cav_processed[f"processed_features_m0"])
                    eval(f"input_list_m0")["camera"].append(selected_cav_processed[f"image_inputs_m0"])
                else:
                    raise ValueError("Not support this type of sensor")
                ##### STAMP #####
                agent_modality_list.append(modality_name)

                if self.visualize or self.kd_flag:
                    # heterogeneous setting do not support disconet' kd
                    projected_lidar_stack.append(selected_cav_processed["projected_lidar"])
                    if (sensor_type == "lidar" or "lidar" in sensor_type) and self.kd_flag:
                        eval(f"input_list_{modality_name}_proj").append(
                            selected_cav_processed[f"processed_features_{modality_name}_proj"]
                        )

                if self.supervise_single and self.heterogeneous:
                    single_label_list.append(selected_cav_processed["single_label_dict"])
                    single_object_bbx_center_list.append(selected_cav_processed["single_object_bbx_center"])
                    single_object_bbx_mask_list.append(selected_cav_processed["single_object_bbx_mask"])

            # generate single view GT label
            if self.supervise_single and self.heterogeneous:
                single_label_dicts = self._collate_label_dict(self.post_processor, single_label_list)
                single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
                single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
                processed_data_dict["ego"].update(
                    {
                        "single_label_dict_torch": single_label_dicts,
                        "single_object_bbx_center_torch": single_object_bbx_center,
                        "single_object_bbx_mask_torch": single_object_bbx_mask,
                    }
                )

            # exculude all repetitve objects, DAIR-V2X
            if self.params["fusion"]["dataset"] == "dairv2x":
                if len(object_stack) == 1:
                    object_stack = object_stack[0]
                else:
                    ego_boxes_np = object_stack[0]
                    cav_boxes_np = object_stack[1]
                    # order = self.params["postprocess"]["order"]
                    order = 'hwl' # hwl for DAIR-V2X
                    ego_corners_np = box_utils.boxes_to_corners_3d(ego_boxes_np, order)
                    cav_corners_np = box_utils.boxes_to_corners_3d(cav_boxes_np, order)
                    ego_polygon_list = list(convert_format(ego_corners_np))
                    cav_polygon_list = list(convert_format(cav_corners_np))
                    iou_thresh = 0.05

                    gt_boxes_from_cav = []
                    for i in range(len(cav_polygon_list)):
                        cav_polygon = cav_polygon_list[i]
                        ious = compute_iou(cav_polygon, ego_polygon_list)
                        if (ious > iou_thresh).any():
                            continue
                        gt_boxes_from_cav.append(cav_boxes_np[i])

                    if len(gt_boxes_from_cav):
                        object_stack_from_cav = np.stack(gt_boxes_from_cav)
                        object_stack = np.vstack([ego_boxes_np, object_stack_from_cav])
                    else:
                        object_stack = ego_boxes_np

                unique_indices = np.arange(object_stack.shape[0])
                object_id_stack = np.arange(object_stack.shape[0])
            else:
                # exclude all repetitive objects, OPV2V-H
                unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
                object_stack = np.vstack(object_stack)
                object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number
            if self.params['fusion']['dataset'] == 'v2xreal':
                object_bbx_center = \
                    np.zeros((self.post_process_params["max_num"], 8))
            else:
                object_bbx_center = np.zeros((self.post_process_params["max_num"], 7))
            mask = np.zeros(self.post_process_params["max_num"])
            object_bbx_center[: object_stack.shape[0], :] = object_stack
            mask[: object_stack.shape[0]] = 1

            for modality_name in self.modality_name_list:

                if eval(f"label_dict_list_{modality_name}"):
                    label_dict_modality = eval(f"label_dict_list_{modality_name}")
                    label_dict = {}
                    for key in label_dict_modality[0]:
                        stacked = torch.stack(
                            [torch.tensor(batch_dict[key]) for batch_dict in label_dict_modality], dim=0
                        )
                        if stacked.shape[0] < self.max_cav:
                            pad_shape = (self.max_cav - stacked.shape[0],) + tuple(stacked.shape[1:])
                            padding = torch.zeros(pad_shape, dtype=stacked.dtype)
                            stacked = torch.cat([stacked, padding], dim=0)
                        label_dict[key] = stacked

                    processed_data_dict["ego"].update({f"label_dict_{modality_name}": label_dict})

                # if eval(f"label_dict_list_{modality_name}"):
                #     # label_dict = self.post_processor.collate_batch(eval(f"label_dict_list_{modality_name}"))
                #     processed_data_dict["ego"].update({f"label_dict_{modality_name}" : eval(f"label_dict_list_{modality_name}")[0]})

            for modality_name in self.modality_name_list + ["m0"]:

                if self.sensor_type_dict[modality_name] == "lidar":
                    merged_feature_dict = merge_features_to_dict(eval(f"input_list_{modality_name}"))
                    processed_data_dict["ego"].update({f"input_{modality_name}": merged_feature_dict})  # maybe None
                elif self.sensor_type_dict[modality_name] == "camera":
                    merged_image_inputs_dict = merge_features_to_dict(
                        eval(f"input_list_{modality_name}"), merge="stack"
                    )
                    processed_data_dict["ego"].update(
                        {f"input_{modality_name}": merged_image_inputs_dict}
                    )  # maybe None
                elif (
                    "camera" in self.sensor_type_dict[modality_name] and "lidar" in self.sensor_type_dict[modality_name]
                ):
                    merged_feature_dict = merge_features_to_dict(eval(f"input_list_{modality_name}")["lidar"])
                    merged_image_inputs_dict = merge_features_to_dict(
                        eval(f"input_list_{modality_name}")["camera"], merge="stack"
                    )
                    if not merged_feature_dict or not merged_image_inputs_dict:
                        processed_data_dict["ego"].update({f"input_{modality_name}": None})
                    else:
                        processed_data_dict["ego"].update(
                            {
                                f"input_{modality_name}": {
                                    "lidar": merged_feature_dict,
                                    "camera": merged_image_inputs_dict,
                                }
                            }
                        )
                else:
                    raise ValueError("Not support this type of sensor")

            if self.kd_flag:
                # heterogenous setting do not support DiscoNet's kd
                # stack_lidar_np = np.vstack(projected_lidar_stack)
                # stack_lidar_np = mask_points_by_range(stack_lidar_np,
                #                             self.params['preprocess'][
                #                                 'cav_lidar_range'])
                # stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
                for modality_name in self.modality_name_list:
                    processed_data_dict["ego"].update(
                        {
                            f"input_{modality_name}_proj": merge_features_to_dict(
                                eval(f"input_list_{modality_name}_proj")
                            )  # maybe None
                        }
                    )

            processed_data_dict["ego"].update({"agent_modality_list": agent_modality_list})

            if self.visible:
                dynamic_bev = ego_cav_base.get("bev_visibility_corp.png", None)
            else:
                dynamic_bev = ego_cav_base.get("bev_dynamic.png", None)
            road_bev = ego_cav_base.get("bev_static.png", None)
            lane_bev = ego_cav_base.get("bev_lane.png", None)

            # generate targets label
            label_dict = self.post_processor.generate_label(
                **self._build_label_kwargs(
                    object_bbx_center,
                    self.anchor_box,
                    mask,
                    dynamic_bev,
                    road_bev,
                    lane_bev,
                    self.num_anchors_per_location,
                )
            )

            label_dict_protocol = self.post_processor_protocol.generate_label(
                **self._build_label_kwargs(
                    object_bbx_center,
                    self.anchor_box_protocol,
                    mask,
                    dynamic_bev,
                    road_bev,
                    lane_bev,
                    self.num_anchors_per_location_protocol,
                )
            )

            processed_data_dict["ego"].update(
                {
                    "object_bbx_center": object_bbx_center,
                    "object_bbx_mask": mask,
                    "object_ids": [object_id_stack[i] for i in unique_indices],
                    "anchor_box": self.anchor_box,
                    "num_anchors_per_location": self.num_anchors_per_location,
                    # Here we include anchor_box of other modality and for the protocol modality. This information
                    # should go to each cav data, but it takes a lot of efforts to make this dataset support all
                    # supervision on all cavs. We will leave it to the future work.
                    "anchor_box_dict": self.anchor_box_dict,
                    "num_anchors_per_location_dict": self.num_anchors_per_location_dict,
                    "label_dict": label_dict,
                    "label_dict_protocol": label_dict_protocol,
                    "cav_num": cav_num,
                    "pairwise_t_matrix": pairwise_t_matrix,
                    "lidar_poses_clean": lidar_poses_clean,
                    "lidar_poses": lidar_poses,
                    "transformation_matrix": torch.from_numpy(np.stack(transformation_matrix_list)),
                    "transformation_matrix_clean": torch.from_numpy(np.stack(transformation_matrix_clean_list)),
                }
            )

            if self.visualize:
                processed_data_dict["ego"].update({"origin_lidar": np.vstack(projected_lidar_stack)})
                processed_data_dict["ego"].update(
                    {
                        "origin_lidar_modality": np.concatenate(
                            [
                                np.ones(len(projected_lidar_stack[i])) * int(agent_modality_list[i][1:])
                                for i in range(len(projected_lidar_stack))
                            ]
                        )
                    }
                )

            processed_data_dict["ego"].update({"sample_idx": idx, "cav_id_list": cav_id_list})

            return processed_data_dict
        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            Process a single CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'
            ego_pose : list, length 6
                The ego vehicle lidar pose under world coordinate.
            ego_pose_clean : list, length 6
                only used for gt box generation

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}
            ego_pose, ego_pose_clean = ego_cav_base["params"]["lidar_pose"], ego_cav_base["params"]["lidar_pose_clean"]

            # calculate the transformation matrix
            transformation_matrix = x1_to_x2(selected_cav_base["params"]["lidar_pose"], ego_pose)  # T_ego_cav
            transformation_matrix_clean = x1_to_x2(selected_cav_base["params"]["lidar_pose_clean"], ego_pose_clean)

            modality_name = selected_cav_base["modality_name"]
            sensor_type = self.sensor_type_dict[modality_name]

            sensor_type_protocol = self.sensor_type_dict["m0"]

            # lidar
            if "lidar" in sensor_type or "lidar" in sensor_type_protocol or self.visualize:
                # process lidar
                lidar_np = selected_cav_base["lidar_np"]
                lidar_np = shuffle_points(lidar_np)
                # remove points that hit itself
                lidar_np = mask_ego_points(lidar_np)
                # project the lidar to ego space
                # x,y,z in ego space
                projected_lidar = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
                if self.proj_first:
                    lidar_np[:, :3] = projected_lidar

                if self.visualize:
                    # filter lidar
                    selected_cav_processed.update({"projected_lidar": projected_lidar})
                if self.kd_flag:
                    lidar_proj_np = copy.deepcopy(lidar_np)
                    lidar_proj_np[:, :3] = projected_lidar

                    selected_cav_processed.update({"projected_lidar": lidar_proj_np})

                    # 2023.8.31, to correct discretization errors. Just replace one point to avoid empty voxels. need fix later.
                    lidar_proj_np[np.random.randint(0, lidar_proj_np.shape[0]), :3] = np.array([0, 0, 0])
                    if "lidar" in sensor_type:
                        processed_lidar_proj = eval(f"self.pre_processor_{modality_name}").preprocess(lidar_proj_np)
                        selected_cav_processed.update(
                            {f"processed_features_{modality_name}_proj": processed_lidar_proj}
                        )
                    """ 
                    TODO: This implementation is neither eligent nor robust. Consider to improve.
                    """
                    if "lidar" in sensor_type_protocol:
                        processed_lidar_proj_protocol = eval(f"self.pre_processor_m0").preprocess(lidar_proj_np)
                        selected_cav_processed.update({f"processed_features_m0_proj": processed_lidar_proj_protocol})

                if "lidar" in sensor_type:
                    processed_lidar = eval(f"self.pre_processor_{modality_name}").preprocess(lidar_np)
                    selected_cav_processed.update({f"processed_features_{modality_name}": processed_lidar})

                """
                In protocol-based hetergenous collaboration, m0 are protocol networks
                Current, there are two scenarios that m0 could be exist.
                    1. we are training the protocol network, in which only m0 exists.
                    2. we are training the adapter, in which only m0 and m* exists.
                In the case of adapter training, both m0 and m* takes the raw data of m* but with 
                their corresponding preprocessor. That is why we need to add m0 to the input dict 
                while m* exist.
                TODO: This implementation is neither eligent nor robust. Consider to improve.
                """
                if "lidar" in sensor_type_protocol:
                    processed_lidar_protocol = eval(f"self.pre_processor_m0").preprocess(lidar_np)
                    selected_cav_processed.update({f"processed_features_m0": processed_lidar_protocol})

            # generate targets label single GT, note the reference pose is itself.
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                [selected_cav_base], selected_cav_base["params"]["lidar_pose"]
            )

            if self.visible:
                dynamic_bev = selected_cav_base.get("bev_visibility_corp.png", None)
            else:
                dynamic_bev = selected_cav_base.get("bev_dynamic.png", None)
            road_bev = selected_cav_base.get("bev_static.png", None)
            lane_bev = selected_cav_base.get("bev_lane.png", None)

            label_dict = self.post_processor_dict[modality_name].generate_label(
                **self._build_label_kwargs(
                    object_bbx_center,
                    getattr(self, f"anchor_box_{modality_name}"),
                    object_bbx_mask,
                    dynamic_bev,
                    road_bev,
                    lane_bev,
                    self.num_anchors_per_location_dict.get(modality_name),
                )
            )

            selected_cav_processed.update(
                {
                    "single_label_dict": label_dict,
                    "single_object_bbx_center": object_bbx_center,
                    "single_object_bbx_mask": object_bbx_mask,
                }
            )

            # camera
            if "camera" in sensor_type:
                camera_data_list = selected_cav_base["camera_data"]
                params = selected_cav_base["params"]
                imgs = []
                rots = []
                trans = []
                intrins = []
                extrinsics = []
                post_rots = []
                post_trans = []
                for idx, img in enumerate(camera_data_list):
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(camera_to_lidar[:3, :3])  # R_wc, we consider world-coord is the lidar-coord
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    img_src = [img]

                    # depth
                    if self.load_depth_file:
                        depth_img = selected_cav_base["depth_data"][idx]
                        img_src.append(depth_img)
                    else:
                        depth_img = None
                    # data augmentation
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        eval(f"self.data_aug_conf_{modality_name}"), self.train
                    )
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src,
                        post_rot,
                        post_tran,
                        resize=resize,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate,
                    )
                    # for convenience, make augmentation matrices 3x3
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    # decouple RGB and Depth

                    img_src[0] = normalize_img(img_src[0])
                    if self.load_depth_file:
                        img_src[1] = img_to_tensor(img_src[1]) * 255

                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    extrinsics.append(torch.from_numpy(camera_to_lidar))
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)

                selected_cav_processed.update(
                    {
                        f"image_inputs_{modality_name}": {
                            "imgs": torch.stack(imgs),  # [Ncam, 3or4, H, W]
                            "intrins": torch.stack(intrins),
                            "extrinsics": torch.stack(extrinsics),
                            "rots": torch.stack(rots),
                            "trans": torch.stack(trans),
                            "post_rots": torch.stack(post_rots),
                            "post_trans": torch.stack(post_trans),
                        }
                    }
                )

            if "camera" in sensor_type_protocol:
                camera_data_list = selected_cav_base["camera_data"]
                params = selected_cav_base["params"]
                imgs = []
                rots = []
                trans = []
                intrins = []
                extrinsics = []
                post_rots = []
                post_trans = []
                for idx, img in enumerate(camera_data_list):
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(camera_to_lidar[:3, :3])  # R_wc, we consider world-coord is the lidar-coord
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    img_src = [img]

                    # depth
                    if self.load_depth_file:
                        depth_img = selected_cav_base["depth_data"][idx]
                        img_src.append(depth_img)
                    else:
                        depth_img = None
                    # data augmentation
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        eval(f"self.data_aug_conf_m0"), self.train
                    )
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src,
                        post_rot,
                        post_tran,
                        resize=resize,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate,
                    )
                    # for convenience, make augmentation matrices 3x3
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    # decouple RGB and Depth

                    img_src[0] = normalize_img(img_src[0])
                    if self.load_depth_file:
                        img_src[1] = img_to_tensor(img_src[1]) * 255

                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    extrinsics.append(torch.from_numpy(camera_to_lidar))
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)

                selected_cav_processed.update(
                    {
                        f"image_inputs_m0": {
                            "imgs": torch.stack(imgs),  # [Ncam, 3or4, H, W]
                            "intrins": torch.stack(intrins),
                            "extrinsics": torch.stack(extrinsics),
                            "rots": torch.stack(rots),
                            "trans": torch.stack(trans),
                            "post_rots": torch.stack(post_rots),
                            "post_trans": torch.stack(post_trans),
                        }
                    }
                )

            # anchor box
            selected_cav_processed.update({"anchor_box": self.anchor_box})
            # note the reference pose ego
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                [selected_cav_base], ego_pose_clean
            )

            selected_cav_processed.update(
                {
                    "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                    "object_bbx_mask": object_bbx_mask,
                    "object_ids": object_ids,
                    "transformation_matrix": transformation_matrix,
                    "transformation_matrix_clean": transformation_matrix_clean,
                }
            )

            return selected_cav_processed

        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {"ego": {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            # inputs_list_m0 = []
            # inputs_list_m1 = []
            # inputs_list_m2 = []
            # inputs_list_m3 = []
            # inputs_list_m4 = []

            inputs_list_m0_proj = []
            inputs_list_m1_proj = []
            inputs_list_m2_proj = []
            inputs_list_m3_proj = []
            inputs_list_m4_proj = []

            agent_modality_list = []
            # used to record different scenario
            record_len = []
            label_dict_list = []
            label_dict_protocol_list = []
            lidar_pose_list = []
            origin_lidar = []
            origin_lidar_modality = []
            lidar_pose_clean_list = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            # disconet
            teacher_processed_lidar_list = []

            ### 2022.10.10 single gt ####
            if self.supervise_single and self.heterogeneous:
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []
                object_bbx_center_single = []
                object_bbx_mask_single = []

            for modality_name in self.modality_name_list:
                exec(f"label_dict_list_{modality_name} = []")

                if self.sensor_type_dict[modality_name] == "lidar" or self.sensor_type_dict[modality_name] == "camera":
                    exec(f"inputs_list_{modality_name} = []")
                    # if "m0" in self.modality_name_list and modality_name != "m0":
                    #     exec(f"inputs_list_m0 = []")
                elif (
                    "camera" in self.sensor_type_dict[modality_name] and "lidar" in self.sensor_type_dict[modality_name]
                ):
                    exec(f"inputs_list_{modality_name} = dict()")
                    eval(f"inputs_list_{modality_name}")["lidar"] = []
                    eval(f"inputs_list_{modality_name}")["camera"] = []
                    # if "m0" in self.modality_name_list and modality_name != "m0":
                    #     exec(f"inputs_list_m0 = dict()")
                    #     eval(f"inputs_list_m0")["lidar"] = []
                    #     eval(f"inputs_list_m0")["camera"] = []
                else:
                    raise ValueError("Not support this type of sensor")

            if self.sensor_type_dict["m0"] == "lidar" or self.sensor_type_dict["m0"] == "camera":
                exec(f"inputs_list_m0 = []")
            elif "camera" in self.sensor_type_dict["m0"] and "lidar" in self.sensor_type_dict["m0"]:
                exec(f"inputs_list_m0 = dict()")
                eval(f"inputs_list_m0")["lidar"] = []
                eval(f"inputs_list_m0")["camera"] = []
            else:
                raise ValueError("Not support this type of sensor")

            for i in range(len(batch)):
                ego_dict = batch[i]["ego"]
                object_bbx_center.append(ego_dict["object_bbx_center"])
                object_bbx_mask.append(ego_dict["object_bbx_mask"])
                object_ids.append(ego_dict["object_ids"])
                lidar_pose_list.append(ego_dict["lidar_poses"])  # ego_dict['lidar_pose'] is np.ndarray [N,6]
                lidar_pose_clean_list.append(ego_dict["lidar_poses_clean"])

                for modality_name in self.modality_name_list + ["m0"]:

                    if f"label_dict_{modality_name}" in ego_dict.keys():
                        eval(f"label_dict_list_{modality_name}").append(ego_dict[f"label_dict_{modality_name}"])

                    if ego_dict[f"input_{modality_name}"] is not None:
                        if (
                            self.sensor_type_dict[modality_name] == "lidar"
                            or self.sensor_type_dict[modality_name] == "camera"
                        ):
                            eval(f"inputs_list_{modality_name}").append(ego_dict[f"input_{modality_name}"])
                        elif (
                            "lidar" in self.sensor_type_dict[modality_name]
                            or "camera" in self.sensor_type_dict[modality_name]
                        ):
                            eval(f"inputs_list_{modality_name}")["lidar"].append(
                                ego_dict[f"input_{modality_name}"]["lidar"]
                            )
                            eval(f"inputs_list_{modality_name}")["camera"].append(
                                ego_dict[f"input_{modality_name}"]["camera"]
                            )
                        else:
                            raise ValueError("Not support this type of sensor")

                agent_modality_list.extend(ego_dict["agent_modality_list"])

                record_len.append(ego_dict["cav_num"])
                label_dict_list.append(ego_dict["label_dict"])
                label_dict_protocol_list.append(ego_dict["label_dict_protocol"])
                pairwise_t_matrix_list.append(ego_dict["pairwise_t_matrix"])

                if self.visualize:
                    origin_lidar.append(ego_dict["origin_lidar"])
                    origin_lidar_modality.append(ego_dict["origin_lidar_modality"])

                if self.kd_flag:
                    # hetero setting do not support disconet' kd
                    # teacher_processed_lidar_list.append(ego_dict['teacher_processed_lidar'])
                    for modality_name in self.modality_name_list:
                        if ego_dict[f"input_{modality_name}_proj"] is not None:
                            eval(f"inputs_list_{modality_name}_proj").append(ego_dict[f"input_{modality_name}_proj"])

                ### 2022.10.10 single gt ####
                if self.supervise_single and self.heterogeneous:
                    pos_equal_one_single.append(ego_dict["single_label_dict_torch"]["pos_equal_one"])
                    neg_equal_one_single.append(ego_dict["single_label_dict_torch"]["neg_equal_one"])
                    targets_single.append(ego_dict["single_label_dict_torch"]["targets"])
                    object_bbx_center_single.append(ego_dict["single_object_bbx_center_torch"])
                    object_bbx_mask_single.append(ego_dict["single_object_bbx_mask_torch"])

            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            # 2023.2.5
            for modality_name in self.modality_name_list + ["m0"]:
                if len(eval(f"inputs_list_{modality_name}")) != 0:
                    if self.sensor_type_dict[modality_name] == "lidar":
                        merged_feature_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}"))
                        processed_lidar_torch_dict = eval(f"self.pre_processor_{modality_name}").collate_batch(
                            merged_feature_dict
                        )
                        output_dict["ego"].update({f"inputs_{modality_name}": processed_lidar_torch_dict})

                    elif self.sensor_type_dict[modality_name] == "camera":
                        merged_image_inputs_dict = merge_features_to_dict(
                            eval(f"inputs_list_{modality_name}"), merge="cat"
                        )
                        output_dict["ego"].update({f"inputs_{modality_name}": merged_image_inputs_dict})
                    elif (
                        "camera" in self.sensor_type_dict[modality_name]
                        and "lidar" in self.sensor_type_dict[modality_name]
                    ):
                        merged_feature_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}")["lidar"])
                        processed_lidar_torch_dict = eval(f"self.pre_processor_{modality_name}").collate_batch(
                            merged_feature_dict
                        )
                        merged_image_inputs_dict = merge_features_to_dict(
                            eval(f"inputs_list_{modality_name}")["camera"], merge="cat"
                        )
                        output_dict["ego"].update(
                            {
                                f"inputs_{modality_name}": {
                                    "lidar": processed_lidar_torch_dict,
                                    "camera": merged_image_inputs_dict,
                                }
                            }
                        )

            output_dict["ego"].update({"agent_modality_list": agent_modality_list})

            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
            lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
            label_torch_dict = self._collate_label_dict(self.post_processor, label_dict_list)
            label_torch_dict_protocol = self._collate_label_dict(
                self.post_processor_protocol, label_dict_protocol_list
            )

            # for centerpoint
            label_torch_dict.update({"object_bbx_center": object_bbx_center, "object_bbx_mask": object_bbx_mask})

            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # add pairwise_t_matrix to label dict
            label_torch_dict["pairwise_t_matrix"] = pairwise_t_matrix
            label_torch_dict["record_len"] = record_len

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            anchor_payload = {}
            if self.num_anchors_per_location is None:
                anchor_payload["anchor_box"] = self.anchor_box_torch
            else:
                anchor_payload["all_anchors"] = self.anchor_box_torch
                anchor_payload["num_anchors_per_location"] = self.num_anchors_per_location
            if self.num_anchors_per_location_dict:
                anchor_payload["num_anchors_per_location_dict"] = self.num_anchors_per_location_dict

            output_dict["ego"].update(
                {
                    "object_bbx_center": object_bbx_center,
                    "object_bbx_mask": object_bbx_mask,
                    "record_len": record_len,
                    "label_dict": label_torch_dict,
                    "label_dict_protocol": label_torch_dict_protocol,
                    "object_ids": object_ids[0],
                    "pairwise_t_matrix": pairwise_t_matrix,
                    "lidar_pose_clean": lidar_pose_clean,
                    "lidar_pose": lidar_pose,
                    **anchor_payload,
                }
            )

            if self.visualize:
                origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict["ego"].update({"origin_lidar": origin_lidar})
                output_dict["ego"].update({"origin_lidar_modality": torch.from_numpy(np.array(origin_lidar_modality))})

            if self.kd_flag:
                # teacher_processed_lidar_torch_dict = \
                #     self.pre_processor.collate_batch(teacher_processed_lidar_list)
                # output_dict['ego'].update({'teacher_processed_lidar':teacher_processed_lidar_torch_dict})
                for modality_name in self.modality_name_list:
                    if (
                        len(eval(f"inputs_list_{modality_name}_proj")) != 0
                        and "lidar" in self.sensor_type_dict[modality_name]
                    ):
                        merged_feature_proj_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}_proj"))
                        processed_lidar_torch_proj_dict = eval(f"self.pre_processor_{modality_name}").collate_batch(
                            merged_feature_proj_dict
                        )
                        output_dict["ego"].update({f"inputs_{modality_name}_proj": processed_lidar_torch_proj_dict})

            if self.supervise_single and self.heterogeneous:
                output_dict["ego"].update(
                    {
                        "label_dict_single": {
                            "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                            "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                            "targets": torch.cat(targets_single, dim=0),
                            # for centerpoint
                            "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                            "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0),
                        },
                        "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                        "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0),
                    }
                )

            return output_dict

        def collate_batch_test(self, batch):
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            if batch[0] is None:
                return None
            output_dict = self.collate_batch_train(batch)
            if output_dict is None:
                return None

            # check if anchor box in the batch
            if batch[0]["ego"].get("num_anchors_per_location") is not None:
                output_dict["ego"].update(
                    {
                        "all_anchors": self.anchor_box_torch,
                        "num_anchors_per_location": batch[0]["ego"]["num_anchors_per_location"],
                    }
                )
            elif batch[0]["ego"]["anchor_box"] is not None:
                output_dict["ego"].update({"anchor_box": self.anchor_box_torch})

            if batch[0]["ego"]["anchor_box_dict"] is not None:
                output_dict["ego"].update({"anchor_box_dict": self.anchor_box_torch_dict})
                if "num_anchors_per_location_dict" in batch[0]["ego"]:
                    output_dict["ego"].update(
                        {"num_anchors_per_location_dict": batch[0]["ego"]["num_anchors_per_location_dict"]}
                    )

            # save the transformation matrix (4, 4) to ego vehicle
            # transformation is only used in post process (no use.)
            # we all predict boxes in ego coord.

            if batch[0]["ego"]["transformation_matrix"] is not None:
                output_dict["ego"].update(
                    {
                        "transformation_matrix": batch[0]["ego"]["transformation_matrix"].float(),
                        "transformation_matrix_clean": batch[0]["ego"]["transformation_matrix_clean"].float(),
                    }
                )
            else:
                transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
                transformation_matrix_clean_torch = torch.from_numpy(np.identity(4)).float()
                output_dict["ego"].update(
                    {
                        "transformation_matrix": transformation_matrix_torch,
                        "transformation_matrix_clean": transformation_matrix_clean_torch,
                    }
                )

            output_dict["ego"].update(
                {
                    "sample_idx": batch[0]["ego"]["sample_idx"],
                    "cav_id_list": batch[0]["ego"]["cav_id_list"],
                    "agent_modality_list": batch[0]["ego"]["agent_modality_list"],
                }
            )

            return output_dict

        def post_process(self, data_dict, output_dict, fusion="intermediate", agent_idx=0):
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
            if "ego" in output_dict.keys():
                inner_output = output_dict["ego"]
                if isinstance(inner_output, tuple):
                    inner_output = inner_output[0]
                if isinstance(inner_output, dict) and "cls_preds" not in inner_output:
                    ego_modality = self.ego_modality
                    modality_output = inner_output.get(ego_modality)
                    if modality_output is None and len(inner_output) == 1:
                        modality_output = next(iter(inner_output.values()))
                    if modality_output is None:
                        raise KeyError(f"Missing ego modality output {ego_modality} for post_process.")
                    ego_output_dict = {"ego": modality_output}
                    ego_content = dict(data_dict["ego"])
                    ego_tfm = ego_content.get("transformation_matrix")
                    if torch.is_tensor(ego_tfm) and ego_tfm.dim() == 3 and ego_tfm.shape[-2:] == (4, 4):
                        ego_content["transformation_matrix"] = ego_tfm[0]
                    elif torch.is_tensor(ego_tfm) and ego_tfm.shape != (4, 4):
                        ego_content["transformation_matrix"] = torch.eye(4, dtype=ego_tfm.dtype, device=ego_tfm.device)
                    ego_tfm_clean = ego_content.get("transformation_matrix_clean")
                    if torch.is_tensor(ego_tfm_clean) and ego_tfm_clean.dim() == 3 and ego_tfm_clean.shape[-2:] == (4, 4):
                        ego_content["transformation_matrix_clean"] = ego_tfm_clean[0]
                    elif torch.is_tensor(ego_tfm_clean) and ego_tfm_clean.shape != (4, 4):
                        ego_content["transformation_matrix_clean"] = torch.eye(4, dtype=ego_tfm_clean.dtype, device=ego_tfm_clean.device)
                    ego_data_dict = {"ego": ego_content}
                    pred_box_tensor, pred_score = self.post_processor.post_process(ego_data_dict, ego_output_dict)
                    gt_box_tensor, gt_label_tensor = self.post_processor.generate_gt_bbx(ego_data_dict)
                    return pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor
                if hasattr(self.post_processor, "post_process_stamp"):
                    pred_box_tensor, pred_score = self.post_processor.post_process_stamp(
                        data_dict, output_dict, agent_idx=agent_idx
                    )
                else:
                    pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
                gt_box_tensor, gt_label_tensor = self.post_processor.generate_gt_bbx(data_dict)
                return pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor
            else:
                if fusion == "intermediate":
                    return self.post_process_intermediate(data_dict, output_dict)
                elif fusion == "late":
                    pred_box_tensor, pred_score = self.post_process_late(data_dict, output_dict)
                    gt_box_tensor, gt_label_tensor = self.post_processor.generate_gt_bbx(data_dict)
                    return pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor

        def post_process_intermediate(self, data_dict, output_dict):

            pred_box_tensor_list = []
            pred_score_list = []
            gt_list = []

            counting_dict = defaultdict(int)
            for agent_idx, m in enumerate(data_dict["ego"]["agent_modality_list"]):
                if m not in output_dict:
                    warnings.warn(f"No {m} in output_dict. Skip. Only intentional for partial cav fusion")
                    pred_box_tensor_list.append(None)
                    pred_score_list.append(None)
                    gt_list.append(None)
                    continue
                if m in data_dict:
                    cav_content = data_dict[m]
                else:
                    warnings.warn(f"No {m} in data_dict. Using ego instead.")
                    cav_content = {
                        "anchor_box": data_dict["ego"]["anchor_box_dict"][m],
                        "transformation_matrix": data_dict["ego"][
                            "transformation_matrix"
                        ],  # Batch_id 0 (assume batch size 1) and cav_id 0 (ego vehicle), end up with [L, 4, 4]
                    }
                    
                    ret = self.post_processor_dict[m].post_process_intermediate(
                        cav_content, output_dict, m, feat_idx=counting_dict[m], agent_idx=agent_idx
                    )

                if ret:
                    pred_box_tensor, pred_score = ret
                    pred_box_tensor_list.append(pred_box_tensor)
                    pred_score_list.append(pred_score)
                else:
                    pred_box_tensor_list.append(None)
                    pred_score_list.append(None)

                gt = self.post_processor_dict[m].generate_gt_intermediate(
                    data_dict, modality=m, feat_idx=counting_dict[m], agent_idx=agent_idx
                )
                gt_list.append(gt)
                counting_dict[m] += 1

            return [
                (pred_box_tensor, pred_score, gt_box_tensor)
                for pred_box_tensor, pred_score, gt_box_tensor in zip(pred_box_tensor_list, pred_score_list, gt_list)
            ]

        def post_process_late(self, data_dict, output_dict):
            # late fusion only
            pred_box3d_list = []
            pred_box2d_list = []

            counting_dict = {m: 0 for m in self.modality_name_list}

            for agent_idx, m in enumerate(data_dict["ego"]["agent_modality_list"]):
                if m not in output_dict:
                    warnings.warn(f"No {m} in output_dict. Skip. Only intentional for partial cav fusion")
                    continue
                if m in data_dict:
                    cav_content = data_dict[m]
                else:
                    warnings.warn(f"No {m} in data_dict. Using ego instead.")
                    cav_content = {
                        "anchor_box": data_dict["ego"]["anchor_box_dict"][m],
                        "transformation_matrix": data_dict["ego"][
                            "transformation_matrix"
                        ],  # Batch_id 0 (assume batch size 1) and cav_id 0 (ego vehicle), end up with [L, 4, 4]
                    }

                if output_dict[m].get("cls_preds", None) is None:
                    # This happens mostly likely for protocol modality during late fusion. If not, please pay attentions.
                    warnings.warn(
                        f"Skip {m} for post processing. It is designed for late fusion, if not, please check."
                    )
                    continue
                single_output_dict = {
                    "cls_preds": output_dict[m]["cls_preds"][counting_dict[m] : counting_dict[m] + 1],
                    "reg_preds": output_dict[m]["reg_preds"][counting_dict[m] : counting_dict[m] + 1],
                    "dir_preds": output_dict[m]["dir_preds"][counting_dict[m] : counting_dict[m] + 1],
                }
                pred_box_tensor, pred_score = self.post_processor_dict[m].post_process_single(
                    cav_content, single_output_dict, agent_idx=agent_idx
                )
                pred_box3d_list.append(pred_box_tensor)
                pred_box2d_list.append(pred_score)
                counting_dict[m] += 1

            pred_box_tensor, pred_score = self.post_processor.post_process_output(pred_box3d_list, pred_box2d_list)
            return pred_box_tensor, pred_score

    return Intermediateheteradapter3classFusionDataset
