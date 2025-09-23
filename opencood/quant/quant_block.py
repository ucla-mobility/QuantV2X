import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DoubleConv, DownsampleConv
from opencood.models.heter_encoders import SECOND, CamEncode, CamEncode_Resnet101, PillarVFE, PointPillar, LiftSplatShoot
from opencood.models.sub_modules.feature_alignnet import AlignNet
from efficientnet_pytorch import EfficientNet
from opencood.models.sub_modules.pillar_vfe import PFNLayer
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.resblock import ResNetModified
from .quant_layer import QuantModule, UniformAffineQuantizer, QuantSpconvModule, StraightThrough
from opencood.models.sub_modules.resblock import BasicBlock, Bottleneck
from opencood.models.regnet import ResBottleneckBlock
from opencood.models.mobilenetv2 import InvertedResidual
from opencood.models.mnasnet import _InvertedResidual
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.utils.camera_utils import bin_depths
from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum, depth_discretization
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.pyramid_fuse import weighted_fuse
from opencood.models.fuse_modules.fusion_in_one import V2XViTFusion
from opencood.models.fuse_modules.fuse_utils import regroup as Regroup
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.sub_modules.v2xvit_basic import V2XTransformer, V2XTEncoder, V2XFusionBlock
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
    get_discretized_transformation_matrix
from opencood.models.sub_modules.hmsa import HGTCavAttention
from einops import rearrange
from opencood.models.sub_modules.split_attn import SplitAttn
from opencood.models.sub_modules.mswin import BaseWindowAttention, PyramidWindowAttention
from opencood.models.sub_modules.base_transformer import BaseTransformer, PreNorm, FeedForward
from opencood.models.sub_modules.naive_compress import NaiveCompressor

try: # spconv1
    from spconv import SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor
except: # spconv2
    from spconv.pytorch import  SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor


class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """
    def __init__(self):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        self.ignore_reconstruction = False
        self.trained = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m,(QuantSpconvModule, QuantModule)):
                m.set_quant_state(weight_quant, act_quant)


class QuantBasicBlock(BaseQuantBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, basic_block: BasicBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params, act_quant_params)
        self.conv1.norm_function = basic_block.bn1
        self.conv1.activation_function = basic_block.relu
        self.conv2 = QuantModule(basic_block.conv2, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv2.norm_function = basic_block.bn2

        if basic_block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(basic_block.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
            self.downsample.norm_function = basic_block.downsample[1]
        self.activation_function = basic_block.relu
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantBottleneck(BaseQuantBlock):
    def __init__(self, bottleneck: Bottleneck, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params, act_quant_params)
        self.conv1.norm_function = bottleneck.bn1
        self.conv1.activation_function = bottleneck.relu
        self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params, act_quant_params)
        self.conv2.norm_function = bottleneck.bn2
        self.conv2.activation_function = bottleneck.relu
        self.conv3 = QuantModule(bottleneck.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv3.norm_function = bottleneck.bn3

        if bottleneck.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(bottleneck.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
            self.downsample.norm_function = bottleneck.downsample[1]
        # modify the activation function to ReLU
        self.activation_function = bottleneck.relu
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantResBottleneckBlock(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
    """

    def __init__(self, bottleneck: ResBottleneckBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(bottleneck.f.a, weight_quant_params, act_quant_params)
        self.conv1.norm_function = bottleneck.f.a_bn
        self.conv1.activation_function = bottleneck.f.a_relu
        self.conv2 = QuantModule(bottleneck.f.b, weight_quant_params, act_quant_params)
        self.conv2.norm_function = bottleneck.f.b_bn
        self.conv2.activation_function = bottleneck.f.b_relu
        self.conv3 = QuantModule(bottleneck.f.c, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv3.norm_function = bottleneck.f.c_bn

        if bottleneck.proj_block:
            self.downsample = QuantModule(bottleneck.proj, weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
            self.downsample.norm_function = bottleneck.bn
        else:
            self.downsample = None
        # copying all attributes in original block
        self.proj_block = bottleneck.proj_block

        self.activation_function = bottleneck.relu
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if not self.proj_block else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantInvertedResidual(BaseQuantBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """

    def __init__(self, inv_res: InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()

        self.use_res_connect = inv_res.use_res_connect
        self.expand_ratio = inv_res.expand_ratio
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].norm_function = inv_res.conv[1]
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].norm_function = inv_res.conv[4]
        else:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[6], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].norm_function = inv_res.conv[1]
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].norm_function = inv_res.conv[4]
            self.conv[1].activation_function = nn.ReLU6()
            self.conv[2].norm_function = inv_res.conv[7]
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class _QuantInvertedResidual(BaseQuantBlock):
    def __init__(self, _inv_res: _InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()

        self.apply_residual = _inv_res.apply_residual
        self.conv = nn.Sequential(
            QuantModule(_inv_res.layers[0], weight_quant_params, act_quant_params),
            QuantModule(_inv_res.layers[3], weight_quant_params, act_quant_params),
            QuantModule(_inv_res.layers[6], weight_quant_params, act_quant_params, disable_act_quant=True),
        )
        self.conv[0].activation_function = nn.ReLU()
        self.conv[0].norm_function = _inv_res.layers[1]
        self.conv[1].activation_function = nn.ReLU()
        self.conv[1].norm_function = _inv_res.layers[4]
        self.conv[2].norm_function = _inv_res.layers[7]
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        if self.apply_residual:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantBaseBEVBackbone(BaseQuantBlock):
    def __init__(self, basebevbackbone: BaseBEVBackbone, weight_quant_params={}, act_quant_params={}):
        super().__init__() 

        self.num_levels = basebevbackbone.num_levels
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        for idx in range(self.num_levels):
            base_block = basebevbackbone.blocks[idx]
            wrapped_block = nn.Sequential(
                base_block[0],  # Keep ZeroPad2d
                QuantModule(base_block[1], weight_quant_params, act_quant_params),
            )
            wrapped_block[1].norm_function = base_block[2]
            wrapped_block[1].activation_function = base_block[3]

            for k in range(1, len(base_block) // 3):
                conv_layer = QuantModule(base_block[4 + (k-1)*3], weight_quant_params, act_quant_params)
                conv_layer.norm_function = base_block[5 + (k-1)*3]  # BatchNorm2D
                conv_layer.activation_function = base_block[6 + (k-1)*3]  # ReLU
                wrapped_block.extend([conv_layer])

            self.blocks.append(wrapped_block)

        for idx in range(len(basebevbackbone.deblocks)):
            base_deblock = basebevbackbone.deblocks[idx]
            wrapped_deblock = nn.Sequential(
                QuantModule(base_deblock[0], weight_quant_params, act_quant_params),
            )
            wrapped_deblock[0].norm_function = base_deblock[1]  # BatchNorm2D
            wrapped_deblock[0].activation_function = base_deblock[2] # ReLU
            self.deblocks.append(wrapped_deblock)

        # Preserve the number of BEV features
        self.num_bev_features = basebevbackbone.num_bev_features

    def forward(self, x):
        ups = []
        ret_dict = {}

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(x.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        return x  # Output [N, C, H, W]

    def get_multiscale_feature(self, spatial_features):
        """
        Before multiscale intermediate fusion.
        """
        feature_list = []
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            feature_list.append(x)

        return feature_list

    def decode_multiscale_feature(self, x):
        """
        After multiscale intermediate fusion.
        """
        ups = []
        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)
        return x


class QuantResNetModified(BaseQuantBlock):
    def __init__(
        self,
        resnet_modified: ResNetModified,
        weight_quant_params={},
        act_quant_params={}
    ) -> None:
        super().__init__()

        # Copy all attributes from original ResNetModified
        self._norm_layer = resnet_modified._norm_layer
        self.layernum = resnet_modified.layernum
        self.block = resnet_modified.block  # BasicBlock or Bottleneck

        # Map to corresponding quantized blocks
        block_mapping = {
            BasicBlock: QuantBasicBlock,
            Bottleneck: QuantBottleneck
        }

        if self.block not in block_mapping:
            raise ValueError(f"Unsupported block type: {self.block}. Please add it to block_mapping.")

        quant_block = block_mapping[self.block]

        # Convert each layer into a quantized version
        for i in range(self.layernum):
            original_layer = getattr(resnet_modified, f"layer{i}")
            quant_layer = self._make_quant_layer(
                original_layer, quant_block,
                weight_quant_params=weight_quant_params,
                act_quant_params=act_quant_params
            )
            setattr(self, f"layer{i}", quant_layer)

    def _make_quant_layer(self, original_layer, quant_block, weight_quant_params={}, act_quant_params={}):
        """Wraps each layer in QuantModule while keeping original weights."""
        quantized_layers = []
        for layer in original_layer:
            if isinstance(layer, self.block):  # BasicBlock or Bottleneck
                quantized_layers.append(
                    quant_block(layer, weight_quant_params, act_quant_params)
                )
            else:
                quantized_layers.append(layer)  # Keep other layers as they are

        return nn.Sequential(*quantized_layers)

    def _forward_impl(self, x, return_interm=True):
        interm_features = []
        for i in range(self.layernum):
            x = getattr(self, f"layer{i}")(x)
            interm_features.append(x)

        return interm_features if return_interm else x

    def forward(self, x):
        return self._forward_impl(x)


class QuantResNetBEVBackbone(BaseQuantBlock):
    def __init__(self, resnet_bev_backbone: ResNetBEVBackbone, weight_quant_params={}, act_quant_params={}):
        super().__init__()
        self.model_cfg = resnet_bev_backbone.model_cfg
        self.num_levels = resnet_bev_backbone.num_levels
        self.num_bev_features = resnet_bev_backbone.num_bev_features
        
        self.resnet = QuantResNetModified(resnet_bev_backbone.resnet, weight_quant_params, act_quant_params)
        
        self.deblocks = nn.ModuleList()
        for deblock in resnet_bev_backbone.deblocks:
            quant_deblock = nn.Sequential(
                QuantModule(deblock[0], weight_quant_params, act_quant_params),
            )
            quant_deblock[0].norm_function = deblock[1]
            quant_deblock[0].activation_function = deblock[2]
            self.deblocks.append(quant_deblock)

    def forward(self, spatial_features):
        x = self.resnet(spatial_features)
        ups = []
        
        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)
        
        return x
    
    def get_multiscale_feature(self, spatial_features):
        x = self.resnet(spatial_features)
        return x

    def decode_multiscale_feature(self, x):
        ups = []
        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])
        
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        
        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)
        
        return x
    
    def get_layer_i_feature(self, spatial_features, layer_i):
        return eval(f"self.resnet.layer{layer_i}")(spatial_features)


class QuantPyramidFusion(QuantResNetBEVBackbone):
    def __init__(self, pyramid_fusion: PyramidFusion, weight_quant_params={}, act_quant_params={}):
        super().__init__(pyramid_fusion, weight_quant_params, act_quant_params)
        
        self.stage = pyramid_fusion.stage
        if pyramid_fusion.model_cfg["resnext"]:
            Bottleneck.expansion = 1
            self.resnet = QuantResNetModified(
                pyramid_fusion.resnet,
                weight_quant_params,
                act_quant_params
            )
        
        self.align_corners = pyramid_fusion.align_corners
        
        for i in range(self.num_levels):
            quant_conv = QuantModule(
                getattr(pyramid_fusion, f"single_head_{i}"), weight_quant_params, act_quant_params
            )
            setattr(self, f"single_head_{i}", quant_conv)

    def forward_single(self, spatial_features):
        feature_list = self.get_multiscale_feature(spatial_features)
        occ_map_list = []
        
        for i in range(self.num_levels):
            occ_map = getattr(self, f"single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)
        
        final_feature = self.decode_multiscale_feature(feature_list)
        
        return final_feature, occ_map_list
    
    def forward_collab(self, spatial_features, record_len, affine_matrix, agent_modality_list=None, cam_crop_info=None):
        crop_mask_flag = False
        if cam_crop_info is not None and len(cam_crop_info) > 0:
            crop_mask_flag = True
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {}
            for cam_modality in cam_modality_set:
                mask_list = [1 if x == cam_modality else 0 for x in agent_modality_list] 
                mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
                cam_agent_mask_dict[cam_modality] = mask_tensor

                # e.g. {m2: [0,0,0,1], m4: [0,1,0,0]}

        feature_list = self.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])  # [N, 1, H, W]
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, H, W = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4 # There may be unstable response values at the edges.
                    crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4 # There may be unstable response values at the edges.

                    start_h = int(H//2-crop_H//2)
                    end_h = int(H//2+crop_H//2)
                    start_w = int(W//2-crop_W//2)
                    end_w = int(W//2+crop_W//2)

                    cam_crop_mask[cam_agent_mask_dict[cam_modality],:,start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[cam_agent_mask_dict[cam_modality]] = 1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]

                score = score * cam_crop_mask

            fused_feature_list.append(weighted_fuse(feature_list[i], score, record_len, affine_matrix, self.align_corners))
        fused_feature = self.decode_multiscale_feature(fused_feature_list)

        return fused_feature, occ_map_list 
    
    def forward(self, spatial_features, record_len=None, affine_matrix=None, agent_modality_list=None, cam_crop_info=None):
        """
        Unified forward method to switch between 'single' and 'collab' mode.
        If in 'single' mode, only spatial_features is required.
        If in 'collab' mode, additional parameters are needed.
        """
        if self.stage == "single":
            return self.forward_single(spatial_features)
        elif self.stage == "collab":
            if record_len is None or affine_matrix is None:
                raise ValueError("record_len and affine_matrix are required for forward_collab()")
            return self.forward_collab(spatial_features, record_len, affine_matrix, agent_modality_list, cam_crop_info)
        

class QuantDoubleConv(BaseQuantBlock):
    """
    Quantized version of DoubleConv.
    Wraps Conv2D layers inside QuantModule while keeping activation functions.
    """
    def __init__(self, double_conv: DoubleConv, weight_quant_params={}, act_quant_params={}):
        super().__init__()

        self.double_conv = nn.Sequential(
            QuantModule(double_conv.double_conv[0], weight_quant_params, act_quant_params),  # First Conv2D
            QuantModule(double_conv.double_conv[2], weight_quant_params, act_quant_params),  # Second Conv2D
        )
        self.double_conv[0].activation_function = double_conv.double_conv[1] # First ReLU
        self.double_conv[1].activation_function = double_conv.double_conv[3] # Second ReLU

    def forward(self, x):
        x = self.double_conv[0](x)
        x = self.double_conv[1](x)
        return x


class QuantDownsampleConv(BaseQuantBlock):
    def __init__(self, downsample_conv: DownsampleConv, weight_quant_params={}, act_quant_params={}):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for layer in downsample_conv.layers:
            self.layers.append(
                QuantDoubleConv(layer, weight_quant_params, act_quant_params)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class QuantPFNLayer(BaseQuantBlock):
    """
    Quantized version of PFNLayer in PillarVFE.
    Wraps Linear layers inside QuantModule while keeping BatchNorm and ReLU.
    """
    def __init__(self, pfn_layer: PFNLayer, weight_quant_params={}, act_quant_params={}):
        super().__init__()

        self.last_vfe = pfn_layer.last_vfe
        self.use_norm = pfn_layer.use_norm
        self.part = pfn_layer.part

        if self.use_norm:
            self.linear = QuantModule(pfn_layer.linear, weight_quant_params, act_quant_params)
            self.linear.norm_function = pfn_layer.norm
            # self.linear.activation_function = nn.ReLU(inplace=True)
        else:
            self.linear = QuantModule(pfn_layer.linear, weight_quant_params, act_quant_params)
            # self.linear.activation_function = nn.ReLU(inplace=True)

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part])
                               for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        x = F.relu(x)
        if self.use_act_quant:
            x = self.act_quantizer(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class QuantPillarVFE(nn.Module):
    """
    Quantized version of PillarVFE.
    Replaces all PFNLayer instances with QuantPFNLayer.
    """
    def __init__(self, pillar_vfe: PillarVFE, weight_quant_params={}, act_quant_params={}):
        super().__init__()

        self.with_distance = pillar_vfe.with_distance
        self.use_absolute_xyz = pillar_vfe.use_absolute_xyz

        self.voxel_x = pillar_vfe.voxel_x
        self.voxel_y = pillar_vfe.voxel_y
        self.voxel_z = pillar_vfe.voxel_z
        self.x_offset = pillar_vfe.x_offset
        self.y_offset = pillar_vfe.y_offset
        self.z_offset = pillar_vfe.z_offset

        self.pfn_layers = nn.ModuleList([
            QuantPFNLayer(layer, weight_quant_params, act_quant_params)
            for layer in pillar_vfe.pfn_layers
        ])

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num,
                               dtype=torch.int,
                               device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator
    
    def forward(self, batch_dict):
        """encoding voxel feature using point-pillar method
        Args:
            voxel_features: [M, 32, 4]
            voxel_num_points: [M,]
            voxel_coords: [M, 4]
        Returns:
            features: [M,64], after PFN
        """
        voxel_features, voxel_num_points, coords = \
            batch_dict['voxel_features'], batch_dict['voxel_num_points'], \
            batch_dict['voxel_coords']
        points_mean = \
            voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
            voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2,
                                     keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count,
                                           axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features

        return batch_dict


class QuantPointPillar(nn.Module):
    """
    Quantized version of PointPillar.
    Wraps `PillarVFE` with `QuantPillarVFE` while keeping `PointPillarScatter` unchanged.
    """
    def __init__(self, point_pillar: PointPillar, weight_quant_params={}, act_quant_params={}):
        super().__init__()

        self.pillar_vfe = QuantPillarVFE(point_pillar.pillar_vfe, weight_quant_params, act_quant_params)
        self.scatter = point_pillar.scatter  # Scatter has no learnable parameters

    def forward(self, data_dict, modality_name):
        voxel_features = data_dict[f'inputs_{modality_name}']['voxel_features']
        voxel_coords = data_dict[f'inputs_{modality_name}']['voxel_coords']
        voxel_num_points = data_dict[f'inputs_{modality_name}']['voxel_num_points']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        lidar_feature_2d = batch_dict['spatial_features']  # H0, W0
        return lidar_feature_2d


''' QuantLSS Camera Encoder still under development and testing'''
class QuantCamEncode_Resnet101(BaseQuantBlock):
    def __init__(self, camencode: CamEncode_Resnet101, weight_quant_params={}, act_quant_params={}):
        super().__init__()
        self.D = camencode.D
        self.C = camencode.C
        self.downsample = camencode.downsample
        self.d_min = camencode.d_min
        self.d_max = camencode.d_max
        self.num_bins = camencode.num_bins
        self.mode = camencode.mode
        self.use_gt_depth = camencode.use_gt_depth
        self.depth_supervision = camencode.depth_supervision
        self.conv1 = QuantModule(camencode.conv1, weight_quant_params, act_quant_params)
        self.conv1.norm_function = camencode.bn1
        self.conv1.activation_function = camencode.relu
        self.maxpool = camencode.maxpool
        self.layer1 = []
        for layer in camencode.layer1:
            if isinstance(layer, BasicBlock):
                self.layer1.append(
                    QuantBasicBlock(layer, weight_quant_params, act_quant_params)
                )
            else:
                self.layer1.append(
                    QuantBottleneck(layer, weight_quant_params, act_quant_params)
                )
        self.layer1 = nn.Sequential(*self.layer1)

        self.layer2 = []
        for layer in camencode.layer2:
            if isinstance(layer, BasicBlock):
                self.layer2.append(
                    QuantBasicBlock(layer, weight_quant_params, act_quant_params)
                )
            else:
                self.layer2.append(
                    QuantBottleneck(layer, weight_quant_params, act_quant_params) # lss only contains BottleNeck within its two layers
                )
        self.layer2 = nn.Sequential(*self.layer2)
        
        if not self.use_gt_depth:
            self.depth_head = QuantModule(camencode.depth_head, weight_quant_params, act_quant_params)
        self.image_head = QuantModule(camencode.image_head, weight_quant_params, act_quant_params)
        
    def get_depth_dist(self, x, eps=1e-5):  # 对深度维进行softmax，得到每个像素不同深度的概率
        return F.softmax(x, dim=1)
    
    def get_gt_depth_dist(self, x):  # 对深度维进行onehot，得到每个像素不同深度的概率
        """
        Args:
            x: [B*N, H, W]
        Returns:
            x: [B*N, D, fH, fW]
        """
        target = self.training
        x = x.clamp(max=self.d_max) # save memory
        # [B*N, H, W], indices (float), value: [0, num_bins)
        depth_indices, mask = bin_depths(x, self.mode, self.d_min, self.d_max, self.num_bins, target=target)
        depth_indices = depth_indices[:, self.downsample//2::self.downsample, self.downsample//2::self.downsample]
        onehot_dist = F.one_hot(depth_indices.long()).permute(0,3,1,2) # [B*N, num_bins, fH, fW]

        if not target:
            mask = mask[:, self.downsample//2::self.downsample, self.downsample//2::self.downsample].unsqueeze(1)
            onehot_dist *= mask

        return onehot_dist, depth_indices
    
    def resnet101_forward(self, x):
        x = self.conv1(x) # bns and relu are included in conv1
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def get_resnet_features(self, x):  # 使用resnet101提取特征
        #x: 16 x 3 x 480 x 640
        return self.resnet101_forward(x)

    def forward(self, x):
        """
        Returns:
            log_depth : [B*N, D, fH, fW], or None if not used latter
            depth_gt_indices : [B*N, fH, fW], or None if not used latter
            new_x : [B*N, C, D, fH, fW]
        """
        #x: 16 x 3 x 480 x 640
        #print(x.shape)
        x_img = x[:,:3,:,:].clone()
        features = self.get_resnet_features(x_img)  # depth: B*N x D x fH x fW(24 x 41 x 8 x 22)  x: B*N x C x D x fH x fW(24 x 64 x 41 x 8 x 22)
        x_img_feature = self.image_head(features)
        
        if self.depth_supervision or self.use_gt_depth: # depth data must exist
            x_depth = x[:,3,:,:]
            depth_gt, depth_gt_indices = self.get_gt_depth_dist(x_depth)

        if self.use_gt_depth:
            new_x = depth_gt.unsqueeze(1) * x_img_feature.unsqueeze(2) # new_x: 24 x 64 x 41 x 8 x 18
            return None, new_x
        else:
            depth_logit = self.depth_head(features)
            depth = self.get_depth_dist(depth_logit)
            new_x = depth.unsqueeze(1) * x_img_feature.unsqueeze(2) # new_x: 24 x 64 x 41 x 8 x 18
            if self.depth_supervision:
                return (depth_logit, depth_gt_indices), new_x
            else:
                return None, new_x


class QuantLiftSplatShoot(nn.Module):
    def __init__(self, lift_splat_shoot: LiftSplatShoot, weight_quant_params={}, act_quant_params={}):
        super().__init__()
        self.grid_conf = lift_splat_shoot.grid_conf
        self.data_aug_conf = lift_splat_shoot.data_aug_conf
        self.dx = lift_splat_shoot.dx
        self.bx = lift_splat_shoot.bx
        self.nx = lift_splat_shoot.nx
        self.depth_supervision = lift_splat_shoot.depth_supervision
        self.downsample = lift_splat_shoot.downsample
        self.camC = lift_splat_shoot.camC
        self.frustum = lift_splat_shoot.frustum
        self.use_quickcumsum = lift_splat_shoot.use_quickcumsum
        self.D = lift_splat_shoot.D
        self.camera_encoder_type = lift_splat_shoot.camera_encoder_type
        if self.camera_encoder_type == 'EfficientNet':
            raise NotImplementedError("EfficientNet is not supported in QuantLiftSplatShoot yet.")
        elif self.camera_encoder_type == 'Resnet101':
            self.camencode = QuantCamEncode_Resnet101(lift_splat_shoot.camencode, weight_quant_params, act_quant_params)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  # B:4(batchsize)    N: 4(相机数目)

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],  # points[:, :, :, :, :, 2:3] ranges from [4, 45) meters
                            points[:, :, :, :, :, 2:3]
                            ), 5)  # 将像素坐标(u,v,d)变成齐次坐标(du,dv,d)
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)  # 将像素坐标d[u,v,1]^T转换到车体坐标系下的[x,y,z]
        
        return points  # B x N x D x H x W x 3 (4 x 4 x 41 x 16 x 22 x 3) 

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape  # B: 4  N: 4  C: 3  imH: 256  imW: 352

        x = x.view(B*N, C, imH, imW)  # B和N两个维度合起来  x: 16 x 4 x 256 x 352
        depth_items, x = self.camencode(x) # 进行图像编码  x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)  #将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 16 x 22)
        x = x.permute(0, 1, 3, 4, 5, 2)  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)

        return x, depth_items
    
    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B*N*D*H*W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)  # 将图像展平，一共有 B*N*D*H*W 个点

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  # 将[-48,48] [-10 10]的范围平移到 [0, 240), [0, 1) 计算栅格坐标并取整
        geom_feats = geom_feats.view(Nprime, 3)  # 将像素映射关系同样展平  geom_feats: B*N*D*H*W x 3 
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: B*N*D*H*W x 4, geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~240  y: 0~240  z: 0
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept] 
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)  # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # modify griddify (B x C x Z x Y x X) by Yifan Lu 2022.10.7
        # ------> x
        # |
        # |
        # y
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)  # final: 4 x 64 x Z x Y x X
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x  # 将x按照栅格坐标放到final中

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维

        return final  # final: 4 x 64 x 240 x 240  # B, C, H, W

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3 (4 x N x 42 x 16 x 22 x 3)
        x_img, depth_items = self.get_cam_feats(x)  # 提取图像特征并预测深度编码 x: B x N x D x fH x fW x C(4 x N x 42 x 16 x 22 x 64)
        x = self.voxel_pooling(geom, x_img)  # x: 4 x 64 x 240 x 240
        return x, depth_items
    
    def forward(self, data_dict, modality_name):
        image_inputs_dict = data_dict[f'inputs_{modality_name}']
        x, rots, trans, intrins, post_rots, post_trans = \
                        image_inputs_dict['imgs'], \
                        image_inputs_dict['rots'], \
                        image_inputs_dict['trans'], \
                        image_inputs_dict['intrins'], \
                        image_inputs_dict['post_rots'], \
                        image_inputs_dict['post_trans']
        x, depth_items = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)  # 将图像转换到BEV下，x: B x C x 240 x 240 (4 x 64 x 240 x 240)
        if self.depth_supervision:
            self.depth_items = depth_items
        return x
    

class QuantVoxelBackBone8x(BaseQuantBlock):
    def __init__(self, voxel_backbone: VoxelBackBone8x, weight_quant_params={}, act_quant_params={}):
        super().__init__()

        self.model_cfg = voxel_backbone.model_cfg
        self.sparse_shape = voxel_backbone.sparse_shape
        self.num_point_features = voxel_backbone.num_point_features
        self.backbone_channels = voxel_backbone.backbone_channels

        def quantize_sparse_seq(sparse_seq):
            quant_seq = nn.Sequential()
            for i, layer in enumerate(sparse_seq):
                if isinstance(layer, (SubMConv3d, SparseConv3d)):
                    quant_layer = QuantSpconvModule(layer, weight_quant_params, act_quant_params)
                    quant_layer.is_sparse_conv = True
                    quant_seq.add_module(f"quant_conv_{i}", quant_layer)
                elif isinstance(layer, nn.BatchNorm1d):
                    quant_seq[-1].norm_function = layer
                elif isinstance(layer, nn.ReLU):
                    if isinstance(quant_seq[-1], QuantSpconvModule):
                        quant_seq[-1].activation_function = layer
                    else:
                        quant_seq[-2].activation_function = layer
                # elif isinstance(layer, StraightThrough):
                #     continue
                elif isinstance(layer, (nn.Sequential, SparseSequential)):
                    quant_seq.add_module(f"layer_{i}", quantize_sparse_seq(layer))

            return quant_seq

        # Convert all layers
        self.conv_input = quantize_sparse_seq(voxel_backbone.conv_input)
        self.conv1 = quantize_sparse_seq(voxel_backbone.conv1)
        self.conv2 = quantize_sparse_seq(voxel_backbone.conv2)
        self.conv3 = quantize_sparse_seq(voxel_backbone.conv3)
        self.conv4 = quantize_sparse_seq(voxel_backbone.conv4)
        self.conv_out = quantize_sparse_seq(voxel_backbone.conv_out)

    def forward(self, input_sp_tensor):
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        return out


class QuantSECOND(nn.Module):
    def __init__(self, second: SECOND, weight_quant_params={}, act_quant_params={}):
        super().__init__()
        # Preserve original modules
        self.vfe = second.vfe
        self.map_to_bev = second.map_to_bev

        # Replace spconv block with quantized version
        self.spconv_block = QuantVoxelBackBone8x(
            second.spconv_block,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params
        )

    def forward(self, data_dict, modality_name):
        voxel_features = data_dict[f'inputs_{modality_name}']['voxel_features']
        voxel_coords = data_dict[f'inputs_{modality_name}']['voxel_coords']
        voxel_num_points = data_dict[f'inputs_{modality_name}']['voxel_num_points']
        batch_size = voxel_coords[:, 0].max().item() + 1

        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'batch_size': batch_size
        }

        batch_dict = self.vfe(batch_dict)
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        input_sp_tensor = SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.spconv_block.sparse_shape,
            batch_size=batch_size
        )
        
        encoded_spconv_tensor = self.spconv_block(input_sp_tensor)
        batch_dict['encoded_spconv_tensor'] = encoded_spconv_tensor
        batch_dict = self.map_to_bev(batch_dict)

        return batch_dict['spatial_features']


class QuantHGTCavAttention(BaseQuantBlock):
    def __init__(self, hgt_cav_attention: HGTCavAttention, weight_quant_params={}, act_quant_params={}):
        super().__init__()

        self.heads = hgt_cav_attention.heads
        self.scale = hgt_cav_attention.scale
        self.num_types = hgt_cav_attention.num_types
        self.attend = hgt_cav_attention.attend
        self.drop_out = hgt_cav_attention.drop_out

        # Quantize linear projections for each node type
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()

        for t in range(self.num_types):
            self.k_linears.append(
                QuantModule(hgt_cav_attention.k_linears[t], weight_quant_params, act_quant_params)
            )
            self.q_linears.append(
                QuantModule(hgt_cav_attention.q_linears[t], weight_quant_params, act_quant_params)
            )
            self.v_linears.append(
                QuantModule(hgt_cav_attention.v_linears[t], weight_quant_params, act_quant_params)
            )
            self.a_linears.append(
                QuantModule(hgt_cav_attention.a_linears[t], weight_quant_params, act_quant_params)
            )

        self.relation_att = hgt_cav_attention.relation_att
        self.relation_msg = hgt_cav_attention.relation_msg

    def to_qkv(self, x, types):
        # x: (B,H,W,L,C)
        # types: (B,L)
        q_batch = []
        k_batch = []
        v_batch = []

        for b in range(x.shape[0]):
            q_list = []
            k_list = []
            v_list = []

            for i in range(x.shape[-2]):
                # (H,W,1,C)
                q_list.append(
                    self.q_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
                k_list.append(
                    self.k_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
                v_list.append(
                    self.v_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
            # (1,H,W,L,C)
            q_batch.append(torch.cat(q_list, dim=2).unsqueeze(0))
            k_batch.append(torch.cat(k_list, dim=2).unsqueeze(0))
            v_batch.append(torch.cat(v_list, dim=2).unsqueeze(0))
        # (B,H,W,L,C)
        q = torch.cat(q_batch, dim=0)
        k = torch.cat(k_batch, dim=0)
        v = torch.cat(v_batch, dim=0)
        return q, k, v
    
    def get_relation_type_index(self, type1, type2):
        return type1 * self.num_types + type2

    def get_hetero_edge_weights(self, x, types):
        w_att_batch = []
        w_msg_batch = []

        for b in range(x.shape[0]):
            w_att_list = []
            w_msg_list = []

            for i in range(x.shape[-2]):
                w_att_i_list = []
                w_msg_i_list = []

                for j in range(x.shape[-2]):
                    e_type = self.get_relation_type_index(types[b, i],
                                                          types[b, j])
                    w_att_i_list.append(self.relation_att[e_type].unsqueeze(0))
                    w_msg_i_list.append(self.relation_msg[e_type].unsqueeze(0))
                w_att_list.append(torch.cat(w_att_i_list, dim=0).unsqueeze(0))
                w_msg_list.append(torch.cat(w_msg_i_list, dim=0).unsqueeze(0))

            w_att_batch.append(torch.cat(w_att_list, dim=0).unsqueeze(0))
            w_msg_batch.append(torch.cat(w_msg_list, dim=0).unsqueeze(0))

        # (B,M,L,L,C_head,C_head)
        w_att = torch.cat(w_att_batch, dim=0).permute(0, 3, 1, 2, 4, 5)
        w_msg = torch.cat(w_msg_batch, dim=0).permute(0, 3, 1, 2, 4, 5)
        return w_att, w_msg

    def to_out(self, x, types):
        out_batch = []
        for b in range(x.shape[0]):
            out_list = []
            for i in range(x.shape[-2]):
                out_list.append(
                    self.a_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
            out_batch.append(torch.cat(out_list, dim=2).unsqueeze(0))
        out = torch.cat(out_batch, dim=0)
        return out

    def forward(self, x, mask, prior_encoding):
        # x: (B, L, H, W, C) -> (B, H, W, L, C)
        # mask: (B, H, W, L, 1)
        # prior_encoding: (B,L,H,W,3)
        x = x.permute(0, 2, 3, 1, 4)
        # mask: (B, 1, H, W, L, 1)
        mask = mask.unsqueeze(1)
        # (B,L)
        velocities, dts, types = [itm.squeeze(-1) for itm in
                                  prior_encoding[:, :, 0, 0, :].split(
                                      [1, 1, 1], dim=-1)]
        types = types.to(torch.int)
        dts = dts.to(torch.int)
        qkv = self.to_qkv(x, types)
        # (B,M,L,L,C_head,C_head)
        w_att, w_msg = self.get_hetero_edge_weights(x, types)

        # q: (B, M, H, W, L, C)
        q, k, v = map(lambda t: rearrange(t, 'b h w l (m c) -> b m h w l c',
                                          m=self.heads), (qkv))
        # attention, (B, M, H, W, L, L)
        att_map = torch.einsum(
            'b m h w i p, b m i j p q, bm h w j q -> b m h w i j',
            [q, w_att, k]) * self.scale
        # add mask
        att_map = att_map.masked_fill(mask == 0, -float('inf'))
        # softmax
        att_map = self.attend(att_map)

        # out:(B, M, H, W, L, C_head)
        v_msg = torch.einsum('b m i j p c, b m h w j p -> b m h w i j c',
                             w_msg, v)
        out = torch.einsum('b m h w i j, b m h w i j c -> b m h w i c',
                           att_map, v_msg)

        out = rearrange(out, 'b m h w l c -> b h w l (m c)',
                        m=self.heads)
        out = self.to_out(out, types)
        out = self.drop_out(out)
        # (B L H W C)
        out = out.permute(0, 3, 1, 2, 4)
        return out


class QuantBaseWindowAttention(BaseQuantBlock):
    def __init__(self, base_attn: BaseWindowAttention, weight_quant_params={}, act_quant_params={}):
        super().__init__()
        self.heads = base_attn.heads
        self.scale = base_attn.scale
        self.window_size = base_attn.window_size
        self.relative_pos_embedding = base_attn.relative_pos_embedding

        # Quantized QKV projection
        self.to_qkv = QuantModule(base_attn.to_qkv, weight_quant_params, act_quant_params)

        # Relative positional embedding
        if self.relative_pos_embedding:
            self.relative_indices = base_attn.relative_indices
            self.pos_embedding = base_attn.pos_embedding  # This remains a parameter
        else:
            self.pos_embedding = base_attn.pos_embedding  # Also a parameter in this case

        # Output projection (first Linear layer only needs quantization)
        self.to_out = nn.Sequential(
            QuantModule(base_attn.to_out[0], weight_quant_params, act_quant_params),
            base_attn.to_out[1]  # Dropout stays as-is
        )

    def forward(self, x):
        b, l, h, w, c, m = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        new_h = h // self.window_size
        new_w = w // self.window_size

        # q : (b, l, m, new_h*new_w, window_size^2, c_head)
        q, k, v = map(
            lambda t: rearrange(t,
                                'b l (new_h w_h) (new_w w_w) (m c) -> b l m (new_h new_w) (w_h w_w) c',
                                m=m, w_h=self.window_size,
                                w_w=self.window_size), qkv)
        # b l m h window_size window_size
        dots = torch.einsum('b l m h i c, b l m h j c -> b l m h i j',
                            q, k, ) * self.scale
        # consider prior knowledge of the local window
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0],
                                       self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b l m h i j, b l m h j c -> b l m h i c', attn, v)
        # b l h w c
        out = rearrange(out,
                        'b l m (new_h new_w) (w_h w_w) c -> b l (new_h w_h) (new_w w_w) (m c)',
                        m=self.heads, w_h=self.window_size,
                        w_w=self.window_size,
                        new_w=new_w, new_h=new_h)
        out = self.to_out(out)

        return out


class QuantSplitAttn(BaseQuantBlock):
    def __init__(self, split_attn: SplitAttn, weight_quant_params={}, act_quant_params={}):
        super().__init__()
        self.input_dim = split_attn.input_dim

        self.fc1 = QuantModule(split_attn.fc1, weight_quant_params, act_quant_params)
        self.fc1.norm_function = split_attn.bn1
        self.fc1.activation_function = split_attn.act1

        self.fc2 = QuantModule(split_attn.fc2, weight_quant_params, act_quant_params)
        self.rsoftmax = split_attn.rsoftmax

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, window_list):
        # window list: [(B, L, H, W, C) * 3]
        assert len(window_list) == 3, 'only 3 windows are supported'

        sw, mw, bw = window_list
        B, L = sw.shape[0], sw.shape[1]

        # Global average pooling
        x_gap = sw + mw + bw
        x_gap = x_gap.mean((2, 3), keepdim=True)  # B, L, 1, 1, C

        x_gap = self.fc1(x_gap)                   # B, L, 1, 1, C
        if self.use_act_quant:
            x_gap = self.act_quantizer(x_gap) # act quantization
        x_attn = self.fc2(x_gap)                  # B, L, 1, 1, 3C

        x_attn = self.rsoftmax(x_attn).view(B, L, 1, 1, -1)

        out = (
            sw * x_attn[:, :, :, :, 0:self.input_dim] +
            mw * x_attn[:, :, :, :, self.input_dim:2*self.input_dim] +
            bw * x_attn[:, :, :, :, 2*self.input_dim:]
        )

        return out


class QuantPyramidWindowAttention(BaseQuantBlock):
    def __init__(self, base_attn: PyramidWindowAttention, weight_quant_params={}, act_quant_params={}):
        super().__init__()

        assert isinstance(base_attn.pwmsa, nn.ModuleList)
        self.fuse_mehod = base_attn.fuse_mehod
        self.pwmsa = nn.ModuleList()

        # Convert all BaseWindowAttention to QuantBaseWindowAttention
        for sub_attn in base_attn.pwmsa:
            self.pwmsa.append(
                QuantBaseWindowAttention(sub_attn, weight_quant_params, act_quant_params)
            )

        # Wrap split attention if used
        if self.fuse_mehod == 'split_attn':
            self.split_attn = QuantSplitAttn(base_attn.split_attn, weight_quant_params, act_quant_params)
        elif self.fuse_mehod == 'split_attn128':
            self.split_attn = QuantSplitAttn(base_attn.split_attn, weight_quant_params, act_quant_params)
        elif self.fuse_mehod == 'split_attn64':
            self.split_attn = QuantSplitAttn(base_attn.split_attn, weight_quant_params, act_quant_params)

    def forward(self, x):
        if self.fuse_mehod == 'naive':
            output = None
            for wmsa in self.pwmsa:
                output = wmsa(x) if output is None else output + wmsa(x)
            return output / len(self.pwmsa)

        elif self.fuse_mehod.startswith('split_attn'):
            window_list = [wmsa(x) for wmsa in self.pwmsa]
            return self.split_attn(window_list)


class QuantV2XFusionBlock(BaseQuantBlock):
    def __init__(self, fusion_block: V2XFusionBlock, weight_quant_params={}, act_quant_params={}):
        super().__init__()
        self.num_blocks = fusion_block.num_blocks
        self.layers = nn.ModuleList()

        for i in range(self.num_blocks):
            # Get original attention blocks
            cav_attn = fusion_block.layers[i][0].fn  # PreNorm(wrapped HGTCavAttention)
            pwin_attn = fusion_block.layers[i][1].fn  # PreNorm(wrapped PyramidWindowAttention)

            # Quantize them
            quant_cav_attn = QuantHGTCavAttention(cav_attn, weight_quant_params, act_quant_params)
            quant_pwin_attn = QuantPyramidWindowAttention(pwin_attn, weight_quant_params, act_quant_params)

            # Wrap them again in PreNorm
            dim = fusion_block.layers[i][0].norm.normalized_shape[0]
            wrapped_cav_attn = PreNorm(dim, quant_cav_attn)
            wrapped_pwin_attn = PreNorm(dim, quant_pwin_attn)

            self.layers.append(nn.ModuleList([wrapped_cav_attn, wrapped_pwin_attn]))

    def forward(self, x, mask, prior_encoding):
        for cav_attn, pwin_attn in self.layers:
            x = cav_attn(x, mask=mask, prior_encoding=prior_encoding) + x
            x = pwin_attn(x) + x
        return x


class QuantFeedForward(BaseQuantBlock):
    def __init__(self, feedforward: FeedForward, weight_quant_params={}, act_quant_params={}):
        super().__init__()

        layers = feedforward.net # a sequantial of layers
        self.fc1 = QuantModule(layers[0], weight_quant_params, act_quant_params)
        self.fc1.activation_function = layers[1]  # GELU
        self.dropout1 = layers[2]

        self.fc2 = QuantModule(layers[3], weight_quant_params, act_quant_params, disable_act_quant=True)
        self.dropout2 = layers[4]

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        if self.use_act_quant:
            x = self.act_quantizer(x)
        return x


class QuantV2XTEncoder(BaseQuantBlock):
    def __init__(self, v2xt_encoder: V2XTEncoder, weight_quant_params={}, act_quant_params={}):
        super().__init__()

        self.downsample_rate = v2xt_encoder.downsample_rate
        self.discrete_ratio = v2xt_encoder.discrete_ratio
        self.use_roi_mask = v2xt_encoder.use_roi_mask
        self.use_RTE = v2xt_encoder.use_RTE
        self.RTE_ratio = v2xt_encoder.RTE_ratio

        self.sttf = v2xt_encoder.sttf

        if self.use_RTE:
            self.rte = v2xt_encoder.rte

        self.layers = nn.ModuleList()
        for block, ff in v2xt_encoder.layers:
            quant_block = QuantV2XFusionBlock(block, weight_quant_params, act_quant_params)
            quant_ff = QuantFeedForward(ff.fn, weight_quant_params, act_quant_params)
            self.layers.append(
                nn.ModuleList([
                    quant_block,
                    PreNorm(ff.norm.normalized_shape[0], quant_ff)
                ])
            )

    def forward(self, x, mask, spatial_correction_matrix):

        # transform the features to the current timestamp
        # velocity, time_delay, infra
        # (B,L,H,W,3)
        prior_encoding = x[..., -3:]
        # (B,L,H,W,C)
        x = x[..., :-3]
        if self.use_RTE:
            # dt: (B,L)
            dt = prior_encoding[:, :, 0, 0, 1].to(torch.int)
            x = self.rte(x, dt)
        x = self.sttf(x, mask, spatial_correction_matrix)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask else get_roi_and_cav_mask(x.shape,
                                                                  mask,
                                                                  spatial_correction_matrix,
                                                                  self.discrete_ratio,
                                                                  self.downsample_rate)
        for attn, ff in self.layers:
            x = attn(x, mask=com_mask, prior_encoding=prior_encoding)
            x = ff(x) + x
        return x


class QuantV2XTransformer(BaseQuantBlock):
    def __init__(self, v2xvittransformer: V2XTransformer, weight_quant_params={}, act_quant_params={}):
        super().__init__()
        self.encoder = QuantV2XTEncoder(
            v2xvittransformer.encoder,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params
        )
    
    def forward(self, x, mask, spatial_correction_matrix):
        output = self.encoder(x, mask, spatial_correction_matrix)
        output = output[:, 0]
        return output


class QuantV2XViTFusion(BaseQuantBlock):
    def __init__(self, v2xvit: V2XViTFusion, weight_quant_params={}, act_quant_params={}):
        super().__init__()
        self.fusion_net = QuantV2XTransformer(
            v2xvit.fusion_net,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params
        )

    def forward(self, x, record_len, affine_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        affine_matrix : torch.Tensor
            normalized affine matrix from 'normalize_pairwise_tfm'
            shape: (B, L, L, 2, 3) 
        """
        _, C, H, W = x.shape
        B, L = affine_matrix.shape[:2]

        regroup_feature, mask = Regroup(x, record_len, L)
        prior_encoding = \
            torch.zeros(len(record_len), L, 3, 1, 1).to(record_len.device)
        
        # prior encoding should include [velocity, time_delay, infra], but it is not supported by all basedataset.
        # it is possible to modify the xxx_basedataset.py and intermediatefusiondataset.py to retrieve these information
        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                               regroup_feature.shape[3],
                                               regroup_feature.shape[4])

        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)
        regroup_feature_new = []

        for b in range(B):
            ego = 0
            regroup_feature_new.append(warp_affine_simple(regroup_feature[b], affine_matrix[b, ego], (H, W)))
        regroup_feature = torch.stack(regroup_feature_new)

        # b l c h w -> b l h w c
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # transformer fusion. In perfect setting, there is no delay. 
        # it is possible to modify the xxx_basedataset.py and intermediatefusiondataset.py to retrieve these information
        spatial_correction_matrix = torch.eye(4).expand(len(record_len), L, 4, 4).to(record_len.device)
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2)
        
        return fused_feature
    

class QuantNaiveCompressor(BaseQuantBlock):
    """
    Quantized version of NaiveCompressor.
    Wraps Conv2D layers inside QuantModule while preserving BN and ReLU.
    """
    def __init__(self, naive_compressor: NaiveCompressor, weight_quant_params={}, act_quant_params={}):
        super().__init__()
        self.encoder = nn.Sequential(
            QuantModule(naive_compressor.encoder[0], weight_quant_params, act_quant_params),
        )
        self.encoder[0].norm_function = naive_compressor.encoder[1]
        self.encoder[0].activation_function = naive_compressor.encoder[2]
        self.decoder = nn.Sequential(
            QuantModule(naive_compressor.decoder[0], weight_quant_params, act_quant_params),
        )
        self.decoder[0].norm_function = naive_compressor.decoder[1]
        self.decoder[0].activation_function = naive_compressor.decoder[2]

        self.decoder.append(
            QuantModule(naive_compressor.decoder[3], weight_quant_params, act_quant_params)
        )
        self.decoder[1].norm_function = naive_compressor.decoder[4]
        self.decoder[1].activation_function = naive_compressor.decoder[5]

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


specials = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    ResBottleneckBlock: QuantResBottleneckBlock,
    InvertedResidual: QuantInvertedResidual,
    _InvertedResidual: _QuantInvertedResidual
}

opencood_specials = {
    PyramidFusion: QuantPyramidFusion,
    ResNetBEVBackbone: QuantResNetBEVBackbone,
    BaseBEVBackbone: QuantBaseBEVBackbone,
    DownsampleConv: QuantDownsampleConv,
    PointPillar: QuantPointPillar,
    LiftSplatShoot: QuantLiftSplatShoot,
    SECOND: QuantSECOND,
    V2XViTFusion: QuantV2XViTFusion, # v2xvit
    NaiveCompressor: QuantNaiveCompressor,
}

specials_unquantized = [
    # nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.Dropout,
    # nn.ConvTranspose2d, nn.ReLU, nn.Identity,
    # PointPillarScatter, AlignNet, DoubleConv
]

specials_unquantized_names = [
                            # 'encoder_m1', 
                            #   'backbone_m1',
                              'aligner_m1', 
                            #   'encoder_m2', 
                            #   'backbone_m2',
                              'aligner_m2', 
                            #   'pyramid_backbone',
                            #   'shrink_conv',  
                            #   'cls_head', 
                            #   'reg_head', 
                            #   'dir_head',
                            # 'fusion_net',
                            # 'shrinker_m1',
                            # 'shrinker_m2',
                              'codebook',
                              ] # blocks we don't quantize