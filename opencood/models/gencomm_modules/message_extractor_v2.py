# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.ops import DeformConv2d

# class BEVDeformableExtractor(nn.Module):
#     def __init__(self, in_channels=128, out_channels=2):
#         super().__init__()
#         # Replace multi-scale convs with Deformable Convs
#         self.offset1 = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
#         self.offset2 = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
#         self.offset3 = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)

#         self.dcn1 = DeformConv2d(in_channels, 64, kernel_size=3, padding=1)
#         self.dcn2 = DeformConv2d(in_channels, 64, kernel_size=3, padding=1)
#         self.dcn3 = DeformConv2d(in_channels, 64, kernel_size=3, padding=1)

#         self.fuse = nn.Sequential(
#             nn.Conv2d(192, 64, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(64, out_channels, kernel_size=1)
#         )

#         self.attn = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(192, 48, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(48, 192, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         offset1 = self.offset1(x)
#         offset2 = self.offset2(x)
#         offset3 = self.offset3(x)

#         b1 = self.dcn1(x, offset1)
#         b2 = self.dcn2(x, offset2)
#         b3 = self.dcn3(x, offset3)

#         concat = torch.cat([b1, b2, b3], dim=1)
#         attn = self.attn(concat)
#         enhanced = concat * attn

#         return self.fuse(enhanced)

# class MessageExtractorv2(nn.Module):
#     def __init__(self, in_channels=128, out_channels=2):
#         super(MessageExtractorv2, self).__init__()

#         self.bev_extractor = BEVDeformableExtractor(in_channels, out_channels)

#     def forward(self, bev_feature,):

#         enhanced_feature = self.bev_extractor(bev_feature)
#         return enhanced_feature
    
# if __name__ == '__main__':
#     # 测试前景增强模块
#     in_channels = 128
#     reduction_ratio = 16
#     bev_feature = torch.randn(4, in_channels, 100, 352)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class BEVDeformableExtractor(nn.Module):
    def __init__(self, in_channels=128, out_channels=2):
        super().__init__()
        # Replace multi-scale convs with Deformable Convs
        self.offset1 = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
        # self.offset2 = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
        # self.offset3 = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)

        self.dcn1 = DeformConv2d(in_channels, 64, kernel_size=3, padding=1)
        # self.dcn2 = DeformConv2d(in_channels, 64, kernel_size=3, padding=1)
        # self.dcn3 = DeformConv2d(in_channels, 64, kernel_size=3, padding=1)

        self.fuse = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        offset1 = self.offset1(x)
        # offset2 = self.offset2(x)
        # offset3 = self.offset3(x)

        b1 = self.dcn1(x, offset1)
        # b2 = self.dcn2(x, offset2)
        # b3 = self.dcn3(x, offset3)

        # concat = torch.cat([b1, b2, b3], dim=1)
        attn = self.attn(b1)
        enhanced = b1 * attn

        return self.fuse(enhanced)

class MessageExtractorv2(nn.Module):
    def __init__(self, in_channels=128, out_channels=2):
        super(MessageExtractorv2, self).__init__()

        self.bev_extractor = BEVDeformableExtractor(in_channels, out_channels)

    def forward(self, bev_feature,):

        enhanced_feature = self.bev_extractor(bev_feature)
        return enhanced_feature
    
if __name__ == '__main__':
    # 测试前景增强模块
    in_channels = 128
    reduction_ratio = 16
    bev_feature = torch.randn(4, in_channels, 100, 352)

    #params_calculation
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(MessageExtractorv2(in_channels, 2))
    print(f"Number of parameters in MessageExtractorv2: {num_params}")
