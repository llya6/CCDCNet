import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss
# RCB (Residual Convolution Block)

class RES_Dilated_Conv(nn.Module):
    """
    Wide-Focus Residual Block with dilated convolutions.
    """

    def __init__(self, in_channels, out_channels):
        super(RES_Dilated_Conv,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", bias=False) #dilation=1
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", dilation=2, bias=False) #dilation=2
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", dilation=3, bias=False) #dilation=3
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False) #dilation=1
        
        # Ensure input/output have same number of channels for residual connection
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        
        # Optional: BatchNorm for better training stability
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # First dilated conv branch
        x1 = self.conv1(x)
        x1 = self.bn1(x1) 
        x1 = F.gelu(x1) 
        x1 = F.dropout(x1, p=0.1) 

        # Second dilated conv branch
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, p=0.1)

        # Third dilated conv branch
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = F.gelu(x3)
        x3 = F.dropout(x3, p=0.1)

        # Combine the branches
        added = torch.add(x1, x2)
        added = torch.add(added, x3)

        # Apply final convolution
        x_out = self.conv4(added)
        x_out = self.bn4(x_out)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, p=0.1)

        # Add residual connection (shortcut)
        residual = self.shortcut(x)  # Either identity or 1x1 conv
        x_out += residual  # Residual connection
        return x_out


class RS_Dblock(nn.Module):
    def __init__(self, channel):
        super(RS_Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1, bias=False)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2, bias=False)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4, bias=False)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8, bias=False)

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x):
        dilate1_out = self.act(self.dilate1(x))
        dilate1_out = self.dropout(dilate1_out)
        dilate1_out += x 

        dilate2_out = self.act(self.dilate2(dilate1_out))
        dilate2_out = self.dropout(dilate2_out)
        dilate2_out += dilate1_out  

        dilate3_out = self.act(self.dilate3(dilate2_out))
        dilate3_out = self.dropout(dilate3_out)
        dilate3_out += dilate2_out  

        dilate4_out = self.act(self.dilate4(dilate3_out))
        dilate4_out = self.dropout(dilate4_out)
        dilate4_out += dilate3_out  

        out = x + dilate4_out
        return out


class Upsample(nn.Module):

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels):
        super(DecoderBlock, self).__init__()

        self.identity = nn.Sequential(
            Upsample(2, mode="bilinear"),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        )

        # self.decode = nn.Sequential(
        #     Upsample(2, mode="bilinear"),
        #     # nn.BatchNorm2d(input_channels),
        #     DSC(input_channels, input_channels),
        #     # nn.BatchNorm2d(input_channels),
        #     nn.ReLU(inplace=True),
        #     DSC(input_channels, output_channels),
        #     # nn.BatchNorm2d(output_channels),
        # )
        self.decodeD = nn.Sequential(
            Upsample(2, mode="bilinear"),
            ResBlock_CBAM(input_channels, output_channels//4),
            nn.BatchNorm2d(output_channels),
        )
        
        self.pam = PAM_Module(input_channels)
        self.cam = CAM_Module(input_channels)
    def forward(self, x):
        residual = self.identity(x)
        out = self.decodeD(x)
        out = F.interpolate(out, size=residual.shape[2:], mode='bilinear', align_corners=False)
        out += residual

        return out


class GatedFusion(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.gate = nn.Sequential(
            # nn.BatchNorm2d(in_channels),
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
            # nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        G = self.gate(out)

        PG = x * G
        FG = y * (1 - G)

        return FG + PG
    

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple learnable scale parameter (replacement for mmcv's Scale)
class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale

# ConvModule replacement
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, norm_cfg=True, act_cfg=True):
        super(ConvModule, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)]
        if norm_cfg:
            layers.append(nn.BatchNorm2d(out_channels))
        if act_cfg:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

# SpatialQKVBlock
class SpatialQKVBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super(SpatialQKVBlock, self).__init__()
        self.conv_q = ConvModule(in_channels, channels, 1, norm_cfg=False, act_cfg=False)
        self.conv_k = ConvModule(in_channels, channels, 1, norm_cfg=False, act_cfg=False)
        self.conv_v = ConvModule(in_channels, channels, 1, norm_cfg=False, act_cfg=False)

    def forward(self, x):
        N, C, H, W = x.shape
        x_q = self.conv_q(x).reshape(N, -1, H * W).permute(0, 2, 1).contiguous()
        x_k = self.conv_k(x).reshape(N, -1, H * W)
        x_v = self.conv_v(x).reshape(N, -1, H * W).permute(0, 2, 1).contiguous()
        return x_q, x_k, x_v

# SpatialAttBlock
class SpatialAttBlock(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(SpatialAttBlock, self).__init__()
        self.qkv_opt = SpatialQKVBlock(in_channels, channels)
        self.qkv_sar = SpatialQKVBlock(in_channels, channels)
        self.gamma_opt = Scale(0)
        self.gamma_sar = Scale(0)
        self.project_opt = ConvModule(channels, out_channels, kernel_size=1, norm_cfg=False, act_cfg=False)
        self.project_sar = ConvModule(channels, out_channels, kernel_size=1, norm_cfg=False, act_cfg=False)

    def forward(self, x_opt, x_sar):
        N, C, H, W = x_opt.shape
        q_opt, k_opt, v_opt = self.qkv_opt(x_opt)
        q_sar, k_sar, v_sar = self.qkv_sar(x_sar)

        att_opt = F.softmax(torch.bmm(q_opt, k_opt), dim=-1)
        att_sar = F.softmax(torch.bmm(q_sar, k_sar), dim=-1)
        att = torch.bmm(att_opt, att_sar)

        s_opt = torch.bmm(att, v_opt).permute(0, 2, 1).contiguous().reshape(N, -1, H, W)
        s_opt = self.project_opt(s_opt)
        s_opt = x_opt + self.gamma_opt(s_opt)

        s_sar = torch.bmm(att, v_sar).permute(0, 2, 1).contiguous().reshape(N, -1, H, W)
        s_sar = self.project_sar(s_sar)
        s_sar = x_sar + self.gamma_sar(s_sar)

        return s_opt + s_sar

# ChannelQKVBlock
class ChannelQKVBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super(ChannelQKVBlock, self).__init__()
        self.conv_q = ConvModule(in_channels, channels, 1, norm_cfg=False, act_cfg=False)
        self.conv_k = ConvModule(in_channels, channels, 1, norm_cfg=False, act_cfg=False)
        self.conv_v = ConvModule(in_channels, channels, 1, norm_cfg=False, act_cfg=False)

    def forward(self, x):
        N, C, H, W = x.shape
        x_q = self.conv_q(x).reshape(N, -1, H * W)
        x_k = self.conv_k(x).reshape(N, -1, H * W).permute(0, 2, 1).contiguous()
        x_v = self.conv_v(x).reshape(N, -1, H * W)
        return x_q, x_k, x_v

# ChannelAttBlock
class ChannelAttBlock(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(ChannelAttBlock, self).__init__()
        self.qkv_opt = ChannelQKVBlock(in_channels, channels)
        self.qkv_sar = ChannelQKVBlock(in_channels, channels)
        self.gamma_opt = Scale(0)
        self.gamma_sar = Scale(0)
        self.project_opt = ConvModule(channels, out_channels, kernel_size=1, norm_cfg=False, act_cfg=False)
        self.project_sar = ConvModule(channels, out_channels, kernel_size=1, norm_cfg=False, act_cfg=False)

    def forward(self, x_opt, x_sar):
        N, C, H, W = x_opt.shape
        q_opt, k_opt, v_opt = self.qkv_opt(x_opt)
        q_sar, k_sar, v_sar = self.qkv_sar(x_sar)

        att_opt = F.softmax(torch.bmm(q_opt, k_opt), dim=-1)
        att_sar = F.softmax(torch.bmm(q_sar, k_sar), dim=-1)
        att = torch.bmm(att_opt, att_sar)

        s_opt = torch.bmm(att, v_opt).reshape(N, -1, H, W)
        s_opt = self.project_opt(s_opt)
        s_opt = x_opt + self.gamma_opt(s_opt)

        s_sar = torch.bmm(att, v_sar).reshape(N, -1, H, W)
        s_sar = self.project_sar(s_sar)
        s_sar = x_sar + self.gamma_sar(s_sar)

        return s_opt + s_sar

# AttFuseBlock
class AttFuseBlock(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(AttFuseBlock, self).__init__()
        self.sab = SpatialAttBlock(in_channels, channels, out_channels)
        self.cab = ChannelAttBlock(in_channels, channels, out_channels)

    def forward(self, x_opt, x_sar):
        sab_feat = self.sab(x_opt, x_sar)
        cab_feat = self.cab(x_opt, x_sar)
        return sab_feat + cab_feat
    

class ChannelAttentionModule(nn.Module): 
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveMaxPool2d(1) 

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False), 
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False) 
        )
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x)) 
        # print(avgout.shape) 
        maxout = self.shared_MLP(self.max_pool(x)) 
        return self.sigmoid(avgout + maxout) 


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3) 
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x): 
        avgout = torch.mean(x, dim=1, keepdim=True) 
        maxout, _ = torch.max(x, dim=1, keepdim=True) 
        out = torch.cat([avgout, maxout], dim=1) 
        out = self.sigmoid(self.conv2d(out)) 
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel) 
        self.spatial_attention = SpatialAttentionModule() 

    def forward(self, x):
        out = self.channel_attention(x) * x 
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out 
        return out


class ResBlock_CBAM(nn.Module):
    def __init__(self,in_places, places, stride=1,downsampling=False, expansion = 4):
        super(ResBlock_CBAM,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential( 
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False), 
            # nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False), 
            # nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False), 
            # nn.BatchNorm2d(places*self.expansion),
        )
        self.cbam = CBAM(channel=places*self.expansion) 

        if self.downsampling: 
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x) 
        # print(x.shape)
        out = self.cbam(out) 
        if self.downsampling:
            residual = self.downsample(x)

        out += residual 
        out = self.relu(out) 
        return out



from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding


# AlignDecoderBlock
class AlignDecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(AlignDecoderBlock, self).__init__()
        self.up = ConvModule(input_channels, output_channels, kernel_size=1)
        self.identity_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        self.decode = nn.Sequential(
            # nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(output_channels)
        )
        
        self.decodeD =ResBlock_CBAM(input_channels, output_channels//4)
        self.up1 = CAB(input_channels)
        
    def forward(self, low_feat, high_feat):
        f = self.up1(low_feat, high_feat)

        out = self.decodeD(f)

        return out



import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CCDC(nn.Module):
    def __init__(self,
                 enc_opt_dims=[64, 256, 512, 1024, 2048],
                 enc_sar_dims=[64, 64, 128, 256, 512],
                 center_block="dblock",
                 side_dim=64,
                 att_dim_factor=2,
                 norm_cfg=dict(type='BN')
                 ):
        super(CCDC, self).__init__()

        self.backbone_opt = smp.encoders.get_encoder(
            name='resnet50',  # equivalent to TIMMBackbone with resnet50
            in_channels=4,
            depth=5,
            weights="imagenet"
        )

        self.backbone_sar = smp.encoders.get_encoder(
            name='resnet18',  # equivalent to TIMMBackbone with resnet18
            in_channels=1,
            depth=5,
            weights=None  # Assuming no pretrained weights for SAR input
        )


        # bridge module (center block)
        if center_block == 'dblock':
            self.center_opt = RS_Dblock(enc_opt_dims[-1])  
            self.center_sar = RS_Dblock(enc_sar_dims[-1])
        else:
            self.center_opt = nn.Identity()
            self.center_sar = nn.Identity()

        self.side1_rgb = RES_Dilated_Conv(enc_opt_dims[0], side_dim) 
        self.side2_rgb = RES_Dilated_Conv(enc_opt_dims[1], side_dim)
        self.side3_rgb = RES_Dilated_Conv(enc_opt_dims[2], side_dim)
        self.side4_rgb = RES_Dilated_Conv(enc_opt_dims[3], side_dim)
        self.side5_rgb = RES_Dilated_Conv(enc_opt_dims[4], side_dim)

        self.side1_sar = RES_Dilated_Conv(enc_sar_dims[0], side_dim)
        self.side2_sar = RES_Dilated_Conv(enc_sar_dims[1], side_dim)
        self.side3_sar = RES_Dilated_Conv(enc_sar_dims[2], side_dim)
        self.side4_sar = RES_Dilated_Conv(enc_sar_dims[3], side_dim)
        self.side5_sar = RES_Dilated_Conv(enc_sar_dims[4], side_dim)
        
        self.fuse1 = GatedFusion(side_dim) 
        self.fuse2 = GatedFusion(side_dim)
        self.fuse3 = GatedFusion(side_dim)
        self.fuse4 = GatedFusion(side_dim)
        self.fuse5 = AttFuseBlock(side_dim, side_dim // att_dim_factor, side_dim)
        self.final_fuse = GatedFusion(side_dim)
        
        self.decode1_rgb = DecoderBlock(side_dim, side_dim)  
        self.decode2_rgb = AlignDecoderBlock(side_dim, side_dim) 
        self.decode3_rgb = AlignDecoderBlock(side_dim, side_dim)
        self.decode4_rgb = AlignDecoderBlock(side_dim, side_dim)
        self.decode5_rgb = AlignDecoderBlock(side_dim, side_dim)
        
        self.decode1_sar = DecoderBlock(side_dim, side_dim)
        self.decode2_sar = AlignDecoderBlock(side_dim, side_dim)
        self.decode3_sar = AlignDecoderBlock(side_dim, side_dim)
        self.decode4_sar = AlignDecoderBlock(side_dim, side_dim)
        self.decode5_sar = AlignDecoderBlock(side_dim, side_dim)
        
        
        
    def forward(self, x):
        # Split RGB (4 channels) and SAR (1 channel)
        x_img, x_aux = torch.split(x, (4, 1), dim=1)

        # RGB encoding
        x0,x1, x2, x3, x4, x5 = self.backbone_opt(x_img)
        
        x5 = self.center_opt(x5)
        x1_side = self.side1_rgb(x1)
        x2_side = self.side2_rgb(x2)
        x3_side = self.side3_rgb(x3)
        x4_side = self.side4_rgb(x4)
        x5_side = self.side5_rgb(x5)

        # SAR encoding
        y0,y1, y2, y3, y4, y5 = self.backbone_sar(x_aux)
        y5 = self.center_sar(y5)
        y1_side = self.side1_sar(y1)
        y2_side = self.side2_sar(y2)
        y3_side = self.side3_sar(y3)
        y4_side = self.side4_sar(y4)
        y5_side = self.side5_sar(y5)

        # Cross-modal fusion
        y5_side = self.fuse5(x5_side, y5_side)
        y4_side = self.fuse4(x4_side, y4_side)
        y3_side = self.fuse3(x3_side, y3_side)
        y2_side = self.fuse2(x2_side, y2_side)
        y1_side = self.fuse1(x1_side, y1_side)

        # RGB decoding
        out_rgb = self.decode5_rgb(x4_side, x5_side)
        out_rgb = self.decode4_rgb(x3_side, out_rgb)
        out_rgb = self.decode3_rgb(x2_side, out_rgb)
        out_rgb = self.decode2_rgb(x1_side, out_rgb)
        out_rgb = self.decode1_rgb(out_rgb)

        # SAR decoding
        out_sar = self.decode5_sar(y4_side, y5_side)
        out_sar = self.decode4_sar(y3_side, out_sar)
        out_sar = self.decode3_sar(y2_side, out_sar)
        out_sar = self.decode2_sar(y1_side, out_sar)
        out_sar = self.decode1_sar(out_sar)

        # Final fusion
        f_final = self.final_fuse(out_rgb, out_sar)

        
        
        return f_final


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)
        


class CCDC_m(nn.Module):
    def __init__(self,num_classes=21843):
        super(CCDC_m, self).__init__()
        self.num_classes = num_classes
        self.ccdc = CCDC()
        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=self.num_classes,
            kernel_size=3,
        )

    def forward(self, images, masks=None):  
        
        images = self.ccdc(images)  # (B, n_patch, hidden)
        logits = self.segmentation_head(images) 
        if masks != None:  
            loss1 = DiceLoss(mode='binary')(logits, masks)  
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)  
            
            return logits, loss1, loss2   
        return logits


class CAB(nn.Module):
    def __init__(self, features, norm_cfg=dict(type='BN', requires_grad=True)):
        super(CAB, self).__init__()

        # Replacing build_norm_layer with nn.BatchNorm2d
        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),  # Batch Normalization
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen2 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),  # Batch Normalization
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen1[2].weight.data.zero_()
        self.delta_gen2[2].weight.data.zero_()

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[w / s, h / s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)

        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample(low_stage, (h, w), delta2)

        high_stage += low_stage
        return high_stage

# ------------------#
# ResBlock+CBAM
# ------------------#
import torch
import torch.nn as nn
import torchvision




class CAB_Attention(nn.Module):
    def __init__(self, features, norm_cfg=dict(type='BN', requires_grad=True)):
        super(CAB_Attention, self).__init__()

        # Channel Attention Module
        # self.cam = CAM_Module(features)

        # Position Attention Module
        # self.pam = PAM_Module(features * 2)

        # Replacing build_norm_layer with nn.BatchNorm2d
        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),  # Batch Normalization
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen2 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),  # Batch Normalization
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen1[2].weight.data.zero_()
        self.delta_gen2[2].weight.data.zero_()

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[w / s, h / s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)

        high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)

        concat = torch.cat((low_stage, high_stage), 1)
        

        
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        
        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample(low_stage, (h, w), delta2)

        high_stage += low_stage
        return high_stage

    
class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
        # self.SE = SEBlock(self.chanel_in, reduction_ratio=16)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        
        
        # x=self.SE(x)
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        # out=x1*out
        
        return out

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C,height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


