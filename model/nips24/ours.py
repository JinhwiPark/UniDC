import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import BpOps
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from collections import OrderedDict
import functools


class BpDist(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, xy, idx, Valid, num, H, W):
        """
        """
        assert xy.is_contiguous()
        assert Valid.is_contiguous()
        _, Cc, M = xy.shape
        B = Valid.shape[0]
        N = H * W
        args = torch.zeros((B, num, N), dtype=torch.long, device=xy.device)
        IPCnum = torch.zeros((B, Cc, num, N), dtype=xy.dtype, device=xy.device)
        for b in range(B):
            Pc = torch.masked_select(xy, Valid[b:b + 1].view(1, 1, N)).reshape(1, 2, -1) # sparse depth (x,y) 좌표
            BpOps.Dist(Pc, IPCnum[b:b + 1], args[b:b + 1], H, W)
            idx_valid = torch.masked_select(idx, Valid[b:b + 1].view(1, 1, N)) # sparse depth (H*W) 좌표
            args[b:b + 1] = torch.index_select(idx_valid, 0, args[b:b + 1].reshape(-1)).reshape(1, num, N)
        return IPCnum, args

    @staticmethod
    @custom_bwd
    def backward(ctx, ga=None, gb=None):
        return None, None, None, None


class BpConvLocal(Function):
    @staticmethod
    def forward(ctx, input, weight):
        assert input.is_contiguous()
        assert weight.is_contiguous()
        ctx.save_for_backward(input, weight)
        output = BpOps.Conv2dLocal_F(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input, grad_weight = BpOps.Conv2dLocal_B(input, weight, grad_output)
        return grad_input, grad_weight


bpdist = BpDist.apply
bpconvlocal = BpConvLocal.apply

class Dist(nn.Module):
    """
    """

    def __init__(self, num):
        super().__init__()
        """
        """
        self.num = num

    def forward(self, S, xx, yy):
        """
        """
        # import pdb;pdb.set_trace()
        num = self.num
        B, _, height, width = S.shape
        N = height * width
        S = S.reshape(B, 1, N)
        Valid = (S > 1e-3)
        xy = torch.stack((xx, yy), axis=0).reshape(1, 2, -1).float()
        idx = torch.arange(N, device=S.device).reshape(1, 1, N)
        Ofnum, args = bpdist(xy, idx, Valid, num, height, width)
        return Ofnum, args

class GenKernel(nn.Module): # CSPNGenerateAccelerate
    def __init__(self, in_channels, pk, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.conv = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, pk * pk - 1, norm_layer=norm_layer, act=nn.Identity),
        )

    def forward(self, fout):
        weight = self.conv(fout) # weigt = guide
        weight_sum = torch.sum(weight.abs(), dim=1, keepdim=True)
        weight = torch.div(weight, weight_sum + self.eps)
        weight_mid = 1 - torch.sum(weight, dim=1, keepdim=True)
        weight_pre, weight_post = torch.split(weight, [weight.shape[1] // 2, weight.shape[1] // 2], dim=1)
        weight = torch.cat([weight_pre, weight_mid, weight_post], dim=1).contiguous()
        return weight

class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1, padding_mode='zeros',
                 act=nn.ReLU, stride=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False, padding_mode=padding_mode)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv = nn.Sequential(OrderedDict([('conv', conv)]))
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', act())

    def forward(self, x):
        out = self.conv(x)
        return out
    
class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, act=nn.ReLU):
        super().__init__()
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        else:
            bias = False
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4,
                                       stride=2, padding=1, bias=bias)
        self.bn = norm_layer(out_channels)
        self.relu = act()

    def forward(self, x):
        out = self.conv(x.contiguous())
        out = self.bn(out)
        out = self.relu(out)
        return out



class CSPN(nn.Module):
    """
    implementation of CSPN++
    """

    def __init__(self, in_channels, pt, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        self.pt = pt
        self.weight3x3 = GenKernel(in_channels, 3, norm_layer=norm_layer, act=act, eps=eps)
        self.weight5x5 = GenKernel(in_channels, 5, norm_layer=norm_layer, act=act, eps=eps)
        self.weight7x7 = GenKernel(in_channels, 7, norm_layer=norm_layer, act=act, eps=eps)
        self.convmask = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=nn.Sigmoid),
        )
        self.convck = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )
        self.convct = nn.Sequential(
            Basic2d(in_channels + 3, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, fout, hn, h0):
        """
        fout, hn, h0 = (fout, dout, Sp)
        """
        weight3x3 = self.weight3x3(fout) # guidance for 3x3  [B,9,H,W]
        weight5x5 = self.weight5x5(fout) # guidance for 5x5  [B,25,H,W]
        weight7x7 = self.weight7x7(fout) # guidance for 7x7  [B,49,H,W]
        mask3x3, mask5x5, mask7x7 = torch.split(self.convmask(fout) * (h0 > 1e-3).float(), 1, dim=1)
        conf3x3, conf5x5, conf7x7 = torch.split(self.convck(fout), 1, dim=1)
        hn3x3 = hn5x5 = hn7x7 = hn #init_depth
        hns = [hn, ]
        for i in range(self.pt):
            hn3x3 = (1. - mask3x3) * bpconvlocal(hn3x3, weight3x3) + mask3x3 * h0
            hn5x5 = (1. - mask5x5) * bpconvlocal(hn5x5, weight5x5) + mask5x5 * h0
            hn7x7 = (1. - mask7x7) * bpconvlocal(hn7x7, weight7x7) + mask7x7 * h0
            if i == self.pt // 2 - 1: # 중앙?
                hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5 + conf7x7 * hn7x7)
        hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5 + conf7x7 * hn7x7) # 처음 중앙 마지막 있음
        hns = torch.cat(hns, dim=1) # 세개를 합침?
        wt = self.convct(torch.cat([fout, hns], dim=1)) # 첫 feature랑 합쳐서 weight를 만듬
        hn = torch.sum(wt * hns, dim=1, keepdim=True) # 처음 중앙 마지막 depth를 combine해서 최종 depth 예측..
        return hn


class HyperbolicGenKernel(nn.Module): 
    def __init__(self, in_channels, pk, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        self.eps = eps

        from model.nips24.hyptorch.hyperbolicCNN import HypCNN_curvature_generator
        self.conv = HypCNN_curvature_generator(in_channels, pk * pk - 1, hyp_mode='naive', norm_layer=norm_layer, act=act)

    def forward(self, fout, c):
        
        weight = self.conv(fout, c) # weigt = guide
        weight_sum = torch.sum(weight.abs(), dim=1, keepdim=True)
        weight = torch.div(weight, weight_sum + self.eps)
        weight_mid = 1 - torch.sum(weight, dim=1, keepdim=True)
        weight_pre, weight_post = torch.split(weight, [weight.shape[1] // 2, weight.shape[1] // 2], dim=1)
        weight = torch.cat([weight_pre, weight_mid, weight_post], dim=1).contiguous()
        return weight
    
class HyperbolicCSPN(nn.Module):
    def __init__(self, in_channels, pt, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        
        self.pt = pt
        
        self.curvature_generator_3x3 = nn.Sequential(
            Basic2d(in_channels, in_channels, kernel_size=3, padding=1, norm_layer=norm_layer, act=nn.ReLU, stride=1),
            Basic2d(in_channels, 1, kernel_size=1, padding=1, norm_layer=norm_layer, act=nn.Sigmoid, stride=1)
            )

        self.curvature_generator_5x5 = nn.Sequential(
            Basic2d(in_channels, in_channels, kernel_size=5, padding=2, norm_layer=norm_layer, act=nn.ReLU, stride=1),
            Basic2d(in_channels, 1, kernel_size=1, padding=1, norm_layer=norm_layer, act=nn.Sigmoid, stride=1)
            )
        self.curvature_generator_7x7 = nn.Sequential(
            Basic2d(in_channels, in_channels, kernel_size=7, padding=3, norm_layer=norm_layer, act=nn.ReLU, stride=1),
            Basic2d(in_channels, 1, kernel_size=1, padding=1, norm_layer=norm_layer, act=nn.Sigmoid, stride=1)
            )


        self.weight3x3 = HyperbolicGenKernel(in_channels, 3, norm_layer=norm_layer, act=None, eps=eps)
        self.weight5x5 = HyperbolicGenKernel(in_channels, 5, norm_layer=norm_layer, act=None, eps=eps)
        self.weight7x7 = HyperbolicGenKernel(in_channels, 7, norm_layer=norm_layer, act=None, eps=eps)

        self.convmask = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=nn.Sigmoid),
        )
        self.convck = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )
        self.convct = nn.Sequential(
            Basic2d(in_channels + 3, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, fout, hn, h0):
        """
        fout, hn, h0 = (fout, dout, Sp)
        """
        
        curvature3x3 = self.curvature_generator_3x3(fout).mean() + 1e-4
        curvature5x5 = self.curvature_generator_5x5(fout).mean() + 1e-4
        curvature7x7 = self.curvature_generator_7x7(fout).mean() + 1e-4

        weight3x3 = self.weight3x3(fout, curvature3x3) # guidance for 3x3  [B,9,H,W]
        weight5x5 = self.weight5x5(fout, curvature5x5) # guidance for 5x5  [B,25z   ,H,W]
        weight7x7 = self.weight7x7(fout, curvature7x7) # guidance for 7x7  [B,49,H,W]
        mask3x3, mask5x5, mask7x7 = torch.split(self.convmask(fout) * (h0 > 1e-3).float(), 1, dim=1)
        conf3x3, conf5x5, conf7x7 = torch.split(self.convck(fout), 1, dim=1)
        hn3x3 = hn5x5 = hn7x7 = hn #init_depth
        hns = [hn, ]
        for i in range(self.pt):
            hn3x3 = (1. - mask3x3) * bpconvlocal(hn3x3, weight3x3) + mask3x3 * h0
            hn5x5 = (1. - mask5x5) * bpconvlocal(hn5x5, weight5x5) + mask5x5 * h0
            hn7x7 = (1. - mask7x7) * bpconvlocal(hn7x7, weight7x7) + mask7x7 * h0
            if i == self.pt // 2 - 1: # 중앙?
                hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5 + conf7x7 * hn7x7)
        hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5 + conf7x7 * hn7x7)
        hns = torch.cat(hns, dim=1) 
        wt = self.convct(torch.cat([fout, hns], dim=1))
        hn = torch.sum(wt * hns, dim=1, keepdim=True)
        return hn, curvature3x3, curvature5x5, curvature7x7
    

class Coef(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels=3, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=1, padding=padding, bias=True)

    def forward(self, x):
        feat = self.conv(x)
        XF, XB, XW = torch.split(feat, [1, 1, 1], dim=1)
        return XF, XB, XW
    
   
def pinv(S, K, xx, yy):
    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    S = S.view(S.shape[0], 1, -1)
    xx = xx.reshape(1, 1, -1)
    yy = yy.reshape(1, 1, -1)
    Px = S * (xx - cx) / fx
    Py = S * (yy - cy) / fy
    Pz = S
    Pxyz = torch.cat([Px, Py, Pz], dim=1).contiguous()
    return Pxyz

class UpCat(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, kernel_size=3, padding=1,
                 padding_mode='zeros', act=nn.ReLU):
        super().__init__()
        self.upf = Basic2dTrans(in_channels, out_channels, norm_layer=norm_layer, act=act)
        self.conv = Basic2d(out_channels*2, out_channels,
                            norm_layer=norm_layer, kernel_size=kernel_size,
                            padding=padding, padding_mode=padding_mode, act=act)

    def forward(self, feature, feature_accmulated):
        fout = self.upf(feature_accmulated) # fout' = fout + dout
        fout = self.conv(torch.cat([fout, feature], dim=1)) # fout'' = fout' + XI (image feature) ?
        return fout

class HyperbolicProp(nn.Module):
    def __init__(self, hyperbolic_config, Cfi, Cfp=3, Cfo=2, act=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        """
        """
        self.dist = lambda x: (x * x).sum(1)
        self.convXF = nn.Sequential(
            Basic2d(in_channels=Cfi, out_channels=Cfi, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=Cfi, out_channels=Cfi, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
        )
        self.convXL = nn.Sequential(
            Basic2d(in_channels=Cfi, out_channels=Cfi, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=Cfi, out_channels=Cfi, norm_layer=norm_layer, act=nn.Identity, kernel_size=1,
                    padding=0),
        )
        self.act = act()
        self.coef = Coef(Cfi, 3)

        self.euc = nn.Sequential(
            Basic2d(in_channels=5, out_channels=32, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=32, out_channels=16, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
        )

        from model.nips24.hyptorch.hyperbolicMLP import HypMLP

        self.hyp = HypMLP(in_channels=64, out_channels=16, return_euc_feature=False, config=hyperbolic_config)

        self.curvature_generator_3x3 = nn.Sequential(
            Basic2d(32, 16, kernel_size=3, padding=1, norm_layer=norm_layer, act=nn.ReLU, stride=1),
            Basic2d(16, 1, kernel_size=1, padding=1, norm_layer=norm_layer, act=nn.Sigmoid, stride=1)
            )
    def forward(self, If, Pf, Ofnum, args):
        """
        If.shape >> torch.Size([1, 32, 256, 320]) : 픽셀에 feature
        Pf.shape >> torch.Size([1, 3, 81920]) : xyz Camera coordinate 좌표
        Ofnum.shape >>  torch.Size([1, 2, 4, 81920]) : 모든 픽셀에 대해 가장 가까운 4(=num)개의 offset
        args.shape >> torch.Size([1, 4, 81920]) : 모든 픽셀에 대해 가장 가까운 4(=num)개의 (H*W)상의 index        
        """
        
        num = args.shape[-2]
        B, Cfi, H, W = If.shape # feature channel = Cfi = 32
        N = H * W
        B, Cfp, M = Pf.shape # Sparse Depth coordinate = Cfp = 3

        # N == M


        If = If.view(B, Cfi, 1, N) # [1, 32=Cfi, 1, H*W]
        Pf = Pf.view(B, Cfp, 1, M) # [1, 3 =Cfp, 1, H*W]


        # (B, Cfi, 1, N) -> (B, Cfi, 4, N) 
        Ifnum = If.expand(B, Cfi, num, N)  ## Ifnum is BxCfixnumxN 
        
        # torch.Size([1, 32, 4, 81920])
        IPfnum = torch.gather(input=If.expand(B, Cfi, num, N), dim=-1, index=args.view(B, 1, num, N).expand(B, Cfi, num, N))  ## IPfnum is BxCfixnumxN

        # torch.Size([1, 3, 4, 81920])
        Pfnum = torch.gather(input=Pf.expand(B, Cfp, num, M), dim=-1, index=args.view(B, 1, num, N).expand(B, Cfp, num, N))  ## Pfnum is BxCfpxnumxN

        generated_curvature = self.curvature_generator_3x3(If).mean() + 1e-4
        
        X_hyperbolic = self.hyp(torch.cat([Ifnum, IPfnum], dim=1), c=generated_curvature)
        X_euclidean = self.euc(torch.cat([Pfnum, Ofnum], dim=1))

        XF = self.convXF(torch.cat([X_hyperbolic, X_euclidean], dim=1))
        XF = self.act(XF + self.convXL(XF))
        Alpha, Beta, Omega = self.coef(XF)
        Omega = torch.softmax(Omega, dim=2)
        # Pfnum[:, -1:] >> z값 (depth)
        dout = torch.sum(((Alpha + 1) * Pfnum[:, -1:] + Beta) * Omega, dim=2, keepdim=True)
        return dout.view(B, 1, H, W), generated_curvature
    
class Net(nn.Module):
    """
    network
    """

    def __init__(self, args):
        super().__init__()

        self.args = args

        hyperbolic_config = {'non_linearity': False, 'train_c': False, 'batchnorm': False, 'curvature':args.hyperbolic_c, 'clipping':args.clipping, 'clip_param':args.clip_param}

        self.dist = Dist(num=4)
        self.prop = HyperbolicProp(hyperbolic_config, 32)
        self.cspn = HyperbolicCSPN(32, pt=self.args.prop_time)
        
        from model.ours.ZoeDepth.zoedepth.models.base_models.midas import MidasCore
        self.backbone  = MidasCore.build(midas_model_type="MiDaS_small", use_pretrained_midas=True, train_midas=True, fetch_features=True, freeze_bn=True, img_size=[self.args.patch_height, self.args.patch_width])
        
        for name, var in self.backbone.named_parameters():
            if not 'bias' in name:
                var.requires_grad = False
        
        self.upconv1 = UpCat(512,256)
        self.upconv2 = UpCat(256,128)
        self.upconv3 = UpCat(128,64)
        self.upconv4 = UpCat(64,64)
        self.upconv5 = UpCat(64,32)
        
    def forward(self, sample):

        I = sample['rgb']
        Sp = sample['dep']
        K = sample['K']
        # import pdb;pdb.set_trace()
        # print(Sp.nonzero().shape)
        # print(Sp.nonzero().shape)
        # print(sample['gt'].nonzero().shape)
        
        
        foudnation_rel_dep, foundation_feature = self.backbone(I, denorm=False, return_rel_depth=True)

        foudnation_rel_dep = foudnation_rel_dep.unsqueeze(1)
        auumulated_feature = self.upconv1(foundation_feature[2], foundation_feature[1]) # 256,16,20 + 1,8,10 + 512,8,10 -> 256,16,20
        auumulated_feature = self.upconv2(foundation_feature[3], auumulated_feature) # 128,32,40 + 1,16,20 + 256,16,20 -> 128,32,40
        auumulated_feature = self.upconv3(foundation_feature[4], auumulated_feature) # 64,64,80 + 1,32,40 + 128,32,40  -> 64,64,80
        auumulated_feature = self.upconv4(foundation_feature[5], auumulated_feature) # 64,128,160 + 1,64,80 + 64,64,80 -> 64,128,160
        auumulated_feature = self.upconv5(foundation_feature[0], auumulated_feature) # 32,256,320 + 1,128,160 + 64,128,160 -> 32,256,320
    
        Kp = K.clone()
        Kp[:, :2] = Kp[:, :2] / 2
        B, _, height, width = Sp.shape
        xx, yy = torch.meshgrid(torch.arange(width, device=Sp.device), torch.arange(height, device=Sp.device), indexing='xy')
        Pxyz = pinv(Sp, Kp, xx, yy)
        Ofnum, args = self.dist(Sp, xx, yy)
        init_depth, curvature_0 = self.prop(auumulated_feature, Pxyz, Ofnum, args) # dout size X2

        output, curvature_1, curvature_2, curvature_3 = self.cspn(auumulated_feature, init_depth, Sp)

        c_list = [curvature_0,curvature_1,curvature_2,curvature_3]

        return {'pred': output, 'pred_init': init_depth, 'pred_inter': None, 'guidance': None, 'confidence': None, 'foundation_depth': foudnation_rel_dep, 'curvatures':c_list, 'ac': auumulated_feature}