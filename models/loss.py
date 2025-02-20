import torchvision
import torch
from torch import nn


def fft_map(x):
    fft_x = torch.fft.fftn(x)
    fft_x_real = fft_x.real
    fft_x_imag = fft_x.imag

    return fft_x_real, fft_x_imag


class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=[2, 7, 16, 25, 34], use_input_norm=True, use_range_norm=False):
        super(VGGFeatureExtractor, self).__init__()
        """
        use_input_norm: If True, x: [0, 1] --> (x - mean) / std
        use_range_norm: If True, x: [0, 1] --> x: [-1, 1]
        """
        model = torchvision.models.vgg19(pretrained=True)  # The parameter 'pretrained' will be removed, please use 'weights' instead.
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        if self.use_input_norm:  # True
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.list_outputs = isinstance(feature_layer, list)
        if self.list_outputs:  # True
            self.features = nn.Sequential()
            feature_layer = [-1] + feature_layer
            for i in range(len(feature_layer)-1):
                self.features.add_module('child'+str(i), nn.Sequential(*list(model.features.children())[(feature_layer[i]+1):(feature_layer[i+1]+1)]))
        else:
            self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])

        print(self.features)

        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_range_norm:  # False
            x = (x + 1.0) / 2.0
        if self.use_input_norm:  # True
            x = (x - self.mean) / self.std
        if self.list_outputs:
            output = []
            for child_model in self.features.children():
                x = child_model(x)
                output.append(x.clone())
            return output
        else:
            return self.features(x)


class PerceptualLoss(nn.Module):
    """VGG Perceptual loss
    """
    def __init__(self, feature_layer=[2, 7, 16, 25, 34], weights=[0.1, 0.1, 1.0, 1.0, 1.0], lossfn_type='l1', use_input_norm=True, use_range_norm=False):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer, use_input_norm=use_input_norm, use_range_norm=use_range_norm)
        self.lossfn_type = lossfn_type  # L1
        self.weights = weights  # [0.1, 0.1, 1.0, 1.0, 1.0]
        if self.lossfn_type == 'l1':
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()
        print(f'feature_layer: {feature_layer}  with weights: {weights}')

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x_vgg, gt_vgg = self.vgg(x), self.vgg(gt.detach())
        loss = 0.0
        if isinstance(x_vgg, list):
            n = len(x_vgg)
            for i in range(n):
                loss += self.weights[i] * self.lossfn(x_vgg[i], gt_vgg[i])
        else:
            loss += self.lossfn(x_vgg, gt_vgg.detach())
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss


class TotalLoss(nn.Module):
    def __init__(self, args):
        super(TotalLoss, self).__init__()
        self.alpha = args.LOSS.ALPHA
        self.beta = args.LOSS.BETA
        self.gamma = args.LOSS.GAMMA
        self.use_mm_loss = args.LOSS.USE_MM_LOSS
        self.w = 0.9

        self.base_loss = CharbonnierLoss(args.LOSS.CHARBONNIER_EPS)
        self.perceptual_loss = PerceptualLoss()

    def forward(self, outputs, targets, complement=None, complement_target=None):
        rec_gt_real, rec_gt_imag = fft_map(targets)
        rec_real, rec_imag = fft_map(outputs)

        loss_image = self.base_loss(outputs, targets)
        loss_freq = (self.base_loss(rec_real, rec_gt_real) + self.base_loss(rec_imag, rec_gt_imag)) / 2
        loss_perc = self.perceptual_loss(outputs, targets)

        rec_loss = self.alpha * loss_image + self.beta * loss_freq + self.gamma * loss_perc

        if self.use_mm_loss:
            com_gt_real, com_gt_imag = fft_map(complement_target)
            com_real, com_imag = fft_map(complement)

            loss_image_com = self.base_loss(complement, complement_target)
            loss_freq_com = (self.base_loss(com_real, com_gt_real) + self.base_loss(com_imag, com_gt_imag)) / 2
            loss_perc_com = self.perceptual_loss(complement, complement_target)

            com_loss = self.alpha * loss_image_com + self.beta * loss_freq_com + self.gamma * loss_perc_com

            loss = self.w * rec_loss + (1 - self.w) * com_loss

            return {'loss_rec_image': loss_image, 'loss_rec_freq': loss_freq, 'loss_rec_perc': loss_perc,
                    'loss_com_image': loss_image_com, 'loss_com_freq': loss_freq_com, 'loss_com_perc': loss_perc_com,
                    'rec_loss': rec_loss, 'com_loss': com_loss, 'loss': loss}
        else:
            return {'loss_rec_image': loss_image, 'loss_rec_freq': loss_freq, 'loss_rec_perc': loss_perc,
                    'rec_loss': rec_loss, 'loss': rec_loss}


class LossWrapper(nn.Module):
    def __init__(self, args):
        super(LossWrapper, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.cl1_loss = nn.L1Loss()
        self.use_cl1_loss = args.LOSS.USE_MM_LOSS
        self.w = 0.9

    def forward(self, outputs, targets, complement=None, complement_target=None):
        l1_loss = self.l1_loss(outputs, targets)
        if self.use_cl1_loss:
            cl1_loss = self.cl1_loss(complement, complement_target)
            loss = self.w * l1_loss + (1 - self.w) * cl1_loss
            return {'l1_loss': l1_loss, 'cl1_loss': cl1_loss, 'loss': loss}
        else:
            loss = l1_loss
            return {'l1_loss': l1_loss, 'loss': loss}


def build_criterion(args):
    if args.LOSS.TYPE == 'L1':
        return LossWrapper(args)
    else:
        return TotalLoss(args)
