import torch
from torch import nn
import numpy as np
# from .downsampler import * 
from torch.nn import functional
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ExclusionLoss(nn.Module):

    def __init__(self, level=3):
        """
        两个梯度的差别
        参考了以下论文:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            # alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            # alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))
            alphay = 1
            alphax = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            # gradx_loss.append(torch.mean(((gradx1_s ** 2) * (gradx2_s ** 2))) ** 0.25)
            # grady_loss.append(torch.mean(((grady1_s ** 2) * (grady2_s ** 2))) ** 0.25)
            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady



l1_loss = nn.L1Loss().to(device)
def pre_loss(input_img, left_out, right_out, mask_out, fg_hint, bg_hint):
    loss = 0
    # loss += l1_loss(mask_out, torch.ones_like(mask_out) / 2)

    normalizer = l1_loss(fg_hint, torch.zeros(fg_hint.shape).cuda())
    loss += l1_loss(fg_hint * input_img, fg_hint * left_out) / normalizer

    normalizer = l1_loss(bg_hint, torch.zeros(bg_hint.shape).cuda())
    loss += l1_loss(bg_hint * input_img, bg_hint * right_out) / normalizer

    loss += l1_loss((fg_hint - bg_hint + 1) / 2, mask_out)

    return loss

excl_loss = ExclusionLoss()
def total_loss(epoch, input_img, left_out, right_out, mask_out, fg_hint, bg_hint):
    loss = 0
    epoch = min(epoch, 1000)
    loss += 0.5 * reconst_loss(mask_out * left_out + (1 - mask_out) * right_out, input_img) + \
            (0.001 * (epoch // 100)) * reg_loss(mask_out) + 0.5 * excl_loss(left_out, right_out)

    if epoch <= 1000:  #
        normalizer = l1_loss(fg_hint, torch.zeros(fg_hint.shape).cuda())
        loss += l1_loss(fg_hint * input_img, fg_hint * left_out) / normalizer

        normalizer = l1_loss(bg_hint, torch.zeros(bg_hint.shape).cuda())
        loss += l1_loss(bg_hint * input_img, bg_hint * right_out) / normalizer

    return loss

def reconst_loss(input_img, recomp_img):
    '''
    重构损失
    :param recomp_img:
    :param self:
    :return:
    '''
    return l1_loss(input_img, recomp_img)

def reg_loss(mask):
    '''
    正则损失(作为限制mask的先验)
    :param mask:
    :return:
    '''
    return 1 / l1_loss(mask, torch.ones_like(mask) / 2)

