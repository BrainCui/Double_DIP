import torch
import torch.nn as nn
import torchvision
from draw import *
import numpy as np
import matplotlib.image as mpimg
from net.DIP import DIP
from net.losses import *
# from net import skip, skip_mask
from PIL import Image
import argparse
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='output/')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Segmentation(object):
    def __init__(self):
        self.left_net = DIP(out_channels=3).to(device)
        self.right_net = DIP(out_channels=3).to(device)
        self.mask_net = DIP(out_channels=1).to(device)

        # pad = 'reflection'
        # pad = 'zero'
        # left_net = skip(
        #     2, 3,
        #     num_channels_down=[8, 16, 32],
        #     num_channels_up=[8, 16, 32],
        #     num_channels_skip=[0, 0, 0],
        #     upsample_mode='bilinear',
        #     filter_size_down=3,
        #     filter_size_up=3,
        #     need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        # self.left_net = left_net.type(torch.cuda.FloatTensor)

        # right_net = skip(
        #     2, 3,
        #     num_channels_down=[8, 16, 32],
        #     num_channels_up=[8, 16, 32],
        #     num_channels_skip=[0, 0, 0],
        #     upsample_mode='bilinear',
        #     filter_size_down=3,
        #     filter_size_up=3,
        #     need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        # self.right_net = right_net.type(torch.cuda.FloatTensor)

        # mask_net = skip_mask(
        #     2, 1,
        #     num_channels_down=[8, 16, 32],
        #     num_channels_up=[8, 16, 32],
        #     num_channels_skip=[0, 0, 0],
        #     filter_size_down=3,
        #     filter_size_up=3,
        #     upsample_mode='bilinear',
        #     need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        # self.mask_net = mask_net.type(torch.cuda.FloatTensor)

        self.parameters = [p for p in self.left_net.parameters()] + \
                          [p for p in self.right_net.parameters()] + \
                          [p for p in self.mask_net.parameters()]

    def train(self, input_img, fg_hint, bg_hint, epochs_1, epochs_2, learn_rate):
        input_img = input_img.unsqueeze(0).to(device)
        fg_hint = fg_hint.unsqueeze(0).to(device)
        bg_hint = bg_hint.unsqueeze(0).to(device)
        
        
        
        plot_image_grid("fg_bg", [torch_to_np(fg_hint * input_img), torch_to_np(bg_hint * input_img)], output_path=args.output_path)

        self.width = input_img.shape[2]
        self.height = input_img.shape[3]
        optimizer = torch.optim.Adam(self.parameters, lr=learn_rate)

        print('optimize Mask to central value:')
        # 先将mask优化至中间值(每个像素都是0.5)
        for epoch in range(epochs_1):
            optimizer.zero_grad()

            left_out, right_out, mask_out = self.forward_all(epoch, epochs_1)

            loss = pre_loss(input_img, left_out, right_out, mask_out, fg_hint, bg_hint)
            loss.backward(retain_graph=True)
            optimizer.step()
            print('\tEpoch  {}  loss = {:.7f}'.format(epoch + 1, loss))

        print('optimize Double-DIP:')
        writer = SummaryWriter(log_dir="./log")
        for epoch in range(epochs_2):
            optimizer.zero_grad()

            left_out, right_out, mask_out = self.forward_all(epoch, epochs_2)

            loss = total_loss(epoch, input_img, left_out, right_out, mask_out, fg_hint, bg_hint)
            loss.backward(retain_graph=True)
            optimizer.step()
            print('\tEpoch  {}  loss = {:.7f}'.format(epoch + 1, loss))
            if epoch % 500 == 0:
                self.plot(str(epoch), input_img, left_out, right_out, mask_out)
            writer.add_scalar('train/G_loss', loss, epoch + 1, walltime=epoch + 1)
        self.plot('final', input_img, left_out, right_out, mask_out)

    def forward_all(self, epoch, max_epoch):
        if epoch == max_epoch - 1:
            pert = 0
        elif epoch < 1000:
            pert = (1 / 1000.) * (epoch // 100)
        else:
            pert = 1 / 1000.
        noize_left = torch.ones((1, 2, self.width, self.height)).uniform_(-0.5, 0.5).to(device)
        noize_right = torch.ones((1, 2, self.width, self.height)).uniform_(-0.5, 0.5).to(device)
        noize_mask = torch.ones((1, 2, self.width, self.height)).uniform_(-0.5, 0.5).to(device)
        noize_left += (noize_left.clone().normal_() * pert).to(device)
        noize_right += (noize_right.clone().normal_() * pert).to(device)
        noize_mask += (noize_mask.clone().normal_() * pert).to(device)
        left_out = self.left_net(noize_left).to(device)
        right_out = self.right_net(noize_right).to(device)
        mask_out = self.mask_net(noize_mask).to(device)
        return left_out, right_out, mask_out



    def plot(self, name, input_img, left_out, right_out, mask_out):
        # input_img = input_img.cpu().squeeze()
        plot_image_grid("left_right_{}".format(name),
                        [np.clip(torch_to_np(left_out), 0, 1),
                         np.clip(torch_to_np(right_out), 0, 1)], output_path=args.output_path)
        mask_out_np = torch_to_np(mask_out)
        plot_image_grid("learned_mask_{}".format(name),
                        [np.clip(mask_out_np, 0, 1), 1 - np.clip(mask_out_np, 0, 1)], output_path=args.output_path)

        plot_image_grid("learned_image_{}".format(name),
                        [np.clip(mask_out_np * torch_to_np(left_out) + (1 - mask_out_np) * torch_to_np(right_out),
                                 0, 1), torch_to_np(input_img)], output_path=args.output_path)


seg = Segmentation()
downsample = (64, 48)
input_img = Image.open('./data/zebra.bmp')     # 读取图像
input_img = input_img.resize(downsample)       # 下采样到标准大小
input_img = np.array(input_img).astype(np.float32) / 255    # 归一化，[0, 255] -> [0, 1]
input_img = torch.from_numpy(input_img.transpose(2, 0, 1))

fg_hint = Image.open('./data/fg_hint.bmp')
fg_hint = fg_hint.resize(downsample)
fg_hint = np.array(fg_hint)
print(fg_hint)
fg_hint = fg_hint.astype(np.float32)
fg_hint = torch.from_numpy(fg_hint).unsqueeze(0)
print(fg_hint.shape)
print(torch.min(fg_hint))
print(torch.max(fg_hint))

bg_hint = Image.open('./data/bg_hint.bmp')
bg_hint = bg_hint.resize(downsample)
bg_hint = np.array(bg_hint)
print(bg_hint)
bg_hint = bg_hint.astype(np.float32)
bg_hint = torch.from_numpy(bg_hint).unsqueeze(0)

print(bg_hint.shape)
print(torch.min(bg_hint))
print(torch.max(bg_hint))

seg.train(input_img, fg_hint, bg_hint, epochs_1=2000, epochs_2=50000, learn_rate=0.001)
