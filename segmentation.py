import torch
import torch.nn as nn
import torchvision
from draw import *
import numpy as np
import matplotlib.image as mpimg
from net.DIP import DIP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Segmentation(object):
    def __init__(self):
        self.left_net = DIP(out_channels=3).to(device)
        self.right_net = DIP(out_channels=3).to(device)
        self.mask_net = DIP(out_channels=1).to(device)
        self.parameters = [p for p in self.left_net.parameters()] + \
                          [p for p in self.right_net.parameters()] + \
                          [p for p in self.mask_net.parameters()]
        self.l1_loss = nn.L1Loss().to(device)

    def train(self, input_img, epochs_1, epochs_2, learn_rate):
        input_img = input_img.unsqueeze(0).to(device)
        width = input_img.shape[2]
        height = input_img.shape[3]
        optimizer = torch.optim.Adam(self.parameters, lr=learn_rate)


        print('optimize Mask to central value:')
        # 先将mask优化至中间值(每个像素都是0.5)
        for epoch in range(epochs_1):
            optimizer.zero_grad()
            noize_mask = torch.ones((1, 2, width, height)).uniform_(-0.5, 0.5).to(device)
            mask_out = self.mask_net(noize_mask).to(device)
            loss = self.l1_loss(mask_out, torch.ones_like(mask_out) / 2)
            loss.backward(retain_graph=True)
            optimizer.step()
            print('\tEpoch  {}  loss = {:.7f}'.format(epoch + 1, loss))

        print('optimize Double-DIP:')
        for epoch in range(epochs_2):
            optimizer.zero_grad()
            if epoch == epochs_2 - 1:
                pert = 0
            elif iteration < 1000:
                pert = (1 / 1000.) * (epoch // 100)
            else:
                pert = 1 / 1000.
            noize_left = torch.ones((1, 2, width, height)).uniform_(-0.5, 0.5).to(device)
            noize_right = torch.ones((1, 2, width, height)).uniform_(-0.5, 0.5).to(device)
            noize_mask = torch.ones((1, 2, width, height)).uniform_(-0.5, 0.5).to(device)
            noize_left += (noize_left.clone().normal_() * pert).to(device)
            noize_right += (noize_right.clone().normal_() * pert).to(device)
            noize_mask += (noize_mask.clone().normal_() * pert).to(device)
            
            left_out = self.left_net(noize_left).to(device) 
            right_out = self.right_net(noize_right).to(device)
            mask_out = self.mask_net(noize_mask).to(device)
            # print('forward finished')
            loss = 0.5 * self.reconst_loss(mask_out * left_out + (1 - mask_out) * right_out, input_img) + \
                (0.001 * (epoch // 100)) * self.reg_loss(mask_out)
            loss.backward(retain_graph=True)
            optimizer.step()
            print('\tEpoch  {}  loss = {:.7f}'.format(epoch + 1, loss))
            if (epoch % 500 == 0):
                self.plot(str(epoch), input_img, left_out, right_out, mask_out)


    def reconst_loss(self, input_img, recomp_img):
        '''
        重构损失
        :param recomp_img:
        :param self:
        :return:
        '''
        return self.l1_loss(input_img, recomp_img)

    def reg_loss(self, mask):
        '''
        正则损失(作为限制mask的先验)
        :param mask:
        :return:
        '''
        return 1 / self.l1_loss(mask, torch.ones_like(mask) / 2)

    def plot(self, name, input_img, left_out, right_out, mask_out):
        # input_img = input_img.cpu().squeeze()
        plot_image_grid("left_right_{}".format(name),
                        [np.clip(torch_to_np(left_out), 0, 1),
                         np.clip(torch_to_np(right_out), 0, 1)])
        mask_out_np = torch_to_np(mask_out)
        plot_image_grid("learned_mask_{}".format(name),
                        [np.clip(mask_out_np, 0, 1), 1 - np.clip(mask_out_np, 0, 1)])

        plot_image_grid("learned_image_{}".format(name),
                        [np.clip(mask_out_np * torch_to_np(left_out) + (1 - mask_out_np) * torch_to_np(right_out),
                                 0, 1), torch_to_np(input_img)])


seg = Segmentation()
input_img = mpimg.imread('./data/zebra.bmp').astype(np.float32) / 255
input_img = torch.from_numpy(input_img.transpose(2, 0, 1))
seg.train(input_img, epochs_1=2000, epochs_2=5000, learn_rate=0.001)
