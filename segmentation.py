import torch
import torch.nn as nn
import torchvision
from draw import *
import numpy as np
import matplotlib.pyplot as plt
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

    def train(self, input_img, epochs, learn_rate):
        input_img = input_img.unsqueeze(0).to(device)
        width = input_img.shape[2]
        height = input_img.shape[3]
        optimizer = torch.optim.Adam(self.parameters, lr=learn_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()
            noize_left = torch.ones((1, 2, width, height)).uniform_(-0.5, 0.5).to(device)
            noize_right = torch.ones((1, 2, width, height)).uniform_(-0.5, 0.5).to(device)
            noize_mask = torch.ones((1, 2, width, height)).uniform_(-0.5, 0.5).to(device)
            left_out = self.left_net(noize_left).to(device)
            right_out = self.right_net(noize_right).to(device)
            mask_out = self.mask_net(noize_mask).to(device)
            loss = 0.5 * self.reconst_loss(mask_out * left_out + (1 - mask_out) * right_out, input_img) + \
                self.reg_loss(mask_out)
            loss.backward()
            optimizer.step()
            print('Epoch  {}  loss = {:.7f}'.format(epoch + 1, loss))
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
print(input_img)
seg.train(input_img, epochs=1000, learn_rate=0.001)
