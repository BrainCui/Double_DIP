import torch.nn as nn
import torch

class Concat(nn.Module):
    def __init__(self, model_1, model_2):
        super(Concat, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2

    def forward(self, input):
        out_1 = self.model_1(input)
        out_2 = self.model_2(input)

        if out_1.shape[2] != out_2.shape[2] or out_1.shape[3] != out_2.shape[3]:  # 如果图像大小不相等
            min_shape_2 = min([out_1.shape[2], out_2.shape[2]])
            min_shape_3 = min([out_1.shape[3], out_2.shape[3]])

            diff2 = (out_1.size(2) - min_shape_2) // 2
            diff3 = (out_1.size(3) - min_shape_3) // 2
            out_1 = out_1[:, :, diff2 : min_shape_2 + diff2, diff3 : min_shape_3 + diff3]

            diff2 = (out_2.size(2) - min_shape_2) // 2
            diff3 = (out_2.size(3) - min_shape_3) // 2
            out_2 = out_2[:, :, diff2: min_shape_2 + diff2, diff3: min_shape_3 + diff3]

        return torch.cat((out_1, out_2), dim=1)

