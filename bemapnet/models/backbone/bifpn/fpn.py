import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_outs, tgt_shape):
        """
        Args:
            in_channels_list (list): Backbone各层输出的输入通道数列表，例如[256, 512, 1024, 2048]
            out_channels (int): FPN输出的通道数
            num_outs (int): FPN输出的特征图数量
        """
        super(FPN, self).__init__()
        self.tgt_shape = tgt_shape
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            l_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            self.lateral_convs.append(l_conv)

        self.fpn_convs = nn.ModuleList()
        for _ in range(num_outs):
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): 从backbone提取的不同尺度特征图（从大到小），形状如[(N, C1, H1, W1), (N, C2, H2, W2), ...]
        Returns:
            outputs (list[Tensor]): FPN输出的多尺度特征图，形状如[(N, C, H1, W1), (N, C, H2, W2), ...]
        """
        assert len(inputs) == len(self.lateral_convs)

        lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        for j in range(len(lateral_outputs) - 1, 0, -1):
            lateral_outputs[j - 1] += F.interpolate(lateral_outputs[j], scale_factor=2, mode='nearest')

        outputs = [self.fpn_convs[k](lateral_outputs[k]) for k in range(len(self.fpn_convs))]
        upsampled_features = [F.interpolate(feature, size=self.tgt_shape, mode='nearest') for feature in outputs]
        concatenated_features = torch.cat(upsampled_features, dim=1)  # 在通道维度上拼接

        return concatenated_features


# 测试FPN模块
if __name__ == '__main__':
    in_channels_list = [512, 1024, 2048]
    out_channels = 256
    num_outs = 3
    tgt_shape = (21, 49)

    fpn = FPN(in_channels_list, out_channels, num_outs, tgt_shape)

    inputs = [
        torch.randn(1, 512, 60, 100),
        torch.randn(1, 1024, 30, 50),
        torch.randn(1, 2048, 15, 25)
    ]

    outputs = fpn(inputs)

    for i, output in enumerate(outputs):
        print(f'Output {i}: {output.shape}')
