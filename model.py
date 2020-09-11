import torch.nn.functional as F
from torchsummary import summary
import torch.nn as nn
import torch
import math


class SPP1d(nn.Module):
    """
    一维的空间金字塔池化
    """
    def __init__(self, num_levels):
        super(SPP1d, self).__init__()

        self.num_levels = num_levels

    def forward(self, x):
        # 定义命名
        names = locals()
        # 定义子金字塔列表
        spp_list = []

        # num为样本个数, length为样本宽度
        num, _, length = x.size()

        x_flatten = 0
        for count in range(self.num_levels):
            level = count + 1
            kernel_size = math.ceil(length / level)
            stride = math.ceil(length / level)
            pooling = math.floor((kernel_size*level-length+1)/2)

            tensor = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=pooling)(x)

            # 展开并拼接
            if count == 0:
                # 对子金字塔池化层进行转移
                names['sub_SPP_' + str(count)] = tensor.view(num, -1)
                # 加入spp列表豪华套餐
                spp_list.append(eval('sub_SPP_' + str(count)))
                # 额外赋值
                x_flatten = tensor.view(num, -1)
            else:
                # 对子金字塔池化层进行转移
                names['sub_SPP_' + str(count)] = tensor.view(num, -1)
                # 加入spp列表豪华套餐
                spp_list.append(eval('sub_SPP_' + str(count)))
                # 合并为一层
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)

        # 返回
        return spp_list[0], spp_list[1], spp_list[2], spp_list[3], x_flatten
        # return spp_list, x_flatten


class PickingNet(nn.Module):
    """
    最基本的到时提取网络(SPP型)
    """
    def __init__(self):
        super(PickingNet, self).__init__()
        self.Convolution_1 = nn.Conv1d(1, 32, kernel_size=21, stride=1, padding=21//2, bias=False)
        self.Convolution_2 = nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=15//2, bias=False)
        self.Convolution_3 = nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=11//2, bias=False)

        self.ReLu = nn.ReLU(inplace=True)
        self.Sigmoid = nn.Sigmoid()

        self.MaxPooling = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # 金字塔池化层
        self.SpatialPyramidPooling1d = SPP1d(4)
        # 做一个BN层,衔接展开之后的一维特征,可加可不加
        self.BatchNorm1d = nn.BatchNorm1d(128 * 10, affine=False)

        self.Dropout = nn.Dropout(0.2)

        self.Dense_1 = nn.Linear(128 * 10, 512)
        self.Dense_2 = nn.Linear(512, 256)
        self.Dense_3 = nn.Linear(256, 1)

    def forward(self, x):
        convolution_1 = self.Convolution_1(x)
        convolution_1 = self.ReLu(convolution_1)
        max_pooling_1 = self.MaxPooling(convolution_1)

        convolution_2 = self.Convolution_2(max_pooling_1)
        convolution_2 = self.ReLu(convolution_2)
        max_pooling_2 = self.MaxPooling(convolution_2)

        convolution_3 = self.Convolution_3(max_pooling_2)
        convolution_3 = self.ReLu(convolution_3)
        # max_pooling_3 = self.MaxPooling(convolution_3)

        _, __, ___, ____,  spp = self.SpatialPyramidPooling1d(convolution_3)

        batch_norm = self.BatchNorm1d(spp)

        dense_1 = self.Dense_1(batch_norm)
        dense_1 = self.ReLu(dense_1)
        dense_1 = self.Dropout(dense_1)

        dense_2 = self.Dense_2(dense_1)
        dense_2 = self.ReLu(dense_2)
        dense_2 = self.Dropout(dense_2)

        dense_3 = self.Dense_3(dense_2)
        dense_3 = self.Sigmoid(dense_3)

        return spp, dense_3


class UNet4MMDPickingNet(nn.Module):
    """
        U-Net做辅助网络的框架(4MMD)
    """
    def __init__(self):
        super(UNet4MMDPickingNet, self).__init__()
        self.Convolution_1 = nn.Conv1d(1, 32, kernel_size=21, stride=1, padding=21//2, bias=False)
        self.Convolution_2 = nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=15//2, bias=False)
        self.Convolution_3 = nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=11//2, bias=False)

        self.Convolution_node = nn.Conv1d(128, 64, kernel_size=11, stride=1, padding=11 // 2, bias=False)

        self.Convolution_4 = nn.Conv1d(128 + 64, 128, kernel_size=11, stride=1, padding=11 // 2, bias=False)
        self.Convolution_5 = nn.Conv1d(128 + 64, 64, kernel_size=11, stride=1, padding=11 // 2, bias=False)
        self.Convolution_6 = nn.Conv1d(64 + 32, 1, kernel_size=11, stride=1, padding=11 // 2, bias=False)

        self.Dense_1 = nn.Linear(128 * 10, 512)
        self.Dense_2 = nn.Linear(512, 256)
        self.Dense_3 = nn.Linear(256, 1)

        self.UpSampling = nn.Upsample(scale_factor=2, mode='nearest')

        self.ReLu = nn.ReLU(inplace=True)
        self.Sigmoid = nn.Sigmoid()

        self.MaxPooling = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # 金字塔池化层
        self.SpatialPyramidPooling1d = SPP1d(4)
        # 做一个BN层,衔接展开之后的一维特征
        self.BatchNorm1d = nn.BatchNorm1d(128 * 10, affine=False)

        self.Dropout = nn.Dropout(0.2)

    def forward(self, x_source, x_target):
        # 编码网络
        # 第一个卷积层
        convolution_1_source = self.Convolution_1(x_source)
        convolution_1_source = self.ReLu(convolution_1_source)
        max_pooling_1_source = self.MaxPooling(convolution_1_source)

        convolution_1_target = self.Convolution_1(x_target)
        convolution_1_target = self.ReLu(convolution_1_target)
        max_pooling_1_target = self.MaxPooling(convolution_1_target)

        # 第二个卷积层
        convolution_2_source = self.Convolution_2(max_pooling_1_source)
        convolution_2_source = self.ReLu(convolution_2_source)
        max_pooling_2_source = self.MaxPooling(convolution_2_source)

        convolution_2_target = self.Convolution_2(max_pooling_1_target)
        convolution_2_target = self.ReLu(convolution_2_target)
        max_pooling_2_target = self.MaxPooling(convolution_2_target)

        # 第三个卷积层
        convolution_3_source = self.Convolution_3(max_pooling_2_source)
        convolution_3_source = self.ReLu(convolution_3_source)
        max_pooling_3_source = self.MaxPooling(convolution_3_source)

        convolution_3_target = self.Convolution_3(max_pooling_2_target)
        convolution_3_target = self.ReLu(convolution_3_target)
        max_pooling_3_target = self.MaxPooling(convolution_3_target)

        # 节点卷积层
        convolution_node_source = self.Convolution_node(max_pooling_3_source)
        convolution_node_source = self.ReLu(convolution_node_source)

        convolution_node_target = self.Convolution_node(max_pooling_3_target)
        convolution_node_target = self.ReLu(convolution_node_target)

        # 解码网络
        # 第一个上采样层
        up_sampling_source = self.UpSampling(convolution_node_source)
        concatenate_temp_source = torch.cat((up_sampling_source, convolution_3_source), 1)
        convolution_4_source = self.Convolution_4(concatenate_temp_source)
        convolution_4_source = self.ReLu(convolution_4_source)

        up_sampling_target = self.UpSampling(convolution_node_target)
        concatenate_temp_target = torch.cat((up_sampling_target, convolution_3_target), 1)
        convolution_4_target = self.Convolution_4(concatenate_temp_target)
        convolution_4_target = self.ReLu(convolution_4_target)

        # 第二个上采样层
        up_sampling_source = self.UpSampling(convolution_4_source)
        concatenate_temp_source = torch.cat((up_sampling_source, convolution_2_source), 1)
        convolution_5_source = self.Convolution_5(concatenate_temp_source)
        convolution_5_source = self.ReLu(convolution_5_source)

        up_sampling_target = self.UpSampling(convolution_4_target)
        concatenate_temp_target = torch.cat((up_sampling_target, convolution_2_target), 1)
        convolution_5_target = self.Convolution_5(concatenate_temp_target)
        convolution_5_target = self.ReLu(convolution_5_target)

        # 第三个上采样层
        up_sampling_source = self.UpSampling(convolution_5_source)
        concatenate_temp_source = torch.cat((up_sampling_source, convolution_1_source), 1)
        convolution_6_source = self.Convolution_6(concatenate_temp_source)
        convolution_6_source = self.Sigmoid(convolution_6_source)

        up_sampling_target = self.UpSampling(convolution_5_target)
        concatenate_temp_target = torch.cat((up_sampling_target, convolution_1_target), 1)
        convolution_6_target = self.Convolution_6(concatenate_temp_target)
        convolution_6_target = self.Sigmoid(convolution_6_target)

        # 预测器
        # 走时提取分支
        spp_s_1, spp_s_2, spp_s_3, spp_s_4, spp_source = self.SpatialPyramidPooling1d(convolution_3_source)
        spp_t_1, spp_t_2, spp_t_3, spp_t_4, spp_target = self.SpatialPyramidPooling1d(convolution_3_target)

        # batch_norm = self.BatchNorm1d(spp_source)

        dense_1 = self.Dense_1(spp_source)
        dense_1 = self.ReLu(dense_1)
        # dense_1 = self.Dropout(dense_1)

        dense_2 = self.Dense_2(dense_1)
        dense_2 = self.ReLu(dense_2)
        # dense_2 = self.Dropout(dense_2)

        dense_3 = self.Dense_3(dense_2)
        dense_3 = self.Sigmoid(dense_3)

        return spp_s_1, spp_s_2, spp_s_3, spp_s_4, spp_t_1, spp_t_2, spp_t_3, spp_t_4,\
            spp_source, spp_target, convolution_6_source, convolution_6_target, dense_3


if __name__ == '__main__':
    # 调用DPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化测试网络
    test_net = UNet4MMDPickingNet().to(device)

    # torch.save(test_net, 'net.pkl')

    # 打印网络参数
    summary(test_net, [(1, 400), (1, 600)])
    # summary(test_net, (1, 400))
