import scipy.io as sio
import torch
from torch.utils.data import TensorDataset, DataLoader


# 无标签的回波数据集读取以及有标签的地震数据提取
def borehole_data_loader(args):
    # 读取指定文件
    borehole_data = sio.loadmat(args.borehole_data)
    # 读取井壁回波数据
    noise_borehole_data = borehole_data['noise_signal_2_network']
    borehole_label = borehole_data['label_2_network']
    # 获取样本数目
    num_borehole_instances = len(noise_borehole_data)
    # 打印出样本的数目
    print('Num pulse-echo (borehole):', num_borehole_instances)
    # 转换类型
    noise_borehole_data = torch.from_numpy(noise_borehole_data).type(torch.FloatTensor)
    borehole_label = torch.from_numpy(borehole_label).type(torch.FloatTensor)
    # 增加维度
    noise_borehole_data = noise_borehole_data.view(num_borehole_instances, 1, -1)
    borehole_label = borehole_label.view(num_borehole_instances, 1)

    # 构成训练用的数据集
    dataset = TensorDataset(noise_borehole_data,
                            borehole_label)

    # 返回DataLoader
    # 不打乱数据集
    return DataLoader(dataset=dataset,
                      batch_size=args.batch_size,
                      shuffle=False,
                      pin_memory=True)







