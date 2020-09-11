# 导入data_loader构造函数
from build_dataset import borehole_data_loader
# 导入相关网络
from model import PickingNet
# 导入log和模型保存函数
# from auxiliary.logger import Logger, ModelSaver
# 导入领域自适应loss
# from auxiliary.mmd import mmd_loss

# 导入torch相关
import torch

# 导入时间模块用于计时
import time
# 导入进度条库
from tqdm import tqdm

# 导入可迭代对象
# from collections.abc import Iterable

# 参数解析包
import argparse
# 导入绘图库
import matplotlib.pyplot as plt
# 导入t-SNE绘图
# from auxiliary.t_sne import FeatureVisualize
# 导入numpy
import numpy as np

# 保存结果
import scipy.io as scio


def config():
    # 添加描述.且设置在每个选项的帮助信息后面输出他们对应的缺省值
    parser = argparse.ArgumentParser(description='Test networks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--borehole_data', type=str, default='10dB_borehole_data.mat',
                        help='Borehole dataset for testing')

    # 优化设置
    parser.add_argument('--batch_size', '-b', type=int, default=1000, help='Batch size')
    # parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train')
    # parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--pr', type=float, default=0.05, help='Picking threshold')
    # parser.add_argument('--sr', type=float, default=0.7, help='Save threshold')
    # parser.add_argument('--cr', type=float, default=0.4, help='Correct threshold')
    # parser.add_argument('--lambda_1', type=float, default=1.2, help='MSE coefficient')
    # parser.add_argument('--lambda_2', type=float, default=10, help='MMD coefficient')

    # 检查点及中转路径
    parser.add_argument('--json_name', type=str, default='unet_4MMD_reconstruction_logger_log', help='log name')
    parser.add_argument('--model_name', type=str, default='unet_4MMD_reconstruction', help='model name')
    parser.add_argument('--save', '-s', type=str, default='checkpoints/unet_4MMD_reconstruction',
                        help='Folder to save checkpoints')
    # parser.add_argument('--save_steps', '-ss', type=int, default=50, help='steps to save checkpoints.')
    # parser.add_argument('--base_net_temp', '-sss', type=str,
    #                     default='checkpoints/unet_4MMD_reconstruction/base_net_temp.pth',
    #                     help='Temporary weights')
    parser.add_argument('--base_net_best', type=str,
                        default='base_net_best.pth',
                        help='Best weights')
    parser.add_argument('--output_save', type=str,
                        default='output.mat',
                        help='Results')

    args = parser.parse_args()
    return args


# 完整训练流程
if __name__ == '__main__':
    # 设置参数解析
    args_for_test = config()
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建网络并送入GPU
    picking_net = PickingNet()
    picking_net.to(device)

    # 获取data_loader
    test_data_loader = borehole_data_loader(args_for_test)
    # 获取网络层映射表
    base_net_state_dict = torch.load(args_for_test.base_net_best)
    picking_state_dict = picking_net.state_dict()
    # 提取相同层
    new_state_dict = {k: v for k, v in base_net_state_dict.items() if k in picking_state_dict}
    # 更新参数
    picking_state_dict.update(new_state_dict)
    # 加载参数
    picking_net.load_state_dict(picking_state_dict)

    # 验证评估
    # eval模式
    picking_net.eval()
    # 重新赋值为0
    count = 0
    correct_final = 0

    # 输出保存
    predict_list = []

    for (noise_echo_test, labels_test) in tqdm(test_data_loader):
        with torch.no_grad():
            # 计数+1用于求取每一代的平均值
            count += 1
            # 数据送入GPU
            noise_echo_test_GPU, labels_test_GPU = noise_echo_test.to(device), labels_test.to(device)
            _, predicts = picking_net(noise_echo_test_GPU)
            # print(predicts.size(), labels_test_GPU.size())
            # 计算准确率
            correct_num = (torch.abs(predicts - labels_test_GPU) < args_for_test.pr).sum()
            correct_final += correct_num.item() / len(labels_test)

            output = predicts.data.cpu().squeeze(dim=1).numpy()
            predict_list.extend(list(output))

    correct_final = correct_final / count

    # 绘制图像
    predict_array = np.array(predict_list)
    predict_mat = np.mat(predict_array.reshape((307, 370), order='F'))

    plt.matshow(predict_mat)

    plt.show()
    # 打印结果
    time.sleep(0.5)
    print('test_correct:', correct_final)
    time.sleep(0.5)
    # 保存预测结果
    scio.savemat(args_for_test.output_save, {'predicts': predict_list})

