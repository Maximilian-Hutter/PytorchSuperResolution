from models import SuperResolution
import numpy as np
import torch
import torch.nn as nn
import argparse
import socket
import torch.backends.cudnn as cudnn
import time
from torchsummary import summary
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch EDSR')
    parser.add_argument('--train_data_path', type=str, default="D:/Data/test/", help=("path for the data"))
    parser.add_argument('--height', type=int, default=360, help=("set the height of the image in pixels"))
    parser.add_argument('--width', type=int, default=640, help=("set the width of the image in pixels"))
    parser.add_argument('--imgchannels', type=int, default=3, help=("set the channels of the Image (default = RGB)"))
    parser.add_argument('--augment_data', type=bool, default=False, help=("if true augment train data"))
    parser.add_argument('--batchsize', type=int, default=4, help=("set batch Size"))
    parser.add_argument('--gpu_mode', type=bool, default=True) 
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
    parser.add_argument('--mini_batch',type=int, default=16, help='mini batch size')
    parser.add_argument('--resume',type=bool, default=False, help='resume training/ load checkpoint')
    parser.add_argument('--model_type', type=str, default="EDSR", help="set type of model")
    parser.add_argument('--filters', type=int, default=16, help="set number of filters")
    parser.add_argument('--bottleneck', type=int, default=16, help="set number of filters")
    parser.add_argument('--n_resblock', type=int, default=8, help="set number of filters")
    parser.add_argument('--scale', type=int, default=2, help="set number of filters")
    parser.add_argument('--beta1',type=float, default=0.9, help='decay of first order momentum of gradient')
    parser.add_argument('--beta2',type=float, default=0.999, help='decay of first order momentum of gradient')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0004, help="learning rate")
    parser.add_argument('--snapshots', type=int, default=10, help="number of epochs until a checkpoint is made")
    parser.add_argument('--loss_weight', type=float, default=1, help="set weight for loss addition")
    parser.add_argument
    opt = parser.parse_args()
    np.random.seed(opt.seed)    # set seed to default 123 or opt
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    gpus_list = range(opt.gpus)
    hostname = str(socket.gethostname)
    cudnn.benchmark = True
    print(opt)  # print the chosen parameters

    # defining shapes
    size = (opt.height, opt.width)

    Net = Net = SuperResolution(opt.filters, opt.bottleneck, opt.n_resblock, opt.scale).cuda()


    start = time.time()
    summary(Net, (3, 64, 64))
    end = time.time()

    proctime = end-start
    print(proctime)

    # pytorch_params = sum(p.numel() for p in Net.parameters())
    # print("Network parameters: {}".format(pytorch_params))

    # def print_network(net):
    #     num_params = 0
    #     for param in net.parameters():
    #         num_params += param.numel()
    #     print(net)
    #     print('Total number of parameters: %d' % num_params)

    # print('===> Building Model ')
    # Net = Net


    # print('----------------Network architecture----------------')
    # print_network(Net)
    # print('----------------------------------------------------')