import numpy as np
import torch
import torch as nn
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import socket
import argparse
from prefetch_generator import BackgroundGenerator
import time
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import itertools
import torchvision.transforms as T
import os
from PIL import ImageOps
from PIL import Image
from tqdm import tqdm
from feature_extract import FeatureExtractor
import logging


# import own code
from get_data import ImageDataset
from models import SuperResolution
from criterions import *
import myutils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch EDSR') #D:/Data/div2k/DIV2K_train_HR/
    parser.add_argument('--train_data_path', type=str, default="D:/Data/div2k/DIV2K_train_HR/", help=("path for the data"))
    parser.add_argument('--height', type=int, default=1000, help=("set the height of the image in pixels"))
    parser.add_argument('--width', type=int, default=1000, help=("set the width of the image in pixels"))
    parser.add_argument('--imgchannels', type=int, default=3, help=("set the channels of the Image (default = RGB)"))
    parser.add_argument('--augment_data', type=bool, default=False, help=("if true augment train data"))
    parser.add_argument('--batchsize', type=int, default=1, help=("set batch Size"))
    parser.add_argument('--gpu_mode', type=bool, default=True) 
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
    parser.add_argument('--mini_batch',type=int, default=16, help='mini batch size')
    parser.add_argument('--resume',type=bool, default=False, help='resume training/ load checkpoint')
    parser.add_argument('--model_type', type=str, default="EDSR", help="set type of model")
    parser.add_argument('--filters', type=int, default=8, help="set number of filters")
    parser.add_argument('--bottleneck', type=int, default=8, help="set number of filters")
    parser.add_argument('--n_resblock', type=int, default=3, help="set number of filters")
    parser.add_argument('--scale', type=int, default=2, help="set number of filters")
    parser.add_argument('--beta1',type=float, default=0.9, help='decay of first order momentum of gradient')
    parser.add_argument('--beta2',type=float, default=0.999, help='decay of first order momentum of gradient')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help="learning rate")
    parser.add_argument('--snapshots', type=int, default=10, help="number of epochs until a checkpoint is made")
    parser.add_argument('--loss_weight', type=float, default=1, help="set weight for loss addition")
    parser.add_argument('--crit_lambda', type=float, default=1, help="loss mult for l1loss")
    parser.add_argument('--cont_lambda', type=float, default=1, help="loss mult for feature loss")
    parser.add_argument('--lum_lambda', type=float, default=1, help="loss mult for lum loss")
    parser.add_argument('--blk_lambda', type=float, default=2.3, help="loss mult for blk loss")
    parser.add_argument('--wht_lambda', type=float, default=2.3, help="loss mult for wht loss")
    parser.add_argument
    opt = parser.parse_args()
    np.random.seed(opt.seed)    # set seed to default 123 or opt
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    gpus_list = range(opt.gpus)
    hostname = str(socket.gethostname)
    cudnn.benchmark = True
    print(opt)  # print the chosen parameters

    # dataloader

    size = (opt.height, opt.width)

    print('==> Loading Datasets')
    dataloader = DataLoader(ImageDataset(opt.train_data_path,size, opt.scale,opt.augment_data), batch_size=opt.batchsize, shuffle=True, num_workers=opt.threads)

    # instantiate model

    #Generator = ESRGANplus(opt.channels, filters=opt.filters,hr_shape=hr_shape, n_resblock = opt.n_resblock, upsample = opt.upsample)
    Net = SuperResolution(opt.filters, opt.bottleneck, opt.n_resblock, opt.scale)
    feature_extractor = FeatureExtractor()
    feature_extractor.eval()


    #parameters
    pytorch_params = sum(p.numel() for p in Net.parameters())
    print("Network parameters: {}".format(pytorch_params))

    # loss
    criterion = torch.nn.L1Loss(reduction="mean")
    content_criterion = torch.nn.L1Loss(reduction="mean")
    #abs_crit = abs_criterion()
    #lum_crit = luminance_criterion()

    # run on gpu
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    if cuda:
        Net = Net.cuda(gpus_list[0])
        criterion = criterion.cuda(gpus_list[0])
        feature_extractor = feature_extractor.cuda(gpus_list[0])
        content_criterion = content_criterion.cuda(gpus_list[0])

    optimizer = optim.Adam(Net.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    # load checkpoint/load model
    star_n_iter = 0
    start_epoch = 0
    if opt.resume:
        checkpoint = torch.load(opt.save_folder) ## look at what to load
        start_epoch = checkpoint['epoch']
        start_n_iter = checkpoint['n_iter']
        optimizer.load_state_dict(checkpoint['optim'])
        print("last checkpoint restored")

    def checkpointG(epoch):
        model_out_path = opt.save_folder+str(epoch)+opt.model_type+".pth".format(epoch)
        torch.save(Net.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    # Tensorboard
    writer = SummaryWriter()

    # define Tensor
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    Net = Net.to(memory_format=torch.channels_last)  # faster train time with Computer vision models
    torch.autograd.profiler.emit_nvtx(enabled=False)
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, opt.nEpochs):
        epoch_loss = 0
        Net.train()
        epoch_time = time.time()
        correct = 0
        for i, imgs in enumerate(BackgroundGenerator(tqdm(dataloader)),start=0):#:BackgroundGenerator(dataloader,1))):    # put progressbar

            start_time = time.time()
            img = Variable(imgs["img"].type(Tensor))
            img = img.to(memory_format=torch.channels_last)  # faster train time with Computer vision models
            label = Variable(imgs["label"].type(Tensor))

            if cuda:    # put variables to gpu
                img = img.to(gpus_list[0])
                label = label.to(gpus_list[0])

            # start train
            for param in Net.parameters():
                param.grad = None

            with torch.cuda.amp.autocast():
                generated_image = Net(img)
                #gen_features = feature_extractor(generated_image).detach()
                #real_features = feature_extractor(label).detach()
                #content_loss = content_criterion(gen_features, real_features)
                crit = criterion(generated_image, label)
                lum = luminance_criterion(generated_image, label)
                loss = opt.cont_lambda + opt.crit_lambda *  crit + lum * opt.lum_lambda# + wht * opt.wht_lambda + blk * opt.blk_lambda   # add a vgg net feature extraction loss
            


            if i == 1:
                if opt.batchsize == 1:
                    myutils.save_trainimg(generated_image, epoch)


            train_acc = torch.sum(generated_image == label)
            epoch_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #compute time and compute efficiency and print information
            process_time = time.time() - start_time
            #print("process time: {}, Number of Iteration {}/{}".format(round(process_time, 3),i , (len(dataloader)-1)))
            #pbar.set_description("Compute efficiency. {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time),epoch, opt.epoch))


        if (epoch+1) % (opt.snapshots) == 0:
            checkpointG(epoch)

        epoch_time = time.time() - epoch_time 
        Accuracy = 100*train_acc / len(dataloader)
        writer.add_scalar('loss', epoch_loss, global_step=epoch)
        writer.add_scalar('accuracy',Accuracy, global_step=epoch)
        print("===> Epoch {} Complete: Avg. loss: {:.4f} Accuracy {}, Epoch Time: {:.3f} seconds \n".format(epoch, ((epoch_loss/2) / len(dataloader)), Accuracy, epoch_time))

    def print_network(net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)

    print('===> Building Model ', opt.model_type)
    if opt.model_type == 'EDSR':
        Net = Net


    print('----------------Network architecture----------------')
    print_network(Net)
    print('----------------------------------------------------')