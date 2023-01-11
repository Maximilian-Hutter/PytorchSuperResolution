import torch
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode, pil_to_tensor
from models import *
import torchvision.transforms as T
from torchvision import transforms, utils
from PIL import Image
import argparse
import time
import torchvision


parser = argparse.ArgumentParser(description='PyTorch ESRGANplus')
parser.add_argument('--modelpath', type=str, default="weights/9EDSR.pth", help=("path to the model .pth files"))
parser.add_argument('--inferencepath', type=str, default='D:/Data/test/', help=("Path to image folder"))
parser.add_argument('--imagename', type=str, default='NH.png', help=("filename of the image"))
parser.add_argument('--gpu_mode', type=bool, default=True, help=('enable cuda'))
parser.add_argument('--channels',type=int, default=3, help='number of channels R,G,B for img / number of input dimensions 3 times 2dConv for img')
parser.add_argument('--filters', type=int, default=8, help="set number of filters")
parser.add_argument('--bottleneck', type=int, default=8, help="set number of filters")
parser.add_argument('--n_resblock', type=int, default=3, help="set number of filters")
parser.add_argument('--scale', type=int, default=2, help="set number of filters")
    
      

opt = parser.parse_args()

PATH = opt.modelpath
imagepath = (opt.inferencepath + opt.imagename)
image = Image.open(imagepath)
image = image.resize((int(512/opt.scale), int(512/opt.scale)))
image.save('results/SD.png')

transformtotensor = transforms.Compose([transforms.ToTensor()])
image = transformtotensor(image)

image = image.unsqueeze(0)

image= image.to(torch.float32)

model=SuperResolution(opt.filters, opt.bottleneck, opt.n_resblock, opt.scale)

if opt.gpu_mode == False:
    device = torch.device('cpu')

if opt.gpu_mode:
    	device = torch.device('cuda:0')

model.load_state_dict(torch.load(PATH,map_location=device))

model.eval()
start = time.time()
transform = T.ToPILImage()
out = model(image)
out = transform(out.squeeze(0))
end = time.time()
proctime = end -start
out.save('results/HD.png')
print(proctime)