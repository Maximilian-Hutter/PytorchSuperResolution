import torch
import torch.nn as nn

def luminance_criterion(gen_img, target, alpha=0.05):

    loss =  torch.mean(torch.mean((gen_img - target)**2, [1, 2, 3]) - alpha * torch.pow(torch.mean(gen_img - target, [1, 2, 3]), 2))
    
    return loss

def blackness_criterion(gen_img, target):   # check for low value pixels =  x < 0.1
    gen_black_pixel = torch.where(gen_img <  0.1, gen_img, 0.)
    target_black_pixel = torch.where(target < 0.1, target, 0.)
    crit = nn.L1Loss()
    loss = crit(gen_black_pixel, target_black_pixel)
    return loss

def whiteness_criterion(gen_img, target):   # check for low value pixels =  x < 0.1
    gen_white_pixel = torch.where(gen_img >  0.9, gen_img, 0.)
    target_white_pixel = torch.where(target > 0.9, target, 0.)
    crit = nn.L1Loss()
    loss = crit(gen_white_pixel, target_white_pixel)

    return loss    