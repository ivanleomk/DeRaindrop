#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
# import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
import argparse
#Models lib
from models import *
#Metrics lib
from metrics import calc_psnr, calc_ssim
import math

THUMBNAIL_MAX_WIDTH = 800
THUMBNAIL_MAX_HEIGHT = 800

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    args = parser.parse_args()
    return args

def align_to_four(img):
    #print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    #align to four
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4
    img = img[0:a_row, 0:a_col]
    #print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img


def predict(image):
    image = np.array(image, dtype='float32')/255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image).cuda()

    out = model(image)[-1]

    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((0, 2, 3, 1))
    out = out[0, :, :, :]*255.
    
    return out




if __name__ == '__main__':
    args = get_args()

    model = Generator().cuda()
    model.load_state_dict(torch.load('./weights/gen.pkl'))

    if args.mode == 'demo':
        input_list = sorted(os.listdir(args.input_dir))
        num = len(input_list)
        for i in range(num):
            print ('Processing image: %s'%(input_list[i]))
            img = cv2.imread(args.input_dir + input_list[i])
            height,width,channels = img.shape
            print("----Encountered image of size {} and width {}".format(height,width))
            if height > THUMBNAIL_MAX_HEIGHT or width > THUMBNAIL_MAX_WIDTH:
                scale_percent = 50 # percent of original size
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                new_height,new_width,_ = img.shape
                print("Image dimensions are now {},{} from {},{}".format(height,width,new_height,new_width))
            img = align_to_four(img)
            result = predict(img)
            img_name = input_list[i].split('.')[0]
            print("Writing {} to function now at {}".format(img_name,args.output_dir + img_name + '.jpg'))
            if not cv2.imwrite(args.output_dir + img_name + '.jpg', result):
                raise Exception("Could not write image {} to file".format(args.output_dir + img_name + '.jpg'))

    elif args.mode == 'test':
        input_list = sorted(os.listdir(args.input_dir))
        gt_list = sorted(os.listdir(args.gt_dir))
        num = len(input_list)
        cumulative_psnr = 0
        cumulative_ssim = 0
        for i in range(num):
            
            img = cv2.imread(args.input_dir + input_list[i])
            

            gt = cv2.imread(args.gt_dir + gt_list[i])
            img = align_to_four(img)
            gt = align_to_four(gt)
            result = predict(img)
            result = np.array(result, dtype = 'uint8')
            cur_psnr = calc_psnr(result, gt)
            cur_ssim = calc_ssim(result, gt)
            print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr, cur_ssim))
            cumulative_psnr += cur_psnr
            cumulative_ssim += cur_ssim
        print('In testing dataset, PSNR is %.4f and SSIM is %.4f'%(cumulative_psnr/num, cumulative_ssim/num))

    else:
        print ('Mode Invalid!')

# CUDA_VISIBLE_DEVICES=0 python predict.py --mode demo --input_dir ./demo/input/ --output_dir ./demo/output/

