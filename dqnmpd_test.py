# -*- coding: utf-8 -*-

from __future__ import print_function, division

import traceback
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
# from model4dMPB import ft_net, ft_net_dense
from mpb_model import ft_net, ft_net_dense
# from dataset_processing import FourDimensionDataset
# from random_mars_dataset_processing import FourDimensionDataset
from random_prw_dataset_processing import FourDimensionDataset

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='2', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/home/zzheng/Downloads/Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--crossname', default='cross', help='crossname' )
parser.add_argument('--classnum', default=1000 ,type=int ,help='classnum' )
parser.add_argument('--lowrankpath', default='' , type=str, help='lowrankpath' )

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        # transforms.Resize((144,144), interpolation=3),
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Ten Crop
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop)
          #      for crop in cr           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])


data_dir = test_dir
image_datasets = {}
# image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
#lowrankpath="/home1/dataset/PRW/lowrank/crop/SS/box_gt/"
#lowrankpath="/data2/swfreiddata/PRW/lowrank/crop/SS/box_gt/"
#lowrankpath="/data2/swfreiddata/PRW/prw_Filter_add/"
#lowrankpath="/data2/swfreiddata/PRW/mask/"
#lowrankpath="/data2/swfreiddata/PRW/mask/"
# lowrankpath="/data2/swfreiddata/PRW/mask/"
# lowrankpath ="/home/songwenfeng/swf/pytorch/data/PRW/masklowrank/prw/"
lowrankpath = opt.lowrankpath
#lowrankpath="/data2/swfreiddata/PRW/alllow/masklowrank/prw/"
# image_datasets['gallery'] = FourDimensionDataset(data_dir,'gallery',\
        # lowrankpath, data_transforms)
# image_datasets['query'] = FourDimensionDataset(data_dir,'query',\
        # lowrankpath, data_transforms)
#__init__(self, data, img_path, img_mask, transform=None,randomflip=False,randomflag=False,size=(256,128))
image_datasets['gallery'] = FourDimensionDataset(data_dir,'gallery' ,\
        lowrankpath, data_transforms,randomflag=False, randomflip=False)
image_datasets['query'] = FourDimensionDataset(data_dir,'query' ,\
        lowrankpath, data_transforms,randomflag=False, randomflip=False)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,\
        shuffle=False, num_workers=4) for x in ['gallery','query']}

class_names = image_datasets['query'].img_paths
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    # print (dataloaders)
    for data in dataloaders:
        try:
            img, label = data
            n, c, h, w = img.size()
            count += n
            print(count)
            if opt.use_dense:
                ff = torch.FloatTensor(n,1000).zero_()
                # ff = torch.FloatTensor(n,1024).zero_()
            else:
                ff = torch.FloatTensor(n,1000).zero_()
                # ff = torch.FloatTensor(n,2048).zero_()
            for i in range(2):
                if(i==1):
                    img = fliplr(img)
                input_img = Variable(img.cuda())
                #print (input_img)
                outputs = model(input_img)
                print (outputs)
                f = outputs.data.cpu()
                print(f.size())
                ff = ff+f
            # norm feature
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features,ff), 0)
        except Exception as e:
            print (e)
            traceback.print_exc()
            continue
    return features

def get_id(img_path):
    # print (img_path)
    camera_id = []
    labels = []
    for path in img_path:
        filename = path.split('/')[-1]
        # print (filename)
        label = path.split('/')[-2]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def get_id_name(img_path):
    # print (img_path)
    camera_id = []
    labels = []
    names = []
    for path in img_path:
        # print (path)
        names.append(path)
        filename = path.split('/')[-1]
        # print (filename)
        label = path.split('/')[-2]
        try:
            camera = filename.split('C')[1]
        except Exception as e:
            camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels, names
gallery_path = image_datasets['gallery'].img_paths
query_path = image_datasets['query'].img_paths

gallery_cam,gallery_label, gallery_names = get_id_name(gallery_path)
query_cam,query_label, query_names = get_id_name(query_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
#classifer_number = 13513
classifer_number = 1000
classifer_number = opt.classnum
if opt.use_dense:
    model_structure = ft_net_dense(classifer_number)
else:
    model_structure = ft_net(classifer_number)
model = load_network(model_structure)

# Remove the final fc layer and classifier layer
# model.model.fc = nn.Sequential()
# model.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
gallery_feature = extract_feature(model,dataloaders['gallery'])
query_feature = extract_feature(model,dataloaders['query'])

crossname = opt.crossname
# Save to Matlab for check
# result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'gallery_names':gallery_names,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam,'query_names':query_names}
scipy.io.savemat(crossname+name+'_pytorch_result.mat',result)

