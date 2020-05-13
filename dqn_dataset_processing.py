import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
from cv2 import imread
from torchvision.transforms import *
import random
import PIL
import scipy.io as sio

def GetFileList(dir,filelist):
    newDir = dir
    if os.path.isfile(dir):
        filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            if s.endswith(".abs") or s.endswith(".sh") or s.endswith(".py") or s.endswith(".prototxt") or s.endswith(".solverstate"):
                continue
            newDir = os.path.join(dir, s)
            GetFileList(newDir, filelist)
    return filelist

class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_path, img_filename, label_filename, transform=None):
        self.img_path = os.path.join(data_path, img_path)
        self.transform = transform
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)
        labels = np.loadtxt(label_filepath, dtype=np.int64)
        self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label
    def __len__(self):
        return len(self.img_filename)


class DQN_Dataset(Dataset):
    # datapath dataset root
    #img_path train val query gallery
    def __init__(self, img_path, transform=None):
        self.img_paths = GetFileList(img_path,[])
        # self.img_paths = os.listdir(img_path)

    def __getitem__(self, index):

        data = sio.loadmat(self.img_paths[index])
        # print (self.img_paths[index])
        # img =  data['feature'][0][:500]
        # print (data['feature'][0][0])
        temp = (1-data['feature'][0][0])*10
        img =  np.asarray([temp, temp,temp])
        # img = np.random.random((1,3))[0]
        # img = np.array([data['reward'][0][0], data['reward'][0][0], data['reward'][0][0]])
        # print (img)
        # img = img.reshape(( 32,32))
        # img =  cv2.resize(img, (28, 28))
        # print (img)
        # equ = cv2.equalizeHist((img*255).astype(int))
        # res = np.hstack((img,equ))
        # img = res
        # img = img.reshape((1, 28,28))
        # equ = cv2.equalizeHist(img)
        # # res = np.hstack((img,equ))
        # img = equ
        # print (img.shape)
        # img = [img, img, img]
        label  = data['reward'][0][0]
        img = torch.tensor(img).float()
        label = torch.tensor(label)


        # label = int(self.img_paths[index].split("/")[-2])
        return img, label
    def __len__(self):
        return len(self.img_paths)

class FourDimensionDataset(Dataset):
    # datapath dataset root
    #img_path train val query gallery
    def __init__(self, data, img_path, img_mask, transform=None):
        self.img_paths = GetFileList(os.path.join(data, img_path),[])
        # self.img_paths = os.listdir(os.path.join(data, img_path))
        self.data = data
        # reading img file from file
        # img_full_path = os.path.join(data, img_path)
        # self.img_masks = os.path.join(data, img_mask)
        self.img_masks = img_mask
        # print (self.img_masks)
        self.transform = transform
        self.target_transform = transform

    def __getitem__(self, index):
        # img = Image.open(os.path.join(self.data, self.img_paths[index]))
        # print (self.img_paths)
        # print ("dataprocessoing")
        # print (self.img_paths[index])
        # print ("===============")
        # print (len(self.img_paths))
        # print (index)
        img = imread(self.img_paths[index])
        #print (self.img_paths[index])
        if img is None:
            img = np.zeros((144,144))
        img = cv2.resize(img,(144,144),interpolation=cv2.INTER_CUBIC)
        # print (img)
        img = PIL.Image.fromarray(img)
        # img = img.convert('RGB')
        # filename1 = "/".join(self.img_paths[index].split("/")[-2:])
        filename1,f2,f3 = self.img_paths[index].split("/")[-3:]
        # filename = ("{:0>4d}"+"_"+f3).format(int(f2))
        filename = filename1+"/"+f3
        #filename = f3
        # print (filename)
        #print (self.img_masks + filename)
        if os.path.isfile(self.img_masks+filename):
            img_mask = imread(self.img_masks + filename)
        else:
            #filename = self.img_paths[index].split("/")[-2]+"/"+self.img_paths[index].split("/")[-1][4:]
            filename = self.img_paths[index].split("/")[-2]+"/"+self.img_paths[index].split("/")[-1]
            if os.path.isfile( (self.img_masks + filename)):
                #print (">>>>>>>>>>>>>>>>>>>>>>>>>", self.img_masks + filename)
                img_mask = imread(self.img_masks + filename)
            else:
                # print (">>>>>>>>>>>>>>>>>>>>>>>>>", self.img_masks + filename)
                img_mask =np.asarray( img.convert('L'))
                # print ("img-------------")
        # print ("------------------------")
        # print (self.img_masks + filename)
        # print ("==================",self.img_masks + filename)
        #print (img_mask)
        if img_mask is None:
            print ("???????????None")
            print (img)
            img_mask=np.asarray( img.convert('L'))
        #img_mask = imread(self.img_masks + filename)
        # print (self.img_masks + filename)
        img_mask = cv2.resize(img_mask,(144,144),interpolation=cv2.INTER_CUBIC)
        img_mask = PIL.Image.fromarray(img_mask).convert('L')
        seed = np.random.randint(2147483647) # make a seed with numpy generator
        random.seed(seed) # apply this seed to img tranfsorms
        if self.transform is not None:
            img = self.transform(img)

        random.seed(seed) # apply this seed to target tranfsorms
        if self.target_transform is not None:
            img_mask = self.target_transform(img_mask)

        # label = torch.ByteTensor(np.array(int(self.img_paths[index].split("/")[-2])))
        #print (img)
        #print (img_mask)
        # print ("---------->>>>>>>>>>>>")
        label = int(self.img_paths[index].split("/")[-2])

        # if (label <= 0):
            # label = 1
        # print ("-------------------------------")
        # print (int(self.img_paths[index].split("/")[-2]))
        # label = torch.ByteTensor(int(self.img_paths[index].split("/")[-2]))
        # label = int(self.img_paths[index].split("/")[-2])

        # label = torch.from_numpy(int(self.img_paths[index].split("/")[-2]))
        # img = torch.from_numpy(np.concatenate((img, img_mask), axis=0))
        img = np.concatenate((img, img_mask), axis=0)
        # print (img.shape)
        # img_mask = img_mask.convert('')
        # if self.transform is not None:
            # img = self.transform(img)
        # label = torch.from_numpy(self.label[index])
        # print (img.size, label.size)
        # if label > 483:
            # print (label)
        return img, label
    def __len__(self):
        return len(self.img_paths)
        # return 934
class FourDimensionDatasetVal(Dataset):
    # datapath dataset root
    #img_path train val query gallery
    def __init__(self, data, img_path, img_mask, transform=None):
        self.img_paths = GetFileList(os.path.join(data, img_path),[])
        # self.img_paths = os.listdir(os.path.join(data, img_path))
        self.data = data
        # reading img file from file
        # img_full_path = os.path.join(data, img_path)
        self.img_masks = os.path.join(data, img_mask)
        self.transform = transform
        self.target_transform = transform

    def __getitem__(self, index):
        # print (index)
        # img = Image.open(os.path.join(self.data, self.img_paths[index]))
        # print (self.img_paths)
        # print (self.img_paths[index])
        # print ("===============")
        # print (len(self.img_paths))
        # print (index)
        img = imread(self.img_paths[index])
        print (self.img_paths[index])
        img = cv2.resize(img,(144,144),interpolation=cv2.INTER_CUBIC)
        # print (img)
        img = PIL.Image.fromarray(img)
        # img = img.convert('RGB')
        filename = "/".join(self.img_paths[index].split("/")[-3:])
        img_mask = imread(self.img_masks + filename)
        # print (self.img_masks + filename)
        # print (img_mask.shape)
        img_mask = cv2.resize(img_mask,(144,144),interpolation=cv2.INTER_CUBIC)
        img_mask = PIL.Image.fromarray(img_mask).convert('L')
        seed = np.random.randint(2147483647) # make a seed with numpy generator
        random.seed(seed) # apply this seed to img tranfsorms
        if self.transform is not None:
            img = self.transform(img)

        random.seed(seed) # apply this seed to target tranfsorms
        if self.target_transform is not None:
            img_mask = self.target_transform(img_mask)

        # label = torch.ByteTensor(np.array(int(self.img_paths[index].split("/")[-2])))
        # print ("---------->>>>>>>>>>>>")
        label = int(self.img_paths[index].split("/")[-2])

        # if (label <= 0):
            # label = 1
        # print ("-------------------------------")
        # print (int(self.img_paths[index].split("/")[-2]))
        # label = torch.ByteTensor(int(self.img_paths[index].split("/")[-2]))
        # label = int(self.img_paths[index].split("/")[-2])

        # label = torch.from_numpy(int(self.img_paths[index].split("/")[-2]))
        # img = torch.from_numpy(np.concatenate((img, img_mask), axis=0))
        img = np.concatenate((img, img_mask), axis=0)
        # print (img.shape)
        # img_mask = img_mask.convert('')
        # if self.transform is not None:
            # img = self.transform(img)
        # label = torch.from_numpy(self.label[index])
        # print (img.size, label.size)
        # print (label)
        return img, label
    def __len__(self):
        return len(self.img_paths)
        # return 483
class MyJointRandomFlipTransform(object):
    def __call__(self, image1, image2,):
        # provide a functional interface for the current transforms
        # so that they can be easily reused, and have the parameters
        # of the transformation if needed
        image1, params = random_horizontal_flip(image1, return_params=True)

        # reuses the same transformations, if wanted
        image2 = random_horizontal_flip(image2, params=params)
        img1 = image1
        img2 = image2
        img1, params = RandomCrop(img1,(64,64),return_params=True)
        img2 = RandomCrop(img1, params=params)
        # no transformation in torchvision for bounding_box, have to do it
        # ourselves
        # if params.flip:
            # bounding_box[:, 1] = image1.size(2) - bounding_box[:, 1]
            # bounding_box[:, 3] = image1.size(2) - bounding_box[:, 3]
        return image1, image2
class MyJointRandomCrop(object):
    def __call__(self, image1, image2,):
        # provide a functional interface for the current transforms
        # so that they can be easily reused, and have the parameters
        # of the transformation if needed
        img1 = image1
        img2 = image2
        img1, params = RandomCrop(img1,(64,64),return_params=True)
        img2 = RandomCrop(img1, params=params)
        # no transformation in torchvision for bounding_box, have to do it
        # ourselves
        # if params.flip:
            # bounding_box[:, 1] = image1.size(2) - bounding_box[:, 1]
            # bounding_box[:, 3] = image1.size(2) - bounding_box[:, 3]
        return img1, img2
