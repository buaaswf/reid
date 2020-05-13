# -*- coding: utf-8 -*-

import scipy.io as sio
import os
import shutil
import sys
# import glob
###########create reward to action files for densenet121#############
def GetFileList(dir,filelist):
    newDir = dir
    if os.path.isfile(dir):
        filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # if s.endswith(".abs") or s.endswith(".sh") or s.endswith(".py") or s.endswith(".prototxt") or s.endswith(".solverstate"):
            # if not s.endswith():
                # continue
            newDir = os.path.join(dir, s)
            GetFileList(newDir, filelist)
    return filelist
def reward2action(reward, file, action):
    print (reward)
    r=sio.loadmat(reward)['reward']
    # print (r)
    dstpath = os.path.join(action, str(r[0][0]))
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    print (file)
    print (dstpath)
    shutil.copy(file,dstpath)
def reward2lowrank(reward,dst,l0,l1,l2):
    print (reward)
    r=sio.loadmat(reward)['reward']
    # id=file.split("/")[-1]
    # print (r)
    # dstpath = os.path.join(reward, str(id))
    dstpath = dst
    print (dst)
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    # print (file)
    # print (dstpath)
# shutil.copy(file,dstpath)
    try:
        if r=='2':
            shutil.copy(l2,dstpath)
        elif r=='1':
            shutil.copy(l1,dstpath)
        else:
            shutil.copy(l0,dstpath)
    except  Exception as e:
        return



def listfiles(rewardpath, imgpath, dst):
    imglist=GetFileList(imgpath,[])
    for img in imglist:
        name=img.split("/")[-1].split(".")[0]+".mat"
        id = img.split("/")[-2]
        reward=os.path.join(rewardpath, str(int(id)), name)
        reward2action(reward, img, dst)
def create_new_lowrank_by_reward(rewardpath, l0path, l1path,l2path, dst):
    imglist=GetFileList(rewardpath,[])
    for img in imglist:
        oriname = img.split("/")[-1]
        oriname=img.split("/")[-1].split(".")[0]+".jpg"
        name=img.split("/")[-1].split(".")[0]+".mat"
        id = img.split("/")[-2]
        reward=os.path.join(rewardpath, id, name)
        l0=os.path.join(l0path, id,oriname)
        l1=os.path.join(l1path, id, oriname)
        l2=os.path.join(l2path, id, oriname)
        dstf= os.path.join(dst,id)
        reward2lowrank(reward,dstf,l0,l1,l2)

if __name__=="__main__":
    # listfiles("/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndataset","/home/songwenfeng/swf/pytorch/data/PRW/PRW_pytorch/train_all/",\
              # "/home/songwenfeng/swf/pytorch/code/dqn/imagenet/actor-critic/timeactionprid/")
    # listfiles("/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndataset","/home/songwenfeng/swf/pytorch/data/PRW/PRW_pytorch/train_all/",\
              # "/home/songwenfeng/swf/pytorch/code/dqn/imagenet/actor-critic/timeactionilids/")
    # create_new_lowrank_by_reward("/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndataset","/home/songwenfeng/swf/pytorch/data/PRW/PRW_pytorch/train_all/",\
              # "/home/songwenfeng/swf/pytorch/data/prw/newlowrank/", "/home/songwenfeng/swf/pytorch/data/prid/")
    # create_new_lowrank_by_reward("/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndataset/",\
                                 # "/home/songwenfeng/swf/pytorch/data/PRW/rgbmasktemporalcontext/frame_out/50/",\
                                 # "/home/songwenfeng/swf/pytorch/data/PRW/rgbmasktemporalcontext/frame_out/20/",\
                                 # "/home/songwenfeng/swf/pytorch/data/PRW/rgbmasktemporalcontext/frame_out/100/",\
                                 # "/home/songwenfeng/swf/pytorch/data/prw/newlowrank/")
    # create_new_lowrank_by_reward("/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndatasetprid/",\
                                 # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/50/",\
                                 # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/100/",\
                                 # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/1000/",\
                                 # "/home/songwenfeng/swf/pytorch/data/prid/newlowrank/")
    # create_new_lowrank_by_reward("/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndatasetmarket/",\
                                 # "/home/songwenfeng/swf/pytorch/data/market1501/frame_out/50/",\
                                 # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/100/",\
                                 # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/1000/",\
                                 # "/home/songwenfeng/swf/pytorch/data/prid/newlowrank/")
    # create_new_lowrank_by_reward("/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndatasetprid/",\
                                 # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/50/",\
                                 # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/100/",\
                                 # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/1000/",\
                                 # "/home/songwenfeng/swf/pytorch/data/prid/newlowrank/")
    # create_new_lowrank_by_reward(sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
    # listfiles("/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndataset","/home/songwenfeng/swf/pytorch/data/PRW/PRW_pytorch/train_all/",\
              # "/home/songwenfeng/swf/pytorch/code/dqn/imagenet/actor-critic/timeactionilids/")
    listfiles(sys.argv[1], sys.argv[2], sys.argv[3])
