# -*- coding: utf-8 -*-
import torch
import numpy as np

import argparse
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
# from torchvision import datasets, transforms
from dqn_dataset_processing  import DQN_Dataset as datasets

# class DQN(object):
    # def __init__(self):
        # self.net=Net()
        # self.memory=np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        # self.optimizer=torch.optim.Adam(self.eval_net.parameters(), lr=LR)
def cosine(f1,f2):
    return (np.dot(f1,f2))/(np.sqrt(np.dot(f1,f1)*np.sqrt(np.dot(f2,f2))))
def generatelabelfordqn(f0,f1,f2,f3, label,names):
    reward=[]
    import os
    # f0 = (f0-np.min(f0))/(np.max(f0)-np.min(f0))
    for i in range(0,len(label)):
        # print (label)
        j = label[i]
        # print (i,j)
        print (f1.shape)
        # print ((f1[i][j]).shape)
        # print ((f0[i][j]))
        print ((f0[i][j]), np.max(f0[i]),np.argmax(f0[i]))
        print ((f1[i][j]), np.max(f1[i]),np.argmax(f1[i]))
        print ((f2[i][j]),np.max(f2[i]),np.argmax(f2[i]))
        print ((f3[i][j]),np.max(f3[i]),np.argmax(f3[i]))
        # print ((f4[i][j]),np.max(f4[i]),np.argmax(f3[i]))
        # print ()
        # if f1[i][j] < f2[i][j] and f2[i][j] < f3[i][j]:
            # reward.append(2)
        # elif f1[i][j] < f2[i][j]:
            # reward.append(1)
        # else:
            # reward.append(0)
        if f0[i][j] < f2[i][j] or f0[i][j]< f3[i][j]: #and f2[i][j] < f3[i][j]:
            reward.append(2)
        elif l1[i][j] > f3[i][j]:
            reward.append(1)
        else:
            reward.append(0)
        # print (reward)
        # outputpath = os.path.join(outputdir)
        print (names[i])
        print (label[i])
        id = str(label[i])
        cos1 = cosine(f2[i],f1[i])
        cos2 = cosine(f3[i],f2[i])
        cos3 = cosine(f1[i],f3[i])

        name = names[i].split("/")[-1].split(".")[0]
        print (name)
        # if name=='/home/songwenfeng/swf/pytorch//380/c2s1_104621.jpg'
        if not os.path.exists(os.path.join("dqndataset",id)):
            os.makedirs(os.path.join("dqndataset",id))
        # sio.savemat(os.path.join("dqndataset",id,name+".mat"),\
                    # {'reward':reward[i],'feature':f0[i]})
        sio.savemat(os.path.join("dqndataset",id,name+".mat"),\
                    # {'reward':reward[i],'feature':([cos1,cos2,cos3])})
                    # {'reward':reward[i],'feature':(f0[i])})
                    # {'reward':reward[i],'feature':(np.sort(f0[i])[-1:])})
                    {'reward':reward[i],'feature':(cos1)})
    # sio.savemat("dqnprw.mat",{'reward':reward,'feature':f0})
        # else:
            # reward.append(0)
        # if f1[i] < f2[i] and f2[i] < f3[i]:
            # reward.append(2)
        # else:
            # reward.append(0)
    return reward
def generatelabelfordqnfromlargetosmall(f0,f1,f2,f3, l1,l2,l3, label,names):
    reward=[]
    import os
    # f0 = (f0-np.min(f0))/(np.max(f0)-np.min(f0))
    for i in range(0,len(label)):
        # print (label)
        j = label[i]
        # print (i,j)
        print (f1.shape)
        print (l1.shape)
        # print ((f1[i][j]).shape)
        # print ((f0[i][j]))
        print ((f0[i][j]), np.max(f0[i]),np.argmax(f0[i]))
        print ((f1[i][j]), np.max(f1[i]),np.argmax(f1[i]))
        print ((f2[i][j]),np.max(f2[i]),np.argmax(f2[i]))
        print ((f3[i][j]),np.max(f3[i]),np.argmax(f3[i]))
        # print ((f4[i][j]),np.max(f4[i]),np.argmax(f3[i]))
        # print ()
        # if f1[i][j] < f2[i][j] and f2[i][j] < f3[i][j]:
            # reward.append(2)
        # elif f1[i][j] < f2[i][j]:
            # reward.append(1)
        # else:
            # reward.append(0)
        # if f0[i][j] < f2[i][j] or f0[i][j]< f3[i][j]: #and f2[i][j] < f3[i][j]:
        if np.max([f0[i][j], f2[i][j],f1[i][j],f3[i][j]])>f0[i][j] or np.max([l1[i][j],l2[i][j],l3[i][j],f0[i][j]])>f0[i][j]:
            if np.max([l1[i][j],l2[i][j],l3[i][j],f0[i][j]])>f0[i][j]:
                reward.append(1)
            if np.max([f0[i][j], f2[i][j],f1[i][j],f3[i][j]])>f0[i][j]:
                reward.append(2)
        else:
            reward.append(0)
        # print (reward)
        # outputpath = os.path.join(outputdir)
        print (names[i])
        print (label[i])
        id = str(label[i])
        cos1 = cosine(f2[i],f1[i])
        cos2 = cosine(f3[i],f2[i])
        cos3 = cosine(f1[i],f3[i])

        name = names[i].split("/")[-1].split(".")[0]
        print (name)
        # if name=='/home/songwenfeng/swf/pytorch//380/c2s1_104621.jpg'
        if not os.path.exists(os.path.join("dqndataset",id)):
            os.makedirs(os.path.join("dqndataset",id))
        # sio.savemat(os.path.join("dqndataset",id,name+".mat"),\
                    # {'reward':reward[i],'feature':f0[i]})
        sio.savemat(os.path.join("dqndataset",id,name+".mat"),\
                    # {'reward':reward[i],'feature':([cos1,cos2,cos3])})
                    # {'reward':reward[i],'feature':(f0[i])})
                    # {'reward':reward[i],'feature':(np.sort(f0[i])[-1:])})
                    {'reward':reward[i],'feature':(cos1)})
    # sio.savemat("dqnprw.mat",{'reward':reward,'feature':f0})
        # else:
            # reward.append(0)
        # if f1[i] < f2[i] and f2[i] < f3[i]:
            # reward.append(2)
        # else:
            # reward.append(0)
    return reward
def generatedata():
    # f1 = sio.loadmat("199dqnallprw6dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    # f2 = sio.loadmat("199dqnallprw5dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    # f3 = sio.loadmat("199daqnallprw4dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    # f0 = sio.loadmat("199dqnallprworidense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    f0 = sio.loadmat("199dqnprworidense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    f1 = sio.loadmat("199dqnprwlarge6dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    f2 =sio.loadmat("199dqnprwlarge5dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    f3 =sio.loadmat("199dqnprwlarge4dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    l1 = sio.loadmat("199dqnprwshrink6dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    l2 = sio.loadmat("199dqnprwshrink5dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    l3 = sio.loadmat("199dqnprwshrink4dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    # f1 = sio.loadmat("crossprw1_6oridqnsmall6pytorch_result.mat")
    # f2 = sio.loadmat("crossprw1_5oridqnsmall5pytorch_result.mat")
    # f3 = sio.loadmat("crossprw1_4oridqnsmall4pytorch_result")
    # f4 = sio.loadmat("crossprw1_2oridqnsmall2pytorch_result.mat")
    # # f0 = sio.loadmat("dqndatasetprwdense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    # f0 = sio.loadmat("crossprw_densenetftdqnoripytorch_result.mat")
    ff0 = f0['gallery_f']
    ff1 = f1['gallery_f']
    ff2 = f2['gallery_f']
    ff3 = f3['gallery_f']
    ll1 = l1['gallery_f']
    ll2 = l2['gallery_f']
    ll3 = l3['gallery_f']
    # ff4 = f4['gallery_f']
    label = f0['gallery_label']
    names = f0['gallery_names']
    # generatelabelfordqn(ff0,ff1,ff2,ff3, label[0], names)
    generatelabelfordqnfromlargetosmall(ff0,ff1,ff2,ff3,ll1,ll2,ll3, label[0], names)
    # generatelabelfordqntemporal(ff0,ff1,ff2,ff3,ll1,ll2,ll3, label[0], names)
    # label = f1['']
def generatedattime():
    # f1 = sio.loadmat("199dqnallprw6dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    # f2 = sio.loadmat("199dqnallprw5dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    # f3 = sio.loadmat("199daqnallprw4dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    # f0 = sio.loadmat("199dqnallprworidense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    # f0 = sio.loadmat("./199dqnprwtime50dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    # f1 = sio.loadmat("./199dqnprwtime20dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    # f2 =sio.loadmat("./199dqnprwtime100dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    # f3 =sio.loadmat("./199dqnprwtime1000dense121com0605flip_lowrank_prw_densenet_mbp_pytorch_result.mat")
    # f0 = sio.loadmat("./dqnpridprw1000densnent0805coco_personmasklowrank_market_densenet_mbp_pytorch_result.mat")
    # f1 = sio.loadmat("./dqnpridprw100densnent0805coco_personmasklowrank_market_densenet_mbp_pytorch_result.mat")
    # f2 = sio.loadmat("./dqnpridprw50densnent0805coco_personmasklowrank_market_densenet_mbp_pytorch_result.mat")
    # f3 = sio.loadmat("./dqnpridprw20densnent0805coco_personmasklowrank_market_densenet_mbp_pytorch_result.mat")
    ############################################prid
    # f0 = sio.loadmat("./dqn50time399priddensnentpersonlowrankpridf2densnet_pytorch_result.mat")
    # f1 = sio.loadmat("./dqn20time399priddensnentpersonlowrankpridf2densnet_pytorch_result.mat")
    # f2 = sio.loadmat("./dqn100time399priddensnentpersonlowrankpridf2densnet_pytorch_result.mat")
    # f3 = sio.loadmat("./dqn1000time399priddensnentpersonlowrankpridf2densnet_pytorch_result.mat")
    #############################################mars
    # f0 = sio.loadmat("./dqnduketime50datasetduke0807predukecoco_dense121personmasklowrank_mars_densenet_mbp_pytorch_result.mat")
    # f1 = sio.loadmat("./dqnduketime20datasetduke0807predukecoco_dense121personmasklowrank_mars_densenet_mbp_pytorch_result.mat")
    # f2 = sio.loadmat("./dqnduketime100timedatasetduke0807predukecoco_dense121personmasklowrank_mars_densenet_mbp_pytorch_result.mat")
    # f3 = sio.loadmat("./dqnduketime1000datasetduke0807predukecoco_dense121personmasklowrank_mars_densenet_mbp_pytorch_result.mat")
    # f0 = sio.loadmat("./dqnmarstime50precocomask_dense121_mars_densenet_mbp_pytorch_result.mat")
    # f1 = sio.loadmat("./dqnmarstime20precocomask_dense121_mars_densenet_mbp_pytorch_result.mat")
    # f2 = sio.loadmat("./dqnmarstime100precocomask_dense121_mars_densenet_mbp_pytorch_result.mat")
    # f3 = sio.loadmat("./dqnmarstime1000precocomask_dense121_mars_densenet_mbp_pytorch_result.mat")
    # f0 = sio.loadmat("./dqntime50vilidpersonlowrankvliddensnet_pytorch_result.mat")
    # f1 = sio.loadmat("./dqn100time20vilidpersonlowrankvliddensnet_pytorch_result.mat")
    # f2 = sio.loadmat("./dqnmarstime100vilidpersonlowrankvliddensnet_pytorch_result.mat")
    # f3 = sio.loadmat("./dqnmarstime1000vilidpersonlowrankvliddensnet_pytorch_result.mat")
    import sys
    f0 = sio.loadmat(sys.argv[1])
    f1 = sio.loadmat(sys.argv[2])
    f2 = sio.loadmat(sys.argv[3])
    f3 = sio.loadmat(sys.argv[4])
    ff0 = f0['gallery_f']
    ff1 = f1['gallery_f']
    ff2 = f2['gallery_f']
    ff3 = f3['gallery_f']
    # ff4 = f4['gallery_f']
    label = f0['gallery_label']
    names = f0['gallery_names']
    # generatelabelfordqn(ff0,ff1,ff2,ff3, label[0], names)
    # generatelabelfordqnfromlargetosmall(ff0,ff1,ff2,ff3,ll1,ll2,ll3, label[0], names)
    generatelabelfordqntemporal(ff0,ff1,ff2,ff3, label[0], names)

N_STATES =1024
N_STATES =3
N_ACTIONS = 3
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        # self.fc2 = nn.Linear(10, 10)
        # self.fc3 = nn.Linear(10, 10)
        # nn.init.kaiming_normal(self.fc1.weight.data)
        # self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(10, N_ACTIONS)
        # self.out.weight.data.normal_(0, 0.1)   # initialization
        # nn.init.kaiming_normal(self.out.weight.data)

    def forward(self, x):
        # print (x)
        x = self.fc1(x)
        # print (self.fc1.weight.data)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        x = F.dropout(x, training=self.training)
        # x = F.relu(x)
        actions_value = self.out(x)
        # print (actions_value)

        # return actions_value
        return F.log_softmax(actions_value, dim=1)
# class Net(nn.Module):
    # def __init__(self, ):
        # super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 100)
        # self.fc3 = nn.Linear(100, 3)
        # # super(Net, self).__init__()

        # # self.fc1 = nn.Linear(N_STATES, 10)
        # # # self.fc2 = nn.Linear(10, 10)
        # # # self.fc3 = nn.Linear(10, 10)
        # # # nn.init.kaiming_normal(self.fc1.weight.data)
        # # # self.fc1.weight.data.normal_(0, 0.1)   # initialization
        # # self.out = nn.Linear(10, N_ACTIONS)
        # # # self.out.weight.data.normal_(0, 0.1)   # initialization
        # # # nn.init.kaiming_normal(self.out.weight.data)
    # def forward(self, x):
        # x = self.conv1(x)
        # x = F.max_pool2d(x, 2)
        # x = F.relu(x)
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # x = self.fc3(x)

        # return F.log_softmax(x, dim=1)

    # def forward(self, x):
        # # print (x)
        # x = self.fc1(x)
        # # print (self.fc1.weight.data)
        # x = F.relu(x)
        # # x = self.fc2(x)
        # # x = F.relu(x)
        # # x = self.fc3(x)
        # x = F.dropout(x, training=self.training)
        # # x = F.relu(x)
        # actions_value = self.out(x)
        # print (actions_value)

        # # return actions_value
        # return F.log_softmax(actions_value, dim=1)
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print (data)
        # print (target)
        optimizer.zero_grad()
        output = model(data)
        # print (output)
        # print (len(data))
        # print (len(train_loader.dataset))
        # pred = log_softmax(output, dim=1)
        # print (output.size())
        # print (target.size())
        # print (pred)
        loss = F.nll_loss(output, target)
        # loss = F.log_softmax(output, target)
        # loss = F.CrossEntropyLoss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print (output)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # print (pred)
            # print (pred, target)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
def generatelabelfordqntemporal(f50,f20,f100,f1000, label,names):
    reward=[]
    import os
    f0=f50
    f1=f20
    f2=f100
    f3=f1000
    # f0 = (f0-np.min(f0))/(np.max(f0)-np.min(f0))
    for i in range(0,len(label)):
        # print (label)
        # j = label[i]
        # print (j)
        # print (i,j)
        # print ((f1[i][j]).shape)
        # print ((f0[i][j]))
        j=np.argmax(f0[i])
        print (np.array_equal(f0,f1))
        print ((f0[i][j]), np.max(f0[i]),np.argmax(f0[i]))
        print ((f1[i][j]), np.max(f1[i]),np.argmax(f1[i]))
        print ((f2[i][j]),np.max(f2[i]),np.argmax(f2[i]))
        print ((f3[i][j]),np.max(f3[i]),np.argmax(f3[i]))
        # print ((f4[i][j]),np.max(f4[i]),np.argmax(f3[i]))
        # print ()
        # if f1[i][j] < f2[i][j] and f2[i][j] < f3[i][j]:
            # reward.append(2)
        # elif f1[i][j] < f2[i][j]:
            # reward.append(1)
        # else:
            # reward.append(0)
        # if f0[i][j] < f2[i][j] or f0[i][j]< f3[i][j]: #and f2[i][j] < f3[i][j]:
        if np.max([f0[i][j], f2[i][j],f3[i][j]])>f0[i][j] or np.max([f0[i][j],f1[i][j]])>f0[i][j]:
            if np.max([f1[i][j],f0[i][j]])>f0[i][j]:
                reward.append(1)
            if np.max([f0[i][j], f2[i][j],f3[i][j]])>f0[i][j]:
                reward.append(2)
        else:
            reward.append(0)
        # print (reward)
        # outputpath = os.path.join(outputdir)
        print (names[i])
        print (label[i])
        id = str(label[i])
        cos1 = cosine(f2[i],f1[i])
        cos2 = cosine(f3[i],f2[i])
        cos3 = cosine(f1[i],f3[i])

        name = names[i].split("/")[-1].split(".")[0]
        print (name)
        # if name=='/home/songwenfeng/swf/pytorch//380/c2s1_104621.jpg'
        if not os.path.exists(os.path.join(sys.argv[5],id)):
            os.makedirs(os.path.join(sys.argv[5],id))
        # sio.savemat(os.path.join("dqndataset",id,name+".mat"),\
                    # {'reward':reward[i],'feature':f0[i]})
        print (os.path.join(sys.argv[5],id,name+".mat"))
        sio.savemat(os.path.join(sys.argv[5],id,name+".mat"),\
                    # {'reward':reward[i],'feature':([cos1,cos2,cos3])})
                    # {'reward':reward[i],'feature':(f0[i])})
                    # {'reward':reward[i],'feature':(np.sort(f0[i])[-1:])})
                    {'reward':reward[i],'feature':(cos1)})
    return reward
def dqn():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = 0
    train_loader = torch.utils.data.DataLoader(
    datasets('./dqndataset'),
    batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets('./dqndataset'),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


    # dqn=Net()
    # model.train()
    # for p in range(0,epoch):

if __name__=="__main__":
        # generatedata()
        generatedattime()
        # dqn()
