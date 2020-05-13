import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from  resnet_4d import resnet50, resnet152
# from  densenet4d import densenet121, densenet169, densenet201
from  densenet4dtwostarts import densenet121, densenet169, densenet201
# from import densenet121, densenet169, densenet201
from inceptionori import inception_v3
import torch.nn.functional as F
# from  densenet4d import densenet121

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        # print (x)
        x = self.add_block(x)
        x = self.classifier(x)
        return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num ):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num ):
        super().__init__()
        # model_ft = models.densenet121(pretrained=False)
        model_ft = densenet121(pretrained=False)
        # print (model_ft)
        # print (model_ft.state_dict().keys())
        # model_ft = densenet121(pretrained=True)
        # model_ft = densenet121(pretrained=False)
        # print (model_ft)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        # print (model_ft)
        self.classifier = ClassBlock(1024, class_num)

    def forward(self, x):
        # print (x)
        # print (self.model)
        x1 = self.model.features1(x[:,0:3,:,])
        x2 = self.model.features2(x[:,1:4,:,:])
        x3 = torch.cat((x1, x2), dim=1)
        x = self.model.features(x3)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x
class ft_net_dense169(nn.Module):

    def __init__(self, class_num ):
        super().__init__()
        # model_ft = models.densenet121(pretrained=False)
        model_ft = densenet169(pretrained=False)
        # print (model_ft)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = ClassBlock(1664, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x
class ft_net_dense201(nn.Module):

    def __init__(self, class_num ):
        super().__init__()
        # model_ft = models.densenet121(pretrained=False)
        model_ft = densenet201(pretrained=False)
        # print (model_ft)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = ClassBlock(1920, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x
# class ft_net_inceptionv3(nn.Module):
    # def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        # super(ft_net_inceptionv3, self).__init__()
        # self.aux_logits = aux_logits
        # self.transform_input = transform_input
        # self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        # self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        # self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        # self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        # self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        # self.Mixed_5b = InceptionA(192, pool_features=32)
        # self.Mixed_5c = InceptionA(256, pool_features=64)
        # self.Mixed_5d = InceptionA(288, pool_features=64)
        # self.Mixed_6a = InceptionB(288)
        # self.Mixed_6b = InceptionC(768, channels_7x7=128)
        # self.Mixed_6c = InceptionC(768, channels_7x7=160)
        # self.Mixed_6d = InceptionC(768, channels_7x7=160)
        # self.Mixed_6e = InceptionC(768, channels_7x7=192)
        # if aux_logits:
            # self.AuxLogits = InceptionAux(768, num_classes)
        # self.Mixed_7a = InceptionD(768)
        # self.Mixed_7b = InceptionE(1280)
        # self.Mixed_7c = InceptionE(2048)
        # self.fc = nn.Linear(2048, num_classes)

        # for m in self.modules():
            # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # import scipy.stats as stats
                # stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                # X = stats.truncnorm(-2, 2, scale=stddev)
                # values = torch.Tensor(X.rvs(m.weight.numel()))
                # values = values.view(m.weight.size())
                # m.weight.data.copy_(values)
            # elif isinstance(m, nn.BatchNorm2d):
                # nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)

    # def forward(self, x):
        # if self.transform_input:
            # x = x.clone()
            # x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            # x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            # x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # # 299 x 299 x 3
        # x = self.Conv2d_1a_3x3(x)
        # # 149 x 149 x 32
        # x = self.Conv2d_2a_3x3(x)
        # # 147 x 147 x 32
        # x = self.Conv2d_2b_3x3(x)
        # # 147 x 147 x 64
        # x = F.max_pool2d(x, kernel_size=3, stride=2)
        # # 73 x 73 x 64
        # x = self.Conv2d_3b_1x1(x)
        # # 73 x 73 x 80
        # x = self.Conv2d_4a_3x3(x)
        # # 71 x 71 x 192
        # x = F.max_pool2d(x, kernel_size=3, stride=2)
        # # 35 x 35 x 192
        # x = self.Mixed_5b(x)
        # # 35 x 35 x 256
        # x = self.Mixed_5c(x)
        # # 35 x 35 x 288
        # x = self.Mixed_5d(x)
        # # 35 x 35 x 288
        # x = self.Mixed_6a(x)
        # # 17 x 17 x 768
        # x = self.Mixed_6b(x)
        # # 17 x 17 x 768
        # x = self.Mixed_6c(x)
        # # 17 x 17 x 768
        # x = self.Mixed_6d(x)
        # # 17 x 17 x 768
        # x = self.Mixed_6e(x)
        # # 17 x 17 x 768
        # if self.training and self.aux_logits:
            # aux = self.AuxLogits(x)
        # # 17 x 17 x 768
        # x = self.Mixed_7a(x)
        # # 8 x 8 x 1280
        # x = self.Mixed_7b(x)
        # # 8 x 8 x 2048
        # x = self.Mixed_7c(x)
        # # 8 x 8 x 2048
        # x = F.avg_pool2d(x, kernel_size=8)
        # # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # # 1 x 1 x 2048
        # x = x.view(x.size(0), -1)
        # # 2048
        # x = self.fc(x)
        # # 1000 (num_classes)
        # if self.training and self.aux_logits:
            # return x, aux
        # return x



class ft_net_inceptionv3(nn.Module):
    def __init__(self, class_num ):
        super().__init__()
        # model_ft = models.densenet121(pretrained=False)
        model_ft = inception_v3(pretrained=False)
        # print (model_ft)
        # model_ft.Mixed_7c.avgpool = nn.AdaptiveAvgPool2d((8,8))
        model_ft.Mixed_7c.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()

        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = ClassBlock(2048, class_num)
        # self.classifier = ClassBlock(512, class_num)

    def forward(self, x):
        # print (x)
        # if self.transform_input:
            # x = x.clone()
            # x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            # x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            # x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            # x[:, 3] = x[:, 3] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.model.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.model.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.model.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.model.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.model.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.model.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.model.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.model.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.model.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6e(x)
        # 17 x 17 x 768
        # if self.training and self.aux_logits:
            # aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.model.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.model.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.model.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=1)
        # x = self.model.avgpool(x)
        # # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # # 1 x 1 x 2048
        # x = x.view(x.size(0), -1)
        # # 2048
        x = self.model.fc(x)
        # # 1000 (num_classes)
        # if self.training and self.aux_logits:
            # return x, aux
        # return x
        # x = self.model.Mixed_7c(x)
        # x = torch.squeeze(x)
        # x = self.classifier(x)
        return x


# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num ):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0,x1),1)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num ):
        super(PCB, self).__init__()

        self.part = 6 # We cut the pool5 to 6 parts
        # model_ft = models.resnet50(pretrained=True)
        model_ft = resnet50(pretrained=False)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, True, False, 256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y
class PCBdensenet(nn.Module):
    def __init__(self, class_num ):
        super(PCBdensenet, self).__init__()

        self.part = 1 # We cut the pool5 to 6 parts
        # model_ft = models.densenet121(pretrained=True)
        model_ft = densenet121(pretrained=False)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        #self.model.layer4[0].downsample[0].stride = (1,1)
        #self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            # setattr(self, name, ClassBlock(1024, class_num, True, False, 256))
            setattr(self, name, ClassBlock(1024, class_num))

    def forward(self, x):
        # x = self.model.conv1(x)
        # x = self.model.bn1(x)
        # x = self.model.relu(x)
        # x = self.model.maxpool(x)


        # x = self.model.layer1(x)
        # x = self.model.layer2(x)
        # x = self.model.layer3(x)
        # x = self.model.layer4(x)
        # x = self.avgpool(x)
        x = self.model.features(x)
        x = self.avgpool(x)
        # print (x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            # print ("==============",x)
            # print (">>>>>>>>>>>>>",x[:,:,i])
            part[i] = torch.squeeze(x[:,:,i])
            # print (part[i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            # print ("---------------",c)
            predict[i] = c(part[i])

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y

class PCB_test(nn.Module):
    def __init__(self,model):
        super(PCB_test,self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),x.size(2))
        return y

# debug model structure
#net = ft_net(751)
# net = ft_net_dense(751)
net = ft_net_inceptionv3(751)
# print(net)
input = Variable(torch.FloatTensor(8, 4, 224, 224))
output = net(input)
print('net output size:')
print(output.shape)
