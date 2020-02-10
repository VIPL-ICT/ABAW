import torch
import torchvision
import torchvision.datasets
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import torch.utils.data as Data
import os
import torch
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image
import pdb
import random
from torch.utils.data.dataset import Dataset
from torchvision.models import resnet

tfilepath = 'EXPR_Set/Training_Set'
tfiles=os.listdir(tfilepath)
tfiles=sorted(tfiles)
tvideo=['null']
tframe=['null']
tvideolength=['null']
tlabel=['null']
tframelabel=['null']
tnumframe=['null']
tnumvideo=len(tfiles)
for tfile in tfiles:
    tmpfile=open('EXPR_Set/Training_Set/'+tfile)
    tmplines = tmpfile.readlines()
    tmplength=len(tmplines)-2
    tvideolength.append(tmplength)
    tmplabel=['null']
    tmpvideo=['null']
    tmpframe=['null']
    tmpframelabel=['null']
    tmpnumframe=0
    for i in range(1,tmplength+1):
        tmplabel.append(int(tmplines[i]))
        if int(tmplines[i])>=0 and os.path.exists('face_imgs/'+tfile[:-4]+'/'+"%05d"%i+'.jpg')==True:
            tmpvideo.append('face_imgs/'+tfile[:-4]+'/'+"%05d"%i+'.jpg')
            tmpnumframe+=1
            tmpframe.append('face_imgs/'+tfile[:-4]+'/'+"%05d"%i+'.jpg')
            tmpframelabel.append(int(tmplines[i]))
        else:
            tmpvideo.append('null')
    tvideo.append(tmpvideo)
    tframe.append(tmpframe)
    tframelabel.append(tmpframelabel)
    tlabel.append(tmplabel)
    tnumframe.append(tmpnumframe)
    if(tmpnumframe==0):
        print("NO!",tfile)


vfilepath = 'EXPR_Set/Validation_Set'
vfiles=os.listdir(vfilepath)
vfiles=sorted(vfiles)
vvideo=['null']
vframe=['null']
vvideolength=['null']
vlabel=['null']
vframelabel=['null']
vnumframe=['null']
vnumvideo=len(vfiles)
vset=[]
vsetlabel=[]
for vfile in vfiles:
    tmpfile=open('EXPR_Set/Validation_Set/'+vfile)
    tmplines = tmpfile.readlines()
    tmplength=len(tmplines)-2
    vvideolength.append(tmplength)
    tmplabel=['null']
    tmpvideo=['null']
    tmpframe=['null']
    tmpframelabel=['null']
    tmpnumframe=0
    for i in range(1,tmplength+1):
        tmplabel.append(int(tmplines[i]))
        if int(tmplines[i])>=0 and os.path.exists('face_imgs/'+vfile[:-4]+'/'+"%05d"%i+'.jpg')==True:
            tmpvideo.append('face_imgs/'+vfile[:-4]+'/'+"%05d"%i+'.jpg')
            tmpnumframe+=1
            tmpframe.append('face_imgs/'+vfile[:-4]+'/'+"%05d"%i+'.jpg')
            tmpframelabel.append(int(tmplines[i]))
        else:
            tmpvideo.append('null')
    vvideo.append(tmpvideo)
    vframe.append(tmpframe)
    vframelabel.append(tmpframelabel)
    vlabel.append(tmplabel)
    vnumframe.append(tmpnumframe)
    if(tmpnumframe<9):
        print("NO! ",vfile)

for i in range(1,vnumvideo+1):  
    for j in range(vnumframe[i]//8):
        tmpset=[]
        tmplabel=[]
        for k in range(1,9):
            tmpset.append(vframe[i][j*8+k])
            tmplabel.append(vframelabel[i][j*8+k])
        vset.append(tmpset)
        vsetlabel.append(tmplabel)
    last=vnumframe[i]%8
    if last!=0:
        tmpset=[]
        tmplabel=[]
        for j in range(1,last+1):
            tmpset.append(vframe[i][vnumframe[i]//8*8+j])
            tmplabel.append(vframelabel[i][vnumframe[i]//8*8+j])
        for j in range(last+1,9):
            tmpset.append(vframe[i][vnumframe[i]//8*8+last])
            tmplabel.append(-1)
        vset.append(tmpset)
        vsetlabel.append(tmplabel)
data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
to_tensor=transforms.Compose([transforms.ToTensor()])
class TrainDataset(Dataset):
    def __init__(self,  transform=None):
        self.length=1000

    def __getitem__(self, index):
        video=np.zeros(( 8, 3,224, 224), dtype=np.float32)
        videonum=random.randint(1,tnumvideo)
        left_frame = random.randint(1,tnumframe[videonum]-7)
        while int(tframe[videonum][left_frame+7][-9:-4])-int(tframe[videonum][left_frame][-9:-4])>12:
            left_frame = random.randint(1,tnumframe[videonum]-7)
        correct_label = tframelabel[videonum][left_frame:(left_frame+8)]
        for idx in range(8):
            img=Image.open(tframe[videonum][left_frame+idx])
            img = img.convert('RGB')
            img = img.resize((224, 224), Image.ANTIALIAS)
            img = data_transform(img)
            img=img.numpy()
            video[ idx, :,:,:] = img
        correct_label=np.array(correct_label)
        return video, correct_label

    def __len__(self):
        return self.length
    
train_dataset=TrainDataset()
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = 4,shuffle = False,num_workers = 2)
class ValidDataset(Dataset):
    def __init__(self,  transform=None):
        self.length=len(vset)

    def __getitem__(self, index):
        video=np.zeros(( 8, 3,224, 224), dtype=np.float32)
        for idx in range(8):
            img=Image.open(vset[index][idx])
            img = img.convert('RGB')
            img = img.resize((224, 224), Image.ANTIALIAS)
            img = data_transform(img)
            img=img.numpy()
            video[ idx, :,:,:] = img
        correct_label=np.array(vsetlabel[index])
        return video, correct_label

    def __len__(self):
        return self.length
    
valid_dataset=ValidDataset()
valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size = 2,shuffle = True,num_workers = 2)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM(planes, 16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.cbam(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM( planes * 4, 16 )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.cbam(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,**kwargs)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet=resnet101()
        self.rnn = nn.LSTM(
            input_size=2048,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out = nn.Linear(128,64)
        self.out2 = nn.Linear(64, 7)
        
    def forward(self, x):
        x=x.view(-1,3,224,224)
        x=self.resnet(x)
        x=x.view(-1,8,2048)
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out)
        out = self.out2(out)
        out = out.view(-1,7)
        return out

net=Net()
net_state_dict=net.state_dict()
save=torch.load("pretrain")
weight={k:v for k,v in save.items() if k in net_state_dict}
net_state_dict.update(weight)
net.load_state_dict(net_state_dict)
net=net.cuda()
net.train()
cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr = 0.0001,momentum = 0.9)
for epoch in range(30000):
    running_loss = 0.0
    net.train()
    for i,data in enumerate(train_loader,0):
        inputs,labels = data
        inputs=inputs.cuda()
        labels=labels.cuda()
        labels=labels.view(32)
        inputs,labels = Variable(inputs),Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = cirterion(outputs,labels)
        loss.backward()
        optimizer.step()
    net.eval()
    total=0
    correct=0
    cnt=0
    conmat = [[0 for i in range(8)] for j in range(8)]
    for data in valid_loader:
        cnt+=1
        if cnt==501:
            break;
        inputs,labels = data
        inputs=inputs.cuda()
        labels=labels.cuda()
        labels=labels.view(16)
        inputs,labels = Variable(inputs),Variable(labels)
        outputs = net(inputs)    
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        for i in range(16):
            conmat[labels[i]][predicted[i]]+=1
    print(correct,total,'%d %%' % (100 * correct / total))
    f1=0
    for expr in range(7):
        total_expr=0
        total_pred=0
        for pred in range(7):
            total_expr+=conmat[expr][pred]
            total_pred+=conmat[pred][expr]
        if total_pred>0:
            p1=conmat[expr][expr]/total_pred
        else:
            p1=0
        if total_expr>0:
            p2=conmat[expr][expr]/total_expr
        else:
            p2=0
        if p1==0 and p2==0:
            pass
        else:
            f1+=2*p1*p2/(p1+p2)/7
    for i in range(7):
        for j in range(7):
            print(conmat[i][j], end=' ')
        print()
    score=0.67*f1+0.33*float(correct) / float(total)
    torch.save(net.state_dict(),"checkpoint/"+str(epoch)+" "+str(correct)+" "+str(total)+" "+str(1000 * correct / total)+" "+str(f1*1000)+" "+str(score*1000))
print('finished training!')


