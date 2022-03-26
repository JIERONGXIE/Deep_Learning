# 加载数据

----

## Dataset

- Dataset是一个包装类，用来将数据包装为Dataset类，然后传入DataLoader中，我们再使用DataLoader这个类来更加快捷的对数据进行操作



-----

## DataLoader

- DataLoader是一个比较重要的类，它为我们提供的常用操作有：batch_size（每个batch的大小），shuffle（是否进行shuffle操作），num_workers（加载数据的时候使用几个子进程）



-------

# 可视化

-------

## 曲线图

``` python
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter('logs')
for i in range(100):
    writer.add_scalar('y=3x',i*3,i)
writer.close()
```

``` shell
终端运行
>>> tensorboard --logdir=logs //参数--port6067
```

-----

## 图片

``` python
from torch.utils.tensorboard import SummaryWriter
from cv2 import cv2 as cv
img=cv.imread('data//9.jpg',-1)
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
writer=SummaryWriter('logs')
writer.add_image('test',img,1,dataformats='HWC')
#writer.add_image('test',img,2,dataformats='HWC')
writer.close()
```

- 图片需要类型numpy.array

----

# torchvision.transfroms

----

## tansfroms.ToTensor()

- 将图片格式改为<class 'torch.Tensor'>

``` python
img_path='data\\'
img=cv.imread(img_path+'9.jpg',-1)
transfer=transforms.ToTensor()#实例化
tensor=transfer(img)
print(type(tensor))
```

运行结果

``` python
<class 'torch.Tensor'>
```

$$
tool=transforms.ToTensor()
\\result=tool(input)
$$

---

# torchvision的数据集使用

-----

``` python
import torchvision
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter('logs')
dataset_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set=torchvision.datasets.CIFAR10(root='./dataset',train=True,download=False,transform=dataset_transform)
test_set=torchvision.datasets.CIFAR10(root='./dataset',train=False,download=False,transform=dataset_transform)
for i in range(10):
    img,target=test_set[i]
    writer.add_image('test_set',img,i)
writer.close()
```

-----

# DataLoader的使用

----

``` python
import torchvision
from torch.utils.data import DataLoader
writer=SummaryWriter('logs')
test_set=torchvision.datasets.CIFAR10(root='./dataset',train=False,download=False,transform=torchvision.transforms.ToTensor())
test_loader=DataLoader(dataset=test_set,batch_size=4,shuffle=True,num_workers=0,drop_last=False)
step=0
for data in test_loader:
    imgs,targets=data
    writer.add_images('test_loader',imgs,step)
    step+=1
writer.close()
```

---

# nn.Module

---

## 池化层

``` python
class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output=self.maxpool1(input)
        return output

input=torch.tensor([
    [1,2,0,3,1],
    [0,1,2,3,1],
    [1,2,1,0,0],
    [5,2,3,1,1],
    [2,1,0,1,1]],dtype=torch.float32)
net=network()
input=torch.reshape(input,(-1,1,5,5))
output=net(input)
print(output)
```

----

# 激活函数

---

``` python
class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()
        self.relu1=ReLU()

    def forward(self,input):
        output=self.relu1(input)
        return output

net=network()
a=torch.tensor([[1,-0.5],[-1,5]])
a=torch.reshape(a,(-1,1,2,2))
print(a.shape)
output=net(a)
print(output)
```

---

# 网络可视化

---

```` python
class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()
        # self.conv1=Conv2d(3,32,5,padding=2)
        # self.maxpool1=MaxPool2d(2)
        # self.conv2=Conv2d(32,32,5,padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten=Flatten()
        # self.linear1=Linear(1024,64)
        # self.linear2=Linear(64,10)
        self.model=Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )
    def forward(self,x):
        x=self.model(x)
        return x

net=network()
x=torch.ones([64,3,32,32])
out=net(x)
print(out.shape)
writer=SummaryWriter('logs',)
writer.add_graph(net,x)
writer.close()
````

---

# 损失函数和反向传播

---

## 损失函数

``` python
input=torch.tensor([1.,2.,3.])
target=torch.tensor([1.,2.,5.])
input=torch.reshape(input,(1,1,1,-1))
target=torch.reshape(target,(1,1,1,-1))
loss=MSELoss()
outut=loss(input,target)
print(outut)

input=torch.tensor([1.,2.,3.])
target=torch.tensor([1.,2.,5.])
input=torch.reshape(input,(1,1,1,-1))
target=torch.reshape(target,(1,1,1,-1))
loss=MSELoss()
outut=loss(input,target)
print(outut)
```

---

## 反向传播

``` python
dataset=datasets.CIFAR10('./data',train=False,transform=torchvision.transforms.ToTensor(),download=False)
dataloader=DataLoader(dataset,batch_size=64)

class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()
        # self.conv1=Conv2d(3,32,5,padding=2)
        # self.maxpool1=MaxPool2d(2)
        # self.conv2=Conv2d(32,32,5,padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten=Flatten()
        # self.linear1=Linear(1024,64)
        # self.linear2=Linear(64,10)
        self.model=Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )
    def forward(self,x):
        x=self.model(x)
        return x

loss=CrossEntropyLoss()
net=network()
optimizer=torch.optim.SGD(net.parameters(),lr=0.01,)
for epoch in range(20):
    running_loss=0.00
    for data in dataloader:
        imgs,targets=data
        output=net(imgs)
        result_loss=loss(output,targets)
        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()
        running_loss+=result_loss
    print(running_loss)
```

----

# 网络模型的使用及修改

---

``` python
vgg16_false=torchvision.models.vgg16(pretrained=False)
vgg16_true=torchvision.models.vgg16(pretrained=True)
#print(vgg16_true)
dataset=datasets.CIFAR10('./data',train=True,transform=torchvision.transforms.ToTensor(),download=True)
# vgg16_true.add_module('add_linear',nn.Linear(1000,10))
vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))
vgg16_false.classifier[6]=nn.Linear(4096,10)
print(vgg16_false)
```

---

# 网络模型的保存和加载

---

``` python
#mothd1
vgg16=torchvision.models.vgg16(pretrained=False)
torch.save(vgg16,'vgg16.pth')
model=torch.load('vgg16.pth')
#method2
vgg16=torchvision.models.vgg16(pretrained=False)
torch.save(vgg16.state_dict(),'vgg16.pth')
vgg16.load_state_dict(torch.load('vgg16.pth'))
```

----

# 完整训练套路

``` python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()
        self.model=Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
        )
    def forward(self,x):
        x=self.model(x)
        return x

writer=SummaryWriter('logs')#可视化

train_data=torchvision.datasets.CIFAR10('./data',train=True,transform=torchvision.transforms.ToTensor(),download=True)#train_dataset_size=50000
test_data=torchvision.datasets.CIFAR10('./data',train=False,transform=torchvision.transforms.ToTensor(),download=True)#test_dataset_size=10000
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

train_dataset_size=len(train_data)
test_dataset_size=len(test_data)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net=network()#网络模型实例化
net=net.to(device)

loss_fn=nn.CrossEntropyLoss()#损失函数实例化
loss_fn=loss_fn.to(device)

learning_rate=1e-2#学习率
optimizer=torch.optim.SGD(net.parameters(),lr=learning_rate)#优化器实例化

total_train_step=0
total_test_step=0

epoch=30
for i in range(epoch):
    print('-------第{}轮-------'.format(i+1))
    total_train_loss=0
    for data in train_dataloader:
        imgs,targets=data
        imgs=imgs.to(device)
        targets=targets.to(device)
        output=net(imgs)
        loss=loss_fn(output,targets)
        total_train_loss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step+=1
        if total_train_step%100==0:
            writer.add_scalar('tain',loss.item(),total_train_step)
        #print('损失',loss)
    print('total_train_loss=',total_train_loss)
    score=0
    accuracy=0
    total_test_loss=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = net(imgs)
            loss = loss_fn(output, targets)
            total_test_loss+=loss
            total_test_step+=1
            score+=(output.argmax(1)==targets).sum()
        accuracy=score/test_dataset_size
        writer.add_scalar('test',loss.item(),total_test_step)
        torch.save('net{}.pth'.format(i))
        print('total_test_loss=',total_test_loss)
        print('accuracy=',accuracy)
```

----

# GPU训练

----
- 网络模型，数据，损失函数调用cuda()

---

# 验证模型套路

----

``` python
img=PIL.Image.open('dog.jpg')
transforms=torchvision.transforms.Compose(torchvision.transforms.Resize((32,32))
                                          ,torchvision.transforms.ToTensor())
img=transform(img)
print(img.shape)
net=torch.load('net8.pth',map_location='cpu')
img=torch.reshape(img,(1,3,32,32))
net.eval()
with torch.no_grad():
    output=net(img)
print(output.argmax(1))
```

------

# 参数管理

----

``` python
from torch import nn

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.module=nn.Sequential(
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,16),
            nn.ReLU(),
        )
    def forward(self,x):
        x=self.module(x)
        return x

net=network()

print(net.module[2].state_dict)
print(net.module[2].state_dict())
```

``` python
<bound method Module.state_dict of Linear(in_features=8, out_features=16, bias=True)>
OrderedDict([('weight', tensor([[-0.3529, -0.1329, -0.2600,  0.1637, -0.0089, -0.2689, -0.0511, -0.2948],
        [ 0.0925, -0.1803, -0.0122,  0.2237,  0.2870,  0.1468, -0.1578, -0.1722],
        [-0.1379,  0.1538, -0.1783, -0.0800, -0.2576, -0.0558,  0.0835, -0.0762],
        [-0.1803,  0.2355, -0.0416, -0.1508,  0.3035, -0.0661,  0.2818, -0.2936],
        [ 0.0608, -0.3311, -0.1519, -0.1157, -0.3264, -0.3000,  0.3025,  0.3006],
        [-0.0900,  0.1071,  0.2498,  0.0836, -0.1884, -0.2895, -0.1821,  0.2641],
        [-0.3044,  0.1846,  0.0444, -0.1614, -0.2014, -0.3049, -0.0868,  0.0452],
        [-0.3379,  0.2790, -0.1209,  0.1119,  0.2173,  0.2162, -0.0185, -0.3372],
        [-0.0984,  0.1159,  0.2904,  0.0063, -0.1140,  0.1263,  0.1275,  0.3013],
        [ 0.1807, -0.0556,  0.3514, -0.1634, -0.0911, -0.2577,  0.1476,  0.2593],
        [-0.1305,  0.0367, -0.2321, -0.2577, -0.0848,  0.2483, -0.2710, -0.1040],
        [ 0.2743,  0.2557,  0.3122, -0.2824, -0.0219, -0.3312,  0.2687, -0.1857],
        [ 0.2776,  0.0518,  0.2161, -0.1943, -0.1237, -0.0102,  0.2867, -0.3390],
        [ 0.3174, -0.2804, -0.2126,  0.2606, -0.1736,  0.1919,  0.2462,  0.2483],
        [-0.2422, -0.0554, -0.0761, -0.0438, -0.0509, -0.1571,  0.3256,  0.2545],
        [-0.2636,  0.2625,  0.2437, -0.1118,  0.0617, -0.3165,  0.1629, -0.1305]])), ('bias', tensor([-0.0291, -0.3192,  0.0111,  0.1115,  0.1402, -0.1091,  0.0517,  0.3357,
         0.3350,  0.1946,  0.3117,  0.1946,  0.1079,  0.0868, -0.2312,  0.3209]))])
```

---

``` python
print(type(net.module[2].bias))
print(net.module[2].bias)
print(net.module[2].bias.data)
```

``` python
<class 'torch.nn.parameter.Parameter'>
Parameter containing:
tensor([ 0.0210, -0.2681, -0.0876,  0.1300, -0.3055,  0.0776, -0.0842, -0.0226,
         0.1469, -0.0279,  0.2842,  0.2559, -0.2186,  0.3160, -0.1442, -0.0847],
       requires_grad=True)
tensor([ 0.0210, -0.2681, -0.0876,  0.1300, -0.3055,  0.0776, -0.0842, -0.0226,
         0.1469, -0.0279,  0.2842,  0.2559, -0.2186,  0.3160, -0.1442, -0.0847])
```

----

``` python
print(net.module[2].weight)
print(net.module[2].weight.grad)
```

``` python
tensor([[ 0.3079, -0.2649, -0.2269, -0.0626,  0.3057, -0.3025, -0.2201, -0.1881],
        [-0.3348, -0.0102, -0.1623,  0.2584, -0.0542,  0.3191,  0.0883,  0.3202],
        [ 0.2986,  0.0963,  0.1117,  0.3445,  0.1583, -0.2509,  0.1283, -0.0747],
        [ 0.2763, -0.2718, -0.3467,  0.2441,  0.2624,  0.2099, -0.3229, -0.3488],
        [ 0.1470,  0.2945, -0.2675,  0.1413, -0.3322,  0.2621,  0.1899, -0.2079],
        [-0.1773,  0.2896, -0.1904, -0.0685, -0.1185,  0.1237,  0.1072, -0.0019],
        [ 0.2755, -0.0768, -0.1566,  0.3039,  0.3058,  0.0117,  0.0640,  0.0386],
        [-0.2277,  0.0832, -0.0609, -0.1996,  0.3392, -0.0028,  0.1056,  0.2739],
        [ 0.2456,  0.0140,  0.1224,  0.1816, -0.1690, -0.1534, -0.1919, -0.1932],
        [-0.2774,  0.0188, -0.1033,  0.1644, -0.0802, -0.1111, -0.2431,  0.3264],
        [ 0.2486,  0.0255, -0.1517, -0.2644,  0.2220,  0.0325, -0.1046, -0.2644],
        [ 0.1634,  0.3090, -0.1353, -0.0991, -0.1370,  0.0774, -0.3256, -0.1291],
        [ 0.1195,  0.1356, -0.3188,  0.2309,  0.2786, -0.1956,  0.3359,  0.0503],
        [ 0.0253, -0.1135, -0.0529, -0.1604, -0.0904,  0.3466, -0.1651,  0.1861],
        [ 0.0215,  0.1457, -0.1645,  0.0347, -0.1618,  0.0726,  0.1869, -0.2557],
        [-0.1433, -0.1218, -0.2849, -0.0572,  0.1411, -0.1803, -0.2320,  0.1749]],
       requires_grad=True)
None
```

----

``` python
print(*[(name,param.shape) for name,param in net.module[2].named_parameters()])
print(*[(name,param.shape) for name,param in net.module.named_parameters()])
print(net.module.state_dict()['2.bias'].data)
```

``` python
('weight', torch.Size([16, 8])) ('bias', torch.Size([16]))
('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([16, 8])) ('2.bias', torch.Size([16]))
tensor([-0.2855, -0.1636,  0.0395, -0.0978, -0.1423, -0.1550,  0.3042, -0.0420,
        -0.2189, -0.3404,  0.1700, -0.0380,  0.3362,  0.2234, -0.3341,  0.0944])
```

----

# 内置初始化

----

``` python
def init_normal(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,mean=0,std=0.01)
        #nn.init.constant_(m.weight,1)
        #nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

net.module.apply(init_normal)
print(net.module[0].weight.data)
```

``` python
tensor([[-0.0075,  0.0113,  0.0016,  0.0054],
        [-0.0102, -0.0010,  0.0032, -0.0056],
        [-0.0210, -0.0172,  0.0041,  0.0152],
        [ 0.0085,  0.0036, -0.0051,  0.0021],
        [-0.0012, -0.0037, -0.0175, -0.0074],
        [ 0.0040, -0.0089, -0.0008,  0.0027],
        [-0.0016,  0.0310,  0.0208,  0.0095],
        [ 0.0059,  0.0126,  0.0131,  0.0010]])
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
tensor([[ 0.2422,  0.0390,  0.6793,  0.6293],
        [ 0.3500, -0.3499,  0.1394,  0.1150],
        [ 0.3454, -0.4659,  0.4386, -0.6473],
        [ 0.6560,  0.0680, -0.1065, -0.5436],
        [ 0.5346, -0.2171,  0.5872,  0.5257],
        [ 0.5366,  0.5652, -0.6058, -0.5469],
        [ 0.0804, -0.0031,  0.2893, -0.4242],
        [-0.6345,  0.4592, -0.4775, -0.2077]])
```

---

# 自定义层

-----

- 没有参数的层

``` python
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()

    def forward(self,x):
        return x-x.mean()
```

- 带参数的图层

``` python
class MyLinear(nn.Module):
    def __init__(self,input,output):
        super(MyLinear, self).__init__()
        self.weight=nn.Parameter(torch.randn(input,output))
        self.bias=nn.Parameter(torch.zeros(output))

    def forward(self,x):
        linear=torch.matmul(x,self.weight.data)+self.bias.data
        return nn.ReLU(linear)
```

------

## 层输入

---

``` python
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.module=nn.Sequential(
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Flatten()
        )
    def forward(self,x):
        x=self.module(x)
        return x

net=network()
x=torch.randn(3,3,3,4)
print(net(x).shape)
```

``` python
torch.Size([3, 144])
```

-----

# LeNet

----

``` python
Conv2d(1,6,5,padding=2),
ReLU(),
nn.AvgPool2d(2,stride=2),

Conv2d(6,16,5,padding=0),
ReLU(),
nn.AvgPool2d(2,stride=2),

Flatten(),

Linear(16*5*5,120),
ReLU(),

Linear(120,84),
ReLU(),

Linear(84,10),
```

----

# AlexNet

------

```` python
nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),
nn.ReLU(),
nn.MaxPool2d(kernel_size=3,stride=2),

nn.Conv2d(96,256,kernel_size=5,padding=2),
nn.ReLU(),
nn.MaxPool2d(kernel_size=3,stride=2),

nn.Conv2d(256,384,kernel_size=3,padding=1),
nn.ReLU(),
nn.Conv2d(384,384,kernel_size=3,padding=1),
nn.ReLU(),
nn.Conv2d(384,256,kernel_size=3,padding=1),
nn.ReLU(),
nn.MaxPool2d(kernel_size=3,stride=2),

nn.Flatten(),

nn.Linear(6400,4096),
nn.ReLU(),
nn.Dropout(p=0.5),

nn.Linear(4096,4096),
nn.ReLU(),
nn.Dropout(p=0.5),

nn.Linear(4096,10)
````

----

# VGG

------

- VGG11
- VGG13
- VGG16
- VGG19

---

## VGG11

``` python
from torch.nn import Conv2d, Linear, Flatten, Sequential, MaxPool2d, ReLU, Dropout

class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()
        self.features=nn.Sequential(#5次池化层操作
            
            Conv2d(3, 64, kernel_size=3,padding=1),
            ReLU(),
            MaxPool2d(2),
            
            Conv2d(64, 128, kernel_size=3,padding=1),
            ReLU(),
            MaxPool2d(2),
            
            Conv2d(128, 256, kernel_size=(3, 3),padding=1),
            ReLU(),
            Conv2d(256, 256, kernel_size=3,padding=1),
            ReLU(),
            MaxPool2d(2),
            
            Conv2d(256, 512, kernel_size=3,padding=1),
            ReLU(),
            Conv2d(512, 512, kernel_size=3,padding=(1, 1)),
            ReLU(),
            MaxPool2d(2),
            
            Conv2d(512, 512, kernel_size=3,padding=1),
            ReLU(),
            Conv2d(512, 512, kernel_size=3,padding=1),
            ReLU(),
            MaxPool2d(2),
        )
        self.avgpool:=AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier=nn.Sequential(
            
            Linear(25088,4096),
            ReLU(),
            Dropout(0.5),
            
            Linear(4096,4096),
            ReLU(),
            Dropout(0.5),
            
            Linear(4096,1000),
        )

    def forward(self,x):
        x=self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x=self.classifier(x)
        return x
```

---

# NiN（网络中的网络）

-----

``` python
class NiN_Network(nn.Module):
    def __init__(self):
        super(NiN_Network, self).__init__()

        def nin_block(input,output,kernel_size,strides,padding):
            return Sequential(
                
            Conv2d(input,output,kernel_size,strides,padding),
            ReLU(),
            Conv2d(output,output,1,strides,padding),
            ReLU(),
            Conv2d(output,output,1,strides,padding),
            ReLU(),
            )

        self.module=Sequential(
            
            nin_block(1,96,kernel_size=11,strides=4,padding=0),
            MaxPool2d(3,stride=2),
            
            nin_block(96,256,kernel_size=5,strides=1,padding=2),
            MaxPool2d(3,stride=2),
            
            nin_block(256,384,kernel_size=3,strides=1,padding=1),
            MaxPool2d(3,stride=2),
            
            Dropout(0.5),
            
            nin_block(384,10,kernel_size=3,strides=1,padding=1),#[batch_size,channel,H,W]，channel是10
            
            AdaptiveAvgPool2d((1,1)),#[batch_size,10,1,1]
            
            Flatten(),#打平后10个通道代表10个类别
        )
    
    def forward(self,x):
        x=self.module(x)
        return x
```

-----

## 一些总结

1、全连接层很占内存 

2、卷积核越大越占内存 

3、层数越多越占内存 

4、模型越占内存越难训练  

5、得出一个结论：多用1*1、3*3 卷积、 AdaptiveAvgPool2d替代全连接，既可以加快速度，又可以达到与全连接、大卷积核一样的效果。还有一个规律，就是图像尺寸减半，同时通道数指数增长，可以很好地保留特征。

----

# GoogLeNet（含并行连接的网络）

----

``` python
class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):#除c1外,c2,c3,c4都有两层
        super(Inception, self).__init__(**kwargs)

        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):

        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))

        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))#高宽减半

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))#高宽减半

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))#高宽减半

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))#高宽减半

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),#通道数当类别数
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```

----

# 批量归一化

-----

``` python
nn.Linear(256, 120), nn.BatchNorm1d(120),
```

------

# ResNet

----

``` python
```

----









