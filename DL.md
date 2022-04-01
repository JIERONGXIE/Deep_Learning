# Deep Learning

---

## 自动求导

---

``` python
x=torch.arange(4.0,requires_grad=True)
#等价于
x=torch.arange(4.0)
x.requires_grad=True
#tensor([0., 1., 2., 3.], requires_grad=True)
-----------------------------------------------------------------------------------------------------
y=2*torch.dot(x,x)
y
#tensor(28., grad_fn=<MulBackward0>)
-----------------------------------------------------------------------------------------------------
y.backward()#向后传播
x.grad
#tensor([ 0.,  4.,  8., 12.])
-----------------------------------------------------------------------------------------------------
x.grad==4*x
#tensor([True, True, True, True])
-----------------------------------------------------------------------------------------------------
x.grad.zero_()
#tensor([0., 0., 0., 0.])
y=x.sum()
y.backward()
x.grad
tensor([1., 1., 1., 1.])
-----------------------------------------------------------------------------------------------------
x.grad.zero_()
y=x*x
u=y.detach()
z=u*x
z.sum().backward()
x.grad
#tensor([0., 1., 4., 9.])
x.grad==u
#tensor([True, True, True, True])
-----------------------------------------------------------------------------------------------------
x.grad.zero_()
y.sum().backward()
x.grad==2*x
#tensor([True, True, True, True])
-----------------------------------------------------------------------------------------------------
def f(a):
    b=2*a
    while b.norm()<1000:
        b=b*2
    if b.sum()>0:
        c=b
    else:
        c=100*b
    return c
a=torch.randn(size=(),requires_grad=True)
d=f(a)
d.backward()
a.grad
#tensor(2048.)
a.grad==d/a
#tensor(True)
```

---

## 线性回归

----

### 线性模型

- n维输入：

$$
x=[x_1,x_2,x_3,...,x_n]^T
$$

- 一个n维权重和一个标量偏差

$$
w=[w_1,w_2,w_3,...,w_n]^T,b
$$

- 输出是输入的加权和

$$
y=w_1x_1+w_2x_2+w_3x_3+...+w_nx_n+b
$$

- 向量版本

$$
y=<w,x>+b
$$

---

``` python
from torch import  nn
net=nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

---

## Softmax回归

---

- 输出经过Softmax

$$
y_{predict}=\frac{e^{o_i}}{\sum_{k}e^{o_{k}}}\\最终输出一个[p^1,p^2,p^3,...p^k]\\分K个类别对应输出向量中的K个值，其和为1，最大值的索引就是预测类别
$$

----

## 多层感知机（MLP，Multilayer Perceptron）

---

### 单层感知机

-----

$$
O=\sigma(<w,x>+b)\\\sigma(x)=\begin{cases}1&x>0\\-1&otherwise\end{cases}
$$

---

- 对比
  - 感知机
    - 只能二分类，-1，1（离散值）
  - 回归
    - 输出实数
  - Softmax
    - 输出概率

----

- 训练感知机

initialize	w=0 and b=0
repeat
	if 
$$
y_i[<w_i,x_i>+b]\leq0\\then\\w<-w+y_ix_i\\b<-b+y_i
$$
end if

until all classified correctly

- 等价于使用批量大小为1的梯度下降，并使用如下的损失函数

$$
\varphi(y,x,w)=max(0,y<w,x>)
$$

----

- 收敛

  - 数据在半径r内

  - 余量ρ分类两类

    - $$
      y(x^Tw+b)\geq\rho\\对于\quad||w||^2+b^2\leq1
      $$

  - 感知机保证在（r^2+1）/（ρ^2）布后收敛

---

- 缺点：单层感知机无法拟合XOR问题，只能产生线性分割面

----

### 多层感知机

----

- 单隐藏层——单分类

$$
输入层：X∈R^n（没有计算）\\
隐藏层：W_1∈R^{m*n},b_1∈R^m（m是指多少个隐藏单元）\\
输出层：W_2∈R^m,b_2∈R
$$

- 隐藏层加激活函数

$$
h=\sigma(w_1x+b_1)\\o=w_2h+b_2
$$

---

- 多类分类

$$
输入层：X∈R^n（没有计算）\\
隐藏层：W_1∈R^{m*n},b_1∈R^m（m是指多少个隐藏单元）\\
输出层：W_2∈R^{m*k},b_2∈R^k
$$

$$
h=\sigma(w_1x+b_1)\\o=w_2h+b_2\\y=Softmax(o)
$$

----

- 多隐藏层

$$
h_1=\sigma(w_1x+b_1)\\h_2=\sigma(w_2h_1+b_2)\\h_3=\sigma(w_3h_2+b_3)\\o=w_4h_3+b_4
$$

- 超参数
  - 隐藏层层数
  - 每层隐藏层的大小

---

![](image\多层感知机.png)

---

## 权重衰退（Weight Decay）

---

- 使用均方范数作为硬性限制

  - 通过限制参数的选择范围来控制模型容量

    - $$
      min\quad\varphi(w,b)\\subject\quad to\\||w||^2\leq \theta
      $$

      

  - 通常不限制偏移b（作用不大）
  - 小的θ意味更强的正则项

![](image\权重衰退.png)

----

![](image\参数更新法则.png)

---

## 丢弃法（Drop Out）

---

![](image\丢弃法动机.png)

![](image\丢弃法.png)

![](image\使用丢弃法.png)

![](image\推理中的丢弃法.png)

![](image\丢弃法总结.png)

------

## 数值稳定性

----

![](image\数值稳定性_神经网络的tidu.png)

![](image\数值稳定性的两个常见问题.png)

![](image\例如MLP.png)

![](image\梯度爆炸.png)

![](image\梯度爆炸的问题.png)

![](image\梯度消失1.png)

![](image\梯度消失2.png)

![](image\梯度消失3.png)

![](image\让训练更加稳定.png)

![](image\让每层的方差是一个常数.png)

![](image\权重初始化.png)

![](image\例子MLP.png)

$$
方差D(X)=E(X^2)-E(X)^2\\E(X)是期望
$$
![](image\正向方差.png)

$$
Var(x)=E(x^2)-E(x)^2\\这里E(x)=0
$$
![](image\反向均值和方差.png)

![](image\Xavier初始.png)

![](image\假设线性的激活函数.png)

![](image\反向方差.png)

----

## 卷积层

----

![](image\重新考察全连接层.png)

![](image\平移不变性.png)

![](image\局部性.png)

![](image\二维卷积层.png)

![](image\卷积例子.png)

![](image\交叉相关和卷积.png)

![](image\一维和三维交叉相关.png)

![](image\卷积总结.png)

------

## 卷积计算

-----

$$
M=\frac{N-Conv+2Padding}{Strides}
$$

------

## LeNet

-----





-----

## AlexNet

----











-----

## NiN

----

![](image\NiN块.png)

- 1*1卷积核类似全连接层，只是输出是和输入维度相同，且有信息融合作用

![](image\NiN架构.png)

![](image\NiN_Networks.png)

![](image\NiN总结.png)

----

## GoogLeNet

-----

![](image\Inception块.png)

- 更少的参数个数和计算时间复杂度

![](image\GoogLeNet.png)

![](image\GoogLeNet_1&2.png)

![](image\GoogLeNet_3.png)

![](image\GoogLeNet_4&5.png)

![](image\GoogLNet_all.png)

---

## 批量归一化（Batch Normalization）

----

![](image\批量归一化绪论.png)

![](image\批量归一化做法.png)

![](image\批量归一化作用.png)

![](image\批量归一化的提出.png)

![](image\批量归一化总结.png)

-----

## ResNet

-------

![](image\残差块.png)

![](image\残差块细节.png)

------

## 深度学习硬件知识

----

![](image\cpu_1.png)

![](image\cpu_2.png)

![](image\cpuVSgpu.png)

![](image\提升GPU利用率.png)

![](image\高性能编程.png)

---

## 数据增广

----

![](image\数据增强.png)

![](image\使用增强数据训练.png)

![](image\翻转.png)

![](image\切割.png)

![](image\颜色.png)

![](image\其他方法.png)

------

## 锚框

----

![](image\锚框.png)

- 直接预测object的bbox是比较难的，因为范围太广了，所以变通一下，改为预测anchor到真实box的偏移量

![](image\IOU-交并比.png)

![](image\赋予锚框标号.png)

![](image\赋予锚框标号_1.png)

- 矩阵的每个元素都是一个锚框与一个边缘框的IOU

![](image\NMS.png)

------

## R-CNN

-----

![](image\R-CNN.png)

![](image\ROI池化层.png)

![](image\Fast RCNN.png)

![](image\Mask R-CNN.png)

![](image\物体检测识别总结.png)

----

## SSD

---

![](image\生成锚框.png)

![](image\SSD模型.png)













