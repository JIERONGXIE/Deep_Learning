# Anaconda

- 查看安装了哪些包

``` 
conda list
```

- 查看当前存在哪些虚拟环境

``` 
conda env list 
conda info -e
```

- 检查更新当前conda

``` 
conda update conda
```

- 创建虚拟环境

``` 
conda create -n test python=3.8
```

- 激活虚拟环境

``` 
Linux:	source activate test
Windows:	activate test
```

- 关闭虚拟环境（即从当前环境退出返回使用PATH环境中的默认python版本）

``` 
Linux:	source deactivate 
Windows:	deactivate
```

- 删除虚拟环境

``` 
conda remove -n your_env_name --all
```

- 删除环境钟的某个包

``` 
conda remove --name $your_env_name  $package_name 
```

----

# Label Image

-----

- 目标文件夹下运行cmd

- 进入YOLO环境

  - ``` shell
    activate yolo


``` shell
pip3 install PyQt5
pip3 install PyQt5_tools
pip3 install lxml
Pyrcc5 -o resources.py resources.qrc
```

- resources.py拷贝到同级的libs目录下
- 删除 labelImg-master \ data \ predefined_classes.txt
- python labelImg.py

<img src="readme_img\Snipaste_2022-04-01_17-47-57.png" style="zoom:60%;" />

- 打开图片目录后
  - 选择改变存放目录，选择存放labels的文件夹
  - 选择YOLO模式

- 注意查看label是否正确

------

# YOLO配置

-----

## coco128.yaml

- 复制 yolov5-master / data / coco128.yaml 到 my_dataset 中
- 改名为 my_dataset

<img src="G:\目标检测_YOLOV5\readme_img\Snipaste_2022-04-01_22-14-16.png" style="zoom:60%;" />

- 注释path
- train 和 val改成存放训练图片的文件夹
- nc改为目标检测的类别数
- names改成目标检测的类别名

----

## yolov5s.yaml

- 复制 yolov5-master / modules / yolov5s.yaml 到 my_dataset 中（以yolov5s为例，可选其他模型）
- 改名为 my_dataset_1

<img src="G:\目标检测_YOLOV5\readme_img\Snipaste_2022-04-01_22-27-28.png" style="zoom:60%;" />

- nc改为
- 目标检测的类别数

----

## 修改train.py

- 修改如下

![](G:\目标检测_YOLOV5\readme_img\Snipaste_2022-04-01_22-32-45.png)

![](G:\目标检测_YOLOV5\readme_img\Snipaste_2022-04-01_22-35-37.png)

![](G:\目标检测_YOLOV5\readme_img\Snipaste_2022-04-01_22-36-05.png)

- 训练结果在 yolov5-master / runs \ train \ exp \ weights中

-----

- 进入yolo虚拟环境进行训练

``` shell
activate yolo
python train.py
```

-----

## 修改detect.py

![](G:\目标检测_YOLOV5\readme_img\Snipaste_2022-04-01_22-43-21.png)

- weights是训练完成后的权重文件
- source是存放将要进行预测的文件夹

-----

``` shell
activate yolo
python detect.py
```

<img src="G:\目标检测_YOLOV5\readme_img\Snipaste_2022-04-01_22-45-35.png" style="zoom:60%;" />

- 在该文件夹下查看预测结果

------

# 训练模型导出

------

- 进入yolo虚拟环境

``` shell
python export.py --weights runs\train\exp9\weights\best.pt --img 640 --batch 1
```

-------

# OpenCV dnn模块

-----

## 模型和数据的加载

----

``` c++
//  加载 Darknet
Net readNetFromDarknet(const String &cfgFile, const String &darknetModel = String());
Net readNetFromDarknet(const std::vector<uchar>& bufferCfg, 
					   const std::vector<uchar>& bufferModel = std::vector<uchar>())
Net readNetFromDarknet(const char *bufferCfg, size_t lenCfg, 
					   const char *bufferModel = NULL, size_t lenModel = 0)

// 加载 Caffe
Net readNetFromCaffe(const String &prototxt, const String &caffeModel = String());
Net readNetFromCaffe(const std::vector<uchar>& bufferProto, 
					 const std::vector<uchar>& bufferModel = std::vector<uchar>())
Net readNetFromCaffe(const char *bufferProto, size_t lenProto,  
					 const char *bufferModel = NULL, size_t lenModel = 0)

// 加载TensorFlow
Net readNetFromTensorflow(const String &model, const String &config = String())
Net readNetFromTensorflow(const std::vector<uchar>& bufferModel, 
						  const std::vector<uchar>& bufferConfig = std::vector<uchar>())
Net readNetFromTensorflow(const char *bufferModel, size_t lenModel, 
						  const char *bufferConfig = NULL, size_t lenConfig = 0)

// 加载 Torch
Net readNetFromTorch(const String &model, bool isBinary = true);
Mat readTorchBlob(const String &filename, bool isBinary = true);

// 加载 Intel
Net readNetFromModelOptimizer(const String &xml, const String &bin);

// 加载 ONNX
Net readNetFromONNX(const String &onnxFile);
Mat readTensorFromONNX(const String& path);

// 通用 
Net readNet(const String& model, const String& config = "", 
			const String& framework = "")
Net readNet(const String& framework, const std::vector<uchar>& bufferModel,
 			const std::vector<uchar>& bufferConfig = std::vector<uchar>())
```

------

### cv2.dnn.readNet

``` python
cv2.dnn.readNet(model,  # 模型
                config = "", 
                framework = "" )
```

- model二进制训练好的网络权重文件，可能来自支持的网络框架，扩展名为如下：

- - *.caffemodel (Caffe,[http://caffe.berkeleyvision.org/](https://link.zhihu.com/?target=http%3A//caffe.berkeleyvision.org/))
  - *.pb (TensorFlow, [https://www.tensorflow.org/](https://link.zhihu.com/?target=https%3A//www.tensorflow.org/))
  - *.t7 | *.net (Torch, [http://torch.ch/](https://link.zhihu.com/?target=http%3A//torch.ch/))
  - *.weights (Darknet, [https://pjreddie.com/darknet/](https://link.zhihu.com/?target=https%3A//pjreddie.com/darknet/))
  - *.bin (DLDT, [https://software.intel.com/openvino-toolkit](https://link.zhihu.com/?target=https%3A//software.intel.com/openvino-toolkit))

- 

- config针对模型二进制的描述文件，不同的框架配置文件有不同扩展名：

- - *.prototxt (Caffe, [http://caffe.berkeleyvision.org/](https://link.zhihu.com/?target=http%3A//caffe.berkeleyvision.org/))
  - *.pbtxt (TensorFlow, [https://www.tensorflow.org/](https://link.zhihu.com/?target=https%3A//www.tensorflow.org/))
  - *.cfg (Darknet, [https://pjreddie.com/darknet/](https://link.zhihu.com/?target=https%3A//pjreddie.com/darknet/))
  - *.xml (DLDT, [https://software.intel.com/openvino-toolkit](https://link.zhihu.com/?target=https%3A//software.intel.com/openvino-toolkit))

- 

- framework显示声明参数，说明模型使用哪个框架训练出来的。

---

``` python
import cv2
import numpy as np

bin_model = "bvlc_googlenet.caffemodel"
protxt = "bvlc_googlenet.prototxt"

# load CNN model
net = cv2.dnn.readNet(bin_model, protxt)

# 获取各层信息
layer_names = net.getLayerNames()

for name in layer_names:
    id = net.getLayerId(name)
    layer = net.getLayer(id)
    print("layer id : %d, type : %s, name: %s"%(id, layer.type, layer.name))

print("successfully")
```

``` python
layer id : 1, type : Convolution, name: conv1/7x7_s2
layer id : 2, type : ReLU, name: conv1/relu_7x7
layer id : 3, type : Pooling, name: pool1/3x3_s2
layer id : 4, type : LRN, name: pool1/norm1
layer id : 5, type : Convolution, name: conv2/3x3_reduce
layer id : 6, type : ReLU, name: conv2/relu_3x3_reduce
layer id : 7, type : Convolution, name: conv2/3x3
layer id : 8, type : ReLU, name: conv2/relu_3x3
layer id : 9, type : LRN, name: conv2/norm2
layer id : 10, type : Pooling, name: pool2/3x3_s2
...
successfully
```

-----

### cv2.dnn.blobFromImage

``` python
cv2.dnn.blobFromImage(
                        image,
                        scalefactor = 1.0,
                        size = Size(),
                        mean = Scalar(),
                        swapRB = false,
                        crop = false,
                        ddepth = CV_32F 
)
```

- 网络模型支持的输入数据是四维的输入，所以要把读取到的Mat对象转换为四维张量

- 对图像进行预处理，包括减均值，比例缩放，裁剪，交换通道等，返回一个4通道的blob（blob可以简单理解为一个N维的数组，用于神经网络的输入）
- image:输入图像（1、3或者4通道）
- 可选参数
- scalefactor : 图像各通道数值的缩放比例，默认为1.0
- size : 输出图像的空间尺寸，如size=（200，300）表示高h=300，宽w=200
- mean : 用于各通道减去的值，以降低光照的影响（e.g. image为bgr3通道的图像，mean=[104.0，177.0，123.0]，表示b通道的值-104，g-177，r-123），b原通道均值 - b最终通道的均值 =104
- swapRB : 是否交换RB通道，默认为False。(cv2.imread读取的是彩图是bgr通道)
- crop : 是否图像裁剪，默认为False。当值为True时，先按比例缩放，然后从中心裁剪成size尺寸
- ddepth : 输出的图像深度，可选CV_32F 或者 CV_8U

```  python
blob = cv2.dnn.blobFromImage(im, 1 / 255.0,swapRB=True, crop=False)
```

----

``` python
import numpy as np
import cv2
#导入bgr彩色图像
img=cv2.imread('e:\\imagesfavour\\9.23.2.jpg')
#求原图各通道均值
mean_ori=[]
for i in range(3):
     mean_ori.append(np.mean(img[:,:,i]))
#各通道分别减去20，30，40
blob=cv2.dnn.blobFromImage(img,scalefactor=1,size=(300,300),mean=[20,30,40])
#求输出blob的各通道均值
mean_blob=[]
for i in range(3):
     mean_blob.append(np.mean(blob[0][i])) 
print('原图各通道均值：{}'.format(mean_ori))
print('输出blob各通道的均值:{}'.format(mean_blob))
print('原图的shape:{}'.format(img.shape))
print('输出blob的shape:{}'.format(blob.shape))
```

``` python
原图各通道均值：[110.10160243055556, 134.75311979166668, 158.2965668402778]
输出blob各通道的均值:[90.09298, 104.75085, 118.2897]
原图的shape:(1200, 1920, 3)
输出blob的shape:(1, 3, 300, 300)
```

-----

