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
