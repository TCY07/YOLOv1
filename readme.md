# YOLOv1

## About

This is a very simple YOLOv1 project. It won't perform extremely well.

A trained model is provided as 'result.model' in 'result.zip'. You can use it for simple testing predict.

## Dependency

pytorch, torchvision, and cuda_tool_kit if you want to use cuda

## Train

### Dataset

Download voc2007 dataset from

http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

Remember to set your path

### Pre-trained model

* If you want to use an existing model (for example, my 'result.model' in result.zip), you can skip this step.

Download pre-trained resnet50 model from

https://download.pytorch.org/models/resnet50-19c8e357.pth

Set pre-trained model path in train.py

To start train your model from pre-trained resnet50, run command

>python train.py

To continue to train an existing model, run command

>python train.py --resume

To see args, run command

>python train.py -h

## Predict

Make sure you have a model. By default model's name is 'result.model'.

Run command

>python predict.py --image_name [filename]

This will generate a new image with boundingbox on it.

---

中文

## 关于本项目

个人学习用，容易上手，但是效果可能不是特别好。

提供了一个已经训练好的模型（在result.zip里面），可以用来进行简单的物体识别任务。

## 依赖的包

>pytorch, torchvision, 如果想使用GPU加速还需安装cuda_tool_kit

## 训练模型

### 下载数据集

直接从下面两个链接下载即可

http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

解压后会得到一个VOCdevkit，整个拿来用就行。图像在里面的JPEGImages文件夹下。

可能需要配置一下代码中的路径。

### 下载预训练模型

* 如果你已经有一个模型（例如我们提供的result.model），并想在其基础上继续训练，可以跳过这一步。

下载resnet50的预训练模型：

https://download.pytorch.org/models/resnet50-19c8e357.pth

在train.py里面设置你的预训练模型路径

从预训练模型开始进行训练：

>python train.py

继续训练已有的模型：

>python train.py --resume

查看可供调整的参数：

>python train.py -h

## 进行预测

需要已有一个模型才能进行预测。你可以把我们提供的result.model放在项目的目录下。

预测指令：

>python predict.py --image_name [filename]

[filename]是图片的路径。

预测完成后将生成一张新图片，图片上检测到的物体将被框出。
