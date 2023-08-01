## yolov3_pytorch

### 文件目录

model\
----model.py\
----decoder.py\
utils\
----loss.py\
----decode.py\
train.py\
train_paral.py\
eval.py

### 对应文件功能：
- model.model.py ：Darknet和内部的类定义
- model.decoder.py ：Decoder，用于对训练好的网络做评估
- utils.loss.py：用于定义训练的损失
- utils.decode.py：对Darknet的输出计算预测的bbox
- train.py：训练代码
- eval.py：用于对训练好的模型评估

### 训练
1. Example of single process training using train.py:\
python train.py --pattern 0 --batchsize 6 --pretrain_path xxx
训练的每个epoch的日志文件和权重会保存。\
pattern = 0表示从头开始训练；1表示导入现有权重训练。

2. Example of training with multi-process training using train_paral.py:\


### 推断
python demo.py

### 验证（暂时未调试过是否有效）
先用eval.py对pytorch模型评估生成文件，跑完这个脚本后需要再跑mAP/main.py进行mAP的计算。

### 注解文件的格式如下：
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14 
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
# make sure that x_max < width and y_max < height

### 参考仓库与博客链接：
1.https://github.com/miemie2013/Pytorch-DIOU-YOLOv3/blob/master/eval.py
2.https://github.com/ultralytics/yolov3/blob/master/models/yolo.py \
3.https://github.com/eriklindernoren/PyTorch-YOLOv3 \
4.从0到1，pytorch实现YOLOv3 https://blog.csdn.net/qq_25737169/article/details/80530579 \
5.【论文解读】Yolo三部曲解读——Yolov3 https://zhuanlan.zhihu.com/p/76802514 \
6.YOLO系列算法(v3v4)损失函数详解 https://blog.csdn.net/qq_27311165/article/details/107008610
