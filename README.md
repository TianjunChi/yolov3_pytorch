# yolov3_pytorch

文件目录

model\
----model.py\
----decoder.py\
utils\
----loss.py\
----decode.py\
train.py\
eval.py

对应文件功能：
model.model.py ：Darknet和内部的类定义\
model.decoder.py ：Decoder，用于对训练好的网络做评估\
utils.loss.py：用于定义训练的损失\
utils.decode.py：对Darknet的输出计算预测的bbox\
train.py：训练代码\
eval.py：用于对训练好的模型评估

备注：\
网络定义格式和loss函数主要参考了https://github.com/miemie2013/Pytorch-DIOU-YOLOv3/ 的代码，并做出部分项目和代码结构的简化与调整。

正在进行...：\
1.可以把这部分代码作为模板，正在阅读loss函数代码和看看是否有必要修改简化loss.py中loss_layer的定义。\
2.简化train.py，把不属于我们任务场景的部分删去，并做测试。

参考仓库与博客链接：\
1.https://github.com/miemie2013/Pytorch-DIOU-YOLOv3/blob/master/eval.py\
2.https://github.com/ultralytics/yolov3/blob/master/models/yolo.py \
3.https://github.com/eriklindernoren/PyTorch-YOLOv3 \
4.从0到1，pytorch实现YOLOv3 https://blog.csdn.net/qq_25737169/article/details/80530579 \
5.【论文解读】Yolo三部曲解读——Yolov3 https://zhuanlan.zhihu.com/p/76802514 \
6.YOLO系列算法(v3v4)损失函数详解 https://blog.csdn.net/qq_27311165/article/details/107008610
