# yolov3_pytorch

文件目录

model\
----model.py\
----decoder.py
utils\
----loss.py\
----decode.py

train.py\
eval.py


model.model.py ：Darknet和内部的类定义\
model.decoder.py ：Decoder，用于对训练好的网络做评估\
utils.loss.py：用于定义训练的损失\
utils.decode.py：对Darknet的输出计算预测的bbox\
train.py：训练代码\
eval.py：用于对训练好的模型评估\
