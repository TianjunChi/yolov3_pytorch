'''
Descripttion: 
version: 
Author: congsir
Date: 2023-07-20 13:36:59
LastEditors: Please set LastEditors
LastEditTime: 2023-07-27 10:29:42
'''

import cv2
import os
import time

from model.decoder import Decode

import torch
import platform
sysstr = platform.system()
print(torch.cuda.is_available())
print(torch.__version__)
# 禁用cudnn就能解决Windows报错问题。Windows用户如果删掉之后不报CUDNN_STATUS_EXECUTION_FAILED，那就可以删掉。
if sysstr == 'Windows':
    torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    # file = 'data/voc_classes.txt'
    file = 'F:\yoloV3_pytorch\Pytorch-DIOU-YOLOv3-master/data/coco_classes.txt'

    model_path = 'F:\yoloV3_pytorch\yolo_bgr_mAP_47.pt' # 预训练好的网络权重
    # model_path = 'ep000360-loss2.108-val_loss2.296.pt'   # 改为训练好的模型的文件名


    # 选一个
    # input_shape = (320, 416)
    #input_shape = (416, 416)
    input_shape = (608, 608)

    _decode = Decode(0.3, 0.45, input_shape, model_path, file, initial_filters=32)

    # detect images in test floder.
    for (root, dirs, files) in os.walk('images/test'):
        if files:
            start = time.time()
            for f in files:
                path = os.path.join(root, f)
                image = cv2.imread(path)
                image = _decode.detect_image(image)
                cv2.imwrite('images/res/' + f, image)
                cv2.imshow('res', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            print('total time: {0:.6f}s'.format(time.time() - start))