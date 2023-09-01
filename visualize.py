'''
Descripttion: 
version: 
Author: congsir
Date: 2023-08-30 14:02:27
LastEditors: Please set LastEditors
LastEditTime: 2023-08-31 14:15:04
'''
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
# a function to load an image from a given file and plot a bounding box on it
def draw(image_path):
    #image = np.expand_dims(image, axis=0)
    #image = torch.from_numpy(image)
    #image = image.permute(0, 3, 1, 2)
    #plt.imshow(image)
    #plt.show()
    #image = image.cuda()
    #bboxes_pr = detect_image(image, _decode, input_shape)
    #image = draw_bbox(image, bboxes_pr, classes, show_label=True)


    # a function to draw given bounding box on image
    def draw_bbox(image, bboxes, show_label=True):
        """
        image: a 3-D numpy array of shape [H, W, 3].
        bboxes: a numpy array of shape [N, 5].
        classes: a list of class names.
        """

        image_h, image_w,_ = image.shape
        print(image_w,image_h)
        """
        bboxes[0][0] = bboxes[0][0] * 320/416
        bboxes[0][1] = bboxes[0][1] * 160/416 
        bboxes[0][2] = bboxes[0][2] * 320/416 
        bboxes[0][3] = bboxes[0][3] * 160/416
        """

        for i, bbox in enumerate(bboxes):
            """
            for j in range(0,4):
                bbox[j] = bbox[j] * 310
            """
            coor = np.array(bbox[:4], dtype=np.int32)
            fontScale = 0.5
            #score = bbox[4]
            class_ind = int(bbox[4])
            bbox_color = (0, 255, 0)
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            print(c1,c2)
            aa = cv2.rectangle(image, c1, c2,bbox_color,  bbox_thick)
            cv2.imshow('this',aa)
            """
            if show_label:
                bbox_mess = '%s' % ('person')
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])),bbox_color, -1)
            """

    # a function to draw given bounding box on image
    def draw_bbox_2(image, bboxes, show_label=True):
        """
        image: a 3-D numpy array of shape [H, W, 3].
        bboxes: a numpy array of shape [N, 5].
        classes: a list of class names.
        """

        image_h, image_w,_ = image.shape
        print(image_w,image_h)
        """
        bboxes[0][0] = bboxes[0][0] * 320/416
        bboxes[0][1] = bboxes[0][1] * 320/416 
        bboxes[0][2] = bboxes[0][2] * 320/416 
        bboxes[0][3] = bboxes[0][3] * 320/416 
        """

        for i, bbox in enumerate(bboxes):
            """
            for j in range(0,4):
                bbox[j] = bbox[j] * 310
            """
            coor = np.array(bbox[:4], dtype=np.int32)
            fontScale = 0.5
            #score = bbox[4]
            class_ind = int(bbox[4])
            bbox_color = (0, 255, 255)
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            print(c1,c2)
            aa = cv2.rectangle(image, c1, c2,bbox_color,  bbox_thick)
            cv2.imshow('this',aa)

    sizes = [(320,320)]# ,(310,310)
    for size in sizes:
        image = cv2.imread(image_path)
        print("image.shape:" ,image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size)
        #image = np.array(image, dtype=np.float32)
        #image = image / 255.                                                                
        draw_bbox_2(image,[[138.45 ,80.53 ,149.94 ,103.58,0]], show_label=True)
        draw_bbox(image,[[132, 84, 144 ,106,0]], show_label=True)
        plt.imshow(image)
        plt.show()

if __name__ == '__main__':
    #draw("E:\yolov3_pytorch\\annotation\data_parallel\images\\train\exp2_154.png")
    draw("E:\yolov3_pytorch\\annotation\data_parallel\images\\val\exp3_110.png")