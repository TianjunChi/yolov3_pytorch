'''
Descripttion: 
version: 
Author: congsir
Date: 2023-09-26 13:41:45
LastEditors: Please set LastEditors
LastEditTime: 2023-09-27 11:15:14
'''
# mAP
import numpy as np
# read the txt file and get the ground truth and predicted bounding box
def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        bboxes = []
        classes = []
        for line in lines:
            line = line.strip().split()
            bboxes.append([float(line[1]), float(line[2])/2, float(line[3]), float(line[4])/2])
            break
            # classes.append(int(line[0]))
        return np.array(bboxes)

def read_pre(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        bboxes = []
        classes = []
        for line in lines:
            line = line.strip().split()
            bboxes.append([float(line[2]), float(line[3]), float(line[4]), float(line[5])])
            break
            # classes.append(int(line[0]))
        return np.array(bboxes)


# calculate the iou of two bounding box
def bboxes_iou(boxes1, boxes2):
    # print(boxes1, boxes2)
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

# traverse the txt file under a certain dir and give the file name
def read_txt_name():
    import os
    txt_name = []
    for root, dirs, files in os.walk('E:\yolov3_pytorch\\mAP\\ground-truth'):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                txt_name.append(os.path.splitext(file)[0])
    return txt_name

txt_file = read_txt_name()
score = []
count = 0
TP = 0
FP = 0
FN = 0
IOU_thres = 0.5
for f in txt_file:
    gt_boxes = read_txt('E:\yolov3_pytorch\\mAP\\ground-truth\\'+f+'.txt')
    pre_boxes = read_pre('E:\yolov3_pytorch\\mAP\\predicted\\'+f+'.txt')
    if gt_boxes.size<=0 or pre_boxes.size<=0:
        print("no result for this picture: "+str(f)+".txt")
        FN += 1
        continue
    tmp_score = bboxes_iou(gt_boxes, pre_boxes)
    if tmp_score > IOU_thres:
        TP += 1
    else:
        FP += 1
    """
    if tmp_score < 0.1:
        count += 1
        print(f+".txt IOU<0.1")
        continue

    """
    score.append(tmp_score)

    # print(gt_boxes, pre_boxes)
score = np.array(score).mean()
print("score: ",score)
# get a mean of the array 
#print("count: ",count)
print("TP: ",TP," FP: ",FP," FN: ",FN)
print("prec"+str(IOU_thres),float(TP)/float((TP+FP)))
print("recall"+str(IOU_thres),float(TP)/float((TP+FN)))


# a function to calculate the precision and recall rate of a certain class and finally calculate the average precision
