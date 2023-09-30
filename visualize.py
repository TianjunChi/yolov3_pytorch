'''
Descripttion: 
version: 
Author: congsir
Date: 2023-08-30 14:02:27
LastEditors: Please set LastEditors
LastEditTime: 2023-09-27 19:11:26
'''
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
# a function to load an image from a given file and plot a bounding box on it
def draw(anno_dir,pre_dir,frame_num):
    #image = np.expand_dims(image, axis=0)
    #image = torch.from_numpy(image)
    #image = image.permute(0, 3, 1, 2)
    #plt.imshow(image)
    #plt.show()
    #image = image.cuda()
    #bboxes_pr = detect_image(image, _decode, input_shape)
    #image = draw_bbox(image, bboxes_pr, classes, show_label=True)

    # return all the txt files under a given directory
    import os 
    def get_all_txt_files(directory):
        txt_files = []
        for file in os.listdir(directory):
            if file.endswith(".txt"):
                txt_files.append(file)
        return txt_files

    def get_all_lines(file_path):
        lines = []
        with open(file_path) as f:
            lines = f.readlines()
        return lines

    gt_anno_list  = get_all_lines(anno_dir) # "E:\yolov3_pytorch\annotation\data_val.txt"
    print("gt_length: ",len(gt_anno_list))
    pre_anno_list = get_all_txt_files(pre_dir)
    # get a random number between 0 and n 
    import random
    def get_random_number(n):
        return random.randint(0, n)
     
    ### random mode
    # num = get_random_number(len(txt_list))

    num = frame_num

    # read a given txt file and get the first line of the file
    def read_first_line(path):
        with open(path) as f:
            lines = f.readlines()
            if len(lines) == 0:
                return False
        return lines[0]
    
    
    # join two paths using os
    
    # gt_path = gt_anno_list[num]
    # print('gt path: ',gt_path)
    pre_path = pre_anno_list[num]
    # print("gt_path: ",gt_path)
    # print("pre_path: ",pre_path)
    # gt_line = read_first_line(os.path.join(anno_dir,gt_path)).replace('\n','') # read the first line of the gt file
    pre_line = read_first_line(os.path.join(pre_dir,str(frame_num)+'.txt'))
    #pre_line = read_first_line(os.path.join(pre_dir,pre_path))
    pre_label = True # draw pre bbox 
    pre_anno = []
    if pre_line == False:
        print("pre_line is empty")
        pre_label = False # do not draw pre bbox
    else:
        pre_line = pre_line.replace('\n','')
        pre_anno = pre_line.split(' ') # a list of one annotation
    
    #print("pre bbox: ",pre_line)

    gt_line = gt_anno_list[num]
    gt_anno = gt_line.split(' ') # a list of one annotation
    #print("gt bbox: ",gt_line)

    # a function to draw given bounding box on image
    def draw_bbox(image, bboxes, offset = 1,show_label=True,valid=True): # gt 格式 = [class , x1,y1,x2,y2]-> offset = 1// pre格式 = [class , score , x1,y1,x2,y2]-> offset = 2
        # 对于GT,按照320*320的格式可视化， 对于Pre结果，按照320*160的格式可视化
        """
        image: a 3-D numpy array of shape [H, W, 3].
        bboxes: a numpy array of shape [N, 5].
        classes: a list of class names.
        return: an image with bounding boxes.
        """
        if valid == False:
            return False,False
        
        image_h, image_w,_ = image.shape
        # print(image_w,image_h)
        """
        bboxes[0][0] = bboxes[0][0] * 320/416
        bboxes[0][1] = bboxes[0][1] * 320/416 
        bboxes[0][2] = bboxes[0][2] * 320/416 
        bboxes[0][3] = bboxes[0][3] * 320/416 
        """     

        for i, bbox in enumerate(bboxes):
            print("bbox: ",bbox)
            
            coor = np.array((0.0,0.0,0.0,0.0), dtype=np.int32)
            if offset == 1: # gt result -> 1
                for j in range(0,4):
                    if j == 1 or j == 3: # 输出的size(320,160),gt size(320,320),所以要/2
                        coor[j] = math.floor(float(bbox[j+offset]))# /2)
                    else:
                        coor[j] = math.floor(float(bbox[j+offset]))
                coor[0] = coor[0]
                coor[1] = coor[1] *2
                coor[2] = coor[2]
                coor[3] = coor[3] *2
                """
                coor[0] = coor[0] - coor[2]/2 
                coor[1] = coor[1] - coor[3]/2
                coor[2] = coor[0] + coor[2]/2
                coor[3] = coor[1] + coor[3]/2 
                coor[1] = coor[1] / 2 
                coor[3] = coor[3] / 2 
                """
                print("gt rectangle: ",math.floor(coor[0]),math.floor(coor[1]),math.floor(coor[2]),math.floor(coor[3]))
            else: # predict result -> 2
                print("confidence score: ",bbox[1])
                conf_score = bbox[1]
                for j in range(0,4):
                    coor[j] = float(bbox[j+offset])
            # print("coor after resize: ",coor)            
                coor[0] = coor[0]
                coor[1] = coor[1] *4
                coor[2] = coor[2]
                coor[3] = coor[3] *4
                print("preditct rectangle: ",coor[0],coor[1],coor[2],coor[3])
            # coor = np.array(bbox[:4], dtype=np.int32)
            fontScale = 0.5
            #score = bbox[4]
            # class_ind = int(bbox[0])
            bbox_color = (0,0,0)
            if offset == 1: # gt result color: green
                bbox_color = (255,0,0)
            else: # prediction color: blue
                bbox_color = (255,255,255) 
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            print(c1,c2)
            output = cv2.rectangle(image, c1, c2,bbox_color,  bbox_thick) # top-left & bottom-right
            #return image,True
            return output,True
            cv2.imshow('this',aa)
            cv2.waitKey(100)  # Wait for 1 second (1000 milliseconds)
            # cv2.destroyAllWindows()


    sizes = [(320,320)]# ,(310,310)
    print('gt anno:',gt_anno)
    print('gt line: ',gt_line)
    image_path = os.path.join("E:\yolov3_pytorch\\annotation\data_parallel\images\\",gt_anno[0])
    # image_path = os.path.join(anno_dir,gt_path).replace('labels','images').replace('txt','png')
    print(image_path)

    for size in sizes:
        image = cv2.imread(image_path)
        print("image.shape:" ,image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size)
        #image = np.array(image, dtype=np.float32)
        #image = image / 255.                           
        gt_image,_ = draw_bbox(image,[gt_anno], 1,show_label=True)
        #"""
        pre_image,success= draw_bbox(image,[pre_anno], 2, show_label=True,valid = pre_label)
        print("success: ",success)
        if success == False:
            cv2.imshow('gt',gt_image)
            #cv2.imwrite('res_'+str(frame_num)+'.png',gt_image)
        else:
            print(gt_image.shape)
            cv2.imshow('gt',gt_image+pre_image)
            cv2.imwrite('single_head_output_float_320_160/single_head_'+str(frame_num)+'.png',gt_image+pre_image)
            # cv2.imshow('pre',pre_image)
        #"""
        #cv2.imshow('gt',gt_image)
        #cv2.imwrite('gt_bbox/'+str(frame_num)+'.png',gt_image)#+pre_image)
        cv2.waitKey(500)  # Wait for 1 second (1000 milliseconds)

        # draw_bbox(image,[[132, 84, 144 ,106,0]], show_label=True)
        #plt.imshow(image)
        #plt.show()

if __name__ == '__main__':
    # draw("E:\yolov3_pytorch\\annotation\data_parallel\images\\train\exp2_154.png")
    gt_path = "E:\yolov3_pytorch\\annotation\data_val.txt"
    pre_path = "E:\yolov3_pytorch\mAP\predicted\\"
    #for i in range(0,484):
    #conf_list = []

    #image_list = [104,105,108,109,110,111,117,142,143,159,223,228,229,230,234,236,237,238,239,244,249,329,332,334,338,346,347,348,351,352,439,448,456,457,470,80,99]
    for i in range(0,484):#506
        draw(gt_path,pre_path,i) # i 代表按照val.txt中的顺序（非顺序排列）的第i张图片 
    
    """
    print('conf_list: ',conf_list)
    res = np.array(conf_list,dtype = float).sum()/len(conf_list)
    print(res)
    """
"""
104.txt IOU<0.1
105.txt IOU<0.1
108.txt IOU<0.1
109.txt IOU<0.1
110.txt IOU<0.1
111.txt IOU<0.1
117.txt IOU<0.1
142.txt IOU<0.1
143.txt IOU<0.1
159.txt IOU<0.1
223.txt IOU<0.1
228.txt IOU<0.1
229.txt IOU<0.1
230.txt IOU<0.1
234.txt IOU<0.1
236.txt IOU<0.1
237.txt IOU<0.1
238.txt IOU<0.1
239.txt IOU<0.1
244.txt IOU<0.1
249.txt IOU<0.1
329.txt IOU<0.1
332.txt IOU<0.1
334.txt IOU<0.1
338.txt IOU<0.1
346.txt IOU<0.1
347.txt IOU<0.1
348.txt IOU<0.1
351.txt IOU<0.1
352.txt IOU<0.1
439.txt IOU<0.1
448.txt IOU<0.1
456.txt IOU<0.1
457.txt IOU<0.1
470.txt IOU<0.1
80.txt IOU<0.1
99.txt IOU<0.1
394.txt IOU<0.1
438.txt IOU<0.1
"""