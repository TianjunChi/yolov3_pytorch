'''
Descripttion: 
version: 
Author: congsir
Date: 2023-09-11 10:58:36
LastEditors: Please set LastEditors
LastEditTime: 2023-09-11 11:19:56
'''
from kmeans import kmeans, avg_iou
CLUSTERS = 9
ANNOTATIONS_PATH = "E:\yolov3_pytorch\\annotation\data_parallel\labels\\train"
import numpy as np
# a function to read all the txt files under the input directory and put the all the txt files in one txt file
def load_dataset(dir):
    # get all the txt files under the input directory
    import os
    file_list = os.listdir(dir)
    
    dataset = []
    # read all the txt files and put them in the output file
    for file in file_list:
        if file.endswith('.txt'):
            for line in open(dir + '/' + file):
                # get the path of the current txt file f
                # get file path of all the files under the current dir
                anno = line.split()
                dataset.append([float(anno[3]),float(anno[4])])
    dataset = np.array(dataset)
    return dataset


data = load_dataset(ANNOTATIONS_PATH)
out = kmeans(data, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}".format(out))
print("Boxes:\n {}".format(out*416))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))

"""
[7,14]
[8,16]
[10,27]
[12,21]
[16,17]
[17,34]
[17,22]
[20,18]
[22,20]
"""