'''
Descripttion: 
version: 
Author: congsir
Date: 2023-08-30 17:26:43
LastEditors: 
LastEditTime: 2023-08-30 17:30:21
'''
import glob
import xml.etree.ElementTree as ET
 
import numpy as np
 
from kmeans import kmeans, avg_iou
 
ANNOTATIONS_PATH = "Annotations"
CLUSTERS = 9
 
 
def load_dataset(path):
    dataset = []
    # read all the txt files under the given dir and add them to the dataset
    for txt_file in glob.glob("{}/*txt".format(path)):
        

    return np.array(dataset)
 
if __name__ == '__main__':
  #print(__file__)
  data = load_dataset(ANNOTATIONS_PATH)
  out = kmeans(data, k=CLUSTERS)
  #clusters = [[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]
  #out= np.array(clusters)/416.0
  print(out)
  print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
  print("Boxes:\n {}-{}".format(out[:, 0]*416, out[:, 1]*416))
 
  ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
  print("Ratios:\n {}".format(sorted(ratios)))