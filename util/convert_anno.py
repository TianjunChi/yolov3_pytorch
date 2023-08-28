'''
Descripttion: 
version: 
Author: congsir
Date: 2023-07-07 14:02:58
LastEditors: Please set LastEditors
LastEditTime: 2023-07-28 15:49:17
'''
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm
import shutil
import os


def coco91_to_coco80_class():
    """Converts 91-index COCO class IDs to 80-index COCO class IDs.

    Returns:
        (list): A list of 91 class IDs where the index represents the 80-index class ID and the value is the
            corresponding 91-index class ID.

    """
    return [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None,
        None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
        None, 73, 74, 75, 76, 77, 78, 79, None]

def convert_coco(labels_dir='../coco/annotations/', use_segments=False, use_keypoints=False, cls91to80=True):

    def make_dirs_OSX(dir):
        dir = Path(dir)
        if os.path.exists(dir / 'labels'):#Path(dir / 'labels').exists():
            shutil.rmtree(dir/'labels')  # delete dir
        for p in dir, dir/'labels':
            p.mkdir(parents=True, exist_ok=True)  # make dir
        return dir
    
    save_dir = make_dirs_OSX('yolo_labels')  # output directory
    coco80 = coco91_to_coco80_class()

    def findAllFile(base):
        for root, ds, fs in os.walk(base):
            for f in fs:
                if f.endswith('.json'):
                    fullname = os.path.join(root, f)
                    yield fullname
    base = labels_dir
    for json_file in findAllFile(base):
        print(json_file)
        json_file = Path(json_file)
        "F:\OSX_train\annotation\ConductMusic\keypoint_annotation.json"
        fn = Path(save_dir) / 'labels'
        fn.mkdir(parents=True, exist_ok=True)
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {f'{x["id"]:d}': x for x in data['images']}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data['annotations']:
            imgToAnns[ann['image_id']].append(ann)
        count = 0
        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {json_file}'):
            count +=1
            img = images[f'{img_id:d}']

            h, w = img['height'], img['width']
            f = '_'.join(img['file_name'].split('/'))
            
            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                if ann['iscrowd']:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue
                
                cls = coco80[ann['category_id'] - 1] if cls91to80 else ann['category_id'] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                ## total length = 56+60+60 = 176
                # add body keypoints+bbox+category(1+4+17*3=56)
                k = np.zeros((1,1))
                if use_keypoints and ann.get('keypoints') is not None:
                    k = (np.array(ann['keypoints']).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                    kl = (np.array(ann['lefthand_kpts']).reshape(-1, 3)[1:] / np.array([w, h, 1])).reshape(-1).tolist()
                    kr = (np.array(ann['righthand_kpts']).reshape(-1, 3)[1:] / np.array([w, h, 1])).reshape(-1).tolist()
                    k = box + k + kl + kr
                    keypoints.append(k)

            # Write
            with open((fn /('ConductMusic_'+f)).with_suffix('.txt'), 'a') as file:
                for i in range(len(bboxes)):
                    #print("i= ",i)
                    if use_keypoints:
                        line = *(keypoints[i]),  # cls, box, keypoints
                        #print("length: ",len(line))
                        #print("line info: ", line)
                    else:
                        line = *(segments[i]
                                 if use_segments and len(segments[i]) > 0 else bboxes[i]),  # cls, box or segments
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')
        print('count: ',count)

if __name__ == "__main__":
    convert_coco(labels_dir='F:\OSX_train\\annotation', use_keypoints=True)
