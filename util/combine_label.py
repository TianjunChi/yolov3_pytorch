'''
Descripttion: 
version: 
Author: congsir
Date: 2023-08-13 14:23:47
LastEditors: Please set LastEditors
LastEditTime: 2023-09-26 13:37:52
'''
# a function to read all the txt files under the input directory and put the all the txt files in one txt file
def combine_txt(dir):
    # get all the txt files under the input directory
    import os
    file_list = os.listdir(dir)
    # open the output file
    f = open('E:\yolov3_pytorch\\annotation\data_val.txt', 'w')
    # read all the txt files and put them in the output file
    for file in file_list:
        if file.endswith('.txt'):
            for line in open(dir + '/' + file):
                # get the path of the current txt file f
                # get file path of all the files under the current dir
                anno = line.split()
                # f.writelines('train' + '/' + str(file).replace('txt','png') + ' ' + ' '.join(anno[1:])+' '+anno[0]+ '\n')  
                f.writelines('val' + '/' + str(file).replace('txt','png') + ' ' # image path
                            + str(round((float(anno[1]) - 0.5*float(anno[3]))*320,2)) + ' ' # x1    
                            + str(round((float(anno[2]) - 0.5*float(anno[4]))*160,2)) + ' ' # y1    
                            + str(round((float(anno[1]) + 0.5*float(anno[3]))*320,2)) + ' ' # x1 + w
                            + str(round((float(anno[2]) + 0.5*float(anno[4]))*160,2)) + ' ' # y1 + h
                            +anno[0] # class 
                            + '\n') 
    f.close()

if __name__ == '__main__':
    combine_txt("E:\yolov3_pytorch\\annotation\data_parallel\labels\\val")
    print('finished')