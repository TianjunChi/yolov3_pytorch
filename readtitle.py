'''
Descripttion: 
version: 
Author: congsir
Date: 2023-08-25 15:39:03
LastEditors: Please set LastEditors
LastEditTime: 2023-08-25 15:41:44
'''
import os 
# a function to read all the file name under a given directory
def read_all_file(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(file)
    print(file_list)
    return file_list

read_all_file("E:\数字人") 