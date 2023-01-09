# Author:NDK
# -*- coding:utf-8 -*-

import os
import os.path
from PIL import Image
import cv2
import numpy as np
import glob
'''
filein: 输入图片
fileout: 输出图片
width: 输出图片宽度
height:输出图片高度
type:输出图片类型（png, gif, jpeg...）
'''
def ResizeImage(filein, fileout):
  img = Image.open(filein)
  (x,y)=img.size
  x_s = int(x/10)
  y_s = int(y/10)
  out = img.resize((x_s, y_s),Image.ANTIALIAS) #resize image with high-quality
  out.save(fileout)
if __name__ == "__main__":
    root_path = "/home/syz/data/label_data/unlabeled_watch"    #操作文件路径
    print(root_path)
    # dir = root_path+"images"+"/"
    dir = root_path
    count = 0
    for root,dir,files in os.walk(dir):
        for file in files:
            filein = root_path+"/"+str(file)
            fileout = "/home/syz/data/label_data/img2/"+str(file)
            ResizeImage(filein, fileout)



#   filein = r'/home/syz/data/label_data/unlabeled_watch/000070.jpg'
#   fileout = r'/home/syz/data/label_data/img2/000070.jpg'
#   width = 200
#   height = 250
#   type = 'png'
#   ResizeImage(filein, fileout, width, height)


# print(root_path)
# # dir = root_path+"images"+"/"
# dir = root_path
# count = 0
# for root,dir,files in os.walk(dir):
#     for file in files:
#         srcImg = cv2.imread(root_path+"/"+str(file))
#         img_ = Image.open(root_path+"/"+str(file))
#         print(root_path+str(file))
#         (x, y) = img_.size
#         print(x,y)
#         x_s = int(x/10)
#         y_s = int(y/10)
#         newImg = img_.resize((x_s,y_s), Image.ANTIALIAS)
#         #newImg = img.resize((50, 50), Image.BILINEAR)   #想调整的大小
#         #cv2.imshow('result', newImg)
#         #cv2.waitKey(-1)
#         # newImg.save('./img2')

#         cv2.imwrite(r'/home/syz/data/label_data/img2'+str(file),newImg)       #  写入文件地址

