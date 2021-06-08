#img_id=01612-012de910494ad529eed4801568631526.jpg
#img_size(w,h)=506,900
#box=(6.282123,391.481079,445.250946,429.958740)score=0.900000


import os
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as numpy

import matplotlib.patches as mpathes

import re
import cv2 
from PIL import Image
from PIL import ImageDraw, ImageFont

# img = mpimg.imread('./yolo_res/imgs/00007-0118c38f1617aa06ff31601567611476.jpg')
# filename = './yolo_res/txts/00007-0118c38f1617aa06ff31601567611476.jpg.txt'
# f = open(filename, 'r')
# boxes = []
# for lines in f:
#     ls = lines.strip('\n')
#     res = ls.split('=')
#     if(res[0] == 'box'):
#         box = re.findall(r'[(](.*?)[)]',res[1])
#         boxes.append(box[0])
# for i in range(len(boxes)):
#     box = boxes[i].split(',')
#     plt.gca().add_patch(plt.Rectangle(xy=(float(box[0]), float(box[1])),width=float(box[2]), height=float(box[3]), edgecolor = 'r', fill=False, linewidth=2))
# plt.imshow(img)
# plt.axis('off')
# plt.subplots_adjust(left=0.09,right=1,wspace=0.25,hspace=0.25,bottom=0.13,top=0.91)
# plt.savefig('000.jpg', bbox_inches='tight')
# plt.show()

# img = cv2.imread('./yolo_res/imgs/00007-0118c38f1617aa06ff31601567611476.jpg')
# filename = './yolo_res/txts/00007-0118c38f1617aa06ff31601567611476.jpg.txt'
# f = open(filename, 'r')
# boxes = []
# for lines in f:
#     ls = lines.strip('\n')
#     res = ls.split('=')
#     if(res[0] == 'box'):
#         box = re.findall(r'[(](.*?)[)]',res[1])
#         boxes.append(box[0])
# for i in range(len(boxes)):
#     box = boxes[i].split(',')
#     x1 = int(float(box[0]))
#     y1 = int(float(box[1]))
#     ptLeftTop = (x1, y1)
#     x2 = int(float(box[0]))+int(float(box[2]))
#     y2 = int(float(box[1]))+int(float(box[3]))
#     # ptRightBottom = (x2, y2)
#     point_color = (0, 0, 255)
#     thickness = 3
#     lineType = 8
#     cv2.rectangle(img, (x1,y1), (x2, y2), point_color, thickness, lineType)
#     # cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
# cv2.imshow("img",img)
# cv2.waitKey (1000)
# cv2.imwrite("000.jpg",img)





imgspath = '/Users/sunyunzhe01/Desktop/model_postprocess/yolo_res/imgs'
txtspath = '/Users/sunyunzhe01/Desktop/model_postprocess/yolo_res/txts/'
imgfiles = os.listdir(imgspath)
for imgfile in imgfiles:
    imgpath = os.path.join(imgspath, imgfile)
    img = cv2.imread(imgpath)
    filename = txtspath + imgfile + '.txt'

    f = open(filename, 'r')

    boxes = []

    for lines in f:
        ls = lines.strip('\n')
        res = ls.split('=')

        if(res[0] == 'box'):
            box = re.findall(r'[(](.*?)[)]',res[1])
            boxes.append(box[0])

    for i in range(len(boxes)):
        box = boxes[i].split(',')
        x1 = int(float(box[0]))
        y1 = int(float(box[1]))
        ptLeftTop = (x1, y1)
        x2 = int(float(box[0]))+int(float(box[2]))
        y2 = int(float(box[1]))+int(float(box[3]))
        # ptRightBottom = (x2, y2)
        point_color = (0, 0, 255)
        thickness = 3
        lineType = 8
        cv2.rectangle(img, (x1,y1), (x2, y2), point_color, thickness, lineType)
        # plt.gca().add_patch(plt.Rectangle(xy=(float(box[0]), float(box[1])),width=float(box[2]), height=float(box[3]), edgecolor = 'r', fill=False, linewidth=2))

    # plt.imshow(img)
    # plt.axis('off')
    # saveimgname = imgfile+"res.jpg"
    # plt.savefig(saveimgname)
    # plt.show()
    cv2.imshow("img",img)
    saveimgname = imgfile+"res.jpg"
    cv2.imwrite("./res/{}".format(saveimgname),img)
    cv2.waitKey (1000)
    
        

    



