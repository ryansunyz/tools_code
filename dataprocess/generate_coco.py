import os
import numpy as np
import cv2
import time
import logging
from collections import defaultdict
import xml.etree.ElementTree as ET
from pycocotools.coco import COCO
import json
#from .coco import CocoDataset


def get_file_list(path, type='.xml'):
    file_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        foldname = maindir.split('/')[-1]
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext == type:
                file_names.append(foldname + '/'+ filename)
    return file_names

'''
'''

from collections import OrderedDict,defaultdict


x_info = {'description': 'qx,R50res,cls11plus.mergeyouhua',
  'url': 'http://xx.com',
  'version': '1.0',
  'year': 2021,
  'contributor': 'sunyunzhe01',
  'date_created': "2021/03/31"}
coco_json = dict(info = x_info, images=[], annotations=[], categories=[])

names = """0 human_face
1 ufo
2 上身
3 上身泳衣和上身内衣
4 下身
5 二维码
6 全身
7 全身泳衣和全身内衣（连体或分体）
8 全身连衣裙
9 圆形码（小程序码）
10 条形码"""
cls_names =[]
for x in names.split("\n"):
    x = x.split(" ")
    cls_names.append(x[1])
print(cls_names)
name_to_id = OrderedDict()
categories = []
for inm,xnm in enumerate(cls_names):
    name_to_id[xnm] = inm
    categories.append({'supercategory': xnm, 'id': inm, 'name': xnm})
coco_json["categories"] = categories
# categories

def xml_to_coco(ann_path):
    """
    convert xml annotations to coco_api
    :param ann_path:
    :return:
    """
    logging.info('loading annotations into memory...')
    tic = time.time()
    ann_file_names = get_file_list(ann_path, type='.xml')
    logging.info("Found {} annotation files.".format(len(ann_file_names)))
    print('annotations file len is ', len(ann_file_names))
    image_info = []
    categories = []
    annotations = []
    for idx, supercat in enumerate(cls_names):
        categories.append({'supercategory': supercat,
                            'id': idx,
                            'name': supercat})
    ann_id = 1
    for idx, xml_name in enumerate(ann_file_names):
        tree = ET.parse(os.path.join(ann_path, xml_name))
        root = tree.getroot()
        file_name = root.find('filename').text
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        #prename = xml_name.split('/')[0]
        jpg_name = xml_name.replace('.xml','.jpg')
        file_name = jpg_name
        info = {'file_name': file_name,
                'height': height,
                'width': width,
                'id': idx + 1}
        image_info.append(info)
        for _object in root.findall('object'):
            category = _object.find('name').text
            if category == 'qr':
                category = '二维码'
            if category == 'cc':
                category = '圆形码（小程序码）'
            if category == 'bar':
                category = '条形码'
            if category not in cls_names:
                logging.warning("WARNING! {} is not in class_names! Pass this box annotation.".format(category))
                print("the filename is ... ", xml_name)
                continue
            for cat in categories:
                if category == cat['name']:
                    cat_id = cat['id']
            xmin = int(_object.find('bndbox').find('xmin').text)
            ymin = int(_object.find('bndbox').find('ymin').text)
            xmax = int(_object.find('bndbox').find('xmax').text)
            ymax = int(_object.find('bndbox').find('ymax').text)
            w = xmax - xmin
            h = ymax - ymin
            if w < 0 or h < 0:
                logging.warning("WARNING! Find error data in file {}! Box w and h should > 0. Pass this box "
                                "annotation.".format(xml_name))
                continue
            coco_box = [max(xmin, 0), max(ymin, 0), min(w, width), min(h, height)]
            ann = {'image_id': idx + 1,
                    'bbox': coco_box,
                    'category_id': cat_id,
                    'iscrowd': 0,
                    'id': ann_id,
                    'area': coco_box[2] * coco_box[3]
                    }
            annotations.append(ann)
            ann_id += 1

    coco_dict = {'info': x_info,
                'images': image_info,
                'categories': categories,
                'annotations': annotations}
    logging.info('Load {} xml files and {} boxes'.format(len(image_info), len(annotations)))
    logging.info('Done (t={:0.2f}s)'.format(time.time() - tic))
    return coco_dict


if __name__ == '__main__':
    annfiles = './anno_train_xml'
    coco_dict = xml_to_coco(annfiles)
    json_file = './syz_train.json'
    json.dump(coco_dict, open(json_file, 'w'), ensure_ascii=False)
