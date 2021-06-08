import os
import json
import cv2
from lxml import etree
import xml.etree.cElementTree as ET
import time
import pandas as pd
from tqdm import tqdm
 
 
 
anno = "x_Merge.cls11_val.json"
xml_dir = "anno_val_xml_1/"
 
#Create anno dir
# dttm = time.strftime("%Y%m%d%H%M%S", time.localtime())
# if os.path.exists(xml_dir):
#     os.rename(xml_dir,xml_dir+dttm)
#os.mkdir(xml_dir)
# Read json file
 
 
import json
with open(anno,'r') as load_f:
    f = json.load(load_f)
 
imgs = f['images']
 
df_cate = pd.DataFrame(f['categories'])
df_cate_sort = df_cate.sort_values(["id"],ascending=True)
categories = list(df_cate_sort['name'])
df_anno = pd.DataFrame(f['annotations'])
 
 
#anno_list = list(annos.itertuples(index=False))
#print(anno_list[0]['id'])
 
 
 
for i in tqdm(range(len(imgs))):
    xml_content = []
    file_name = imgs[i]['file_name']
    # syz relpace file name
    file_name_split = file_name.split("/")
    file_name = file_name_split[-1]
    height = imgs[i]['height']
    img_id = imgs[i]['id']
    width = imgs[i]['width']
 
    xml_content.append("<annotation>")
    xml_content.append("	<folder>VOC2007</folder>")
    xml_content.append("	<filename>"+file_name+"</filename>")
    xml_content.append("	<size>")
    xml_content.append("		<width>"+str(width)+"</width>")
    xml_content.append("		<height>"+str(height)+"</height>")
    xml_content.append("	</size>")
    xml_content.append("	<segmented>0</segmented>")
    #通过img_id找到annotations
    annos = df_anno[df_anno["image_id"].isin([img_id])]
 
    for index, row in annos.iterrows():
        bbox = row["bbox"]
        category_id = row["category_id"]
        cate_name = categories[category_id]
 
        # add new object
        xml_content.append("	<object>")
        xml_content.append("		<name>"+cate_name+"</name>")
        xml_content.append("		<pose>Unspecified</pose>")
        xml_content.append("		<truncated>0</truncated>")
        xml_content.append("		<difficult>0</difficult>")
        xml_content.append("		<bndbox>")
        xml_content.append("			<xmin>"+str(int(bbox[0]))+"</xmin>")
        xml_content.append("			<ymin>"+str(int(bbox[1]))+"</ymin>")
        xml_content.append("			<xmax>"+str(int(bbox[0]+bbox[2]))+"</xmax>")
        xml_content.append("			<ymax>"+str(int(bbox[1]+bbox[3]))+"</ymax>")
        xml_content.append("		</bndbox>")
        xml_content.append("	</object>")
    xml_content.append("</annotation>")
 
    x = xml_content
    xml_content=[x[i] for i in range(0,len(x)) if x[i]!="\n"]
    ### list存入文件  

    save_path = file_name_split[0]
    #print('save_path', save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    
    xml_path = os.path.join(xml_dir+save_path,file_name.replace('.jpg','.xml'))
    with open(xml_path, 'w+',encoding="utf8") as f:
        f.write('\n'.join(xml_content))
    xml_content[:]=[]
