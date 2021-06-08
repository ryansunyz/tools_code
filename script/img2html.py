import codecs
dict_tongji={}
dict_tongji_recall={}
excel_list=[]
import glob
import shutil
from PIL import Image

import os

imgsgt = "./gt/"
imggtnames = os.listdir(imgsgt)
imggtnames = sorted(imggtnames)

imgspath = "./yolo/"
imgnames = os.listdir(imgspath)
imgnames=sorted(imgnames)

imgsttfnetpath = "./ttfnet/"
imgttfnetnames = os.listdir(imgsttfnetpath)
imgttfnetnames=sorted(imgttfnetnames)


show_path = './'



with codecs.open(show_path+'/resultshow.html','w',encoding= u'utf-8',errors='ignore') as wf:
    wf.write("<!doctype html>"+'\n'+
"<mvc:default-servlet-handler/>"+'\n'+
'<html lang="zh">'+'\n'+
"<head>"+'\n'+
'<meta charset="UTF-8">'+'\n'+
            "</head>"+'\n'+
            "<html>"+'\n'+ 
            "<body>"+'\n')
    for v in range(len(imgnames)):
    #for k,v in dict_img_pres_top1.items():
        #img = k
        imggt = imgsgt+imggtnames[v]
        imgname = imgspath+imgnames[v]
        imgttf = imgsttfnetpath+imgttfnetnames[v]

        wf.write(imgname)

        wf.write('<font size="3" color="red">'+u' yolo vs ttfnet '+':'+str(v)+' ')
        wf.write('</font>'+"\n")
        wf.write("<br />")
        #wf.write("<img src='"+img.split('/')[-1]+"' "+'height='+"200"+'/'+">" + "\n")
        wf.write("<img src='"+imggt+"' "+'height='+"200"+'/'+">" + "\n")
        wf.write("<img src='"+imgname+"' "+'height='+"200"+'/'+">" + "\n")
        wf.write("<img src='"+imgttf+"' "+'height='+"200"+'/'+">" + "\n")
        wf.write("<br />")
        wf.write('<font size="3" color="red">')

        wf.write("<br />")
        wf.write("<br />")
        wf.write("<br />")

    wf.write("</body>"+'\n'+"</html>")

