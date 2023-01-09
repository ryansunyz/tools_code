#coding:utf8
#rename files in fold
import os

def rename():
    path = "/home/syz/data/label_data/watch"
    files = os.listdir(path)
    i = 20

    for file in files:
        dir = os.path.join(path, file)
        # if os.path.isdir(dir):
        #     rename(dir)
        #     continue
        file_split = file.split('.')
        i=i+1
        if file_split[1] == "jpg":
            new_dir = os.path.join(path,"0000"+str(i)+'.jpg')
            os.rename(dir,new_dir)
# rename(os.path.dirname(__file__))
rename()
