import numpy as np
import cv2

img = np.fromfile("./det.bin",dtype=np.float32)
img = img.reshape(3,416,416).transpose(1,2,0)

img *= 255
img = img.astype(np.uint8)

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

print(img.shape)
cv2.imwrite("aaa.jpg",img)
