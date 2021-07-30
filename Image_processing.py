
import numpy as np
import cv2
import os
from PIL import Image
from torchvision.transforms import transforms

path = '/home/ingivision/PycharmProjects/STN/github/data/Placa 1C2'
images = os.listdir(path)
ori_images = []
target_images = []
images.sort()

for i in range(len(images)):
  ori_images.append(f"Placa 1C22/{images[i]}")
  target_images.append(f"Placa 1C2/{images[i]}")

path = '/home/ingivision/PycharmProjects/STN/github/data'

for j in range(len(ori_images)):
    img = cv2.imread(f"{path}/{ori_images[j]}",0)
    img = np.asarray(img)
    img = img - np.amin(img)
    img = img/np.amax(img)*255
    gray = img.astype(np.uint8)
    gaussiana = cv2.GaussianBlur(gray, (21,21), 0)
    thresh = cv2.adaptiveThreshold(gaussiana, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 701, 12)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    l = 0
    for i in range(len(contours)):
          if cv2.contourArea(contours[i])>l:
            momentos = cv2.moments(contours[i])
            cx = int(momentos['m10']/momentos['m00'])
            cy = int(momentos['m01']/momentos['m00'])
            if (cx>1296 and cx<2200 and cy> 150 and cy<972):
              worm = contours[i]
              l=cv2.contourArea(contours[i])
    Fondo = 255 * np.ones((1944, 2592), np.uint8)
    worm = worm.reshape((1,len(worm),2))
    img = cv2.fillPoly(Fondo,worm,0)
    cv2.imwrite(f"{path}/Seg_Micro2/{images[j]}", img)
    canny = cv2.Canny(img,0,255)
    canny = 255 - canny
    img = cv2.distanceTransform(img, cv2.DIST_C, 3)
    cv2.imwrite(f"{path}/Dist_Micro2/{images[j]}", img)

recorte = transforms.Compose([
                                          transforms.ToPILImage(),
                                          transforms.CenterCrop((972, 1296)),
        ])

for j in range(len(ori_images)):
    img = cv2.imread(f"{path}/{target_images[j]}",0)
    img = np.asarray(img)
    img = recorte(img)
    img = np.asarray(img)
    img = img - np.amin(img)
    img = img/np.amax(img)*255
    gray = img.astype(np.uint8)
    gaussiana = cv2.GaussianBlur(gray, (21,21), 0)
    thresh = cv2.adaptiveThreshold(gaussiana, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 701, 15)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    l = 0
    for i in range(len(contours)):
          if cv2.contourArea(contours[i])>l:
                worm = contours[i]
                l=cv2.contourArea(contours[i])
    Fondo = 255 * np.ones((972, 1296), np.uint8)
    img = cv2.drawContours(Fondo, worm, -1, (0, 255, 0), 3)
    worm = worm.reshape((1,len(worm),2))
    img = cv2.fillPoly(Fondo,worm,0)
    cv2.imwrite(f"{path}/Seg_Micro1/{images[j]}", np.asarray(img))
    canny = cv2.Canny(img,0,255)
    canny = 255 - canny
    img = cv2.distanceTransform(img, cv2.DIST_C, 3)
    cv2.imwrite(f"{path}/Dist_Micro1/{images[j]}", img)