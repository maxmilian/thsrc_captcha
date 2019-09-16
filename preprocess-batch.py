#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, random, cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import binarize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

WIDTH = 140
HEIGHT = 48

CAPTCHA_FOLDER = "captcha/"
PROCESSED_FOLDER = "processed/"


# In[ ]:


def imgDenoise(filename):
    img = cv2.imread(filename)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 30, 30, 7, 21)
    return dst


# In[ ]:


def img2Gray(img):
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh


# In[ ]:


def findRegression(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[:, 14:WIDTH - 7] = 0
    imagedata = np.where(img == 255)

    X = np.array([imagedata[1]])
    Y = HEIGHT - imagedata[0]
    
    poly_reg = PolynomialFeatures(degree = 2)
    X_ = poly_reg.fit_transform(X.T)
    regr = LinearRegression()
    regr.fit(X_, Y)
    return regr


# In[ ]:


def dePolynomial(img, regr):
    X2 = np.array([[i for i in range(0, WIDTH)]])
    poly_reg = PolynomialFeatures(degree = 2)
    X2_ = poly_reg.fit_transform(X2.T)
    offset = 4

    newimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for ele in np.column_stack([regr.predict(X2_).round(2), X2[0]]):
        pos = HEIGHT - int(ele[0])
        newimg[pos - offset:pos + offset, int(ele[1])] = 255 - newimg[pos - offset:pos + offset, int(ele[1])]

    return newimg


# In[ ]:


def preprocessing(from_filename, to_filename):
    if not os.path.isfile(from_filename):
        return
    img = imgDenoise(from_filename)
    img = img2Gray(img)
    regr = findRegression(img)
    newimg = dePolynomial(img, regr)
    cv2.imwrite(to_filename, newimg)
    return


# In[ ]:


i = 0

# ignore existing image
while True:
    i += 1
    filename = PROCESSED_FOLDER + str(i) + '.jpg'
    if not os.path.isfile(filename):
        i -= 1
        break

print("start to process image from index: " + str(i))

while True:
    i += 1
    filename = CAPTCHA_FOLDER + str(i) + '.jpg'
    if not os.path.isfile(filename):
        break
    preprocessing(filename, PROCESSED_FOLDER + str(i) + '.jpg')
    print("i: " + str(i))

print("completed")


# In[ ]:




