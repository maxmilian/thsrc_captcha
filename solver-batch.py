#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, random, cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import binarize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

WIDTH = 280
HEIGHT = 96

CAPTCHA_FOLDER = "captcha/"
PROCESSED_FOLDER = "processed/"


# In[2]:


def imgDenoise(filename):
    img = cv2.imread(filename)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 30, 30, 7, 21)
#     plt.subplot(121)
#     plt.imshow(img)
#     plt.subplot(122)
#     plt.imshow(dst)
#     plt.show()
    return dst


# In[3]:


def img2Gray(img):
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
#     plt.imshow(thresh)
#     plt.show()
    return thresh


# In[4]:


def findRegression(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[:, 8:WIDTH - 8] = 0
    imagedata = np.where(img == 255)

    X = np.array([imagedata[1]])
    Y = HEIGHT - imagedata[0]
    
    poly_reg = PolynomialFeatures(degree = 2)
    X_ = poly_reg.fit_transform(X.T)
    regr = LinearRegression()
    regr.fit(X_, Y)
    return regr


# In[7]:


def dePolynomial(img, regr):
    X2 = np.array([[i for i in range(0, WIDTH)]])
    poly_reg = PolynomialFeatures(degree = 2)
    X2_ = poly_reg.fit_transform(X2.T)
    
    newimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for ele in np.column_stack([regr.predict(X2_).round(2), X2[0]]):
        pos = HEIGHT - int(ele[0])
        newimg[pos - 3:pos + 3, int(ele[1])] = 255 - newimg[pos - 3:pos + 3, int(ele[1])]

    return newimg


# In[6]:


i = 0

# ignore existing image
while True:
    i += 1
    filename = PROCESSED_FOLDER + str(i) + '.jpg'
    if not os.path.isfile(filename):
        i -= 1
        break

while True:
    i += 1
    filename = CAPTCHA_FOLDER + str(i) + '.jpg'
    if not os.path.isfile(filename):
        break
    img = imgDenoise(filename)
    img = img2Gray(img)
    regr = findRegression(img)
    newimg = dePolynomial(img, regr)
    cv2.imwrite(PROCESSED_FOLDER + str(i) + '.jpg', newimg)
    print("i: " + str(i))

print("completed")


# In[ ]:




