#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time, os
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

full_image = "thsrc.jpg"
WIDTH = 280
HEIGHT = 96
FOLDER = "captcha/"


# In[ ]:


def get_screenshot():
    chrome_options = Options() # 啟動無頭模式
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    executable_path = '/usr/local/bin/chromedriver'
    driver = webdriver.Chrome(executable_path=executable_path, options=chrome_options)
    driver.get('http://irs.thsrc.com.tw/IMINT/')

    driver.find_element_by_id('btn-confirm').click()
    time.sleep(1)

    driver.save_screenshot(full_image)
    element = driver.find_element_by_xpath('//*[@id="BookingS1Form_homeCaptcha_passCode"]')
    location = element.location
    size = element.size
    driver.quit()
    return location, size


# In[ ]:


def refine_coordinate(location, size):
    left = location['x'] * 2
    right = left + size['width'] * 2
    top = location['y'] * 2
    bottom = top + size['height'] * 2
    return (left, top, right, bottom)


# In[ ]:


def crop_and_resize_image(coordinate, i):
    img = Image.open(full_image)
    img = img.crop(coordinate)
    img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    # convert rgba to rgb
    img = img.convert('RGB')
    img.save(FOLDER + str(i) + '.jpg', "JPEG")


# In[ ]:


i = 0

# ignore existing image
while True:
    i += 1
    filename = FOLDER + str(i) + '.jpg'
    if not os.path.isfile(filename):
        i -= 1
        break

while True:
    i += 1
    location, size = get_screenshot()
    coordinate = refine_coordinate(location, size)
    crop_and_resize_image(coordinate, i)
    print("i: " + str(i))


# In[ ]:




