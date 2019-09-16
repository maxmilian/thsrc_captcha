#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time, os
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

full_image = "thsrc.jpg"
WIDTH = 140
HEIGHT = 48
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
    device_pixel_ratio = int(driver.execute_script('return window.devicePixelRatio'))
    driver.quit()
    return location, size, device_pixel_ratio


# In[ ]:


def refine_coordinate(location, size, ratio):
    # Mac Retina 把截圖會把解析度變為 2 倍，需要考慮 ratio
    left = location['x'] * ratio
    right = left + size['width'] * ratio
    top = location['y'] * ratio
    bottom = top + size['height'] * ratio
    return (left, top, right, bottom)


# In[ ]:


def crop_image(coordinate):
    img = Image.open(full_image)
    img = img.crop(coordinate)
    return img


# In[ ]:


i = 0

#ignore existing image
while True:
    i += 1
    filename = FOLDER + str(i) + '.jpg'
    if not os.path.isfile(filename):
        i -= 1
        break

print("start to crawler from index: " + str(i))

while i < 5000:
    i += 1
    location, size, ratio = get_screenshot()
    coordinate = refine_coordinate(location, size, ratio)
    img = crop_image(coordinate)
    img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    # convert rgba to rgb
    img = img.convert('RGB')
    img.save(FOLDER + str(i) + '.jpg', "JPEG")
    print("i: " + str(i))

print("completed")


# In[ ]:




