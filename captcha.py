#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from selenium import webdriver
# driver = webdriver.Firefox()
# driver = webdriver.Chrome()
driver = webdriver.PhantomJS()
driver.get('http://irs.thsrc.com.tw/IMINT/')
driver.find_element_by_id('btn-confirm').click()


# In[ ]:


driver.save_screenshot("captcha/thsrc.jpg")


# In[ ]:


element = driver.find_element_by_xpath('//*[@id="BookingS1Form_homeCaptcha_passCode"]')
location = element.location
location


# In[ ]:


element.size


# In[ ]:


left = location['x']
print("1 left: " + str(left))
left = 400
print("2 left: " + str(left))
right = left + element.size['width']
top = element.location['y']
print("1 top: " + str(top))
top = 550
print("2 top: " + str(top))
bottom = top + element.size['height']
(left, top, right, bottom)


# In[ ]:


from PIL import Image
img = Image.open('captcha/thsrc.jpg')
img = img.crop((left, top, right, bottom))


# In[ ]:


img.show()


# In[ ]:


import time
# convert rgba to rgb
img = img.convert('RGB')
img.save("captcha/" + str(time.time()) + '.jpg', "JPEG")


# In[ ]:




