#!/usr/bin/env python
# coding: utf-8

# In[27]:


from selenium import webdriver
driver = webdriver.Firefox()
#driver.add_cookie({'name' : 'AcceptIRSCookiePolicyTime', 'value' : 'true'})
driver.get('http://irs.thsrc.com.tw/IMINT/')
driver.find_element_by_id('btn-confirm').click()


# In[28]:


driver.save_screenshot('thsrc.jpg')


# In[29]:


#element = driver.find_element_by_id('BookingS1Form_homeCaptcha_passCode')
element = driver.find_element_by_xpath('//*[@id="BookingS1Form_homeCaptcha_passCode"]')
print(element)
location = element.location
location


# In[30]:


element.size


# In[31]:


left = location['x']
right = location['x'] + element.size['width']
top = element.location['y']
bottom = element.location['y']  + element.size['height']
(left, top, right, bottom)


# In[32]:


from PIL import Image
img = Image.open('thsrc.jpg')
img = img.crop((left, top, right, bottom))


# In[33]:


img.show()


# In[34]:


import time
# convert rgba to rgb
img = img.convert('RGB')

img.save("captcha/" + str(time.time()) + '.jpg', "JPEG")


# In[ ]:




