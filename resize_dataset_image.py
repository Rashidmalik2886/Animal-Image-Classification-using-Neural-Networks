
# coding: utf-8

# In[17]:


from PIL import Image
import os, sys
from resizeimage import resizeimage


# In[20]:


path = "F:\\ML\\clg\\Dataset\\"
dirs = os.listdir(path)

def resize_image():
    for d in dirs:
        next_dir = os.path.join(path,d)
        for file in os.listdir(next_dir):
            filepath = os.path.join(next_dir,file)
            print(filepath)
            with Image.open(filepath) as im:
                
                cover = resizeimage.resize_cover(im, [224, 224])
                cover.save(filepath, im.format)


# In[21]:


resize_image()

