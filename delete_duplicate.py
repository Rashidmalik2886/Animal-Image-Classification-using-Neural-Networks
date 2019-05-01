
# coding: utf-8

# In[1]:


import hashlib,os
from scipy.misc import imread, imresize, imshow
import time 
import numpy as np
from hashlib import md5
from pathlib import Path


# # Removing Duplicate Images

# In[2]:


os.getcwd()


# In[3]:


os.chdir(r'F:\\ML\\clg\\Dataset')


# In[4]:


os.getcwd()


# In[5]:


path = os.getcwd()
dirs = os.listdir(path)
print(path)
print(dirs)


# In[6]:


def delete_duplicate_images(files):
    duplicates=[]
    files_list = os.listdir()
    hash_keys = dict()
    for index, filename in enumerate (os.listdir('.')):
        if os.path.isfile(filename):
            with open(filename,'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash not in hash_keys:
                hash_keys[filehash] = index
            else:
                duplicates.append((index,hash_keys[filehash]))
    print(duplicates)
    for index in duplicates:
        
        os.remove(files_list[index[0]])
       


# In[7]:


for d in dirs:
        next_dir = os.path.join(path,d)
        for file in os.listdir(next_dir):
            filepath = os.path.join(next_dir,file)
            filepath = Path(filepath)
            os.chdir(filepath)
            delete_duplicate_images(filepath)

