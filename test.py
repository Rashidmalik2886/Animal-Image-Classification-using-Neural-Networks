
# coding: utf-8

# In[2]:


from keras.models import load_model
import numpy as np


# In[3]:


model = load_model('classifier.h5')
model.summary()


# In[4]:


model.get_weights()


# In[5]:


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[6]:


from keras.preprocessing import image


# In[7]:


model.optimizer


# In[8]:


def my_model():
    test_image = image.load_img(r"C:\Users\Rashid Malik\Desktop\Testing\lion.jpeg", 
                            target_size = (124,124))
    test_image = image.img_to_array(test_image)
    test_image/=255
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    answer = np.argmax(result)
    return result


# In[9]:


test = my_model()


# In[10]:


my_dict = {"cat": 0, 'cow': 1, 'dear': 2, 'dog': 3, 'elephant': 4, 'goat': 5, 'horse': 6, 'lion': 7, 'panda': 8, 'sheep': 9, 'wolf': 10}
print(test)
new_test = test[0]
print(new_test)
act_value = max(new_test)
print(act_value)
val_index = new_test.argmax()


# In[18]:


print(val_index)
def get_key(val):
    for key, value in my_dict.items():
            if val == value:
                    return key

key = get_key(val_index)
print(key)
if act_value > 0.7:
    result = key
elif val_index > 0.4:
    result = "Not Sure but may be " + key
else:
    result = "Sorry, Can't recognise"


# In[19]:


print(result)

