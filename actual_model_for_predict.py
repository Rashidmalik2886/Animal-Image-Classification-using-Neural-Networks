
# coding: utf-8

# In[2]:



# coding: utf-8

# In[ ]:


from keras.models import load_model
import numpy as np
from keras.preprocessing import image


model_path = 'models/classifier.h5'
model = load_model(model_path)
model.get_weights()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

def get_key(val):
    for key, value in my_dict.items():
        if val == value:
                return key

def my_model(path):
    test_image = image.load_img(path, 
                            target_size = (124,124))
    test_image = image.img_to_array(test_image)
    test_image/=255
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    answer = np.argmax(result)
    my_dict = {'cat': 0, 'cow': 1, 'dear': 2, 'dog': 3, 'elephant': 4, 'goat': 5, 'horse': 6, 'lion': 7, 'panda': 8, 'sheep': 9, 'wolf': 10}
#print(test)
    new_test = test[0]
#print(new_test)
    act_value = max(new_test)
#print(act_value)
    if act_value> 0.5:
        val_index = new_test.argmax()
    else:
        val_index = 11
        
    result = get_key(val_index)
    
    return result


my_model(r"C:\Users\Rashid Malik\Desktop\Testing\cat-250px-Gatto_europeo4.jpg")
            

