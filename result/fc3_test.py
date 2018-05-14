import tensorflow as tf 
from tensorflow.python import keras 
import numpy as np 
import pickle as pk 

def load_test_data():
    print("loading training data..")
    file1 = open("D:\\python_proj\\Mnist\\data\\test_image.pkl", "rb")
    list1 = pk.load(file1)
    file1.close()
    for i in range(10000):
        list1[i] = np.reshape(list1[i], (784,))
        list1[i] = np.subtract(np.divide(list1[i], 64), 2)
    file2 = open("D:\\python_proj\\Mnist\\data\\test_label.pkl", "rb")
    list2 = pk.load(file2)
    file2.close()
    return np.array(list1), np.array(list2)

imgs, labels = load_test_data()
labels = keras.utils.to_categorical(labels)
fc3_model = keras.models.load_model("D:\\python_proj\\Mnist\\result\\fc3_res.krmd")

res = fc3_model.evaluate(x=imgs, y=labels, batch_size=64)
print()
print("acc:", res[1])


###
"""
this 3-layer fully connected model
got 97.14% acc on test data
"""
###