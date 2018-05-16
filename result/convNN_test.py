import tensorflow as tf 
from tensorflow.python import keras
import numpy as np 
import pickle as pk 
print("finish import.")

path = "D:\\python_proj\\Mnist\\result\\convNN_res.krmd"
conv_model = keras.models.load_model(path)

def load_test_data():
    print("loading training data..")
    file1 = open("D:\\python_proj\\Mnist\\data\\test_image.pkl", "rb")
    list1 = pk.load(file1)
    file1.close()
    for i in range(10000):
        list1[i] = np.reshape(list1[i], (1,28,28))
        list1[i] = np.subtract(np.divide(list1[i], 64), 2)
    file2 = open("D:\\python_proj\\Mnist\\data\\test_label.pkl", "rb")
    list2 = pk.load(file2)
    file2.close()
    return np.array(list1), np.array(list2)

imgs, labels = load_test_data()
labels = keras.utils.to_categorical(labels)

res = conv_model.evaluate(x=imgs, y=labels)
print()
print("test acc:", res[1])

###
"""
this convolutionary model
got 98.8% acc on test data
"""
###