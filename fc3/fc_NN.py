import tensorflow as tf 
from tensorflow.python import keras 
import numpy as np 
import pickle as pk 
import random
print("finish import.")

def load_train_data():
    print("loading training data..")
    file1 = open("D:\\python_proj\\Mnist\\data\\train_image.pkl", "rb")
    list1 = pk.load(file1)
    file1.close()
    for i in range(len(list1)):
        list1[i] = np.reshape(list1[i], (784,))
        list1[i] = np.subtract(np.divide(list1[i], 64), 2)
    file2 = open("D:\\python_proj\\Mnist\\data\\train_label.pkl", "rb")
    list2 = pk.load(file2)
    file2.close()
    list3 = []
    for i in range(60000):
        list3.append(tuple([list1[i], list2[i]]))
    random.shuffle(list3)
    list1 = []
    list2 = []
    for e in list3:
        list1.append(e[0])
        list2.append(e[1])
    return np.array(list1), np.array(list2)

fc3_model = keras.models.Sequential()
fc3_model.add(keras.layers.Dense(units=600, input_shape=(784,), activation="selu"))
fc3_model.add(keras.layers.Dense(units=200, activation="selu"))
fc3_model.add(keras.layers.Dense(units=10, activation="selu"))
fc3_model.add(keras.layers.Activation("softmax"))

fc3_model.compile(optimizer="Adam", 
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

imgs, labels = load_train_data()
labels = keras.utils.to_categorical(labels, num_classes=10)
least_loss = 2.3

while (True):
    fc3_model.fit(x=imgs[0:50000], y=labels[0:50000], batch_size=64, epochs=1)
    loss, acc = fc3_model.evaluate(x=imgs[50000:60000], y=labels[50000:60000], batch_size=64)
    print()
    print("loss:", loss)
    print("acc:", acc)
    print()
    if loss < least_loss:
        least_loss = loss
    if loss > (least_loss * 1.2):
        break 

fc3_model.save("D:\\python_proj\\Mnist\\result\\fc3_res.krmd")