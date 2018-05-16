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
        list1[i] = np.reshape(list1[i], (1, 28, 28))
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

imgs, labels = load_train_data()
labels = keras.utils.to_categorical(labels)

conv_model = keras.models.Sequential()

conv_model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), 
                padding="valid", data_format="channels_first", 
                activation="selu", input_shape=(1,28,28))) #32,26,26
conv_model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), 
                padding="valid", data_format="channels_first", 
                activation="selu")) #64,24,24
conv_model.add(keras.layers.MaxPool2D(data_format="channels_first")) #64,12,12

conv_model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), 
                padding="valid", data_format="channels_first", 
                activation="selu")) #64,10,10
conv_model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), 
                padding="valid", data_format="channels_first", 
                activation="selu")) #64,8,8
conv_model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), 
                padding="valid", data_format="channels_first", 
                activation="selu")) #64,6,6
conv_model.add(keras.layers.MaxPool2D(data_format="channels_first")) #64,3,3

conv_model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), 
                padding="valid", data_format="channels_first", 
                activation="selu")) #128,1,1
conv_model.add(keras.layers.Flatten())
conv_model.add(keras.layers.Dense(units=10, activation="softmax"))

adam = keras.optimizers.Adam(lr=0.0002)
conv_model.compile(
    optimizer=adam,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
loss_ave, loss_pre = 0, 23.0
acc_ave, acc_pre = 0, 0

while True:
    for i in range(5):
        conv_model.fit(x=imgs[0:50000], y=labels[0:50000], 
                        batch_size=64, epochs=1)
        res = conv_model.evaluate(x=imgs[50000:60000], y=labels[50000:60000], 
                        batch_size=64)
        loss_ave+= res[0]
        acc_ave += res[1]
    loss_ave /= 5; print(); print("loss:", loss_ave)
    acc_ave /= 5; print("acc:", acc_ave)
    if loss_ave > loss_pre * 1.1:
        break
    loss_pre, acc_pre = loss_ave, acc_ave
    loss_ave, acc_ave = 0, 0

conv_model.save("D:\\python_proj\\Mnist\\result\\convNN_res.krmd", 
                overwrite=True)