import struct 
import pickle as pk 
import numpy as np 

def read_train_label():
    file1 = open("./data/train-labels.idx1-ubyte", "rb")
    list1 = []
    a = file1.read(8)
    for i in range(60000):
        a = file1.read(1)
        list1.append(struct.unpack(">B", a)[0])
    file1.close()
    return list1 

def read_train_image():
    file1 = open("./data/train-images.idx3-ubyte", "rb")
    list1 = []
    a = file1.read(16)
    for n in range(60000):
        slist = []
        for i in range(28):
            sslist = []
            for j in range(28):
                x = file1.read(1)
                sslist.append(struct.unpack(">B", x)[0])
            slist.append(sslist)
        np_mat = np.array(slist)
        list1.append(np_mat)
    file1.close()
    return list1

def read_test_label():
    file1 = open("./data/t10k-labels.idx1-ubyte", "rb")
    list1 = []
    a = file1.read(8)
    for i in range(10000):
        a = file1.read(1)
        list1.append(struct.unpack(">B", a)[0])
    file1.close()
    return list1 

def read_test_image():
    file1 = open("./data/t10k-images.idx3-ubyte", "rb")
    list1 = []
    a = file1.read(16)
    for n in range(10000):
        slist = []
        for i in range(28):
            sslist = []
            for j in range(28):
                x = file1.read(1)
                sslist.append(struct.unpack(">B", x)[0])
            slist.append(sslist)
        np_mat = np.array(slist)
        list1.append(np_mat)
    file1.close()
    return list1

def main():
    train_label_list = read_train_label()
    train_image_list = read_train_image()
    test_label_list = read_test_label()
    test_image_list = read_test_image()
    file1 = open("./data/train_label.pkl", "wb")
    pk.dump(train_label_list, file1)
    file1.close()
    file2 = open("./data/train_image.pkl", "wb")
    pk.dump(train_image_list, file2)
    file2.close()
    file3 = open("./data/test_label.pkl", "wb")
    pk.dump(test_label_list, file3)
    file3.close()
    file4 = open("./data/test_image.pkl", "wb")
    pk.dump(test_image_list, file4)
    file4.close()
    return 0

if __name__=="__main__":
    main()