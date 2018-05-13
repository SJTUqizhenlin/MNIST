import numpy as np 
import pickle as pk 
import matplotlib.pyplot as plt 

def plt_show(x, label):
    plt.subplot(1,1,1)
    plt.title("This is {0}".format(label))
    plt.imshow(x, plt.cm.gray)
    plt.axis("off")
    plt.show()

def visualize(list1, list2, n):
    x = list1[n]
    label = list2[n]
    plt_show(x, label)

def fetch_datas():
    print("loading...")
    file1 = open("./data/test_image.pkl", "rb")
    list1 = pk.load(file1)
    file1.close()
    file2 = open("./data/test_label.pkl", "rb")
    list2 = pk.load(file2)
    file2.close()
    file3 = open("./data/train_image.pkl", "rb")
    list3 = pk.load(file3)
    file3.close()
    file4 = open("./data/train_label.pkl", "rb")
    list4 = pk.load(file4)
    file4.close()
    return list1, list2, list3, list4 

def main():
    l1, l2, l3, l4 = fetch_datas()
    print()
    a = int(input("test or train do you want?(1.test;2.train)"))
    while a == 1 or a == 2:
        if a == 1:
            b = int(input("the number of pic you want:(0-9999)"))
            visualize(l1, l2, b)
        if a == 2:
            b = int(input("the number of pic you want:(0-59999)"))
            visualize(l3, l4, b)
        print()
        a = int(input("test or train do you want?(1.test;2.train)"))
    return 0

if __name__=="__main__":
    main()
