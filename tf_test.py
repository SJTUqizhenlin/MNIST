import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import time
print("Import Finished")

x_data = np.random.rand(2, 100)
y_data = np.dot([0.1, 0.2], x_data) + 0.3

b = tf.Variable(tf.zeros((1), dtype=tf.float64))
w = tf.Variable(tf.random_uniform((1,2), minval=0, maxval=1, dtype=tf.float64))
y = tf.matmul(w, x_data) + b 

loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

fig = plt.figure()
sfig1 = fig.add_subplot(1,1,1)
sfig1.set(xlabel="loops", ylabel="loss", title="linear fit")
sfig1.grid()
vis_los = ([], [])
plt.ion()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train)
        vis_los[0].append(i)
        vis_los[1].append(sess.run(loss))
        if i % 100 == 0:
            print(sess.run(w), sess.run(b))
            sfig1.plot(vis_los[0], vis_los[1])
            plt.draw()
            plt.pause(1e-6)
            time.sleep(1.0)
    print(sess.run(w), sess.run(b))
plt.ioff()
plt.show()