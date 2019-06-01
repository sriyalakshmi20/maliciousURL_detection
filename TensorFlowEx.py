import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test  = tf.keras.utils.normalize(x_test,  axis=1)

model1 = tf.keras.models.Sequential()
model1.add(tf.keras.layers.Flatten())

model1.add(tf.keras.layers.Dense(128,
                                 activation=tf.nn.relu))
model1.add(tf.keras.layers.Dense(128, activation=tf.nn.softmax))

model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

model1.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model1.evaluate(x_test,y_test)
print(val_loss, val_acc)