from keras.datasets import cifar10
import numpy as np

from PIL import Image
import pickle

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

X_train = np.empty((x_train.shape[0], 75, 75, 3))
for i in range(x_train.shape[0]):
    X_train[i] = np.array(Image.fromarray(x_train[i]).resize((75, 75)))


with open('X_train.pkl', 'wb') as handle:
    pickle.dump(X_train, handle)