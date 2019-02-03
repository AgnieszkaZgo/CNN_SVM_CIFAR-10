from keras.datasets import cifar10
import numpy as np
from sklearn import svm
from keras.applications.inception_v3 import InceptionV3
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

X_train = np.empty((x_train.shape[0], 75, 75, 3))
for i in range(x_train.shape[0]):
    X_train[i] = np.array(Image.fromarray(x_train[i]).resize((75, 75)))/255

X_test = np.empty((x_test.shape[0], 75, 75, 3))
for i in range(x_test.shape[0]):
    X_test[i] = np.array(Image.fromarray(x_test[i]).resize((75, 75)))/255


model_inc = InceptionV3(weights='imagenet', include_top=False, input_shape=(75, 75, 3), classes=10)

codes_train = model_inc.predict(X_train).reshape(x_train.shape[0], -1)
codes_test = model_inc.predict(X_test).reshape(x_test.shape[0], -1)

scaler = StandardScaler()
scaler.fit(codes_train)
codes_train = scaler.transform(codes_train)
svc = svm.LinearSVC(C=10, dual=False, penalty='l1', max_iter=5000)
svc.fit(codes_train, y_train.flatten())

codes_test = scaler.transform(codes_test)
pred = svc.predict(codes_test)

print('Accuracy on test set:', accuracy_score(pred, y_test.flatten()))

