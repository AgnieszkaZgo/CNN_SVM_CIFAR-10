from keras.datasets import cifar10
import numpy as np
from sklearn import svm
from keras.applications.inception_v3 import InceptionV3
from sklearn.metrics import accuracy_score
from PIL import Image
from sklearn.model_selection import GridSearchCV


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


param_grid = {'penalty': ['l1', 'l2'],
              'C': [1e-4, 1e-3, 0.01, 0.1, 1]}

gs = GridSearchCV(svm.LinearSVC(dual=False, max_iter=5000), param_grid, cv=3, n_jobs=-1, verbose=10, return_train_score=True)
gs.fit(codes_train, y_train.flatten())
print('Best parameters:', gs.best_params_)

best_model = gs.best_estimator_
pred = best_model.predict(codes_test)
pred_2 = best_model.predict(codes_train)

print('Accuracy on test set:', accuracy_score(pred, y_test.flatten()))
print('Accuracy on train set:', accuracy_score(pred_2, y_train.flatten()))