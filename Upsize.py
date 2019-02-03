import pandas as pd
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn import svm
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pickle

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

X_train=np.empty((x_train.shape[0], 75,75,3))
for i in range(x_train.shape[0]):
    X_train[i]=np.array(Image.fromarray(x_train[i]).resize((75,75)))


with open('X_train.pkl', 'wb') as handle:
    pickle.dump(X_train, handle)