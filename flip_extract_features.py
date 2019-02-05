from keras.datasets import cifar10
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from PIL import Image
import pickle

(x_train, _), (_, _) = cifar10.load_data()
ts = 299

x_sh_1 = x_train.shape[0]
X_train = np.empty((x_sh_1, ts, ts, 3))

for i in range(x_sh_1):
    X_train[i] = np.array(Image.fromarray(x_train[i]).transpose(Image.FLIP_LEFT_RIGHT).resize((ts, ts)))/255


model_inc = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

codes_train = model_inc.predict(X_train)

with open('/home/agnieszka/codes_train_flip.pkl', 'wb') as codes:
    pickle.dump(codes_train, codes)

