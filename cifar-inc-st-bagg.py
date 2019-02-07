from keras.datasets import cifar10
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

with open('/home/agnieszka/codes_train.pkl', 'rb') as codes:
    codes_train = pickle.load(codes)

with open('/home/agnieszka/codes_test.pkl', 'rb') as codes:
    codes_test = pickle.load(codes)

(_, y_train), (_, y_test) = cifar10.load_data()

param_grid = [{'model__base_estimator__kernel': ['linear'], 'model__base_estimator__C': [0.001, 0.01, 0.1]},
              {'model__base_estimator__kernel': ['rbf'], 'model__base_estimator__C': [1, 10, 50],
               'model__base_estimator__gamma':['scale'], 'model__max_samples':[0.3, 0.5],
               'model__max_features':[0.3, 0.5]}]

pipe = Pipeline([('scaler', StandardScaler()), ('model', BaggingClassifier(svm.SVC()))])

gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=10, return_train_score=True)
gs.fit(codes_train, y_train.flatten())
print('Best parameters:', gs.best_params_)

best_model_st_bagg = gs.best_estimator_

with open('/home/agnieszka/best_model_st_bagg.sav', 'wb') as model:
    pickle.dump(best_model_st_bagg, model)

pred_1 = best_model_st_bagg.predict(codes_test)
pred_2 = best_model_st_bagg.predict(codes_train)

print('Accuracy on test set:', accuracy_score(pred_1, y_test.flatten()))
print('Accuracy on train set:', accuracy_score(pred_2, y_train.flatten()))
