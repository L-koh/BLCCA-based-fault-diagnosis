import os
os.environ['KERAS_BACKEND']='theano'
from keras.models import load_model, model_from_json
from keras.optimizers import RMSpropfrom model import blcca_model, train_blcca
import pandas as pd
import numpy as np
import warnings
import config as con
import read_data as rd
from plot import *
from ftlib import *
from blcca_test import *
from knn import *


warnings.filterwarnings('ignore')

Train = True
col = 1
addcol = 0

if col>7:
    addcol = 1

if __name__ == '__main__':


    blcca_train_data = pd.read_excel('data/lib_lcca.xls', sheet_name=0).values()

    lcca_a, lcca_b = blcca_train_data[:, :-1], blcca_train_data[:, -1:]
    lcca_knn = KNeighborsClassifier(n_neighbors=3,
                                    metric="l1")
    lcca_knn.fit(lcca_a, lcca_b.reshape([lcca_b.shape[0]]).astype(int))
    a = lcca_knn.predict(lcca_a)

    blcca_view_1, blcca_view_2 = rd.data_prepare(col = col + addcol)

    # test()     #confusion matrix
    testknn()

    path = mkdir(lib=False)

    blcca = blcca_model()
    blcca.summary()

    train_blcca(path, blcca_view_1, blcca_view_2, blcca, train=Train, knn=lcca_knn)

    plt.show()



