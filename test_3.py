import pickle
import time

import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

from gensim.models import KeyedVectors

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

PREPROC_TD_IDF = 0
PREPROC_SKIP_GRAM = 1

def tf_idf_vectors(train, test):
    columns1 = list(train.columns.values)
    columns2 = list(test.columns.values)
    missing = list()
    for column in columns1:
        if not(column in columns2):
            missing.append(column)
    temp = pd.DataFrame(0.0, index=np.arange(len(test)), columns=missing)
    test = pd.concat((test, temp), axis = 1)

    print("Concat done!")

    to_drop = list()

    for column in columns2:
        if not(column in columns1):
            to_drop.append(column)

    test.drop(to_drop, inplace = True,  axis=1)

    test = test.reindex(sorted(test.columns), axis=1)

    print("Done!")

    return test

def skip_gram(train, test, wordvectors):
    lista = test['Sentence'].values.tolist()
    values = list()
    for sentence in lista:
        values.append(sentence.split(" "))

    w2v = KeyedVectors.load("word_vectors/" + wordvectors + ".wordvectors", mmap='r')

    values_vec = list()
    for sentence in values:
        sen_vec = np.array([])
        for word in sentence:
            try:
                word_embedding = w2v[word]
                sen_vec = np.concatenate((sen_vec, word_embedding), axis = None)
            except:
                pass
        values_vec.append(sen_vec.tolist())

    maximum = len(list(train.columns.values))
    for sentence in values_vec:
        diff = maximum - len(sentence)
        while diff > 0:
            sentence.append(-1)
            diff -= 1
        while diff < 0:
            sentence.pop(0)
            diff += 1
    test = pd.DataFrame(values_vec)

    return test

def framework(model, train, data1, preproc, nn):
    if nn:
        clf = keras.models.load_model("models/" + model + ".h5")
    else:
        clf = pickle.load(open("models/" + model + ".sav", 'rb'))

    if preproc == 0:
        df_train = pd.read_csv("payloads/" + train + "/TF_IDF/train.csv")
        X_train = df_train.loc[:, df_train.columns != 'Label']

        df_test_1 = pd.read_csv("payloads/" + data1 + "/TF_IDF/test.csv")
        X_test_1 = df_test_1.loc[:, df_test_1.columns != 'Label']
        y_test_1 = df_test_1['Label'] 

        X_test_1 = tf_idf_vectors(X_train, X_test_1)

    if preproc == 1:
        df_train = pd.read_csv("payloads/" + train + "/skip_gram/train.csv")
        X_train = df_train.loc[:, df_train.columns != 'Label']

        df_test_1 = pd.read_csv("payloads/" + data1 + "/plain/test.csv", delimiter=";")
        X_test_1 = df_test_1.loc[:, df_test_1.columns != 'Label']
        y_test_1 = df_test_1['Label']

        X_test_1 = skip_gram(X_train, X_test_1, train)

    len_1 = len(X_test_1.index)

    if nn:
        X_test_1 = tf.convert_to_tensor(X_test_1)
        start_1 = time.time()
        predict_1 = clf.predict(X_test_1)
        stop_1 = time.time()
        test_pred_1 = np.argmax(predict_1,axis=1)
        f1_score_1 = f1_score(y_test_1, test_pred_1)

    else:
        start_1 = time.time()
        test_pred_1 = clf.predict(X_test_1)
        stop_1 = time.time()
        f1_score_1 = f1_score(y_test_1, test_pred_1)


    runtime_1 = (stop_1 - start_1) / len_1

    print("For: " + model + ":\nF1-score on " + data1 + ": " + str(f1_score_1) + " in " + str(runtime_1))

framework("united_gbc_0.01_1000_2", "united", "company", PREPROC_SKIP_GRAM, False)
framework("united_lr_newton-cg_0.1", "united", "company",  PREPROC_SKIP_GRAM, False)
framework("united_nn_0.001_64_sigmoid", "united", "company", PREPROC_SKIP_GRAM, True)
framework("united_rf_32_10", "united", "company",  PREPROC_SKIP_GRAM, False)
framework("united_svm_linear_10", "united", "company",  PREPROC_SKIP_GRAM, False)

framework("SQLi_1_gbc_0.1_1000_4", "SQLi_1", "company",  PREPROC_TD_IDF, False)
framework("SQLi_1_lr_newton-cg_10", "SQLi_1", "company",  PREPROC_TD_IDF, False)
framework("SQLi_1_nn_0.001_64_sigmoid", "SQLi_1", "company",  PREPROC_TD_IDF, True)
framework("SQLi_1_rf_16_100", "SQLi_1", "company",  PREPROC_TD_IDF, False)
framework("SQLi_1_svm_linear_1", "SQLi_1", "company",  PREPROC_TD_IDF, False)

framework("SQLi_2_gbc_0.1_1000_8", "SQLi_2", "company",  PREPROC_SKIP_GRAM, False)
framework("SQLi_2_rf_1_100", "SQLi_2", "company",  PREPROC_SKIP_GRAM, False)
framework("SQLi_2_svm_poly_10", "SQLi_2", "company",  PREPROC_SKIP_GRAM, False)

framework("SQLi_2_lr_newton-cg_10", "SQLi_2", "company",  PREPROC_TD_IDF, False)
framework("SQLi_2_nn_0.1_64_sigmoid", "SQLi_2", "company",  PREPROC_TD_IDF, True)
