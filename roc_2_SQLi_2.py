from sklearn.metrics import f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

import pickle

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

print("SQLi_2.csv")

df_train = pd.read_csv("payloads/SQLi_2/TF_IDF/train.csv")
X_train = df_train.loc[:, df_train.columns != 'Label']

df_test = pd.read_csv("payloads/united/TF_IDF/test.csv")
X_test = df_test.loc[:, df_test.columns != 'Label']
y_test = df_test['Label'] 

X_test = tf_idf_vectors(X_train, X_test)

clf1 = pickle.load(open("models/SQLi_2_lr_newton-cg_10.sav", 'rb'))

test_pred_logreg = clf1.predict_proba(X_test)[:,1]
test_pred_logreg_o = clf1.predict(X_test)
print(f1_score(y_test, test_pred_logreg_o))

import tensorflow as tf
from tensorflow import keras
model = keras.models.load_model("models/SQLi_2_nn_0.1_64_sigmoid.h5")

X_test_tf = tf.convert_to_tensor(X_test)
test_pred_nn = model.predict(X_test_tf)[:,1]

fpr_logreg, tpr_logreg, _ = roc_curve(y_test,  test_pred_logreg)
auc_logreg = roc_auc_score(y_test, test_pred_logreg)
fpr_nn, tpr_nn, _ = roc_curve(y_test,  test_pred_nn)
auc_nn = roc_auc_score(y_test, test_pred_nn)

df_train = pd.read_csv("payloads/SQLi_2/skip_gram/train.csv")
X_train = df_train.loc[:, df_train.columns != 'Label']

df_test = pd.read_csv("payloads/united/plain/test.csv")
X_test = df_test.loc[:, df_test.columns != 'Label']
y_test = df_test['Label']

X_test = skip_gram(X_train, X_test, "SQLi_2")

clf = pickle.load(open("models/SQLi_2_gbc_0.1_1000_8.sav", 'rb'))

test_pred_grad = clf.predict_proba(X_test)[:,1]
test_pred_grad_o = clf.predict(X_test)
print(f1_score(y_test, test_pred_grad_o))

clf2 = pickle.load(open("models/SQLi_2_rf_1_100.sav", 'rb'))

test_pred_rf = clf2.predict_proba(X_test)[:,1]
test_pred_rf_o = clf2.predict(X_test)
print(f1_score(y_test, test_pred_rf_o))

clf3 = pickle.load(open("models/SQLi_2_svm_poly_10.sav", 'rb'))

test_pred_svm = clf3.predict_proba(X_test)[:,1]
test_pred_svm_o = clf3.predict(X_test)
print(f1_score(y_test, test_pred_svm_o))


fpr_grad, tpr_grad, _ = roc_curve(y_test,  test_pred_grad)
auc_grad = roc_auc_score(y_test, test_pred_grad)
fpr_rf, tpr_rf, _ = roc_curve(y_test,  test_pred_rf)
auc_rf = roc_auc_score(y_test, test_pred_rf)
fpr_svm, tpr_svm, _ = roc_curve(y_test,  test_pred_svm)
auc_svm = roc_auc_score(y_test, test_pred_svm)


fig = plt.figure()
ax = plt.subplot(111)

#create ROC curve
ax.plot(fpr_logreg,tpr_logreg,"y",label="LR AUC="+str(round(auc_logreg, 3)))
ax.plot(fpr_svm,tpr_svm, "b",label="SVM AUC="+str(round(auc_svm, 3)))
ax.plot(fpr_rf,tpr_rf,"m",label="RF AUC="+str(round(auc_rf, 3)))
ax.plot(fpr_grad,tpr_grad,"r",label="GB AUC="+str(round(auc_grad, 3)))
ax.plot(fpr_nn,tpr_nn, "g",label="NN AUC="+str(round(auc_nn, 3)))

plt.xscale("log")
#plt.yscale("log")
plt.ylabel('True Positive Rate', fontsize = 15)
plt.xlabel('False Positive Rate', fontsize = 15)
plt.xlim([0.01, 1])
plt.title("Train: SQLi2 Test: United", fontsize = 15)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 15})

plt.savefig('roc_curves/2_u.png', dpi=2000, bbox_inches='tight')
plt.show()
