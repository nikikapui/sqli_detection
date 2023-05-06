from sklearn.metrics import f1_score, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

print("united.csv")

df_test = pd.read_csv("payloads/united/skip_gram/test.csv")
X_test = df_test.loc[:, df_test.columns != 'Label']
y_test = df_test['Label']

clf = pickle.load(open("models/united_gbc_0.01_1000_2.sav", 'rb'))

test_pred_grad = clf.predict_proba(X_test)[:,1]
test_pred_grad_o = clf.predict(X_test)
print(f1_score(y_test, test_pred_grad_o))

clf1 = pickle.load(open("models/united_lr_newton-cg_0.1.sav", 'rb'))

test_pred_logreg = clf1.predict_proba(X_test)[:,1]
test_pred_logreg_o = clf1.predict(X_test)
print(f1_score(y_test, test_pred_logreg_o))

clf2 = pickle.load(open("models/united_rf_32_10.sav", 'rb'))

test_pred_rf = clf2.predict_proba(X_test)[:,1]
test_pred_rf_o = clf2.predict(X_test)
print(f1_score(y_test, test_pred_rf_o))

clf3 = pickle.load(open("models/united_svm_linear_10.sav", 'rb'))

test_pred_svm = clf3.predict_proba(X_test)[:,1]
test_pred_svm_o = clf3.predict(X_test)
print(f1_score(y_test, test_pred_svm_o))

import tensorflow as tf
from tensorflow import keras
model = keras.models.load_model("models/united_nn_0.001_64_sigmoid.h5")

X_test_tf = tf.convert_to_tensor(X_test)
#y_test_tf = tf.convert_to_tensor(y_test)
test_pred_nn = model.predict(X_test_tf)[:,1]

predict_test = model.predict(X_test_tf)
test_pred = np.argmax(predict_test,axis=1)
f1_test = f1_score(y_test, test_pred)

fpr_grad, tpr_grad, _ = roc_curve(y_test,  test_pred_grad)
auc_grad = roc_auc_score(y_test, test_pred_grad)
fpr_logreg, tpr_logreg, _ = roc_curve(y_test,  test_pred_logreg)
auc_logreg = roc_auc_score(y_test, test_pred_logreg)
fpr_rf, tpr_rf, _ = roc_curve(y_test,  test_pred_rf)
auc_rf = roc_auc_score(y_test, test_pred_rf)
fpr_svm, tpr_svm, _ = roc_curve(y_test,  test_pred_svm)
auc_svm = roc_auc_score(y_test, test_pred_svm)
fpr_nn, tpr_nn, _ = roc_curve(y_test,  test_pred_nn)
auc_nn = roc_auc_score(y_test, test_pred_nn)

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
plt.title("United", fontsize = 15)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 15})

plt.savefig('roc_curves/united.png', dpi=2000, bbox_inches='tight')
plt.show()
