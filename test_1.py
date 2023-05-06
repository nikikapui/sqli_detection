from termios import TIOCPKT_DOSTOP

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
from gensim.models import Word2Vec
import multiprocessing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import backend as K

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import random
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

X_FOLD = 2

PREPROC_TD_IDF = 0
PREPROC_KEYWORD = 1
PREPROC_SKIP_GRAM = 2

MODEL_LOGREG = 0
MODEL_SVM = 1
MODEL_RANDOM_FOREST = 2
MODEL_GRAD_BOOST = 3
MODEL_NEURAL_NETWORK = 4

def tf_idf_vectors(data):
    vectorizer = TfidfVectorizer(token_pattern = r'[a-zA-Z]+')
    vectors = vectorizer.fit_transform(df['Sentence'].tolist())
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.toarray()
    df_2 = pd.DataFrame(dense, columns=feature_names)
    df_2["Label"] = df["Label"]
    #print("TF-IDF Vectors")
    return df_2

def keyword_weights(data):
    result_nk = []
    result_sk = []
    result_len = []
    for element in data['Sentence']:
        keywords = {"union": 9, "truncate": 9, "xp_cmdshell": 9, "load_file": 9, "outfile": 9, "dumpfile": 9, "exec": 9, "select": 3, "update": 3, "insert": 3, "delete": 3, "count": 3, "where": 3, "group": 3, "order": 3, "drop": 3, "table": 3, "master": 3, "net": 3, "and": 1, "or": 1, "by": 1, "from": 1, "into": 1}
        no_of_keywords = 0
        sum_weight_of_keywords = 0
        element = element.lower()
        for keyword, weight in keywords.items():
            element_temp = element
            while element.find(keyword) != -1:
                element = element[(element.find(keyword)+len(keyword)):]
                no_of_keywords += 1
                sum_weight_of_keywords += weight
            element = element_temp
        result_nk.append(no_of_keywords)
        result_sk.append(sum_weight_of_keywords)
        result_len.append(len(element))
    return pd.concat([pd.Series(result_nk, name="Number of Keywords"), pd.Series(result_sk, name = "Sum Weight of Keywords"), pd.Series(result_len, name = "Length"), data["Label"]], axis = 1)

def skip_gram(data):
    lista = data['Sentence'].values.tolist()
    values = list()
    for sentence in lista:
        values.append(sentence.split(" "))
    EMB_DIM = 5
    w2v = gensim.models.Word2Vec(values, vector_size=EMB_DIM, min_count = 1, negative = 15, workers=multiprocessing.cpu_count(), window = 5, sg=1)
    values_vec = list()
    for sentence in values:
        sen_vec = np.array([])
        for word in sentence:
            sen_vec = np.concatenate((sen_vec, w2v.wv[word]), axis = None)
        values_vec.append(sen_vec.tolist())
    list_len = [len(i) for i in values_vec]
    maximum = max(list_len)
    for sentence in values_vec:
        diff = maximum - len(sentence)
        while diff > 0:
            sentence.append(-1)
            diff -= 1
    df_temp = pd.DataFrame(values_vec)
    df_temp['Label'] = data['Label']
    return df_temp

def LogReg(X_train, X_valid, X_test, y_train, y_valid, y_test):
    solvers = ["newton-cg", "lbfgs"]
    c = [0.1, 1, 10]
    result = []
    for i, solver in enumerate(solvers):
        for c_ in c:
            tmp_tr, tmp_va, tmp_te = [0, 0, 0]
            for fold in range(X_FOLD):
                random.seed(fold)
                clf = LogisticRegression(C = c_, solver = solver)
                clf.fit(X_train, y_train)
                train_pred = clf.predict(X_train)
                f1_train = f1_score(y_train, train_pred)
                valid_pred = clf.predict(X_valid)
                f1_valid = f1_score(y_valid, valid_pred)
                test_pred = clf.predict(X_test)
                f1_test = f1_score(y_test, test_pred)
                tmp_tr = tmp_tr + f1_train / X_FOLD
                tmp_va = tmp_va + f1_valid / X_FOLD
                tmp_te = tmp_te + f1_test / X_FOLD
            result.append([tmp_tr, tmp_va, tmp_te, "LG" + " " + solver + " " + str(c_)])
    return sorted(result, key=lambda x:x[1], reverse=True)[0]

def SVM(X_train, X_valid, X_test, y_train, y_valid, y_test):
    kernels = ["linear", "poly", "rbf"]
    c = [0.1, 1, 10]
    result = []
    for kernel in kernels:
        for c_ in c:
            tmp_tr, tmp_va, tmp_te = [0, 0, 0]
            for fold in range(X_FOLD):
                random.seed(fold)
                clf = SVC(C=c_, kernel = kernel)
                clf.fit(X_train, y_train)
                train_pred = clf.predict(X_train)
                f1_train = f1_score(y_train, train_pred)
                valid_pred = clf.predict(X_valid)
                f1_valid = f1_score(y_valid, valid_pred)
                test_pred = clf.predict(X_test)
                f1_test = f1_score(y_test, test_pred)
                tmp_tr = tmp_tr + f1_train / X_FOLD
                tmp_va = tmp_va + f1_valid / X_FOLD
                tmp_te = tmp_te + f1_test / X_FOLD
            result.append([tmp_tr, tmp_va, tmp_te, "SVM" + " " + kernel + " " + str(c_)])
    return sorted(result, key=lambda x:x[1], reverse=True)[0]

def RandomForest(X_train, X_valid, X_test, y_train, y_valid, y_test):
    max_features = [2**i for i in range(int(min(5.0, np.log(len(X_train.columns))) / np.log(2)))]
    n_estimators = [10, 100, 1000]
    result = []
    for max_feature in max_features:
        for n_estimator in n_estimators:
            tmp_tr, tmp_va, tmp_te = [0, 0, 0]
            for fold in range(X_FOLD):
                random.seed(fold)
                clf = RandomForestClassifier(max_features=max_feature, n_estimators=n_estimator)
                clf.fit(X_train, y_train)
                train_pred = clf.predict(X_train)
                f1_train = f1_score(y_train, train_pred)
                valid_pred = clf.predict(X_valid)
                f1_valid = f1_score(y_valid, valid_pred)
                test_pred = clf.predict(X_test)
                f1_test = f1_score(y_test, test_pred)
                tmp_tr = tmp_tr + f1_train / X_FOLD
                tmp_va = tmp_va + f1_valid / X_FOLD
                tmp_te = tmp_te + f1_test / X_FOLD
            result.append([tmp_tr, tmp_va, tmp_te, "RF" + " " + str(max_feature) + " " + str(n_estimator)])
    return sorted(result, key=lambda x:x[1], reverse=True)[0]

def GBC(X_train, X_valid, X_test, y_train, y_valid, y_test):
    learning_rates = [0.001, 0.01, 0.1]
    n_estimators = [10, 100, 1000]
    max_depths = [2, 4, 8]
    result = []
    for learning_rate in learning_rates:
        for n_estimator in n_estimators:
            for max_depth in max_depths:
                tmp_tr, tmp_va, tmp_te = [0, 0, 0]
                for fold in range(X_FOLD):
                    random.seed(fold)
                    clf = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimator, max_depth=max_depth)
                    clf.fit(X_train, y_train)
                    train_pred = clf.predict(X_train)
                    f1_train = f1_score(y_train, train_pred)
                    valid_pred = clf.predict(X_valid)
                    f1_valid = f1_score(y_valid, valid_pred)
                    test_pred = clf.predict(X_test)
                    f1_test = f1_score(y_test, test_pred)
                    tmp_tr = tmp_tr + f1_train / X_FOLD
                    tmp_va = tmp_va + f1_valid / X_FOLD
                    tmp_te = tmp_te + f1_test / X_FOLD
                result.append([tmp_tr, tmp_va, tmp_te, "GB" + " " + str(learning_rate) + " " + str(n_estimator) + " " + " " + str(max_depth)])
    return sorted(result, key=lambda x:x[1], reverse=True)[0]

def NeuralNetwork(X_train, X_valid, X_test, y_train, y_valid, y_test):
    # Parameters
    learning_rates = [0.001, 0.01, 0.1]
    hidden_layers = [64, 128]
    activation_functions = ["relu", "sigmoid"]
    result = []

    for learning_rate in learning_rates:
        for hidden_layer in hidden_layers:
            for activation_function_hidden in activation_functions:
                for activation_functin_output in activation_functions:
                    tmp_tr, tmp_va, tmp_te = [0, 0, 0]
                    for fold in range(X_FOLD):
                        random.seed(fold)
                        # Model
                        inputs = Input(shape=(X_train.shape[1],))
                        # Hidden layer
                        hidden_output = Dense(hidden_layer, activation=activation_function_hidden)(inputs)
                        # Softmax
                        predictions = Dense(2, activation=activation_functin_output)(hidden_output)
                        # Full model
                        model = Model(inputs=inputs, outputs=predictions)
                        # Optimizer
                        optimizer = Adam(learning_rate=learning_rate)
                        # Compilation and fitting
                        model.compile(optimizer=optimizer,
                                    loss='sparse_categorical_crossentropy', # we use this cross entropy variant as the input is not
                                                                            # one-hot encoded
                                    metrics=['accuracy'])

                        X_train = tf.convert_to_tensor(X_train)
                        y_train = tf.convert_to_tensor(y_train)
                        X_test = tf.convert_to_tensor(X_test)
                        y_test = tf.convert_to_tensor(y_test)

                        history = model.fit(x=X_train,y=y_train,epochs=20,batch_size=200)

                        predict_train = model.predict(X_train)
                        train_pred = np.argmax(predict_train,axis=1)
                        f1_train = f1_score(y_train, train_pred)

                        predict_valid = model.predict(X_valid)
                        valid_pred = np.argmax(predict_valid,axis=1)
                        f1_valid = f1_score(y_valid, valid_pred)

                        predict_test = model.predict(X_test)
                        test_pred = np.argmax(predict_test,axis=1)
                        f1_test = f1_score(y_test, test_pred)

                        tmp_tr = tmp_tr + f1_train / X_FOLD
                        tmp_va = tmp_va + f1_valid / X_FOLD
                        tmp_te = tmp_te + f1_test / X_FOLD

                        result.append([tmp_tr, tmp_va, tmp_te, "NN " + str(learning_rate) + " " + str(hidden_layer) + " " + activation_function_hidden + " " + activation_functin_output])
    
    return sorted(result, key=lambda x:x[1], reverse=True)[0]
    

def framework(dataset, preprocessing, model):
    if preprocessing == 0:
        dataset = tf_idf_vectors(dataset)
    elif preprocessing == 1:
        dataset = keyword_weights(dataset)
    elif preprocessing == 2:
        dataset = skip_gram(dataset)
    else: 
        return print("Nincs ilyen preprocesszálás!")
    X_train, X_test, y_train, y_test = train_test_split(dataset.loc[:, dataset.columns != 'Label'], dataset['Label'], stratify=dataset['Label'], test_size = 0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify = y_train, test_size = 0.125)
    if model == 0:
        return LogReg(X_train, X_valid, X_test, y_train, y_valid, y_test)
    elif model == 1:
        return SVM(X_train, X_valid, X_test, y_train, y_valid, y_test)
    elif model == 2:
        return RandomForest(X_train, X_valid, X_test, y_train, y_valid, y_test)
    elif model == 3:
        return GBC(X_train, X_valid, X_test, y_train, y_valid, y_test)
    elif model == 4:
        return NeuralNetwork(X_train, X_valid, X_test, y_train, y_valid, y_test)
    else: 
        print("Nincs ilyen modell!")

dataset = "united"
#dataset = ["united", "SQLi_1", "SQLi_2"]
preprocessing = (PREPROC_TD_IDF, PREPROC_KEYWORD, PREPROC_SKIP_GRAM)
models = (MODEL_LOGREG, MODEL_SVM, MODEL_RANDOM_FOREST, MODEL_GRAD_BOOST)

print(str(dataset))
for preproc in preprocessing:
    print("Preproc: ", str(preproc))
    for model in models:
        df = pd.read_csv(dataset + ".csv", delimiter=";")
        print(framework(df, preproc, model))
