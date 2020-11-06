import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
import seaborn as sns
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

import math
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import os

import ipywidgets as wg
from IPython.display import display
from ipywidgets import Layout
from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import KBinsDiscretizer
# from pyod.models.knn import KNN
# from pyod.models.iforest import IForest
# from pyod.models.pca import PCA as PCA_od
from sklearn import cluster
from scipy import stats
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr

import sys
from sklearn.pipeline import Pipeline
from sklearn import metrics
import calendar
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from flask import Flask, request, jsonify, render_template
from flask import jsonify, request
from flask_restful import Resource
import boto3
from botocore.exceptions import ClientError
import logging
from flask import current_app
import json
import pickle
import random, string
import csv

from nltk.corpus import stopwords




def id_column_detector(data):
    len_samples = data.shape[0]
    id_col = []
    for i in list(data.columns):
        if data[i].nunique() == len_samples:
            id_col.append(i)
    return id_col

def id_column_detector2(data):
    len_samples = data.shape[0]
    id_col = []

    for i in list(data.columns):
        if data[i].dtype in ['int64','float64']:
            if len(data[i].unique()) == len_samples:
                # we extract column and sort it
                features = data[i].sort_values()
                # no we subtract i+1-th value from i-th (calculating increments)
                increments = features.diff()[1:]
                if sum(np.abs(increments-1) < 1e-7) == len_samples-1:
                    id_col.append(i)
    return id_col

def convert_to_datetime(data):
    import pandas as pd

    data = data.apply(lambda col: pd.to_datetime(col, errors='ignore')
              if col.dtypes == object
              else col,
              axis=0)
    print(data.dtypes)
    dat_col_df = data.select_dtypes(include=['datetime64[ns, UTC]'])
    date_col_list = list(dat_col_df.columns)


    # Getting id column name as a string
    date_col = ""
    # traverse in the string
    for ele in date_col_list:
        date_col += ele
    if len(date_col_list)>0:
        #sort dataframe using date
        data = data.sort_values(by=date_col)

    return data,date_col

def fetch_date_features(data,date_col):
    dayofweek = data[date_col].dt.dayofweek
    dayNumber = data[date_col].dt.day
    weekNumber = data[date_col].dt.week
    monthNumber = data[date_col].dt.month
    yearNumber = data[date_col].dt.year

    data['dayofweek']=dayofweek
    data['dayNumber']=dayNumber
    data['weekNumber']=weekNumber
    data['monthNumber']=monthNumber
    data['yearNumber']=yearNumber

    #Dropping id column
    data.drop([date_col], axis=1, inplace=True)

    return data

def separate_target(data,features,target):
    X=[]
    Y=[]
    for i in features:
        if i!=target:
            X.append(i)
        else:
            Y.append(i)
    data_pr = data[X] #Independent data
    target_pr = data[Y] #Dependent Data
    return data_pr,target_pr

def get_categorical_features(data):
    cat_feat = data.select_dtypes(include=['O']).columns.values
    return cat_feat

def get_numerical_features(data):
    numerical_features = data.select_dtypes(include=[np.number]).columns.values
    return numerical_features


def classify_or_regression(target_pr):
    c1 = target_pr.dtypes == 'int64'
    c11 = str(c1).split('\n')
    c111 = c11[0].split('    ')
    print(str(c111[1]))


    c2 = target_pr.nunique() <= 20
    c22 = str(c2).split('\n')
    c222 = c22[0].split('    ')
    print(str(c222[1]))



    c3 = target_pr.dtypes.name in ['object', 'bool', 'category']
    c333 = str(c3)
    print(c3)


    c4 = target_pr.dtypes == 'float64'
    c44 = str(c4).split('\n')
    c444 = c44[0].split('    ')
    print(str(c444[1]))

    if (c111[1] == 'True' or c444[1]=='True') and c222[1]=='False' and c333=='False':
        ml_usecase ='regression'
    else:
        ml_usecase ='classification'
    return ml_usecase

def categorical_feature_imputer(data_pr,cat_feat):
    # Introducing 'unknown' variable for missing values
    notnull_cat_df = data_pr[list(cat_feat)].fillna('unknown')
    return notnull_cat_df

# Filling the numerical values with mean for Numerical variables
def numerical_feature_imputer(data_pr,cleaned_num_feat):
    for i in cleaned_num_feat:
        mean_val = data_pr[i].mean()
        data_pr[i].fillna(mean_val, inplace=True)

    return data_pr

def decontracted(phrase):
    # specific
    phrase = re.sub(r"!", " ", phrase)
    phrase = re.sub(r" ", "_", phrase)
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# Clean data from notnull_cat_df and insert to data_pr
from tqdm import tqdm
def clean_text_data(data_pr,notnull_cat_df,not_null_cat_feat_lst):
    for i in not_null_cat_feat_lst:
        cleaned = []
        for titles in tqdm(notnull_cat_df[i]):
            title = decontracted(str(titles))
            title = title.replace('\\r', ' ')
            title = title.replace('\\"', ' ')
            title = title.replace('\\n', ' ')
            title = re.sub('[^A-Za-z0-9]+', ' ', title)
            title = ' '.join(f for f in title.split() if f not in stop_words)
            title = title.replace(" ", "_")
            title = re.sub(r'^\s*$', 'unknown', title) #If still any places are empty due to removal of stopwords
            cleaned.append(title.lower().strip())

        data_pr['clean'+i] = cleaned
    return data_pr

#Combined both in function
def vectorize_variables(X_train,X_test,cleaned_cat_feat):
    from sklearn.preprocessing import LabelEncoder,StandardScaler

    le = LabelEncoder()
    X_train[cleaned_cat_feat] = X_train[cleaned_cat_feat].apply(le.fit_transform)
    X_test[cleaned_cat_feat] = X_test[cleaned_cat_feat].apply(le.fit_transform)

    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    vectorizer = StandardScaler()
    X_train_cols = list(X_train.columns)
    X_test_cols = list(X_test.columns)
    X_train[X_train_cols] = vectorizer.fit_transform(X_train[X_train_cols])
    X_test[X_test_cols] = vectorizer.transform(X_test[X_test_cols])

    return X_train,X_test

# After Text preprocessing do this step
def fetch_text_features(data_pr,cleaned_cat_feat):
    import random
    cleaned_text_feat = []
    for i in cleaned_cat_feat:
        n = random.randint(0, 20)
#         print(data_pr[i][n])
        tokn = str(data_pr[i][n]).split('_')

        #Need to decide this later
        if len(tokn)>5:
            cleaned_text_feat.append(i)
    return cleaned_text_feat

def fetch_categorical_features(cleaned_cat_feat,cleaned_text_feat):
    for i in cleaned_text_feat:
        cleaned_cat_feat.remove(i)
    return cleaned_cat_feat

def get_word2tfidf(data_pr,cleaned_text_feat):
    from sklearn.feature_extraction.text import TfidfVectorizer
    # merge texts
   # cleaned_text_feat = ['cleanmake','cleanmodel']
    questions = []
    for i in cleaned_text_feat:
        questions.extend(list(data_pr[i]))
    tfidf = TfidfVectorizer(lowercase=False, )
    tfidf.fit_transform(questions)

    # dict key:word and value:tf-idf score
    word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

    return word2tfidf

#do this for each feature and store them in dataframe
def find_tfidf_w2v(data_pr, word2tfidf, cleaned_text_feat):
    from tqdm import tqdm
    import spacy
    # en_vectors_web_lg, which includes over 1 million unique vectors.
    nlp = spacy.load('en_core_web_sm')

    for i in cleaned_text_feat:
        vecs1 = []
        # https://github.com/noamraph/tqdm
        # tqdm is used to print the progress bar
        for qu1 in tqdm(list(data_pr[i])):
            doc1 = nlp(qu1)
            # 384 is the number of dimensions of vectors
            mean_vec1 = np.zeros([len(doc1) , len(doc1[0].vector)])
            for word1 in doc1:
                # word2vec
                vec1 = word1.vector
                # fetch df score
                try:
                    idf = word2tfidf[str(word1)]
                except:
                    idf = 0
                # compute final vec
                mean_vec1 += vec1 * idf
            mean_vec1 = mean_vec1.mean(axis=0)
            vecs1.append(mean_vec1)
        data_pr['vect_'+i] = list(vecs1)
    return data_pr

def create_dataframe_textvectored(id_col,data_pr,sequence_text_feat):
    import pandas as pd
    from functools import reduce
    df_list = [data_pr]
    id=data_pr.index
    for i in sequence_text_feat:
        text_vect_df = pd.DataFrame(data_pr[i].values.tolist(), index= id)
        text_vect_df[id_col]=data_pr[id_col]

        df_list.append(text_vect_df)

    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=[id_col],how='outer'), df_list)

    return df_merged

def compare_all_classification_models(X_train,X_test,y_train,y_test,verbose = True):
    #import sklearn dependencies
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import RidgeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.model_selection import StratifiedKFold
    folds_shuffle_param = False
    import sklearn.metrics as metrics

    seed= 10
    n_jobs_param = -1 #To use all the cores

    available_estimators = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada',
                            'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost']





    lr = LogisticRegression(random_state=seed) #dont add n_jobs_param here. It slows doesn Logistic Regression somehow.
    knn = KNeighborsClassifier(n_jobs=n_jobs_param)
    nb = GaussianNB()
    dt = DecisionTreeClassifier(random_state=seed)
    svm = SGDClassifier(max_iter=1000, tol=0.001, random_state=seed, n_jobs=n_jobs_param)
    rbfsvm = SVC(gamma='auto', C=1, probability=True, kernel='rbf', random_state=seed)
    gpc = GaussianProcessClassifier(random_state=seed, n_jobs=n_jobs_param)
    mlp = MLPClassifier(max_iter=500, random_state=seed)
    ridge = RidgeClassifier(random_state=seed)
    rf = RandomForestClassifier(n_estimators=10, random_state=seed, n_jobs=n_jobs_param)
    qda = QuadraticDiscriminantAnalysis()
    ada = AdaBoostClassifier(random_state=seed)
    gbc = GradientBoostingClassifier(random_state=seed)
    lda = LinearDiscriminantAnalysis()
    et = ExtraTreesClassifier(random_state=seed, n_jobs=n_jobs_param)
    xgboost = XGBClassifier(random_state=seed, verbosity=0, n_jobs=n_jobs_param)
    catboost = CatBoostClassifier(random_state=seed, silent = True, thread_count=n_jobs_param)


    model_dict = { 'Logistic Regression' : 'lr',
               'Linear Discriminant Analysis' : 'lda',
               'Ridge Classifier' : 'ridge',
               'Extreme Gradient Boosting' : 'xgboost',
               'Ada Boost Classifier' : 'ada',
               'CatBoost Classifier' : 'catboost',
               'Gradient Boosting Classifier' : 'gbc',
               'Random Forest Classifier' : 'rf',
               'Naive Bayes' : 'nb',
               'Extra Trees Classifier' : 'et',
               'Decision Tree Classifier' : 'dt',
               'K Neighbors Classifier' : 'knn',
               'Quadratic Discriminant Analysis' : 'qda',
               'SVM - Linear Kernel' : 'svm',
               'Gaussian Process Classifier' : 'gpc',
               'MLP Classifier' : 'mlp',
               'SVM - Radial Kernel' : 'rbfsvm'}

    model_library = [lr, knn, nb, dt, svm, rbfsvm, gpc, mlp, ridge, rf, qda, ada, gbc, lda, et, xgboost, catboost]

    model_names = ['Logistic Regression',
                   'K Neighbors Classifier',
                   'Naive Bayes',
                   'Decision Tree Classifier',
                   'SVM - Linear Kernel',
                   'SVM - Radial Kernel',
                   'Gaussian Process Classifier',
                   'MLP Classifier',
                   'Ridge Classifier',
                   'Random Forest Classifier',
                   'Quadratic Discriminant Analysis',
                   'Ada Boost Classifier',
                   'Gradient Boosting Classifier',
                   'Linear Discriminant Analysis',
                   'Extra Trees Classifier',
                   'Extreme Gradient Boosting',
                   'CatBoost Classifier']


    # Store metric values for all models
    acc_list = []
    auc_list =[]
    recall_list =[]
    precision_list =[]
    f1_list =[]
    kappa_list =[]
    mcc_list =[]
    time_list = []

    name_counter = 0
    for model in model_library:
        if hasattr(model, 'predict_proba'):
            time_start=time.time()
            print('Training model on if condition : ',model_names[name_counter])
            model.fit(X_train,y_train)
            time_end=time.time()
            pred_prob = model.predict_proba(X_test)
            pred_prob = pred_prob[:,1]
            pred_ = model.predict(X_test)
            sca = metrics.accuracy_score(y_test,pred_)
            #if y_train.value_counts().count() > 2  implies Multiclass
            if y_train.value_counts().count() > 2:
                sc = 0
                recall = metrics.recall_score(y_test,pred_, average='macro')
                precision = metrics.precision_score(y_test,pred_, average = 'weighted')
                f1 = metrics.f1_score(y_test,pred_, average='weighted')
            else:
                try:
                    sc = metrics.roc_auc_score(y_test,pred_prob)
                except:
                    sc = 0
                    print("model has no predict_proba attribute. AUC set to 0.00")
                recall = metrics.recall_score(y_test,pred_)
                precision = metrics.precision_score(y_test,pred_)
                f1 = metrics.f1_score(y_test,pred_)
        else:
            time_start=time.time()
            print('Training model on else condition : ',model_names[name_counter])
            model.fit(X_train,y_train)
            time_end=time.time()
            print("model has no predict_proba attribute. pred_prob set to 0.00")
            pred_prob = 0.00
            pred_ = model.predict(X_test)
            sca = metrics.accuracy_score(y_test,pred_)
            if y_train.value_counts().count() > 2:
                sc = 0
                recall = metrics.recall_score(y_test,pred_, average='macro')
                precision = metrics.precision_score(y_test,pred_, average = 'weighted')
                f1 = metrics.f1_score(y_test,pred_, average='weighted')
            else:
                try:
                    sc = metrics.roc_auc_score(y_test,pred_prob)
                except:
                    sc = 0
                    print("model has no predict_proba attribute. AUC set to 0.00")
                recall = metrics.recall_score(y_test,pred_)
                precision = metrics.precision_score(y_test,pred_)
                f1 = metrics.f1_score(y_test,pred_)
        mcc = metrics.matthews_corrcoef(y_test,pred_)
        kappa = metrics.cohen_kappa_score(y_test,pred_)
        time_taken = time_end - time_start

        acc_list.append(sca)
        auc_list.append(sc)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
        kappa_list.append(kappa)
        mcc_list.append(mcc)
        time_list.append(time_taken)
        name_counter = name_counter+1

    print("Creating metrics dataframe")

    compare_models_ = pd.DataFrame({'Model':model_names, 'Accuracy':acc_list, 'AUC':auc_list,
                   'Recall':recall_list, 'Prec.':precision_list,
                   'F1':f1_list, 'Kappa': kappa_list, 'MCC':mcc_list, 'Time Taken' :time_list})

    return compare_models_
def model_name_assigner(estimator, ml_usecase):
    model_dict_regr = {'lr':'Linear Regression',
                   'lasso':'Lasso Regression',
                   'ridge':'Ridge Regression',
                   'en': 'Elastic Net',
                   'lar' : 'Least Angle Regression',
                   'llar':'Lasso Least Angle Regression',
                   'omp':'Orthogonal Matching Pursuit',
                   'br':'Bayesian Ridge',
                   'ard':'Automatic Relevance Determination',
                   'par':'Passive Aggressive Regressor',
                   'ransac':'Random Sample Consensus',
                   'tr':'TheilSen Regressor',
                   'huber':'Huber Regressor',
                   'kr':'Kernel Ridge',
                   'svm':'Support Vector Machine',
                   'knn':'K Neighbors Regressor',
                   'dt':'Decision Tree',
                   'rf':'Random Forest',
                   'et':'Extra Trees Regressor',
                   'ada':'AdaBoost Regressor',
                   'gbr':'Gradient Boosting Regressor',
                   'mlp':'Multi Level Perceptron',
                   'xgboost':'Extreme Gradient Boosting',
                   'catboost':'CatBoost Regressor'}
    model_dict_clf = {'lr':'Logistic Regression',
               'lda':'Linear Discriminant Analysis',
               'ridge':'Ridge Classifier',
               'xgboost': 'Extreme Gradient Boosting',
               'ada':'Ada Boost Classifier',
               'catboost':'CatBoost Classifier',
               'gbc':'Gradient Boosting Classifier',
               'rf':'Random Forest Classifier',
               'nb':'Naive Bayes',
               'et':'Extra Trees Classifier',
               'dt':'Decision Tree Classifier',
               'knn':'K Neighbors Classifier',
               'qda':'Quadratic Discriminant Analysis',
               'svm':'SVM - Linear Kernel',
               'gpc':'Gaussian Process Classifier',
               'mlp':'MLP Classifier',
               'rbfsvm':'SVM - Radial Kernel'}
    if ml_usecase == 'regression':
        result = model_dict_regr[estimator]
    else:
        result = model_dict_clf[estimator]
    return result
def compare_all_regression_models(X_train,X_test,y_train,y_test,verbose = True):

    #import sklearn dependencies
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    from sklearn.linear_model import Lars
    from sklearn.linear_model import LassoLars
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from sklearn.linear_model import BayesianRidge
    from sklearn.linear_model import ARDRegression
    from sklearn.linear_model import PassiveAggressiveRegressor
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import TheilSenRegressor
    from sklearn.linear_model import HuberRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.model_selection import KFold
    folds_shuffle_param = False
    import sklearn.metrics as metrics

    seed= 10
    n_jobs_param = -1 #To use all the cores

    available_estimators = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par',
                            'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr',
                            'mlp', 'xgboost', 'catboost']

    #creating model object
    lr = LinearRegression(n_jobs=n_jobs_param)
    lasso = Lasso(random_state=seed)
    ridge = Ridge(random_state=seed)
    en = ElasticNet(random_state=seed)
    lar = Lars()
    llar = LassoLars()
    omp = OrthogonalMatchingPursuit()
    br = BayesianRidge()
    ard = ARDRegression()
    par = PassiveAggressiveRegressor(random_state=seed)
    ransac = RANSACRegressor(min_samples=0.5, random_state=seed)
    tr = TheilSenRegressor(random_state=seed, n_jobs=n_jobs_param)
    huber = HuberRegressor()
    kr = KernelRidge()
    svm = SVR()
    knn = KNeighborsRegressor(n_jobs=n_jobs_param)
    dt = DecisionTreeRegressor(random_state=seed)
    rf = RandomForestRegressor(random_state=seed, n_jobs=n_jobs_param)
    et = ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs_param)
    ada = AdaBoostRegressor(random_state=seed)
    gbr = GradientBoostingRegressor(random_state=seed)
    mlp = MLPRegressor(random_state=seed)
    xgboost = XGBRegressor(random_state=seed, n_jobs=n_jobs_param, verbosity=0)
    catboost = CatBoostRegressor(random_state=seed, silent = True, thread_count=n_jobs_param)


    model_dict = {'Linear Regression' : 'lr',
                   'Lasso Regression' : 'lasso',
                   'Ridge Regression' : 'ridge',
                   'Elastic Net' : 'en',
                   'Least Angle Regression' : 'lar',
                   'Lasso Least Angle Regression' : 'llar',
                   'Orthogonal Matching Pursuit' : 'omp',
                   'Bayesian Ridge' : 'br',
                   'Automatic Relevance Determination' : 'ard',
                   'Passive Aggressive Regressor' : 'par',
                   'Random Sample Consensus' : 'ransac',
                   'TheilSen Regressor' : 'tr',
                   'Huber Regressor' : 'huber',
                   'Kernel Ridge' : 'kr',
                   'Support Vector Machine' : 'svm',
                   'K Neighbors Regressor' : 'knn',
                   'Decision Tree' : 'dt',
                   'Random Forest' : 'rf',
                   'Extra Trees Regressor' : 'et',
                   'AdaBoost Regressor' : 'ada',
                   'Gradient Boosting Regressor' : 'gbr',
                   'Multi Level Perceptron' : 'mlp',
                   'Extreme Gradient Boosting' : 'xgboost',
                   'CatBoost Regressor' : 'catboost'}

    model_library = [lr, lasso, ridge, en, lar, llar, omp, br, par, ransac,  huber,
                         svm, knn, dt, rf, et, ada, gbr, xgboost, catboost]


    model_names = ['Linear Regression',
                       'Lasso Regression',
                       'Ridge Regression',
                       'Elastic Net',
                       'Least Angle Regression',
                       'Lasso Least Angle Regression',
                       'Orthogonal Matching Pursuit',
                       'Bayesian Ridge',
                       'Passive Aggressive Regressor',
                       'Random Sample Consensus',

                       'Huber Regressor',
                       'Support Vector Machine',
                       'K Neighbors Regressor',
                       'Decision Tree',
                       'Random Forest',
                       'Extra Trees Regressor',
                       'AdaBoost Regressor',
                       'Gradient Boosting Regressor',
                       'Extreme Gradient Boosting',
                       'CatBoost Regressor']


    # Store metric values for all models
    score_mae = []
    score_mse =[]
    score_rmse =[]
    score_rmsle =[]
    score_r2 =[]
    time_list= []

    name_counter = 0
    model_store = []
    for model in model_library:
        print("Initializing " + str(model_names[name_counter]))
        fold_num = 1
        time_start=time.time()
        print("Fitting Model")
        model.fit(X_train,y_train)
        print("Evaluating Metrics")
        time_end=time.time()

        pred_ = model.predict(X_test)


        print("Compiling Metrics")
        mae = metrics.mean_absolute_error(y_test,pred_)
        mse = metrics.mean_squared_error(y_test,pred_)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(y_test,pred_)
        rmsle = np.sqrt(np.mean(np.power(np.log(np.array(abs(pred_))+1) - np.log(np.array(abs(y_test))+1), 2)))
        time_taken = time_end - time_start

        score_mae.append(mae)
        score_mse.append(mse)
        score_rmse.append(rmse)
        score_rmsle.append(rmsle)
        score_r2.append(r2)
        time_list.append(time_taken)

        name_counter = name_counter+1


    print("Creating metrics dataframe")
    compare_models_ = pd.DataFrame({'Model':model_names, 'MAE':score_mae, 'MSE':score_mse,
                       'RMSE':score_rmse, 'R2':score_r2, 'RMSLE':score_rmsle, 'Time Taken' :time_list})
    return compare_models_

import random
def preprocessor(train_data,test_data,target):
    
    #Better to remove the rows corresponding to the missing values in target data
    train_data = train_data[train_data[target].notnull()]
    test_data = test_data[test_data[target].notnull()]
    
    # dropping ALL duplicte values 
    train_data = train_data.drop_duplicates(keep='first')
    test_data = test_data.drop_duplicates(keep='first')
    print(train_data.shape,test_data.shape)


    train_data ,date_col_train= convert_to_datetime(train_data)
    test_data ,date_col_test= convert_to_datetime(test_data)
    if date_col_train!='' and date_col_test!='':
        #Fetching date features and adding to dataframe
        train_data = fetch_date_features(train_data,date_col_train)
        test_data = fetch_date_features(test_data,date_col_test)
    

    #separating target from independent features
    train_features = train_data.columns.values
    test_features = test_data.columns.values
    train_data_pr,train_target_pr = separate_target(train_data,  train_features,target)
    test_data_pr,test_target_pr = separate_target(test_data,  test_features,target)
    print(train_data_pr.columns, test_data_pr.columns)
    #find id column in train and test
    id_train = id_column_detector(train_data_pr)
    id_test = id_column_detector(test_data_pr)
    print(id_train,id_test)
    if len(id_train)==0:
        print('No id column found..  Assigning new id column!')
        train_data_pr= train_data_pr.reset_index()
        test_data_pr= test_data_pr.reset_index()
        id_train = id_column_detector(train_data_pr)
        id_test = id_column_detector(test_data_pr)
    elif len(id_train)==1:
        print('id column found !')
        train_data_pr.set_index(id_train)
        test_data_pr.set_index(id_test)
    else:
        print('More than one matching id column found.. More rigorous algorithms activated !')
        id_train = id_column_detector2(train_data_pr)
        id_test = id_column_detector2(test_data_pr)
        print(id_train,id_test)
        train_data_pr.set_index(id_train)
        test_data_pr.set_index(id_test)

    
    print(id_train,id_test)
    # Getting id column name as a string
    id_col = ""
    # traverse in the string
    for ele in id_train:
        id_col += ele
        
        

    #Get the categorical and numerical features list
    cat_feat = get_categorical_features(train_data_pr)
    numerical_features = get_numerical_features(train_data_pr)
    #if there are inf or -inf then replace them with NaN
    train_data_pr.replace([np.inf,-np.inf],np.NaN,inplace=True)
    test_data_pr.replace([np.inf,-np.inf],np.NaN,inplace=True)

    #Impute categorical featurs by introducing another category ='unknown'
    train_notnull_cat_df = categorical_feature_imputer(train_data_pr,cat_feat)
    test_notnull_cat_df = categorical_feature_imputer(test_data_pr,cat_feat)

    #Preprocssing text data
    train_data_pr = clean_text_data(train_data_pr,train_notnull_cat_df,cat_feat)
    test_data_pr = clean_text_data(test_data_pr,test_notnull_cat_df,cat_feat)

    #Drop unclean categorical features from data_pr
    train_data_pr.drop(cat_feat, axis=1, inplace=True)
    test_data_pr.drop(cat_feat, axis=1, inplace=True)

    #Getting the cleaned categorical features list
    cleaned_cat_feat = list(map(lambda x:'clean'+x,cat_feat))

    #Getting the cleaned text feature list
    print('Numerical features: ',numerical_features)
    cleaned_text_feat = fetch_text_features(train_data_pr,cleaned_cat_feat)
    print('Text features detected: ',cleaned_text_feat)

    #Getting the cleaned categorical features list after removing text feature from it
    cleaned_cat_feat = fetch_categorical_features(cleaned_cat_feat,cleaned_text_feat)
    print('Categorical Features : ',cleaned_cat_feat)

    if len(cleaned_text_feat)>0:
        #Get word to tfidf mapping for train and test data
        train_word2tfidf = get_word2tfidf(train_data_pr,cleaned_text_feat)
        test_word2tfidf = get_word2tfidf(test_data_pr,cleaned_text_feat)

        #Find TFIDF-W2V for text features
        train_data_pr = find_tfidf_w2v(train_data_pr, train_word2tfidf, cleaned_text_feat)
        test_data_pr = find_tfidf_w2v(test_data_pr, test_word2tfidf, cleaned_text_feat)

        #Getting the cleaned text features list
        sequence_text_feat = list(map(lambda x:'vect_'+x,cleaned_text_feat))
        train_data_pr = create_dataframe_textvectored(id_col,train_data_pr,sequence_text_feat)
        test_data_pr = create_dataframe_textvectored(id_col,test_data_pr,sequence_text_feat)

        train_data_pr.drop(cleaned_text_feat, axis=1, inplace=True)
        test_data_pr.drop(cleaned_text_feat, axis=1, inplace=True)

        train_data_pr.drop(sequence_text_feat, axis=1, inplace=True)
        test_data_pr.drop(sequence_text_feat, axis=1, inplace=True)

    #Impute numerical featurs by mean
    train_data_pr = numerical_feature_imputer(train_data_pr,numerical_features)
    test_data_pr = numerical_feature_imputer(test_data_pr,numerical_features)

    id_for_later_use = test_data_pr[id_train]
   
    #Dropping id column
    train_data_pr.drop(id_train[0], axis=1, inplace=True)
    test_data_pr.drop(id_test[0], axis=1, inplace=True)

    X_train,X_test = vectorize_variables(train_data_pr,test_data_pr,cleaned_cat_feat)
    y_train = train_target_pr
    y_test = test_target_pr
    
#     #Impute if target variable has NaNs
#     mean_val = y_train.mean()
#     y_train.fillna(mean_val, inplace=True)
    
 

    return X_train,X_test,y_train,y_test,id_for_later_use,id_train



def compare_models(X_train,X_test,y_train,y_test):
    #AUTO INFER the ml use case(classification or regression)
    ml_usecase = classify_or_regression(y_train)
    print(ml_usecase)
    if ml_usecase == 'regression':
        df = compare_all_regression_models(X_train,X_test,y_train,y_test)
    else:
        df = compare_all_classification_models(X_train,X_test,y_train,y_test)
    return df,ml_usecase






def choose_classification_model(tr_data_pr,te_data_pr,tr_target_pr,te_target_pr,estimator = None,
                 ensemble = False,
                 method = None,
                 fold = 10,
                 round = 4,
                 cross_validation = True,
                 verbose = True,
                 system = True,fix_imbalance_param=True,
                 **kwargs):
    folds_shuffle_param=False
    seed= 10
    n_jobs_param = -1 #To use all the cores
    #import sklearn dependencies
    import sys
    from IPython.display import display, HTML, clear_output, update_display
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import RidgeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    available_estimators = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada',
                            'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost']

    #only raise exception of estimator is of type string.
    if type(estimator) is str:
        if estimator not in available_estimators:
            sys.exit('(Value Error): Estimator Not Available. Please see docstring for list of available estimators.')

    #checking error for ensemble:
    if type(ensemble) is not bool:
        sys.exit('(Type Error): Ensemble parameter can only take argument as True or False.')

    '''Checking error for method'''
    #1 Check When method given and ensemble is not set to True.
    if ensemble is False and method is not None:
        sys.exit('(Type Error): Method parameter only accepts value when ensemble is set to True.')

    #2 Check when ensemble is set to True and method is not passed.
    if ensemble is True and method is None:
        sys.exit("(Type Error): Method parameter missing. Pass method = 'Bagging' or 'Boosting'.")

    #3 Check when ensemble is set to True and method is passed but not allowed.
    available_method = ['Bagging', 'Boosting']
    if ensemble is True and method not in available_method:
        sys.exit("(Value Error): Method parameter only accepts two values 'Bagging' or 'Boosting'.")

    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')

    #checking round parameter
    if type(round) is not int:
        sys.exit('(Type Error): Round parameter only accepts integer value.')

    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.')

    #checking system parameter
    if type(system) is not bool:
        sys.exit('(Type Error): System parameter can only take argument as True or False.')

    #checking cross_validation parameter
    if type(cross_validation) is not bool:
        sys.exit('(Type Error): cross_validation parameter can only take argument as True or False.')

    #checking boosting conflict with estimators
    boosting_not_supported = ['lda','qda','ridge','mlp','gpc','svm','knn', 'catboost']
    if method == 'Boosting' and estimator in boosting_not_supported:
        sys.exit("(Type Error): Estimator does not provide class_weights or predict_proba function and hence not supported for the Boosting method. Change the estimator or method to 'Bagging'.")

    #cross validation setup starts here
    kf = StratifiedKFold(fold, random_state=seed, shuffle=folds_shuffle_param)

    print("Declaring metric variables")

    score_auc =np.empty((0,0))
    score_acc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    score_mcc =np.empty((0,0))
    score_training_time =np.empty((0,0))
    avgs_auc =np.empty((0,0))
    avgs_acc =np.empty((0,0))
    avgs_recall =np.empty((0,0))
    avgs_precision =np.empty((0,0))
    avgs_f1 =np.empty((0,0))
    avgs_kappa =np.empty((0,0))
    avgs_mcc =np.empty((0,0))
    avgs_training_time =np.empty((0,0))

    print("Importing untrained model")

    if estimator == 'lr':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=seed, **kwargs)
        full_name = 'Logistic Regression'

    elif estimator == 'knn':

        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_jobs=n_jobs_param, **kwargs)
        full_name = 'K Neighbors Classifier'

    elif estimator == 'nb':

        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB(**kwargs)
        full_name = 'Naive Bayes'

    elif estimator == 'dt':

        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=seed, **kwargs)
        full_name = 'Decision Tree Classifier'

    elif estimator == 'svm':

        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(max_iter=1000, tol=0.001, random_state=seed, n_jobs=n_jobs_param, **kwargs)
        full_name = 'SVM - Linear Kernel'

    elif estimator == 'rbfsvm':

        from sklearn.svm import SVC
        model = SVC(gamma='auto', C=1, probability=True, kernel='rbf', random_state=seed, **kwargs)
        full_name = 'SVM - Radial Kernel'

    elif estimator == 'gpc':

        from sklearn.gaussian_process import GaussianProcessClassifier
        model = GaussianProcessClassifier(random_state=seed, n_jobs=n_jobs_param, **kwargs)
        full_name = 'Gaussian Process Classifier'

    elif estimator == 'mlp':

        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(max_iter=500, random_state=seed, **kwargs)
        full_name = 'MLP Classifier'

    elif estimator == 'ridge':

        from sklearn.linear_model import RidgeClassifier
        model = RidgeClassifier(random_state=seed, **kwargs)
        full_name = 'Ridge Classifier'

    elif estimator == 'rf':

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=seed, n_jobs=n_jobs_param, **kwargs)
        full_name = 'Random Forest Classifier'

    elif estimator == 'qda':

        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        model = QuadraticDiscriminantAnalysis(**kwargs)
        full_name = 'Quadratic Discriminant Analysis'

    elif estimator == 'ada':

        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(random_state=seed, **kwargs)
        full_name = 'Ada Boost Classifier'

    elif estimator == 'gbc':

        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(random_state=seed, **kwargs)
        full_name = 'Gradient Boosting Classifier'

    elif estimator == 'lda':

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis(**kwargs)
        full_name = 'Linear Discriminant Analysis'

    elif estimator == 'et':

        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(random_state=seed, n_jobs=n_jobs_param, **kwargs)
        full_name = 'Extra Trees Classifier'

    elif estimator == 'xgboost':

        from xgboost import XGBClassifier
        model = XGBClassifier(random_state=seed, verbosity=0, n_jobs=n_jobs_param, **kwargs)
        full_name = 'Extreme Gradient Boosting'

    elif estimator == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(random_state=seed, silent=True, thread_count=n_jobs_param, **kwargs) # Silent is True to suppress CatBoost iteration results
        full_name = 'CatBoost Classifier'

    else:

        print("Declaring custom model")

        model = estimator

        def get_model_name(e):
            return str(e).split("(")[0]

        model_dict_logging = {'ExtraTreesClassifier' : 'Extra Trees Classifier',
                                'GradientBoostingClassifier' : 'Gradient Boosting Classifier',
                                'RandomForestClassifier' : 'Random Forest Classifier',
                                'XGBClassifier' : 'Extreme Gradient Boosting',
                                'AdaBoostClassifier' : 'Ada Boost Classifier',
                                'DecisionTreeClassifier' : 'Decision Tree Classifier',
                                'RidgeClassifier' : 'Ridge Classifier',
                                'LogisticRegression' : 'Logistic Regression',
                                'KNeighborsClassifier' : 'K Neighbors Classifier',
                                'GaussianNB' : 'Naive Bayes',
                                'SGDClassifier' : 'SVM - Linear Kernel',
                                'SVC' : 'SVM - Radial Kernel',
                                'GaussianProcessClassifier' : 'Gaussian Process Classifier',
                                'MLPClassifier' : 'MLP Classifier',
                                'QuadraticDiscriminantAnalysis' : 'Quadratic Discriminant Analysis',
                                'LinearDiscriminantAnalysis' : 'Linear Discriminant Analysis',
                                'CatBoostClassifier' : 'CatBoost Classifier',
                                'BaggingClassifier' : 'Bagging Classifier',
                                'VotingClassifier' : 'Voting Classifier'}
        mn = get_model_name(estimator)
        if 'catboost' in mn:
            mn = 'CatBoostClassifier'
        if mn in model_dict_logging.keys():
            full_name = model_dict_logging.get(mn)
        else:
            full_name = mn
    print(str(full_name) + ' Imported succesfully')

    #Checking method when ensemble is set to True.
    print("Checking ensemble method")

    if method == 'Bagging':
        print("Ensemble method set to Bagging")
        from sklearn.ensemble import BaggingClassifier
        model = BaggingClassifier(model,bootstrap=True,n_estimators=10, random_state=seed, n_jobs=n_jobs_param)

    elif method == 'Boosting':
        print("Ensemble method set to Boosting")
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(model, n_estimators=10, random_state=seed)


    #For each fold
    fold_num = 1

    for train_i , test_i in kf.split(tr_data_pr,tr_target_pr):
        print("Initializing Fold " + str(fold_num))
        X_train,X_test = tr_data_pr.iloc[train_i], tr_data_pr.iloc[test_i]
        y_train,y_test = tr_target_pr.iloc[train_i], tr_target_pr.iloc[test_i]
        time_start=time.time()
        if fix_imbalance_param:
            print("Initializing SMOTE")
            from imblearn.over_sampling import SMOTE
            resampler = SMOTE(random_state=seed)
            #X_train,y_train = resampler.fit_sample(X_train, y_train)  #Open this later while classification
            print("Resampling completed")



        if hasattr(model, 'predict_proba'):

            print('Fitting model: ',model)
            model.fit(X_train,y_train)
            pred_prob = model.predict_proba(X_test)
            pred_prob = pred_prob[:,1]
            pred_ = model.predict(X_test)
            sca = metrics.accuracy_score(y_test,pred_)
            #if tr_target_pr.value_counts().count() > 2  implies Multiclass
            if tr_target_pr.value_counts().count() > 2:
                sc = 0
                recall = metrics.recall_score(y_test,pred_, average='macro')
                precision = metrics.precision_score(y_test,pred_, average = 'weighted')
                f1 = metrics.f1_score(y_test,pred_, average='weighted')
            else:
                try:
                    sc = metrics.roc_auc_score(y_test,pred_prob)
                except:
                    sc = 0
                    print("model has no predict_proba attribute. AUC set to 0.00")
                recall = metrics.recall_score(y_test,pred_)
                precision = metrics.precision_score(y_test,pred_)
                f1 = metrics.f1_score(y_test,pred_)
        else:
            print('Training model on else condition : ',model)
            model.fit(X_train,y_train)
            print("model has no predict_proba attribute. pred_prob set to 0.00")
            pred_prob = 0.00
            pred_ = model.predict(X_test)
            sca = metrics.accuracy_score(y_test,pred_)
            if tr_target_pr.value_counts().count() > 2:
                sc = 0
                recall = metrics.recall_score(y_test,pred_, average='macro')
                precision = metrics.precision_score(y_test,pred_, average = 'weighted')
                f1 = metrics.f1_score(y_test,pred_, average='weighted')
            else:
                try:
                    sc = metrics.roc_auc_score(y_test,pred_prob)
                except:
                    sc = 0
                    print("model has no predict_proba attribute. AUC set to 0.00")
                recall = metrics.recall_score(y_test,pred_)
                precision = metrics.precision_score(y_test,pred_)
                f1 = metrics.f1_score(y_test,pred_)

        print('Compiling metrics: ')
        time_end=time.time()
        mcc = metrics.matthews_corrcoef(y_test,pred_)
        kappa = metrics.cohen_kappa_score(y_test,pred_)
        training_time= time_end - time_start
        score_acc = np.append(score_acc,sca)
        score_auc = np.append(score_auc,sc)
        score_recall = np.append(score_recall,recall)
        score_precision = np.append(score_precision,precision)
        score_f1 =np.append(score_f1,f1)
        score_kappa =np.append(score_kappa,kappa)
        score_mcc=np.append(score_mcc,mcc)
        score_training_time=np.append(score_training_time,training_time)


        fold_results = pd.DataFrame({'Accuracy':[sca], 'AUC': [sc], 'Recall': [recall],
                                     'Prec.': [precision], 'F1': [f1], 'Kappa': [kappa], 'MCC':[mcc]}).round(round)

        fold_num += 1
    print("Calculating mean and std")

    mean_acc=np.mean(score_acc)
    mean_auc=np.mean(score_auc)
    mean_recall=np.mean(score_recall)
    mean_precision=np.mean(score_precision)
    mean_f1=np.mean(score_f1)
    mean_kappa=np.mean(score_kappa)
    mean_mcc=np.mean(score_mcc)
    mean_training_time=np.sum(score_training_time) #changed it to sum from mean

    std_acc=np.std(score_acc)
    std_auc=np.std(score_auc)
    std_recall=np.std(score_recall)
    std_precision=np.std(score_precision)
    std_f1=np.std(score_f1)
    std_kappa=np.std(score_kappa)
    std_mcc=np.std(score_mcc)
    std_training_time=np.std(score_training_time)

    avgs_acc = np.append(avgs_acc, mean_acc)
    avgs_acc = np.append(avgs_acc, std_acc)
    avgs_auc = np.append(avgs_auc, mean_auc)
    avgs_auc = np.append(avgs_auc, std_auc)
    avgs_recall = np.append(avgs_recall, mean_recall)
    avgs_recall = np.append(avgs_recall, std_recall)
    avgs_precision = np.append(avgs_precision, mean_precision)
    avgs_precision = np.append(avgs_precision, std_precision)
    avgs_f1 = np.append(avgs_f1, mean_f1)
    avgs_f1 = np.append(avgs_f1, std_f1)
    avgs_kappa = np.append(avgs_kappa, mean_kappa)
    avgs_kappa = np.append(avgs_kappa, std_kappa)
    avgs_mcc = np.append(avgs_mcc, mean_mcc)
    avgs_mcc = np.append(avgs_mcc, std_mcc)

    avgs_training_time = np.append(avgs_training_time, mean_training_time)
    avgs_training_time = np.append(avgs_training_time, std_training_time)

    print("Creating metrics dataframe per fold")

    model_results = pd.DataFrame({'Accuracy': score_acc, 'AUC': score_auc, 'Recall' : score_recall, 'Prec.' : score_precision ,
                     'F1' : score_f1, 'Kappa' : score_kappa, 'MCC': score_mcc})
    model_avgs = pd.DataFrame({'Accuracy': avgs_acc, 'AUC': avgs_auc, 'Recall' : avgs_recall, 'Prec.' : avgs_precision ,
                     'F1' : avgs_f1, 'Kappa' : avgs_kappa, 'MCC': avgs_mcc},index=['Mean', 'SD'])


    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)
    print(model_results)

    model_fit_start = time.time()
    print('Finalising Model  ')
    model.fit(tr_data_pr, tr_target_pr)
    print('Model is finalised !! ')
    model_fit_end = time.time()

    model_fit_time = np.array(model_fit_end - model_fit_start).round(2)


    return model,model_results




def choose_regression_model(tr_data_pr,te_data_pr,tr_target_pr,te_target_pr,estimator = None,
                 ensemble = False,
                 method = None,
                 fold = 10,
                 round = 4,
                 cross_validation = True,
                 verbose = True,
                 system = True,
                 **kwargs):

    seed= 10
    n_jobs_param = -1 #To use all the cores
    import sys
    from IPython.display import display, HTML, clear_output, update_display
    #import sklearn dependencies
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    from sklearn.linear_model import Lars
    from sklearn.linear_model import LassoLars
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from sklearn.linear_model import BayesianRidge
    from sklearn.linear_model import ARDRegression
    from sklearn.linear_model import PassiveAggressiveRegressor
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import TheilSenRegressor
    from sklearn.linear_model import HuberRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.model_selection import KFold
    folds_shuffle_param = False
    import sklearn.metrics as metrics


    #checking error for estimator (string)
    available_estimators = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par',
                            'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr',
                            'mlp', 'xgboost', 'catboost']

    #only raise exception of estimator is of type string.
    if type(estimator) is str:
        if estimator not in available_estimators:
            sys.exit('(Value Error): Estimator Not Available. Please see docstring for list of available estimators.')

    #checking error for ensemble:
    if type(ensemble) is not bool:
        sys.exit('(Type Error): Ensemble parameter can only take argument as True or False.')

    '''Checking error for method'''
    #1 Check When method given and ensemble is not set to True.
    if ensemble is False and method is not None:
        sys.exit('(Type Error): Method parameter only accepts value when ensemble is set to True.')

    #2 Check when ensemble is set to True and method is not passed.
    if ensemble is True and method is None:
        sys.exit("(Type Error): Method parameter missing. Pass method = 'Bagging' or 'Boosting'.")

    #3 Check when ensemble is set to True and method is passed but not allowed.
    available_method = ['Bagging', 'Boosting']
    if ensemble is True and method not in available_method:
        sys.exit("(Value Error): Method parameter only accepts two values 'Bagging' or 'Boosting'.")

    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')

    #checking round parameter
    if type(round) is not int:
        sys.exit('(Type Error): Round parameter only accepts integer value.')

    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.')

    #checking system parameter
    if type(system) is not bool:
        sys.exit('(Type Error): System parameter can only take argument as True or False.')

    #checking cross_validation parameter
    if type(cross_validation) is not bool:
        sys.exit('(Type Error): cross_validation parameter can only take argument as True or False.')

    #checking boosting conflict with estimators
    boosting_not_supported = ['lda','qda','ridge','mlp','gpc','svm','knn', 'catboost']
    if method == 'Boosting' and estimator in boosting_not_supported:
        sys.exit("(Type Error): Estimator does not provide class_weights or predict_proba function and hence not supported for the Boosting method. Change the estimator or method to 'Bagging'.")

    #cross validation setup starts here
    kf = KFold(fold, random_state=seed, shuffle=folds_shuffle_param)

    print("Declaring metric variables")

    score_mae =np.empty((0,0))
    score_mse =np.empty((0,0))
    score_rmse =np.empty((0,0))
    score_rmsle =np.empty((0,0))
    score_r2 =np.empty((0,0))
    score_mape =np.empty((0,0))
    score_training_time=np.empty((0,0))
    avgs_mae =np.empty((0,0))
    avgs_mse =np.empty((0,0))
    avgs_rmse =np.empty((0,0))
    avgs_r2 =np.empty((0,0))
    avgs_rmsle =np.empty((0,0))
    avgs_training_time=np.empty((0,0))

    print("Importing untrained model")

    if estimator == 'lr':

        from sklearn.linear_model import LinearRegression
        model = LinearRegression(n_jobs=n_jobs_param, **kwargs)
        full_name = 'Linear Regression'

    elif estimator == 'lasso':

        from sklearn.linear_model import Lasso
        model = Lasso(random_state=seed, **kwargs)
        full_name = 'Lasso Regression'

    elif estimator == 'ridge':

        from sklearn.linear_model import Ridge
        model = Ridge(random_state=seed, **kwargs)
        full_name = 'Ridge Regression'

    elif estimator == 'en':

        from sklearn.linear_model import ElasticNet
        model = ElasticNet(random_state=seed, **kwargs)
        full_name = 'Elastic Net'

    elif estimator == 'lar':

        from sklearn.linear_model import Lars
        model = Lars(**kwargs)
        full_name = 'Least Angle Regression'

    elif estimator == 'llar':

        from sklearn.linear_model import LassoLars
        model = LassoLars(**kwargs)
        full_name = 'Lasso Least Angle Regression'

    elif estimator == 'omp':

        from sklearn.linear_model import OrthogonalMatchingPursuit
        model = OrthogonalMatchingPursuit(**kwargs)
        full_name = 'Orthogonal Matching Pursuit'

    elif estimator == 'br':
        from sklearn.linear_model import BayesianRidge
        model = BayesianRidge(**kwargs)
        full_name = 'Bayesian Ridge Regression'

    elif estimator == 'ard':

        from sklearn.linear_model import ARDRegression
        model = ARDRegression(**kwargs)
        full_name = 'Automatic Relevance Determination'

    elif estimator == 'par':

        from sklearn.linear_model import PassiveAggressiveRegressor
        model = PassiveAggressiveRegressor(random_state=seed, **kwargs)
        full_name = 'Passive Aggressive Regressor'

    elif estimator == 'ransac':

        from sklearn.linear_model import RANSACRegressor
        model = RANSACRegressor(min_samples=0.5, random_state=seed, **kwargs)
        full_name = 'Random Sample Consensus'

    elif estimator == 'tr':

        from sklearn.linear_model import TheilSenRegressor
        model = TheilSenRegressor(random_state=seed, n_jobs=n_jobs_param, **kwargs)
        full_name = 'TheilSen Regressor'

    elif estimator == 'huber':

        from sklearn.linear_model import HuberRegressor
        model = HuberRegressor(**kwargs)
        full_name = 'Huber Regressor'

    elif estimator == 'kr':

        from sklearn.kernel_ridge import KernelRidge
        model = KernelRidge(**kwargs)
        full_name = 'Kernel Ridge'

    elif estimator == 'svm':

        from sklearn.svm import SVR
        model = SVR(**kwargs)
        full_name = 'Support Vector Regression'

    elif estimator == 'knn':

        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_jobs=n_jobs_param, **kwargs)
        full_name = 'Nearest Neighbors Regression'

    elif estimator == 'dt':

        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(random_state=seed, **kwargs)
        full_name = 'Decision Tree'

    elif estimator == 'rf':

        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=seed, n_jobs=n_jobs_param, **kwargs)
        full_name = 'Random Forest Regressor'

    elif estimator == 'et':

        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs_param, **kwargs)
        full_name = 'Extra Trees Regressor'

    elif estimator == 'ada':

        from sklearn.ensemble import AdaBoostRegressor
        model = AdaBoostRegressor(random_state=seed, **kwargs)
        full_name = 'AdaBoost Regressor'

    elif estimator == 'gbr':

        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(random_state=seed, **kwargs)
        full_name = 'Gradient Boosting Regressor'

    elif estimator == 'mlp':

        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(random_state=seed, **kwargs)
        full_name = 'MLP Regressor'

    elif estimator == 'xgboost':

        from xgboost import XGBRegressor
        model = XGBRegressor(random_state=seed, n_jobs=n_jobs_param, verbosity=0, **kwargs)
        full_name = 'Extreme Gradient Boosting Regressor'


    elif estimator == 'catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(random_state=seed, silent = True, thread_count=n_jobs_param, **kwargs)
        full_name = 'CatBoost Regressor'

    else:

        logger.info("Declaring custom model")

        model = estimator

        def get_model_name(e):
            return str(e).split("(")[0]

        model_dict_logging = {'ExtraTreesRegressor' : 'Extra Trees Regressor',
                            'GradientBoostingRegressor' : 'Gradient Boosting Regressor',
                            'RandomForestRegressor' : 'Random Forest',
                            'XGBRegressor' : 'Extreme Gradient Boosting',
                            'AdaBoostRegressor' : 'AdaBoost Regressor',
                            'DecisionTreeRegressor' : 'Decision Tree',
                            'Ridge' : 'Ridge Regression',
                            'TheilSenRegressor' : 'TheilSen Regressor',
                            'BayesianRidge' : 'Bayesian Ridge',
                            'LinearRegression' : 'Linear Regression',
                            'ARDRegression' : 'Automatic Relevance Determination',
                            'KernelRidge' : 'Kernel Ridge',
                            'RANSACRegressor' : 'Random Sample Consensus',
                            'HuberRegressor' : 'Huber Regressor',
                            'Lasso' : 'Lasso Regression',
                            'ElasticNet' : 'Elastic Net',
                            'Lars' : 'Least Angle Regression',
                            'OrthogonalMatchingPursuit' : 'Orthogonal Matching Pursuit',
                            'MLPRegressor' : 'Multi Level Perceptron',
                            'KNeighborsRegressor' : 'K Neighbors Regressor',
                            'SVR' : 'Support Vector Machine',
                            'LassoLars' : 'Lasso Least Angle Regression',
                            'PassiveAggressiveRegressor' : 'Passive Aggressive Regressor',
                            'CatBoostRegressor' : 'CatBoost Regressor',
                            'BaggingRegressor' : 'Bagging Regressor'}
        mn = get_model_name(estimator)
        if 'catboost' in mn:
            mn = 'CatBoostRegressor'
        if mn in model_dict_logging.keys():
            full_name = model_dict_logging.get(mn)
        else:
            full_name = mn
    print(str(full_name) + ' Imported succesfully')

    #Checking method when ensemble is set to True.
    print("Checking ensemble method")

    if method == 'Bagging':
        print("Ensemble method set to Bagging")
        from sklearn.ensemble import BaggingClassifier
        model = BaggingRegressor(model,bootstrap=True,n_estimators=10, random_state=seed)

    elif method == 'Boosting':
        print("Ensemble method set to Boosting")
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostRegressor(model, n_estimators=10, random_state=seed)


    #For each fold
    fold_num = 1

    for train_i , test_i in kf.split(tr_data_pr,tr_target_pr):
        print("Initializing Fold " + str(fold_num))
        #Cross validation
        X_train,X_test = tr_data_pr.iloc[train_i], tr_data_pr.iloc[test_i]
        y_train,y_test = tr_target_pr.iloc[train_i], tr_target_pr.iloc[test_i]
        time_start=time.time()


        print('Fitting model: ',model)
        model.fit(X_train,y_train)
        print("Evaluating Metrics")
        pred_ = model.predict(X_test)

        print("Compiling Metrics")
        time_end=time.time()
        mae = metrics.mean_absolute_error(y_test,pred_)
        mse = metrics.mean_squared_error(y_test,pred_)
        rmse = np.sqrt(mse)
        rmsle = np.sqrt(np.mean(np.power(np.log(np.array(abs(pred_))+1) - np.log(np.array(abs(y_test))+1), 2)))
        r2 = metrics.r2_score(y_test,pred_)
        training_time=time_end-time_start
        score_mae = np.append(score_mae,mae)
        score_mse = np.append(score_mse,mse)
        score_rmse = np.append(score_rmse,rmse)
        score_rmsle = np.append(score_rmsle,rmsle)
        score_r2 =np.append(score_r2,r2)
        score_training_time=np.append(score_training_time,training_time)


        fold_results = pd.DataFrame({'MAE':[mae], 'MSE': [mse], 'RMSE': [rmse], 'R2': [r2],
                                     'RMSLE' : [rmsle]}).round(round)

        fold_num += 1

    print("Calculating mean and std")

    mean_mae=np.mean(score_mae)
    mean_mse=np.mean(score_mse)
    mean_rmse=np.mean(score_rmse)
    mean_rmsle=np.mean(score_rmsle)
    mean_r2=np.mean(score_r2)
    mean_training_time=np.mean(score_training_time)
    std_mae=np.std(score_mae)
    std_mse=np.std(score_mse)
    std_rmse=np.std(score_rmse)
    std_rmsle=np.std(score_rmsle)
    std_r2=np.std(score_r2)
    std_training_time=np.std(score_training_time)

    avgs_mae = np.append(avgs_mae, mean_mae)
    avgs_mae = np.append(avgs_mae, std_mae)
    avgs_mse = np.append(avgs_mse, mean_mse)
    avgs_mse = np.append(avgs_mse, std_mse)
    avgs_rmse = np.append(avgs_rmse, mean_rmse)
    avgs_rmse = np.append(avgs_rmse, std_rmse)
    avgs_rmsle = np.append(avgs_rmsle, mean_rmsle)
    avgs_rmsle = np.append(avgs_rmsle, std_rmsle)
    avgs_r2 = np.append(avgs_r2, mean_r2)
    avgs_r2 = np.append(avgs_r2, std_r2)
    avgs_training_time=np.append(avgs_training_time, mean_training_time)
    avgs_training_time=np.append(avgs_training_time, std_training_time)

    print("Creating metrics dataframe per fold")

    model_results = pd.DataFrame({'MAE': score_mae, 'MSE': score_mse, 'RMSE' : score_rmse, 'R2' : score_r2,
                                  'RMSLE' : score_rmsle})
    model_avgs = pd.DataFrame({'MAE': avgs_mae, 'MSE': avgs_mse, 'RMSE' : avgs_rmse, 'R2' : avgs_r2,
                                'RMSLE' : avgs_rmsle},index=['Mean', 'SD'])


    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)
    print(model_results)

    model_fit_start = time.time()
    print('Finalising Model  ')
    model.fit(tr_data_pr, tr_target_pr)
    print('Model is finalised !! ')
    model_fit_end = time.time()

    model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

    return model,model_results

def choose_models(X_train,X_test,y_train,y_test,estimator):
    # estimator = input('Which model you want ?')
    #AUTO INFER the ml use case(classification or regression)
    ml_usecase = classify_or_regression(y_train)
    print(ml_usecase)
    if ml_usecase == 'regression':
        model,model_results = choose_regression_model(X_train,X_test,y_train,y_test,estimator)
    else:
        model ,model_results= choose_classification_model(X_train,X_test,y_train,y_test,estimator)
    return model,model_results


def tune_classification_model(tr_data_pr,te_data_pr,tr_target_pr,te_target_pr,estimator,
               fold = 10,
               round = 4,
               n_iter = 10,
               custom_grid = None,
               optimize = 'Accuracy',
               choose_better = False,
               verbose = True,fix_imbalance_param=True):

    seed= 10
    #pre-load libraries
    import pandas as pd
    import time, datetime
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display

    #create estimator clone from sklearn.base
    from sklearn.base import clone
    estimator_clone = clone(estimator)

    import random
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import RandomizedSearchCV

    #setting numpy seed
    np.random.seed(seed)
    folds_shuffle_param=False
    n_jobs_param=-1

    #checking estimator if string
    if type(estimator) is str:
        sys.exit('(Type Error): The behavior of tune_model in version 1.0.1 is changed. Please pass trained model object.')

    #restrict VotingClassifier
    if hasattr(estimator,'voting'):
         sys.exit('(Type Error): VotingClassifier not allowed under tune_model().')

    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')

    #checking optimize parameter
    allowed_optimize = ['Accuracy', 'Recall', 'Precision', 'F1', 'AUC', 'MCC']
    if optimize not in allowed_optimize:
        sys.exit('(Value Error): Optimization method not supported. See docstring for list of available parameters.')

    #checking optimize parameter for multiclass
    if tr_target_pr.value_counts().count() > 2:
        if optimize == 'AUC':
            sys.exit('(Type Error): AUC metric not supported for multiclass problems. See docstring for list of other optimization parameters.')

    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.')


    #setting optimize parameter
    if optimize == 'Accuracy':
        optimize = 'accuracy'
        compare_dimension = 'Accuracy'

    elif optimize == 'AUC':
        optimize = 'roc_auc'
        compare_dimension = 'AUC'

    elif optimize == 'Recall':
        if y.value_counts().count() > 2:
            optimize = metrics.make_scorer(metrics.recall_score, average = 'macro')
        else:
            optimize = 'recall'
        compare_dimension = 'Recall'

    elif optimize == 'Precision':
        if y.value_counts().count() > 2:
            optimize = metrics.make_scorer(metrics.precision_score, average = 'weighted')
        else:
            optimize = 'precision'
        compare_dimension = 'Prec.'

    elif optimize == 'F1':
        if y.value_counts().count() > 2:
            optimize = metrics.make_scorer(metrics.f1_score, average = 'weighted')
        else:
            optimize = optimize = 'f1'
        compare_dimension = 'F1'

    elif optimize == 'MCC':
        optimize = 'roc_auc' # roc_auc instead because you cannot use MCC in gridsearchcv
        compare_dimension = 'MCC'

    def get_model_name(e):
        return str(e).split("(")[0]

    mn = get_model_name(estimator)

    if 'catboost' in mn:
        mn = 'CatBoostClassifier'

    model_dict = {'ExtraTreesClassifier' : 'et',
                'GradientBoostingClassifier' : 'gbc',
                'RandomForestClassifier' : 'rf',
                'XGBClassifier' : 'xgboost',
                'AdaBoostClassifier' : 'ada',
                'DecisionTreeClassifier' : 'dt',
                'RidgeClassifier' : 'ridge',
                'LogisticRegression' : 'lr',
                'KNeighborsClassifier' : 'knn',
                'GaussianNB' : 'nb',
                'SGDClassifier' : 'svm',
                'SVC' : 'rbfsvm',
                'GaussianProcessClassifier' : 'gpc',
                'MLPClassifier' : 'mlp',
                'QuadraticDiscriminantAnalysis' : 'qda',
                'LinearDiscriminantAnalysis' : 'lda',
                'CatBoostClassifier' : 'catboost',
                'BaggingClassifier' : 'Bagging'}

    model_dict_logging = {'ExtraTreesClassifier' : 'Extra Trees Classifier',
                        'GradientBoostingClassifier' : 'Gradient Boosting Classifier',
                        'RandomForestClassifier' : 'Random Forest Classifier',
                        'XGBClassifier' : 'Extreme Gradient Boosting',
                        'AdaBoostClassifier' : 'Ada Boost Classifier',
                        'DecisionTreeClassifier' : 'Decision Tree Classifier',
                        'RidgeClassifier' : 'Ridge Classifier',
                        'LogisticRegression' : 'Logistic Regression',
                        'KNeighborsClassifier' : 'K Neighbors Classifier',
                        'GaussianNB' : 'Naive Bayes',
                        'SGDClassifier' : 'SVM - Linear Kernel',
                        'SVC' : 'SVM - Radial Kernel',
                        'GaussianProcessClassifier' : 'Gaussian Process Classifier',
                        'MLPClassifier' : 'MLP Classifier',
                        'QuadraticDiscriminantAnalysis' : 'Quadratic Discriminant Analysis',
                        'LinearDiscriminantAnalysis' : 'Linear Discriminant Analysis',
                        'CatBoostClassifier' : 'CatBoost Classifier',
                        'BaggingClassifier' : 'Bagging Classifier',
                        'VotingClassifier' : 'Voting Classifier'}
    _estimator_ = estimator

    estimator = model_dict.get(mn)

    print("Defining folds")
    kf = StratifiedKFold(fold, random_state=seed, shuffle=folds_shuffle_param)

    print("Declaring metric variables")
    score_auc =np.empty((0,0))
    score_acc =np.empty((0,0))
    score_recall =np.empty((0,0))
    score_precision =np.empty((0,0))
    score_f1 =np.empty((0,0))
    score_kappa =np.empty((0,0))
    score_mcc=np.empty((0,0))
    score_training_time=np.empty((0,0))
    avgs_auc =np.empty((0,0))
    avgs_acc =np.empty((0,0))
    avgs_recall =np.empty((0,0))
    avgs_precision =np.empty((0,0))
    avgs_f1 =np.empty((0,0))
    avgs_kappa =np.empty((0,0))
    avgs_mcc=np.empty((0,0))
    avgs_training_time=np.empty((0,0))

    print("Defining Hyperparameters")
    print("Initializing RandomizedSearchCV")

    #setting turbo parameters
    cv = 3

    if estimator == 'knn':

        from sklearn.neighbors import KNeighborsClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'n_neighbors': range(1,51),
                    'weights' : ['uniform', 'distance'],
                    'metric':["euclidean", "manhattan"]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone, param_distributions=param_grid,
                                        scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       n_jobs=n_jobs_param, iid=False)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'lr':

        from sklearn.linear_model import LogisticRegression

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'C': np.arange(0, 10, 0.001),
                    "penalty": [ 'l1', 'l2'],
                    "class_weight": ["balanced", None]
                        }
        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, cv=cv,
                                        random_state=seed, iid=False, n_jobs=n_jobs_param)
        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'dt':

        from sklearn.tree import DecisionTreeClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {"max_depth": np.random.randint(1, (len(tr_data_pr.columns)*.85),20),
                    "max_features": np.random.randint(1, len(tr_data_pr.columns),20),
                    "min_samples_leaf": [2,3,4,5,6],
                    "criterion": ["gini", "entropy"],
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone, param_distributions=param_grid,
                                       scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       iid=False, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'mlp':

        from sklearn.neural_network import MLPClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'solver' : ['lbfgs', 'sgd', 'adam'],
                    'alpha': np.arange(0, 1, 0.0001),
                    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (100,50,100), (100,100,100)],
                    'activation': ["tanh", "identity", "logistic","relu"]
                    }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, cv=cv,
                                        random_state=seed, iid=False, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'gpc':

        from sklearn.gaussian_process import GaussianProcessClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {"max_iter_predict":[100,200,300,400,500,600,700,800,900,1000]}

        model_grid = RandomizedSearchCV(estimator=estimator_clone, param_distributions=param_grid,
                                       scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'rbfsvm':

        from sklearn.svm import SVC

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'C': np.arange(0, 50, 0.01),
                    "class_weight": ["balanced", None]}

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'nb':

        from sklearn.naive_bayes import GaussianNB

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'var_smoothing': [0.000000001, 0.000000002, 0.000000005, 0.000000008, 0.000000009,
                                            0.0000001, 0.0000002, 0.0000003, 0.0000005, 0.0000007, 0.0000009,
                                            0.00001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.009,
                                            0.004, 0.005, 0.006, 0.007,0.008, 0.009, 0.01, 0.1, 1]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'svm':

        from sklearn.linear_model import SGDClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'penalty': ['l2', 'l1','elasticnet'],
                        'l1_ratio': np.arange(0,1,0.01),
                        'alpha': [0.0001, 0.001, 0.01, 0.0002, 0.002, 0.02, 0.0005, 0.005, 0.05],
                        'fit_intercept': [True, False],
                        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                        'eta0': [0.001, 0.01,0.05,0.1,0.2,0.3,0.4,0.5]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'ridge':

        from sklearn.linear_model import RidgeClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'alpha': np.arange(0,1,0.001),
                        'fit_intercept': [True, False],
                        'normalize': [True, False]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'rf':

        from sklearn.ensemble import RandomForestClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'min_samples_split': [2, 5, 7, 9, 10],
                        'min_samples_leaf' : [1, 2, 4],
                        'max_features' : ['auto', 'sqrt', 'log2'],
                        'bootstrap': [True, False]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'ada':

        from sklearn.ensemble import AdaBoostClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'n_estimators':  np.arange(10,200,5),
                        'learning_rate': np.arange(0,1,0.01),
                        'algorithm' : ["SAMME", "SAMME.R"]
                        }

        if tr_target_pr.value_counts().count() > 2:
            base_estimator_input = _estimator_.estimator.base_estimator
        else:
            base_estimator_input = _estimator_.base_estimator

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'gbc':

        from sklearn.ensemble import GradientBoostingClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'n_estimators': np.arange(10,200,5),
                        'learning_rate': np.arange(0,1,0.01),
                        'subsample' : np.arange(0.1,1,0.05),
                        'min_samples_split' : [2,4,5,7,9,10],
                        'min_samples_leaf' : [1,2,3,4,5],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'max_features' : ['auto', 'sqrt', 'log2']
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'qda':

        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'reg_param': np.arange(0,1,0.01)}

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'lda':

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'solver' : ['lsqr', 'eigen'],
                        'shrinkage': [None, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'et':

        from sklearn.ensemble import ExtraTreesClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'n_estimators': np.arange(10,200,5),
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [int(x) for x in np.linspace(1, 11, num = 1)],
                        'min_samples_split': [2, 5, 7, 9, 10],
                        'min_samples_leaf' : [1, 2, 4],
                        'max_features' : ['auto', 'sqrt', 'log2'],
                        'bootstrap': [True, False]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_


    elif estimator == 'xgboost':

        from xgboost import XGBClassifier

        num_class = tr_target_pr.value_counts().count()

        if custom_grid is not None:
            param_grid = custom_grid

        elif tr_target_pr.value_counts().count() > 2:

            param_grid = {'learning_rate': np.arange(0,1,0.01),
                          'n_estimators': np.arange(10,500,20),
                          'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                          'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                          'colsample_bytree': [0.5, 0.7, 0.9, 1],
                          'min_child_weight': [1, 2, 3, 4],
                          'num_class' : [num_class, num_class]
                         }
        else:
            param_grid = {'learning_rate': np.arange(0,1,0.01),
                          'n_estimators':[10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                          'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                          'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                          'colsample_bytree': [0.5, 0.7, 0.9, 1],
                          'min_child_weight': [1, 2, 3, 4],
                         }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_


    elif estimator == 'lightgbm':

        import lightgbm as lgb

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'num_leaves': [10,20,30,40,50,60,70,80,90,100,150,200],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                        'n_estimators': [10, 30, 50, 70, 90, 100, 120, 150, 170, 200],
                        'min_split_gain' : [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                        'reg_alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        'reg_lambda': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_


    elif estimator == 'catboost':

        from catboost import CatBoostClassifier

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'depth':[3,1,2,6,4,5,7,8,9,10],
                        'iterations':[250,100,500,1000],
                        'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3],
                        'l2_leaf_reg':[3,1,5,10,100],
                        'border_count':[32,5,10,20,50,100,200],
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'Bagging':

        from sklearn.ensemble import BaggingClassifier

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'n_estimators': np.arange(10,300,10),
                        'bootstrap': [True, False],
                        'bootstrap_features': [True, False],
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    print("Random search completed")

    #For each fold
    fold_num = 1

    for train_i , test_i in kf.split(tr_data_pr,tr_target_pr):
        print("Initializing Fold " + str(fold_num))
        X_train,X_test = tr_data_pr.iloc[train_i], tr_data_pr.iloc[test_i]
        y_train,y_test = tr_target_pr.iloc[train_i], tr_target_pr.iloc[test_i]
        time_start=time.time()
        if fix_imbalance_param:
            print("Initializing SMOTE")
            from imblearn.over_sampling import SMOTE
            resampler = SMOTE(random_state=seed)
            #X_train,y_train = resampler.fit_sample(X_train, y_train)  #Open this later while classification
            print("Resampling completed")

        if hasattr(model, 'predict_proba'):

            print('Fitting model: ',model)
            model.fit(X_train,y_train)
            pred_prob = model.predict_proba(X_test)
            pred_prob = pred_prob[:,1]
            pred_ = model.predict(X_test)
            sca = metrics.accuracy_score(y_test,pred_)
            #if tr_target_pr.value_counts().count() > 2  implies Multiclass
            if tr_target_pr.value_counts().count() > 2:
                sc = 0
                recall = metrics.recall_score(y_test,pred_, average='macro')
                precision = metrics.precision_score(y_test,pred_, average = 'weighted')
                f1 = metrics.f1_score(y_test,pred_, average='weighted')
            else:
                try:
                    sc = metrics.roc_auc_score(y_test,pred_prob)
                except:
                    sc = 0
                    print("model has no predict_proba attribute. AUC set to 0.00")
                recall = metrics.recall_score(y_test,pred_)
                precision = metrics.precision_score(y_test,pred_)
                f1 = metrics.f1_score(y_test,pred_)
        else:
            print('Training model on else condition : ',model)
            model.fit(X_train,y_train)
            print("model has no predict_proba attribute. pred_prob set to 0.00")
            pred_prob = 0.00
            pred_ = model.predict(X_test)
            sca = metrics.accuracy_score(y_test,pred_)
            if tr_target_pr.value_counts().count() > 2:
                sc = 0
                recall = metrics.recall_score(y_test,pred_, average='macro')
                precision = metrics.precision_score(y_test,pred_, average = 'weighted')
                f1 = metrics.f1_score(y_test,pred_, average='weighted')
            else:
                try:
                    sc = metrics.roc_auc_score(y_test,pred_prob)
                except:
                    sc = 0
                    print("model has no predict_proba attribute. AUC set to 0.00")
                recall = metrics.recall_score(y_test,pred_)
                precision = metrics.precision_score(y_test,pred_)
                f1 = metrics.f1_score(y_test,pred_)

        print('Compiling metrics: ')
        time_end=time.time()
        mcc = metrics.matthews_corrcoef(y_test,pred_)
        kappa = metrics.cohen_kappa_score(y_test,pred_)
        training_time= time_end - time_start
        score_acc = np.append(score_acc,sca)
        score_auc = np.append(score_auc,sc)
        score_recall = np.append(score_recall,recall)
        score_precision = np.append(score_precision,precision)
        score_f1 =np.append(score_f1,f1)
        score_kappa =np.append(score_kappa,kappa)
        score_mcc=np.append(score_mcc,mcc)
        score_training_time=np.append(score_training_time,training_time)


        fold_results = pd.DataFrame({'Accuracy':[sca], 'AUC': [sc], 'Recall': [recall],
                                     'Prec.': [precision], 'F1': [f1], 'Kappa': [kappa], 'MCC':[mcc]}).round(round)

        fold_num += 1
    print("Calculating mean and std")

    mean_acc=np.mean(score_acc)
    mean_auc=np.mean(score_auc)
    mean_recall=np.mean(score_recall)
    mean_precision=np.mean(score_precision)
    mean_f1=np.mean(score_f1)
    mean_kappa=np.mean(score_kappa)
    mean_mcc=np.mean(score_mcc)
    mean_training_time=np.sum(score_training_time) #changed it to sum from mean

    std_acc=np.std(score_acc)
    std_auc=np.std(score_auc)
    std_recall=np.std(score_recall)
    std_precision=np.std(score_precision)
    std_f1=np.std(score_f1)
    std_kappa=np.std(score_kappa)
    std_mcc=np.std(score_mcc)
    std_training_time=np.std(score_training_time)

    avgs_acc = np.append(avgs_acc, mean_acc)
    avgs_acc = np.append(avgs_acc, std_acc)
    avgs_auc = np.append(avgs_auc, mean_auc)
    avgs_auc = np.append(avgs_auc, std_auc)
    avgs_recall = np.append(avgs_recall, mean_recall)
    avgs_recall = np.append(avgs_recall, std_recall)
    avgs_precision = np.append(avgs_precision, mean_precision)
    avgs_precision = np.append(avgs_precision, std_precision)
    avgs_f1 = np.append(avgs_f1, mean_f1)
    avgs_f1 = np.append(avgs_f1, std_f1)
    avgs_kappa = np.append(avgs_kappa, mean_kappa)
    avgs_kappa = np.append(avgs_kappa, std_kappa)
    avgs_mcc = np.append(avgs_mcc, mean_mcc)
    avgs_mcc = np.append(avgs_mcc, std_mcc)

    avgs_training_time = np.append(avgs_training_time, mean_training_time)
    avgs_training_time = np.append(avgs_training_time, std_training_time)

    print("Creating metrics dataframe per fold")

    model_results = pd.DataFrame({'Accuracy': score_acc, 'AUC': score_auc, 'Recall' : score_recall, 'Prec.' : score_precision ,
                     'F1' : score_f1, 'Kappa' : score_kappa, 'MCC': score_mcc})
    model_avgs = pd.DataFrame({'Accuracy': avgs_acc, 'AUC': avgs_auc, 'Recall' : avgs_recall, 'Prec.' : avgs_precision ,
                     'F1' : avgs_f1, 'Kappa' : avgs_kappa, 'MCC': avgs_mcc},index=['Mean', 'SD'])


    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)
    print(model_results)



    print('Finalising Model ... ')
    if fix_imbalance_param:
            print("Initializing SMOTE")
            from imblearn.over_sampling import SMOTE
            resampler = SMOTE(random_state=seed)
            #tr_data_pr,tr_target_pr = resampler.fit_sample(tr_data_pr,tr_target_pr)  #Open this later while classification
            print("Resampling completed")
    model_fit_start = time.time()
    best_model.fit(tr_data_pr, tr_target_pr)
    print('Model is finalised !! ')
    model_fit_end = time.time()

    model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

    return model,model_results

def tune_regression_model(tr_data_pr,te_data_pr,tr_target_pr,te_target_pr,estimator,
               fold = 10,
               round = 4,
               n_iter = 10,
               custom_grid = None,
               optimize = 'R2',
               choose_better = False,
               verbose = True):
    seed= 10
    #pre-load libraries
    import pandas as pd
    import time, datetime
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display

    #create estimator clone from sklearn.base
    from sklearn.base import clone
    estimator_clone = clone(estimator)


    import random
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import KFold
    from sklearn.model_selection import RandomizedSearchCV

    #setting numpy seed
    np.random.seed(seed)
    folds_shuffle_param=False
    n_jobs_param=-1

    #checking estimator if string
    if type(estimator) is str:
        sys.exit('(Type Error): The behavior of tune_model in version 1.0.1 is changed. Please pass trained model object.')
    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')
    #checking optimize parameter
    allowed_optimize = ['MAE', 'MSE', 'R2', 'RMSE', 'RMSLE', 'MAPE']
    if optimize not in allowed_optimize:
        sys.exit('(Value Error): Optimization method not supported. See docstring for list of available parameters.')
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.')

    #define optimizer
    if optimize == 'MAE':
        optimize = 'neg_mean_absolute_error'
        compare_dimension = 'MAE'
    elif optimize == 'MSE':
        optimize = 'neg_mean_squared_error'
        compare_dimension = 'MSE'
    elif optimize == 'R2':
        optimize = 'r2'
        compare_dimension = 'R2'
    elif optimize == 'MAPE':
        optimize = 'neg_mean_absolute_error' #because mape not present in sklearn
        compare_dimension = 'MAPE'
    elif optimize == 'RMSE':
        optimize = 'neg_mean_squared_error' #because rmse not present in sklearn
        compare_dimension = 'RMSE'
    elif optimize == 'RMSLE':
        optimize = 'neg_mean_squared_error' #because rmsle not present in sklearn
        compare_dimension = 'RMSLE'

    def get_model_name(e):
        return str(e).split("(")[0]

    mn = get_model_name(estimator)

    if 'catboost' in mn:
        mn = 'CatBoostRegressor'

    model_dict = {'ExtraTreesRegressor' : 'et',
                'GradientBoostingRegressor' : 'gbr',
                'RandomForestRegressor' : 'rf',
                'XGBRegressor' : 'xgboost',
                'AdaBoostRegressor' : 'ada',
                'DecisionTreeRegressor' : 'dt',
                'Ridge' : 'ridge',
                'TheilSenRegressor' : 'tr',
                'BayesianRidge' : 'br',
                'LinearRegression' : 'lr',
                'ARDRegression' : 'ard',
                'KernelRidge' : 'kr',
                'RANSACRegressor' : 'ransac',
                'HuberRegressor' : 'huber',
                'Lasso' : 'lasso',
                'ElasticNet' : 'en',
                'Lars' : 'lar',
                'OrthogonalMatchingPursuit' : 'omp',
                'MLPRegressor' : 'mlp',
                'KNeighborsRegressor' : 'knn',
                'SVR' : 'svm',
                'LassoLars' : 'llar',
                'PassiveAggressiveRegressor' : 'par',
                'CatBoostRegressor' : 'catboost',
                'BaggingRegressor' : 'Bagging'}

    model_dict_logging = {'ExtraTreesRegressor' : 'Extra Trees Regressor',
                        'GradientBoostingRegressor' : 'Gradient Boosting Regressor',
                        'RandomForestRegressor' : 'Random Forest',
                        'XGBRegressor' : 'Extreme Gradient Boosting',
                        'AdaBoostRegressor' : 'AdaBoost Regressor',
                        'DecisionTreeRegressor' : 'Decision Tree',
                        'Ridge' : 'Ridge Regression',
                        'TheilSenRegressor' : 'TheilSen Regressor',
                        'BayesianRidge' : 'Bayesian Ridge',
                        'LinearRegression' : 'Linear Regression',
                        'ARDRegression' : 'Automatic Relevance Determination',
                        'KernelRidge' : 'Kernel Ridge',
                        'RANSACRegressor' : 'Random Sample Consensus',
                        'HuberRegressor' : 'Huber Regressor',
                        'Lasso' : 'Lasso Regression',
                        'ElasticNet' : 'Elastic Net',
                        'Lars' : 'Least Angle Regression',
                        'OrthogonalMatchingPursuit' : 'Orthogonal Matching Pursuit',
                        'MLPRegressor' : 'Multi Level Perceptron',
                        'KNeighborsRegressor' : 'K Neighbors Regressor',
                        'SVR' : 'Support Vector Machine',
                        'LassoLars' : 'Lasso Least Angle Regression',
                        'PassiveAggressiveRegressor' : 'Passive Aggressive Regressor',
                        'CatBoostRegressor' : 'CatBoost Regressor',
                        'BaggingRegressor' : 'Bagging Regressor'}
    _estimator_ = estimator

    estimator = model_dict.get(mn)

    print("Defining folds")
    kf = KFold(fold, random_state=seed, shuffle=folds_shuffle_param)


    print("Declaring metric variables")
    score_mae =np.empty((0,0))
    score_mse =np.empty((0,0))
    score_rmse =np.empty((0,0))
    score_rmsle =np.empty((0,0))
    score_r2 =np.empty((0,0))
    score_mape =np.empty((0,0))
    score_training_time=np.empty((0,0))
    avgs_mae =np.empty((0,0))
    avgs_mse =np.empty((0,0))
    avgs_rmse =np.empty((0,0))
    avgs_rmsle =np.empty((0,0))
    avgs_r2 =np.empty((0,0))
    avgs_mape =np.empty((0,0))
    avgs_training_time=np.empty((0,0))

    print("Defining Hyperparameters")
    print("Initializing RandomizedSearchCV")

    #setting turbo parameters
    cv = 3


    if estimator == 'lr':

        from sklearn.linear_model import LinearRegression

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'fit_intercept': [True, False],
                        'normalize' : [True, False]
                        }
        model_grid = RandomizedSearchCV(estimator=estimator_clone, param_distributions=param_grid,
                                        scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                        n_jobs=n_jobs_param, iid=False)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'lasso':

        from sklearn.linear_model import Lasso

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'alpha': np.arange(0,1,0.001),
                        'fit_intercept': [True, False],
                        'normalize' : [True, False],
                        }
        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, cv=cv,
                                        random_state=seed, iid=False,n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'ridge':

        from sklearn.linear_model import Ridge

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {"alpha": np.arange(0,1,0.001),
                        "fit_intercept": [True, False],
                        "normalize": [True, False],
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone, param_distributions=param_grid,
                                       scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       iid=False, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'en':

        from sklearn.linear_model import ElasticNet

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'alpha': np.arange(0,1,0.01),
                        'l1_ratio' : np.arange(0,1,0.01),
                        'fit_intercept': [True, False],
                        'normalize': [True, False]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter, cv=cv,
                                        random_state=seed, iid=False, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'lar':

        from sklearn.linear_model import Lars

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'fit_intercept':[True, False],
                        'normalize' : [True, False],
                        'eps': [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.0005, 0.005, 0.00005, 0.02, 0.007]}

        model_grid = RandomizedSearchCV(estimator=estimator_clone, param_distributions=param_grid,
                                       scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'llar':

        from sklearn.linear_model import LassoLars

        if custom_grid is not None:
            param_grid = custom_grid
        else:
            param_grid = {'alpha': [0.0001,0.001,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                        'fit_intercept':[True, False],
                        'normalize' : [True, False],
                        'eps': [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.0005, 0.005, 0.00005, 0.02, 0.007]}

        model_grid = RandomizedSearchCV(estimator=estimator_clone, param_distributions=param_grid,
                                       scoring=optimize, n_iter=n_iter, cv=cv, random_state=seed,
                                       n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'omp':

        from sklearn.linear_model import OrthogonalMatchingPursuit
        import random

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'n_nonzero_coefs': range(1, len(X_train.columns)+1),
                        'fit_intercept' : [True, False],
                        'normalize': [True, False]}

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'br':

        from sklearn.linear_model import BayesianRidge

        if custom_grid is not None:
            param_grid = custom_grid

        else:

            param_grid = {'alpha_1': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                        'alpha_2': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                        'lambda_1': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                        'lambda_2': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                        'compute_score': [True, False],
                        'fit_intercept': [True, False],
                        'normalize': [True, False]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'ard':

        from sklearn.linear_model import ARDRegression

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'alpha_1': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                        'alpha_2': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                        'lambda_1': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                        'lambda_2': [0.0000001, 0.000001, 0.0001, 0.001, 0.01, 0.0005, 0.005, 0.05, 0.1, 0.15, 0.2, 0.3],
                        'threshold_lambda' : [5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000],
                        'compute_score': [True, False],
                        'fit_intercept': [True, False],
                        'normalize': [True, False]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'par':

        from sklearn.linear_model import PassiveAggressiveRegressor

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'C': np.arange(0,1,0.01), #[0.01, 0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'fit_intercept': [True, False],
                        'early_stopping' : [True, False],
                        #'validation_fraction': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'loss' : ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                        'epsilon' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        'shuffle' : [True, False]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'ransac':

        from sklearn.linear_model import RANSACRegressor

        if custom_grid is not None:
            param_grid = custom_grid

        else:

            param_grid = {'min_samples': np.arange(0,1,0.05), #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'max_trials': np.arange(1,20,1), #[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                        'max_skips': np.arange(1,20,1), #[1,2,3,4,5,6,7,8,9,10],
                        'stop_n_inliers': np.arange(1,25,1), #[1,2,3,4,5,6,7,8,9,10],
                        'stop_probability': np.arange(0,1,0.01), #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'loss' : ['absolute_loss', 'squared_loss'],
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'tr':

        from sklearn.linear_model import TheilSenRegressor

        if custom_grid is not None:
            param_grid = custom_grid

        else:

            param_grid = {'fit_intercept': [True, False],
                        'max_subpopulation': [5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'huber':

        from sklearn.linear_model import HuberRegressor

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'epsilon': [1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.55, 1.6, 1.7, 1.8, 1.9],
                        'alpha': np.arange(0,1,0.0001), #[0.00001, 0.0001, 0.0003, 0.005, 0.05, 0.1, 0.0005, 0.15],
                        'fit_intercept' : [True, False]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'kr':

        from sklearn.kernel_ridge import KernelRidge

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'alpha': np.arange(0,1,0.01) }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'svm':

        from sklearn.svm import SVR

        if custom_grid is not None:
            param_grid = custom_grid

        else:

            param_grid = {'C' : np.arange(0, 10, 0.001),
                        'epsilon' : [1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.55, 1.6, 1.7, 1.8, 1.9],
                        'shrinking': [True, False]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'knn':

        from sklearn.neighbors import KNeighborsRegressor

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'n_neighbors': range(1,51),
                        'weights' :  ['uniform', 'distance'],
                        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                        'leaf_size': [10,20,30,40,50,60,70,80,90]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'dt':

        from sklearn.tree import DecisionTreeRegressor

        if custom_grid is not None:
            param_grid = custom_grid

        else:

            param_grid = {"max_depth": np.random.randint(1, (len(tr_data_pr.columns)*.85),20),
                        "max_features": np.random.randint(1, len(tr_data_pr.columns),20),
                        "min_samples_leaf": [2,3,4,5,6],
                        "criterion": ["mse", "mae", "friedman_mse"],
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'rf':

        from sklearn.ensemble import RandomForestRegressor

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'n_estimators': np.arange(10,300,10),
                        'criterion': ['mse', 'mae'],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'min_samples_split': [2, 5, 7, 9, 10],
                        'min_samples_leaf' : [1, 2, 4, 7, 9],
                        'max_features' : ['auto', 'sqrt', 'log2'],
                        'bootstrap': [True, False]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_


    elif estimator == 'et':

        from sklearn.ensemble import ExtraTreesRegressor

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'n_estimators': np.arange(10,300,10),
                        'criterion': ['mse', 'mae'],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'min_samples_split': [2, 5, 7, 9, 10],
                        'min_samples_leaf' : [1, 2, 4, 5, 7, 9],
                        'max_features' : ['auto', 'sqrt', 'log2'],
                        'bootstrap': [True, False]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'ada':

        from sklearn.ensemble import AdaBoostRegressor

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'n_estimators': np.arange(10,200,5),
                        'learning_rate': np.arange(0.1,1,0.01),
                        'loss' : ["linear", "square", "exponential"]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'gbr':

        from sklearn.ensemble import GradientBoostingRegressor

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                        'n_estimators': np.arange(10,200,5),
                        'learning_rate': np.arange(0,1,0.01),
                        'subsample' : [0.1,0.3,0.5,0.7,0.9,1],
                        'criterion' : ['friedman_mse', 'mse', 'mae'],
                        'min_samples_split' : [2,4,5,7,9,10],
                        'min_samples_leaf' : [1,2,3,4,5,7],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'max_features' : ['auto', 'sqrt', 'log2']
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'mlp':

        from sklearn.neural_network import MLPRegressor

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'learning_rate': ['constant', 'invscaling', 'adaptive'],
                        'solver' : ['lbfgs', 'adam'],
                        'alpha': np.arange(0, 1, 0.0001),
                        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (100,50,100), (100,100,100)],
                        'activation': ["tanh", "identity", "logistic","relu"]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_


    elif estimator == 'xgboost':

        from xgboost import XGBRegressor

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'n_estimators':[10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                        'max_depth': [int(x) for x in np.linspace(1, 11, num = 1)],
                        'colsample_bytree': [0.5, 0.7, 0.9, 1],
                        'min_child_weight': [1, 2, 3, 4]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_


    elif estimator == 'lightgbm':

        import lightgbm as lgb

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'num_leaves': [10,20,30,40,50,60,70,80,90,100,150,200],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                        'n_estimators': [10, 30, 50, 70, 90, 100, 120, 150, 170, 200],
                        'min_split_gain' : [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                        'reg_alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        'reg_lambda': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'catboost':

        from catboost import CatBoostRegressor

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'depth':[3,1,2,6,4,5,7,8,9,10],
                        'iterations':[250,100,500,1000],
                        'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3],
                        'l2_leaf_reg':[3,1,5,10,100],
                        'border_count':[32,5,10,20,50,100,200],
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    elif estimator == 'Bagging':

        from sklearn.ensemble import BaggingRegressor

        if custom_grid is not None:
            param_grid = custom_grid

        else:
            param_grid = {'n_estimators': np.arange(10,300,10),
                        'bootstrap': [True, False],
                        'bootstrap_features': [True, False],
                        }

        model_grid = RandomizedSearchCV(estimator=estimator_clone,
                                        param_distributions=param_grid, scoring=optimize, n_iter=n_iter,
                                        cv=cv, random_state=seed, n_jobs=n_jobs_param)

        model_grid.fit(tr_data_pr,tr_target_pr)
        model = model_grid.best_estimator_
        best_model = model_grid.best_estimator_
        best_model_param = model_grid.best_params_

    print("Random search completed")

    fold_num = 1

    for train_i , test_i in kf.split(tr_data_pr,tr_target_pr):
        print("Initializing Fold " + str(fold_num))
        #Cross validation
        X_train,X_test = tr_data_pr.iloc[train_i], tr_data_pr.iloc[test_i]
        y_train,y_test = tr_target_pr.iloc[train_i], tr_target_pr.iloc[test_i]
        time_start=time.time()


        print('Fitting model: ',model)
        model.fit(X_train,y_train)
        print("Evaluating Metrics")
        pred_ = model.predict(X_test)

        print("Compiling Metrics")
        time_end=time.time()
        mae = metrics.mean_absolute_error(y_test,pred_)
        mse = metrics.mean_squared_error(y_test,pred_)
        rmse = np.sqrt(mse)
        rmsle = np.sqrt(np.mean(np.power(np.log(np.array(abs(pred_))+1) - np.log(np.array(abs(y_test))+1), 2)))
        r2 = metrics.r2_score(y_test,pred_)
        training_time=time_end-time_start
        score_mae = np.append(score_mae,mae)
        score_mse = np.append(score_mse,mse)
        score_rmse = np.append(score_rmse,rmse)
        score_rmsle = np.append(score_rmsle,rmsle)
        score_r2 =np.append(score_r2,r2)
        score_training_time=np.append(score_training_time,training_time)

        fold_results = pd.DataFrame({'MAE':[mae], 'MSE': [mse], 'RMSE': [rmse], 'R2': [r2],
                                     'RMSLE' : [rmsle]}).round(round)

        fold_num += 1

    print("Calculating mean and std")

    mean_mae=np.mean(score_mae)
    mean_mse=np.mean(score_mse)
    mean_rmse=np.mean(score_rmse)
    mean_rmsle=np.mean(score_rmsle)
    mean_r2=np.mean(score_r2)
    mean_training_time=np.mean(score_training_time)
    std_mae=np.std(score_mae)
    std_mse=np.std(score_mse)
    std_rmse=np.std(score_rmse)
    std_rmsle=np.std(score_rmsle)
    std_r2=np.std(score_r2)
    std_training_time=np.std(score_training_time)

    avgs_mae = np.append(avgs_mae, mean_mae)
    avgs_mae = np.append(avgs_mae, std_mae)
    avgs_mse = np.append(avgs_mse, mean_mse)
    avgs_mse = np.append(avgs_mse, std_mse)
    avgs_rmse = np.append(avgs_rmse, mean_rmse)
    avgs_rmse = np.append(avgs_rmse, std_rmse)
    avgs_rmsle = np.append(avgs_rmsle, mean_rmsle)
    avgs_rmsle = np.append(avgs_rmsle, std_rmsle)
    avgs_r2 = np.append(avgs_r2, mean_r2)
    avgs_r2 = np.append(avgs_r2, std_r2)
    avgs_training_time=np.append(avgs_training_time, mean_training_time)
    avgs_training_time=np.append(avgs_training_time, std_training_time)

    print("Creating metrics dataframe per fold")

    model_results = pd.DataFrame({'MAE': score_mae, 'MSE': score_mse, 'RMSE' : score_rmse, 'R2' : score_r2,
                                  'RMSLE' : score_rmsle})
    model_avgs = pd.DataFrame({'MAE': avgs_mae, 'MSE': avgs_mse, 'RMSE' : avgs_rmse, 'R2' : avgs_r2,
                                'RMSLE' : avgs_rmsle},index=['Mean', 'SD'])

    model_results = model_results.append(model_avgs)
    model_results = model_results.round(round)
    print(model_results)

    model_fit_start = time.time()
    print('Finalising Model  ')
    best_model.fit(tr_data_pr, tr_target_pr)
    print('Model is finalised !! ')
    model_fit_end = time.time()



    model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

    return model,model_results

def tune_models(X_train,X_test,y_train,y_test,model_chosen):

    #AUTO INFER the ml use case(classification or regression)
    ml_usecase = classify_or_regression(y_train)
    print(ml_usecase)
    if ml_usecase == 'regression':
        tuned_model ,model_results= tune_regression_model(X_train,X_test,y_train,y_test,model_chosen,fold = 10,  custom_grid = None,optimize = 'R2',verbose = True)
    else:
        tuned_model ,model_results= tune_classification_model(X_train,X_test,y_train,y_test,model_chosen,fold = 10, custom_grid = None,optimize = 'Accuracy',verbose = True)
    return tuned_model,model_results,ml_usecase


def save_model(tuned_model,model_export_filename):
    import joblib as joblib
    # Save the model as a pickle in a file
    joblib.dump(tuned_model, model_export_filename)

def load_model(model_export_filename):
    import joblib as joblib
    # Load the model from the file
    tuned_model_loaded = joblib.load(model_export_filename)
    return tuned_model_loaded

import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)



def forecaster(y_train,y_test):
    import warnings
    warnings.filterwarnings("ignore")
    import datetime
    now = datetime.datetime.now()
    # ddmmYYHMS
    dt_string = now.strftime("%d%m%Y%H%M%S")


    df1 = y_train
    df2 = y_test

    # df1.fillna(0)
    # df2.fillna(0)


    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    df2=scaler.fit_transform(np.array(df2).reshape(-1,1))

    train_data,test_data=df1,df2

    time_step = 30
    X_train, ytrain = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    ytrain = np.nan_to_num(ytrain)
    ytest = np.nan_to_num(ytest)

    print(X_train.shape), print(ytrain.shape)
    print(X_test.shape), print(ytest.shape)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    ### Create the Stacked LSTM model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM

    model=Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=(30,1)))
    model.add(LSTM(64,return_sequences=True))
    model.add(LSTM(64,return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    print(model.summary())

    model.fit(X_train,ytrain,validation_data=(X_test,ytest),epochs=500,batch_size=30,verbose=1)

    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    print(ytrain)
    print(train_predict)

    ### Calculate RMSE performance metrics
    import math
    from sklearn.metrics import mean_squared_error
    train_mse = math.sqrt(mean_squared_error(ytrain,train_predict))
    print('Train MSE: ',train_mse)
    ### Test Data RMSE
    test_mse = math.sqrt(mean_squared_error(ytest,test_predict))
    print('Test MSE: ',test_mse)


    ##Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)

    final_df = np.concatenate((df1, df2), axis=0)

    ### Plotting
    # shift train predictions for plotting
    look_back=30
    trainPredictPlot = numpy.empty_like(final_df)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(final_df)
    testPredictPlot[:, :] = numpy.nan
    #testPredictPlot[look_back:len(test_predict)+look_back, :] = test_predict
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(final_df)-1, :] = test_predict


    #testPredictPlot[len(train_predict)+(look_back*2)+1:len(df2)-1, :] = test_predict
    # plot baseline and predictions
    import matplotlib
    matplotlib.use('agg')
    plt.plot(scaler.inverse_transform(final_df))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    filename_ = './static/forecast_results_viz/forecast_viz'+dt_string+'.jpg'
    #filename_ = './static/forecast_results_viz/forecast_viz.jpg'
    plt.savefig(filename_)
    #plt.show()


    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()


    # demonstrate prediction for next 30 days
    from numpy import array

    lst_output=[]
    n_steps=30
    i=0
    while(i<30):

        if(len(temp_input)>30):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            x_input = np.nan_to_num(x_input)
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1

    #scaler.inverse_transform(lst_output)
    print('Forecasted output for {} days'.format(len(lst_output)))
    lst_output = np.nan_to_num(lst_output)
    print(scaler.inverse_transform(lst_output))
    return scaler.inverse_transform(lst_output),filename_
