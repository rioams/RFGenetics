import streamlit as st
from streamlit_option_menu import option_menu

#memnaca model
import pickle

# import library
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
# import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score
from io import StringIO

def inisialisasi_modelRF(nilai_pohon):
    model = RandomForestClassifier(n_estimators=nilai_pohon, max_depth=None,random_state=0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                  max_features=2, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None,
                                  verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=30, monotonic_cst=None,criterion="gini")
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred),4)
    precision = round(precision_score(y_test, y_pred, average='weighted'),2)
    recall = round(recall_score(y_test, y_pred, average='weighted'),2)
    f1_scores = round(f1_score(y_test, y_pred, average='weighted'),2)
    st.write ("Hasil accuracy:",accuracy)
    st.write ("Hasil precision:",precision)
    st.write ("Hasil recall:",recall)
    st.write ("Hasil f1_scores:",f1_scores)