import streamlit as st
from streamlit_option_menu import option_menu

#memnaca model
import pickle

# import library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score
from io import StringIO

from RF import inisialisasi_modelRF, train_model
from AG import get_selected_features


st.set_option('deprecation.showPyplotGlobalUse', False)

with st.sidebar:
    selected = option_menu("Klasifikasi kualitas Udara", ["RF","RF+AG", "Predict"], default_index=0)

    
def input_data():
    df_training1 = st.file_uploader("Upload Data Training")
    df_testing1 = st.file_uploader("Upload Data Testing")
    css = '''
    <style>
        [data-testid='stFileUploader'] {
            width: max-content;
        }
        [data-testid='stFileUploader'] section {
            padding: 0;
            float: left;
        }
        [data-testid='stFileUploader'] section > input + div {
            display: none;
        }
        [data-testid='stFileUploader'] section + div {
            float: right;
            padding-top: 0;
        }

    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)
    df_training = pd.read_csv(df_training1)
    df_testing = pd.read_csv(df_testing1)

    # train test split
    X_train = df_training.drop('kategori', axis=1)
    X_test = df_testing.drop('kategori', axis=1)
    y_train = df_training['kategori']
    y_test = df_testing['kategori']
        
    return X_train, X_test, y_train, y_test

def save_model(model, filename):
    with open(filename + '.pkl', 'wb') as file:
        pickle.dump(model, file)

def predict(nilai_PM10,nilai_PM25, nilai_SO2, nilai_CO, nilai_O3, nilai_NO2,model_rf,model_rfag,selected_features):
    # Mapping antara hasil klasifikasi dengan label yang diinginkan
    mapping_label = {
        0: "Baik",
        1: "Sedang",
        2: "Tidak Sehat"
    }
    
    user_input_RF = [[nilai_PM10,nilai_PM25, nilai_SO2, nilai_CO, nilai_O3, nilai_NO2]]
    variabel =      [nilai_PM10,nilai_PM25, nilai_SO2, nilai_CO, nilai_O3, nilai_NO2]
    
    # Buat user_input berdasarkan selected_features
    user_input_AG = [[variabel[i] for i in selected_features]]

    # Gunakan model untuk klasifikasi RF
    hasil_klasifikasi_RF = model_rf.predict(user_input_RF)
    hasil_klasifikasi_RFAG = model_rfag.predict(user_input_AG)
    
     # Dapatkan label yang sesuai dari hasil klasifikasi RF
    label_hasil_RF = mapping_label[hasil_klasifikasi_RF[0]]
    label_hasil_RFAG = mapping_label[hasil_klasifikasi_RFAG[0]]

    # Tampilkan hasil klasifikasi RF dengan label yang sesuai        
    st.write ("Hasil Klasifikasi RF:")
    if label_hasil_RF == "Baik":
        st.success(f"Kualitas Udara berstatus {label_hasil_RF}")
    elif label_hasil_RF == "Sedang":
        st.warning(f"Kualitas Udara berstatus {label_hasil_RF}")
    else:
        st.error(f"Kualitas Udara berstatus {label_hasil_RF}")
               
    st.write ("Hasil Klasifikasi RF+AG:")
    st.write ("selected_features:",generate_y(selected_features))
    if label_hasil_RFAG == "Baik":
        st.success(f"Kualitas Udara berstatus {label_hasil_RFAG}")
    elif label_hasil_RFAG == "Sedang":
        st.warning(f"Kualitas Udara berstatus {label_hasil_RFAG}")
    else:
        st.error(f"Kualitas Udara berstatus {label_hasil_RFAG}")

    

def confusion_matrix_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', fmt='d', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Show plot in Streamlit
    st.pyplot()

def generate_y(x):
    # Membuat list y yang diinisialisasi dengan semua nilai 0
    y = [0] * 6
    
    # Mengubah nilai y sesuai dengan nilai x
    for value in x:
        if value < len(y):
            y[value] = 1
    
    return tuple(y)    
    
#RF
if(selected == "RF"):
    st.title('Training Model Random Forest')
    
    try:
        X_train, X_test, y_train, y_test = input_data()
        
    except Exception as e:
        pass
        
    nilai_pohon = st.number_input("Nilai Pohon",1)
   
    training = st.button ("Training")
    
    if training :         
        model_rf = inisialisasi_modelRF(nilai_pohon)
        
        train_model(model_rf, X_train, y_train, X_test, y_test)

        # save the model to disk
        save_model(model_rf,'model_rf')
        
        confusion_matrix_model(model_rf, X_train, y_train, X_test, y_test)

#RF+AG
if(selected == "RF+AG"):
    st.title('Training Model Random Forest + Algoritma Genetika')
    
    try:
        X_train, X_test, y_train, y_test = input_data()
        
    except Exception as e:
        pass
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nilai_pohon = st.number_input("Nilai Pohon",1)
    
    with col2:
        nilai_populasi = st.number_input("Nilai Populasi",10)

    with col3:
        nilai_generasi = st.number_input("Nilai Generasi",1)

    nilai_mr = st.slider('Nilai Mutation Rate',0.1,0.9)
    

    train = st.button ("Hitung train")
    
    if train :     

        selected_features = get_selected_features(X_train, X_test, y_train, y_test,nilai_pohon,nilai_populasi,nilai_generasi,nilai_mr)

        st.write ("selected_features:",generate_y(selected_features))
            
        model_rfag  = inisialisasi_modelRF(nilai_pohon)
        
        train_model(model_rfag, X_train.iloc[:, selected_features], y_train, X_test.iloc[:, selected_features], y_test)
        save_model(model_rfag,'model_rfag')
        
        confusion_matrix_model(model_rfag, X_train.iloc[:, selected_features], y_train, X_test.iloc[:, selected_features], y_test)

#predict
if(selected == "Predict"):
    st.title('Predict')
    
    try:
    # Memuat model dari file jika file ada
        with open('model_rf.pkl', 'rb') as file:
            model_rf = pickle.load(file)
    except FileNotFoundError:
    # Menampilkan pesan jika file tidak ditemukan
            st.write ("Model Random Forest belum ada")
            
    try:
    # Memuat model dari file jika file ada
        with open('model_rfag.pkl', 'rb') as file:
            model_rfag = pickle.load(file)
    except FileNotFoundError:
    # Menampilkan pesan jika file tidak ditemukan
            st.write ("Model Random Forest + Algoritma Genetika belum ada")
            
    # nilai input dari pengguna
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nilai_PM10 = st.number_input("Nilai PM10",0)
        nilai_PM25 = st.number_input("Nilai PM25 ",0)
    
    with col2:
        nilai_SO2 = st.number_input("Nilai SO2",0)
        nilai_CO = st.number_input("Nilai CO",0)

    with col3:
        nilai_O3 = st.number_input("Nilai O3",0)
        nilai_NO2 = st.number_input("Nilai NO2",0)  
    
    with open('selected_features.pkl', 'rb') as file:
        selected_features = pickle.load(file)

    # Tampilkan hasil klasifikasi RF dengan label yang sesuai
    predick = st.button ("predick")

    if predick : 
        predict(nilai_PM10,nilai_PM25, nilai_SO2, nilai_CO, nilai_O3, nilai_NO2,model_rf,model_rfag,selected_features)

