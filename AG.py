
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

from RF import inisialisasi_modelRF

def get_selected_features(X_train, X_test, y_train, y_test,nilai_pohon,nilai_populasi,nilai_generasi,nilai_mr):
    # Fungsi evaluasi untuk algoritma genetika
    def fitness_function(solusi):

        selected_features = np.where(solusi == 1)[0]
        
        # Menangani kasus di mana semua fitur tidak terpilih
        if len(selected_features) == 0 or np.all(solusi == 0):
            return 1

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Mengambil subset fitur dari data latih dan uji
                X_train_subset = X_train.iloc[:, selected_features]
                X_test_subset = X_test.iloc[:, selected_features]

                # Membuat dan melatih model Random Forest dengan parameter tertentu
                model = inisialisasi_modelRF(nilai_pohon)
                model.fit(X_train_subset, y_train)

            # Melakukan prediksi pada data uji
            y_pred = model.predict(X_test_subset)

            # Menghitung akurasi
            akurasi = accuracy_score(y_test, y_pred)

            # Menghitung mean squared error
            mse = mean_squared_error(y_test, y_pred)

            # Menentukan nilai fitness dengan trade-off antara akurasi dan mean squared error
            nilai_fitness = 0.7 * mse + 0.3 * (1 - akurasi)
            return nilai_fitness

        except Exception as e:
            # Menangani segala jenis kesalahan selama pelatihan model
            print(f"Kesalahan selama pelatihan: {e}")
            return 0.0  # Mengembalikan nilai fitness yang lebih rendah

    try:
        # Binary encoding for feature selection
        varbound = np.array([[0, 1]] * X_train.shape[1])

        # Parameter untuk algoritma genetika
        algorithm_param = {'max_num_iteration': nilai_generasi, 'population_size': nilai_populasi, 'mutation_probability': nilai_mr,
                        'elit_ratio': 0.01, 'max_iteration_without_improv': None, 'parents_portion': 0.3,
                        'crossover_type': 'one_point','selection_type': 'roulette','mutation_type': 'uniform_by_center'}

        # Inisialisasi algoritma genetika
        ga_instance = ga(function=fitness_function, dimension=X_train.shape[1], variable_type='bool',
                        variable_boundaries=varbound, function_timeout=100, algorithm_parameters=algorithm_param)
        
    except Exception as e:
        pass
    
    # Jalankan algoritma genetika
    ga_instance.run()
    
    # Dapatkan solusi terbaik
    selected_features = np.where(ga_instance.output_dict['variable'] == 1)[0]
    
    with open('selected_features.pkl', 'wb') as file:
        pickle.dump(selected_features, file)

    return selected_features