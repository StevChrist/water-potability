import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import optuna
import joblib  # Untuk menyimpan/memuat model & scaler
import os

# Nama file untuk menyimpan model dan scaler
MODEL_FILE = 'svm_model.pkl'
SCALER_FILE = 'scaler.pkl'
IMPUTER_FILE = 'imputer.pkl'

# --- 1. Pemuatan dan Pemrosesan Data ---
@st.cache_data # Cache data loading and initial processing
def load_and_preprocess_data(filepath):
    """Memuat dan membersihkan data."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: File '{filepath}' tidak ditemukan. Pastikan file berada di folder yang sama.")
        return None, None, None

    # Pisahkan fitur (X) dan target (y)
    X = df.drop('Potability', axis=1)
    y = df['Potability']

    # Imputasi nilai yang hilang menggunakan median
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

    return X_imputed, y, imputer

# --- 2. Pencarian Hyperparameter dengan Optuna & Pelatihan Model ---
def objective(trial, X_train, y_train, X_val, y_val):
    """Fungsi objektif untuk optimasi Optuna."""
    # Definisikan ruang pencarian hyperparameter
    C = trial.suggest_loguniform('C', 1e-3, 1e3)
    kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']) # Linear bisa ditambahkan jika perlu
    gamma = 'scale' # Default 'scale' seringkali bekerja baik, bisa juga dioptimasi
    if kernel == 'poly':
        degree = trial.suggest_int('degree', 2, 5)
        model = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, random_state=42)
    else:
        model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)

    # Latih model
    model.fit(X_train, y_train)

    # Evaluasi model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

@st.cache_resource # Cache model training
def train_model_with_optuna(X, y):
    """Melatih model SVM menggunakan hyperparameter terbaik dari Optuna."""
    st.write("Memulai pelatihan model (ini mungkin memakan waktu)...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Pisahkan data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Penskalaan fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    status_text.text("Data disiapkan. Memulai pencarian hyperparameter dengan Optuna...")
    progress_bar.progress(20)

    # Jalankan studi Optuna
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    n_trials = 50 # Jumlah percobaan, bisa ditingkatkan untuk hasil lebih baik
    study.optimize(lambda trial: objective(trial, X_train_scaled, y_train, X_test_scaled, y_test),
                     n_trials=n_trials,
                     callbacks=[lambda study, trial: progress_bar.progress(20 + int(80 * trial.number / n_trials))])

    status_text.text("Pencarian hyperparameter selesai. Melatih model final...")
    progress_bar.progress(95)

    # Dapatkan hyperparameter terbaik
    best_params = study.best_params
    st.write("Hyperparameter terbaik ditemukan:", best_params)

    # Latih model final dengan parameter terbaik
    if 'degree' in best_params:
         final_model = SVC(**best_params, random_state=42)
    else:
        # Jika 'degree' tidak ada (misal kernel bukan 'poly'), jangan sertakan
        final_model_params = {k: v for k, v in best_params.items() if k != 'degree'}
        final_model = SVC(**final_model_params, random_state=42)

    final_model.fit(X_train_scaled, y_train)
    status_text.text("Pelatihan model selesai!")
    progress_bar.progress(100)

    # Evaluasi akhir (opsional, untuk info)
    accuracy = final_model.score(X_test_scaled, y_test)
    st.write(f"Akurasi model final pada data uji: {accuracy:.4f}")

    return final_model, scaler

# --- 3. Antarmuka Streamlit ---
def user_input_features(X):
    """Membuat widget input untuk fitur."""
    st.sidebar.header('Masukkan Parameter Kualitas Air')
    features = {}
    for col in X.columns:
        # Gunakan nilai median sebagai default jika memungkinkan
        default_val = float(X[col].median())
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        # Beri sedikit 'padding' agar slider tidak mentok
        step = (max_val - min_val) / 100
        features[col] = st.sidebar.slider(
            col,
            min_value=min_val - (max_val - min_val) * 0.1,
            max_value=max_val + (max_val - min_val) * 0.1,
            value=default_val,
            step=step if step > 0 else 0.01 # Hindari step nol
        )
    return pd.DataFrame(features, index=[0])

# --- Fungsi Utama Aplikasi ---
def main():
    st.set_page_config(page_title="Prediksi Kelayakan Air Minum", layout="wide")
    st.title('💧 Aplikasi Prediksi Kelayakan Air Minum')
    st.write("""
    Aplikasi ini menggunakan model **Support Vector Machine (SVM)** yang dioptimalkan dengan **Optuna**
    untuk memprediksi apakah sampel air layak untuk diminum berdasarkan parameter kualitasnya.
    Masukkan parameter di *sidebar* kiri dan klik 'Prediksi'.
    """)

    # Muat dan proses data
    X, y, imputer = load_and_preprocess_data('water_potability.csv')

    if X is None: # Jika data gagal dimuat, hentikan aplikasi
        return

    # Latih model (atau muat jika sudah ada dan caching bekerja)
    # Cek jika model sudah disimpan, jika tidak, latih dan simpan
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE) or not os.path.exists(IMPUTER_FILE):
        st.info("Model belum dilatih atau tidak ditemukan. Memulai pelatihan...")
        model, scaler = train_model_with_optuna(X, y)
        # Simpan model, scaler, dan imputer
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        joblib.dump(imputer, IMPUTER_FILE) # Simpan imputer juga
        st.success("Model berhasil dilatih dan disimpan.")
    else:
        st.info("Memuat model yang sudah dilatih...")
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        # Imputer sudah ada dari load_and_preprocess_data, tapi kita bisa load jika perlu
        # imputer = joblib.load(IMPUTER_FILE)
        st.success("Model berhasil dimuat.")

    # Tampilkan input pengguna
    user_df = user_input_features(X)

    # Tampilkan input pengguna dalam bentuk tabel
    st.subheader('Parameter yang Anda Masukkan:')
    st.dataframe(user_df)

    # Tombol Prediksi
    if st.sidebar.button('Prediksi Kelayakan Air'):
        # Penskalaan input pengguna
        # Pastikan tidak ada NaN (meskipun slider seharusnya tidak menghasilkan NaN)
        user_df_scaled = scaler.transform(user_df)

        # Lakukan prediksi
        prediction = model.predict(user_df_scaled)
        # predict_proba tidak selalu ada di SVM, jadi kita gunakan predict saja

        st.subheader('Hasil Prediksi:')
        if prediction[0] == 1:
            st.success('✅ **Layak Diminum** (Potable)')
            st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdW1tdzh0OHF1dnd1dHZrdnB1d3h4bTN6emJ0MXNjaHBvMjJ0dW5vcyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKDXS18Lh03l61O/giphy.gif", caption="Air Bersih", width=200)
        else:
            st.error('❌ **Tidak Layak Diminum** (Not Potable)')
            st.warning("Disarankan untuk tidak mengonsumsi air dengan parameter ini tanpa pengolahan lebih lanjut.")
            st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdW53Z3d0b2Y1OXd0emg4M3N1OGc0cTZqMmRtbDV0cWw3Z3c0YW13cyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26gJAtomCg824c44E/giphy.gif", caption="Air Tercemar", width=200)

    st.sidebar.markdown("---")
    st.sidebar.info("Aplikasi ini dibuat sebagai contoh dan tidak boleh digunakan untuk keputusan medis atau konsumsi air tanpa verifikasi profesional.")

if __name__ == '__main__':
    main()