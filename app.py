import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score
import optuna
from sklearn.impute import SimpleImputer # Digunakan jika ada NaN setelah split

# Menonaktifkan peringatan spesifik Optuna yang mungkin muncul di konsol Streamlit
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(page_title="Prediksi Kelayakan Air Minum", layout="wide")

# --- Fungsi Cache untuk Data dan Model ---
@st.cache_data
def load_data(file_path="potability.csv"):
    """Memuat dan melakukan pra-pemrosesan awal pada data."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File dataset '{file_path}' tidak ditemukan. Pastikan file berada di direktori yang benar.")
        return None, None, None, None

    # Penanganan Nilai NaN (menggunakan mean imputation untuk semua kolom numerik)
    # Kolom target 'Potability' juga mungkin memiliki NaN, penting untuk ditangani jika ada.
    # Pertama, pastikan kolom numerik tidak memiliki NaN sebelum memisahkan X dan y
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Hapus baris dimana target 'Potability' adalah NaN, karena tidak bisa diimputasi untuk target
    if 'Potability' in df.columns and df['Potability'].isnull().any():
        df.dropna(subset=['Potability'], inplace=True)
        st.info("Baris dengan nilai NaN pada kolom 'Potability' telah dihapus.")

    # Imputasi untuk fitur
    imputer = SimpleImputer(strategy='mean')
    for col in numeric_cols:
        if col != 'Potability' and df[col].isnull().any(): # Jangan imputasi target
            df[col] = imputer.fit_transform(df[[col]])

    # Periksa lagi apakah masih ada NaN di fitur setelah imputasi
    if df.drop(columns=['Potability'], errors='ignore').isnull().sum().any():
        st.warning("Masih ada nilai NaN di fitur setelah imputasi. Periksa kembali data Anda.")
        # Untuk keamanan, isi sisa NaN jika ada (meskipun idealnya tidak terjadi)
        df.fillna(df.mean(numeric_only=True), inplace=True)


    X = df.drop('Potability', axis=1)
    y = df['Potability']

    # Pastikan semua kolom input ada di X
    expected_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    missing_cols = [col for col in expected_cols if col not in X.columns]
    if missing_cols:
        st.error(f"Kolom berikut tidak ditemukan dalam dataset: {', '.join(missing_cols)}")
        return None, None, None, None

    X = X[expected_cols] # Pastikan urutan kolom sesuai

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

# --- Fungsi Optuna untuk Setiap Model ---

# SVM dengan Optuna
@st.cache_resource(ttl=3600) # Cache resource untuk objek berat seperti model terlatih
def train_svm_optuna(X_train, y_train, X_test, y_test, n_trials=10): # Kurangi n_trials untuk kecepatan demo
    def objective_svm(trial):
        C = trial.suggest_float('C', 1e-2, 1e2, log=True)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']) # linear bisa lambat
        degree = trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3 # Hanya jika kernel poly

        model = SVC(C=C, gamma=gamma, kernel=kernel, degree=degree, random_state=42, probability=True)
        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))

    study_svm = optuna.create_study(direction='maximize')
    study_svm.optimize(objective_svm, n_trials=n_trials)
    best_params_svm = study_svm.best_params
    
    final_svm = SVC(**best_params_svm, random_state=42, probability=True)
    final_svm.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, final_svm.predict(X_test))
    return final_svm, accuracy, best_params_svm

# Random Forest dengan Optuna
@st.cache_resource(ttl=3600)
def train_rf_optuna(X_train, y_train, X_test, y_test, n_trials=10):
    def objective_rf(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))

    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(objective_rf, n_trials=n_trials)
    best_params_rf = study_rf.best_params

    final_rf = RandomForestClassifier(**best_params_rf, random_state=42, n_jobs=-1)
    final_rf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, final_rf.predict(X_test))
    return final_rf, accuracy, best_params_rf

# KNN dengan Optuna
@st.cache_resource(ttl=3600)
def train_knn_optuna(X_train, y_train, X_test, y_test, n_trials=10):
    def objective_knn(trial):
        n_neighbors = trial.suggest_int('n_neighbors', 3, 20)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
        
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric, n_jobs=-1)
        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))

    study_knn = optuna.create_study(direction='maximize')
    study_knn.optimize(objective_knn, n_trials=n_trials)
    best_params_knn = study_knn.best_params
    
    final_knn = KNeighborsClassifier(**best_params_knn, n_jobs=-1)
    final_knn.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, final_knn.predict(X_test))
    return final_knn, accuracy, best_params_knn

# Logistic Regression dengan Optuna
@st.cache_resource(ttl=3600)
def train_lr_optuna(X_train, y_train, X_test, y_test, n_trials=10):
    def objective_lr(trial):
        C = trial.suggest_float('C', 1e-2, 1e2, log=True)
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga']) # saga mendukung l1 dan l2
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2']) if solver == 'saga' else 'l2'
        
        if solver == 'liblinear' and penalty == 'l1': # liblinear hanya support l1 atau l2
             pass # ok
        elif solver == 'liblinear' and penalty == 'l2':
             pass # ok
        elif solver == 'saga': # saga support keduanya
             pass # ok
        else: # kombinasi tidak valid, coba solver lain atau default ke l2 untuk liblinear
            if solver == 'liblinear': penalty = 'l2'


        model = LogisticRegression(C=C, solver=solver, penalty=penalty, random_state=42, max_iter=1000, n_jobs=-1) # max_iter ditambah untuk saga
        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))

    study_lr = optuna.create_study(direction='maximize')
    study_lr.optimize(objective_lr, n_trials=n_trials)
    best_params_lr = study_lr.best_params

    # Pastikan solver dan penalty kompatibel untuk model final
    if best_params_lr['solver'] == 'liblinear' and best_params_lr.get('penalty') not in ['l1', 'l2']:
        best_params_lr['penalty'] = 'l2' # Default jika tidak sesuai


    final_lr = LogisticRegression(**best_params_lr, random_state=42, max_iter=1000, n_jobs=-1)
    final_lr.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, final_lr.predict(X_test))
    return final_lr, accuracy, best_params_lr

# XGBoost dengan Optuna
@st.cache_resource(ttl=3600)
def train_xgb_optuna(X_train, y_train, X_test, y_test, n_trials=10):
    def objective_xgb(trial):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': trial.suggest_float('eta', 1e-3, 0.3, log=True), # learning_rate
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True), # min_split_loss
            'lambda': trial.suggest_float('lambda', 1e-3, 1.0, log=True), # L2 regularization
            'alpha': trial.suggest_float('alpha', 1e-3, 1.0, log=True),   # L1 regularization
            'use_label_encoder': False,
            'seed': 42
        }
        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train, verbose=False) # verbose=False agar tidak print log training XGBoost
        return accuracy_score(y_test, model.predict(X_test))

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=n_trials)
    best_params_xgb = study_xgb.best_params
    
    # Update parameter untuk model final
    final_params_xgb = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'seed': 42
    }
    final_params_xgb.update(best_params_xgb)

    final_xgb = xgb.XGBClassifier(**final_params_xgb)
    final_xgb.fit(X_train, y_train, verbose=False)
    accuracy = accuracy_score(y_test, final_xgb.predict(X_test))
    return final_xgb, accuracy, best_params_xgb

# --- Load Data dan Latih Model (atau ambil dari cache) ---
data_load_state = st.text('Memuat data dan melatih model (mungkin perlu waktu saat pertama kali)...')
X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = load_data()

if X_train_scaled is not None:
    n_trials_optuna = 10 # Jumlah trial Optuna, bisa dinaikkan jika ingin hasil lebih optimal tapi lebih lama

    # Melatih atau memuat model dari cache
    model_svm, acc_svm, params_svm = train_svm_optuna(X_train_scaled, y_train, X_test_scaled, y_test, n_trials=n_trials_optuna)
    model_rf, acc_rf, params_rf = train_rf_optuna(X_train_scaled, y_train, X_test_scaled, y_test, n_trials=n_trials_optuna)
    model_knn, acc_knn, params_knn = train_knn_optuna(X_train_scaled, y_train, X_test_scaled, y_test, n_trials=n_trials_optuna)
    model_lr, acc_lr, params_lr = train_lr_optuna(X_train_scaled, y_train, X_test_scaled, y_test, n_trials=n_trials_optuna)
    model_xgb, acc_xgb, params_xgb = train_xgb_optuna(X_train_scaled, y_train, X_test_scaled, y_test, n_trials=n_trials_optuna)
    
    data_load_state.text('Data dan model berhasil dimuat/dilatih!')

    # --- UI Streamlit ---
    st.title("💧 Prediksi Kelayakan Air Minum menggunakan Machine Learning")
    st.markdown("""
    Aplikasi ini memprediksi apakah sampel air layak minum berdasarkan parameter kualitas fisik dan kimiawinya.
    Model dilatih menggunakan dataset dari Kaggle dan dioptimalkan dengan Optuna.
    """)

    st.sidebar.header("Masukkan Parameter Kualitas Air:")

    # Input dari pengguna
    input_data = {}
    # Urutan input disamakan dengan urutan kolom saat training
    # 'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    
    # Mendapatkan nilai rata-rata dari data training untuk default input
    # Perlu X_train sebelum scaling untuk nilai default yang masuk akal
    _, _, _, _, _, original_cols = load_data() # Memuat ulang untuk mendapatkan X (belum discaled)
    
    # Jika load_data berhasil dan mengembalikan original_cols (X.columns)
    # dan X_train_scaled juga berhasil dimuat (artinya data asli ada)
    # Maka kita bisa coba mendapatkan X_train asli untuk nilai default
    # Ini agak rumit karena scaler dilatih di X_train, jadi kita butuh X_train asli
    # Untuk kesederhanaan, kita akan gunakan placeholder atau nilai tengah jika tidak mudah didapat
    
    # Menggunakan nilai median dari data training sebelum scaling sebagai default
    # Ini memerlukan pemuatan data mentah lagi sebelum dibagi dan discale, hanya untuk mendapatkan median fitur.
    # Atau, kita bisa menggunakan nilai tengah dari rentang yang mungkin.
    # Untuk contoh ini, kita akan set default secara manual atau biarkan kosong.
    # Jika ingin default dari data, pastikan scaler.mean_ dan scaler.scale_ dapat digunakan untuk inverse_transform
    # atau simpan X_train asli.

    # Contoh nilai default (bisa disesuaikan berdasarkan statistik dataset Anda jika diinginkan)
    default_values = {
        'ph': 7.0, 'Hardness': 190.0, 'Solids': 20000.0, 'Chloramines': 7.0,
        'Sulfate': 330.0, 'Conductivity': 420.0, 'Organic_carbon': 14.0,
        'Trihalomethanes': 65.0, 'Turbidity': 4.0
    }
    
    for col_name in feature_names:
        # Menentukan min_value dan max_value yang wajar, bisa berdasarkan data asli jika dianalisis
        min_val = 0.0
        # max_val bisa diset sangat tinggi atau berdasarkan data
        if col_name == "Solids":
            max_val = 65000.0
        elif col_name == "Hardness" or col_name == "Sulfate" or col_name == "Conductivity":
            max_val = 1000.0
        elif col_name == "ph" or col_name == "Organic_carbon" or col_name == "Turbidity":
            max_val = 20.0
        elif col_name == "Chloramines" :
            max_val = 15.0
        elif col_name == "Trihalomethanes":
             max_val = 130.0
        else:
            max_val = None # Tidak ada batas atas spesifik

        input_data[col_name] = st.sidebar.number_input(
            label=f"{col_name.replace('_', ' ').title()}",
            value=default_values.get(col_name, 0.0), # Gunakan default jika ada
            min_value=min_val,
            max_value=max_val,
            step=0.1 if col_name in ['ph', 'Chloramines', 'Organic_carbon', 'Turbidity'] else 1.0,
            format="%.2f" if col_name in ['ph', 'Chloramines', 'Organic_carbon', 'Turbidity', 'Hardness', 'Solids', 'Sulfate', 'Conductivity', 'Trihalomethanes'] else "%d"
        )


    if st.sidebar.button("💧 Prediksi Kelayakan"):
        input_df = pd.DataFrame([input_data])
        
        # Pastikan urutan kolom input_df sama dengan saat training scaler
        input_df = input_df[feature_names]

        input_scaled = scaler.transform(input_df)

        # Prediksi dari setiap model
        pred_svm_proba = model_svm.predict_proba(input_scaled)[0][1] # Probabilitas kelas 1 (Layak)
        pred_rf_proba = model_rf.predict_proba(input_scaled)[0][1]
        pred_knn_proba = model_knn.predict_proba(input_scaled)[0][1]
        pred_lr_proba = model_lr.predict_proba(input_scaled)[0][1]
        pred_xgb_proba = model_xgb.predict_proba(input_scaled)[0][1]

        pred_svm = model_svm.predict(input_scaled)[0]
        pred_rf = model_rf.predict(input_scaled)[0]
        pred_knn = model_knn.predict(input_scaled)[0]
        pred_lr = model_lr.predict(input_scaled)[0]
        pred_xgb = model_xgb.predict(input_scaled)[0]

        # Menampilkan hasil
        st.subheader("🔬 Hasil Prediksi Model:")
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            **Support Vector Machine (SVM) + Optuna:**
            - Probabilitas Layak Minum: **{pred_svm_proba*100:.2f}%**
            - Prediksi: **{'Layak Minum 👍' if pred_svm == 1 else 'Tidak Layak Minum 👎'}**
            - Akurasi Model (pada test set): **{acc_svm*100:.2f}%**
            <details><summary>Parameter Terbaik Optuna SVM</summary><p><small>{params_svm}</small></p></details>
            """, unsafe_allow_html=True)
            st.markdown("---")

            st.markdown(f"""
            **K-Nearest Neighbors (KNN) + Optuna:**
            - Probabilitas Layak Minum: **{pred_knn_proba*100:.2f}%**
            - Prediksi: **{'Layak Minum 👍' if pred_knn == 1 else 'Tidak Layak Minum 👎'}**
            - Akurasi Model (pada test set): **{acc_knn*100:.2f}%**
            <details><summary>Parameter Terbaik Optuna KNN</summary><p><small>{params_knn}</small></p></details>
            """, unsafe_allow_html=True)
            st.markdown("---")
            
            st.markdown(f"""
            **XGBoost + Optuna:**
            - Probabilitas Layak Minum: **{pred_xgb_proba*100:.2f}%**
            - Prediksi: **{'Layak Minum 👍' if pred_xgb == 1 else 'Tidak Layak Minum 👎'}**
            - Akurasi Model (pada test set): **{acc_xgb*100:.2f}%**
            <details><summary>Parameter Terbaik Optuna XGBoost</summary><p><small>{params_xgb}</small></p></details>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            **Random Forest + Optuna:**
            - Probabilitas Layak Minum: **{pred_rf_proba*100:.2f}%**
            - Prediksi: **{'Layak Minum 👍' if pred_rf == 1 else 'Tidak Layak Minum 👎'}**
            - Akurasi Model (pada test set): **{acc_rf*100:.2f}%**
            <details><summary>Parameter Terbaik Optuna Random Forest</summary><p><small>{params_rf}</small></p></details>
            """, unsafe_allow_html=True)
            st.markdown("---")

            st.markdown(f"""
            **Logistic Regression + Optuna:**
            - Probabilitas Layak Minum: **{pred_lr_proba*100:.2f}%**
            - Prediksi: **{'Layak Minum 👍' if pred_lr == 1 else 'Tidak Layak Minum 👎'}**
            - Akurasi Model (pada test set): **{acc_lr*100:.2f}%**
            <details><summary>Parameter Terbaik Optuna Logistic Regression</summary><p><small>{params_lr}</small></p></details>
            """, unsafe_allow_html=True)

        # Kesimpulan berdasarkan mayoritas atau rata-rata probabilitas
        predictions = [pred_svm, pred_rf, pred_knn, pred_lr, pred_xgb]
        final_pred_majority = 1 if sum(predictions) >= 3 else 0 # Mayoritas dari 5 model

        st.markdown("---")
        st.subheader("📜 Kesimpulan Umum:")
        if final_pred_majority == 1:
            st.success("Berdasarkan mayoritas prediksi model, air **LAYAK** diminum. ✅")
        else:
            st.error("Berdasarkan mayoritas prediksi model, air **TIDAK LAYAK** diminum. ❌")

        st.info("Catatan: Harap diingat, prediksi kelayakan air ini adalah estimasi berdasarkan model dan memiliki potensi kesalahan. Jangan jadikan hasil ini sebagai satu-satunya dasar; pengujian laboratorium resmi adalah standar utama untuk menentukan kelayakan konsumsi air.")

else:
    if X_train_scaled is None:
        st.error("Gagal memuat data atau model. Silakan periksa file dataset dan coba lagi.")
    else:
        st.info("Masukkan parameter kualitas air di sidebar dan klik 'Prediksi Kelayakan'.")

st.sidebar.markdown("---")
st.sidebar.markdown("Proyek Capstone")