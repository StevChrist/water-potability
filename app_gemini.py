import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from sklearn.impute import SimpleImputer
import optuna
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import time # Untuk demo progress bar jika perlu
from streamlit_option_menu import option_menu # Import library baru

# --- Konfigurasi Aplikasi ---
st.set_page_config(page_title="AquaCheck Pro - Analisis Kualitas Air",
                   page_icon="💧",
                   layout="wide",
                   initial_sidebar_state="expanded")

# --- Nama File & Data Path ---
DATA_FILE = 'water_potability.csv'
MODEL_FILE = 'svm_model.pkl'
SCALER_FILE = 'scaler.pkl'
IMPUTER_FILE = 'imputer.pkl'
EVAL_DATA_FILE = 'eval_data.pkl'
LOGO_URL = "https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" # Contoh Logo
GITHUB_URL = "https://github.com/your_username/your_repo" # Ganti dengan link Github Anda

# --- Tooltips / Bantuan untuk Fitur Input ---
feature_tooltips = {
    'ph': 'Tingkat keasaman atau kebasaan air (0-14). Ideal: 6.5 - 8.5.',
    'Hardness': 'Kandungan mineral terlarut (kalsium & magnesium).',
    'Solids': 'Total padatan terlarut (TDS) dalam ppm.',
    'Chloramines': 'Disinfektan berbasis klorin & amonia dalam ppm. Standar: < 4 ppm.',
    'Sulfate': 'Sulfat dalam ppm. Tinggi bisa menyebabkan rasa pahit.',
    'Conductivity': 'Kemampuan air menghantarkan listrik (μS/cm).',
    'Organic_carbon': 'Total Karbon Organik (TOC) dalam ppm.',
    'Trihalomethanes': 'Produk sampingan desinfeksi klorin (μg/L).',
    'Turbidity': 'Kekeruhan air (NTU).'
}

# --- CSS Kustom untuk Tombol ---
def load_css():
    st.markdown("""
        <style>
            /* Target tombol utama (yang type="primary") */
            button[data-testid="baseButton-primary"] {
                background-color: #0068C9; /* Warna biru Streamlit */
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: bold;
                border: none;
                transition: background-color 0.3s ease;
            }
            button[data-testid="baseButton-primary"]:hover {
                background-color: #00509E; /* Warna biru lebih gelap saat hover */
            }
            /* Target tombol sekunder (Reset) */
            button[data-testid="baseButton-secondary"] {
                 border-radius: 10px;
            }
            /* Styling Sidebar (opsional, sesuaikan) */
            [data-testid="stSidebar"] {
                 /* background-color: #f0f2f6; */
            }
        </style>
    """, unsafe_allow_html=True)

# --- 1. Fungsi Pemuatan & Pemrosesan Data ---
@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"Error: File '{filepath}' tidak ditemukan. Pastikan ada di folder yang sama.")
        st.stop() # Hentikan eksekusi jika file tidak ada
    except Exception as e:
        st.error(f"Terjadi error saat memuat data: {e}")
        st.stop()

@st.cache_data
def preprocess_data(_df):
    if _df is None: return None, None, None, None, None, None
    df_clean = _df.copy()
    imputer = SimpleImputer(strategy='median')
    features_to_impute = df_clean.drop('Potability', axis=1).columns
    df_clean[features_to_impute] = imputer.fit_transform(df_clean[features_to_impute])
    X = df_clean.drop('Potability', axis=1)
    y = df_clean['Potability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, df_clean, imputer

# --- 2. Fungsi Pelatihan Model & Optuna ---
@st.cache_resource
def train_and_save_model(X_train, y_train, X_test, y_test, _imputer):
    """
    Melatih model jika file tidak ada, atau memuatnya jika sudah ada.
    Membuat file .pkl yang diperlukan secara otomatis.
    """
    all_files_exist = (os.path.exists(MODEL_FILE) and
                       os.path.exists(SCALER_FILE) and
                       os.path.exists(EVAL_DATA_FILE) and
                       os.path.exists(IMPUTER_FILE))

    if all_files_exist:
        # st.info("Memuat model, scaler, dan data evaluasi...")
        try:
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            eval_data = joblib.load(EVAL_DATA_FILE)
            imputer_loaded = joblib.load(IMPUTER_FILE)
            # st.success("Berhasil dimuat.")
            return model, scaler, eval_data, imputer_loaded
        except Exception as e:
            st.warning(f"Gagal memuat file .pkl ({e}). Memulai pelatihan ulang...")
            for f in [MODEL_FILE, SCALER_FILE, EVAL_DATA_FILE, IMPUTER_FILE]:
                if os.path.exists(f): os.remove(f)

    st.warning("Memulai proses pelatihan model (ini mungkin butuh beberapa menit)...")
    progress_bar = st.progress(0, text="Mempersiapkan data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    progress_bar.progress(10, text="Mencari hyperparameter (Optuna)...")

    def objective(trial, X_train_s, y_train_s, X_val_s, y_val_s):
        C = trial.suggest_loguniform('C', 1e-3, 1e3)
        kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
        gamma = 'scale'
        degree = trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3
        model_trial = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, probability=True, random_state=42)
        model_trial.fit(X_train_s, y_train_s)
        return accuracy_score(y_val_s, model_trial.predict(X_val_s))

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    n_trials = 30 # Anda bisa mengurangi jumlah trial untuk pengujian lebih cepat
    for i in range(n_trials):
        study.optimize(lambda trial: objective(trial, X_train_scaled, y_train, X_test_scaled, y_test), n_trials=1)
        progress_bar.progress(10 + int(80 * (i + 1) / n_trials), text=f"Optuna Trial {i+1}/{n_trials}")

    progress_bar.progress(90, text="Melatih model final...")
    best_params = study.best_params
    st.write("Hyperparameter terbaik:", best_params)
    if 'degree' not in best_params and best_params.get('kernel') == 'poly': best_params['degree'] = 3
    final_model = SVC(**best_params, probability=True, random_state=42)
    final_model.fit(X_train_scaled, y_train)

    y_pred = final_model.predict(X_test_scaled)
    y_proba = final_model.predict_proba(X_test_scaled)[:, 1]
    eval_data = {'X_test_scaled': X_test_scaled, 'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba}

    joblib.dump(final_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(eval_data, EVAL_DATA_FILE)
    joblib.dump(_imputer, IMPUTER_FILE)

    progress_bar.progress(100, text="Selesai!")
    st.success("Model berhasil dilatih dan disimpan.")

    return final_model, scaler, eval_data, _imputer

# --- 3. Halaman-Halaman Aplikasi ---

## Halaman Dashboard Eksplorasi
def show_exploration_page(df, X_imputed):
    st.title("📊 Dashboard Eksplorasi Data Kualitas Air")
    st.markdown("Jelajahi dataset kualitas air melalui visualisasi interaktif.")
    st.markdown("---")
    total_samples = len(df)
    missing_values = df.isnull().sum().sum()
    potable_count = df['Potability'].sum()
    not_potable_count = total_samples - potable_count
    col1, col2, col3= st.columns(3)
    col1.metric("Total Sampel", f"{total_samples}")
    col2.metric("Layak Minum (1)", f"{potable_count}", f"{potable_count/total_samples:.1%}")
    col3.metric("Tidak Layak Minum (0)", f"{not_potable_count}", f"-{not_potable_count/total_samples:.1%}") 
    # col4.metric("Total Missing Values", f"{missing_values} (Sebelum Imputasi)")
    st.markdown("---")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown("##### Distribusi Kelayakan")
        potability_counts = df['Potability'].value_counts().reset_index()
        potability_counts.columns = ['Potability', 'Count']
        potability_counts['Potability'] = potability_counts['Potability'].map({0: 'Tidak Layak', 1: 'Layak'})
        fig_pie = px.pie(potability_counts, values='Count', names='Potability', hole=0.3, color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pie.update_layout(showlegend=True, margin=dict(t=20, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        st.markdown("##### Matriks Korelasi")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
        sns.heatmap(X_imputed.corr(), annot=True, cmap='viridis', fmt=".2f", linewidths=.5, ax=ax_corr, annot_kws={"size": 8})
        st.pyplot(fig_corr)
    st.markdown("---")
    st.subheader("Distribusi Fitur")
    features = X_imputed.columns
    for i in range(0, len(features), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(features):
                feature = features[i+j]
                fig_hist = px.histogram(X_imputed, x=feature, title=f'{feature}', marginal="box", color_discrete_sequence=['#1f77b4'])
                fig_hist.update_layout(title_font_size=14, height=300, margin=dict(t=40, b=40, l=40, r=20))
                cols[j].plotly_chart(fig_hist, use_container_width=True)


## Halaman Prediksi Manual
def show_prediction_page(X_imputed, model, scaler, imputer): # Tambah imputer jika diperlukan
    st.title("🔬 Prediksi Manual Kelayakan Air")
    st.write("Gunakan slider atau input box untuk memasukkan nilai parameter air dan dapatkan prediksi.")

    # Inisialisasi session state jika belum ada
    if 'inputs' not in st.session_state:
        st.session_state.inputs = {col: float(X_imputed[col].median()) for col in X_imputed.columns}

    # Fungsi untuk mereset input
    def reset_inputs():
        st.session_state.inputs = {col: float(X_imputed[col].median()) for col in X_imputed.columns}

    # Tampilkan input saat ini
    current_inputs_df = pd.DataFrame(st.session_state.inputs, index=[0])
    # st.subheader("Parameter yang Anda Masukkan:")
    # st.dataframe(current_inputs_df.style.format("{:.2f}"))

    # Form untuk input dan tombol prediksi
    with st.form(key='prediction_form'):
        st.header('Masukkan / Ubah Parameter Air')
        inputs_form = {}
        cols_input = st.columns(3) # Buat 3 kolom untuk input
        col_index = 0

        for col in X_imputed.columns:
            min_val = float(X_imputed[col].min())
            max_val = float(X_imputed[col].max())
            median_val = float(X_imputed[col].median())
            with cols_input[col_index]:
                # Gunakan nilai dari session_state
                inputs_form[col] = st.number_input(
                    label=col,
                    min_value=min_val,
                    max_value=max_val,
                    value=st.session_state.inputs[col], # Ambil dari session state
                    step=0.1, # Sesuaikan step jika perlu
                    help=f"{feature_tooltips.get(col, 'Masukkan nilai fitur.')} (Median: {median_val:.2f})",
                    key=f"input_{col}" # Kunci unik untuk input
                )
            col_index = (col_index + 1) % 3 # Pindah ke kolom berikutnya

        # Tombol submit dan reset di dalam form
        submit_col, reset_col = st.columns([1, 5]) # Atur rasio kolom
        with submit_col:
            submitted = st.form_submit_button('✨ Prediksi!', type="primary", use_container_width=True)
        with reset_col:
            st.form_submit_button('🔄 Reset', on_click=reset_inputs, type="secondary", use_container_width=False) # Tombol reset


    # Lakukan prediksi JIKA form disubmit
    if submitted:
        # Update session state dengan nilai dari form saat disubmit
        st.session_state.inputs = inputs_form.copy()
        user_df = pd.DataFrame(inputs_form, index=[0])

        with st.spinner('Melakukan prediksi...'):
            try:
                # Tidak perlu imputasi karena user memasukkan semua nilai
                user_df_scaled = scaler.transform(user_df)
                prediction = model.predict(user_df_scaled)
                prediction_proba = model.predict_proba(user_df_scaled)

                st.subheader('Hasil Prediksi:')
                prob_potable = prediction_proba[0][1]
                prob_not_potable = prediction_proba[0][0]

                if prediction[0] == 1:
                    st.success(f'✅ **Layak Diminum** (Potable)')
                    st.progress(prob_potable)
                    # st.write(f"Keyakinan Model: {prob_potable:.2%}")
                else:
                    st.error(f'❌ **Tidak Layak Diminum** (Not Potable)')
                    # st.progress(prob_not_potable)
                    # st.write(f"Keyakinan Model: {prob_not_potable:.2%}")

                # Tambahkan Gauge Chart (opsional)
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_potable * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Skor Kelayakan Air (0-100)"},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps' : [
                            {'range': [0, 50], 'color': 'red'},
                            {'range': [50, 100], 'color': 'green'}],
                        'threshold' : {'line': {'color': "orange", 'width': 4}, 'thickness': 0.75, 'value': 50}
                        }))
                st.plotly_chart(fig_gauge, use_container_width=True)


            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")

## Halaman Evaluasi Model
def show_evaluation_page(eval_data):
    st.title("📈 Evaluasi Kinerja Model SVM")
    st.write("Melihat seberapa baik model SVM kita bekerja pada data uji.")
    y_test = eval_data['y_test']
    y_pred = eval_data['y_pred']
    y_proba = eval_data['y_proba']
    st.subheader("Laporan Klasifikasi")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose())
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Tidak Layak', 'Layak'],
                    yticklabels=['Tidak Layak', 'Layak'])
        ax_cm.set_xlabel('Prediksi')
        ax_cm.set_ylabel('Aktual')
        st.pyplot(fig_cm)
    with col2:
        st.subheader("Kurva ROC-AUC")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=f'SVM (AUC = {roc_auc:.2f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                     line=dict(dash='dash'), name='Random Chance'))
        fig_roc.update_layout(title='Receiver Operating Characteristic (ROC)',
                              xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate',
                              height=400)
        st.plotly_chart(fig_roc, use_container_width=True)
    st.info(f"Model SVM mencapai **Akurasi {accuracy_score(y_test, y_pred):.2%}** dan **AUC {roc_auc:.2f}** pada data uji.")


## Halaman Penjelasan Metode & Tentang
def show_methodology_page():
    st.title("📖 Penjelasan Metode & Tentang Aplikasi")
    st.markdown("""
    **AquaCheck Pro** adalah aplikasi web interaktif yang dibangun sebagai bagian dari proyek *capstone*.
    Tujuannya adalah untuk memprediksi kelayakan air minum berdasarkan parameter kimia dan fisikanya.
    """)

    st.subheader("Sumber Dataset")
    st.markdown("""
    Kami menggunakan dataset 'Water Potability' yang bersumber dari [Kaggle](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability).
    Dataset ini berisi 3276 sampel air dengan 9 fitur input dan 1 fitur target (Potability).
    *Missing values* ditangani menggunakan **Imputasi Median**.
    """)

    st.subheader("Model: Support Vector Machine (SVM)")
    st.markdown("""
    Model utama yang digunakan adalah **Support Vector Machine (SVM)**, sebuah algoritma *supervised learning*
    untuk tugas klasifikasi. SVM bekerja dengan mencari *hyperplane* optimal yang memaksimalkan margin antara kelas data.
    *(Anda bisa menambahkan gambar ilustrasi SVM di sini menggunakan st.image)*
    """)

    st.subheader("Optimasi: Optuna")
    st.markdown("""
    Untuk menemukan *hyperparameter* terbaik (seperti C, kernel, degree), kami menggunakan **Optuna**,
    sebuah *framework* optimasi *hyperparameter* canggih yang menggunakan algoritma pencarian efisien.
    *(Anda bisa menambahkan gambar ilustrasi Optuna di sini menggunakan st.image)*
    """)

    show_feature_details_section()

    # st.subheader("Tentang Pembuat")
    # st.info(f"""
    # * **Nama**: [Masukkan Nama Anda]
    # * **NIM**: [Masukkan NIM Anda]
    # * **Institusi**: [Masukkan Nama Universitas/Kampus Anda]
    # * **Kontak**: [Masukkan Email atau Kontak Lain]
    # * **Kode Sumber**: [Link Github]({GITHUB_URL})
    # """)


def show_feature_details_section():
    """Menampilkan bagian penjelasan detail untuk setiap fitur."""

    st.subheader("🔬 Penjelasan Detail Fitur Input")
    st.markdown("""
    Berikut adalah penjelasan lebih mendalam mengenai 9 parameter kualitas air yang digunakan oleh model
    untuk memprediksi kelayakan air minum. Memahami parameter ini akan membantu Anda
    menginterpretasikan hasil prediksi dan pentingnya setiap pengukuran.
    """)

    # --- Data Detail Fitur ---
    feature_details = {
        'ph': {
            'name': 'pH',
            'desc': 'pH adalah ukuran tingkat keasaman atau kebasaan air. Skalanya berkisar dari 0 (sangat asam) hingga 14 (sangat basa/alkali), dengan 7 sebagai titik netral.',
            'unit': 'Skala pH (0-14)',
            'ideal_range': '6.5 - 8.5 (Standar umum air minum)',
            'impact': """
            * **Terlalu Rendah (Asam):** Dapat bersifat korosif terhadap pipa logam, melarutkan logam berat ke dalam air, dan memberikan rasa asam.
            * **Terlalu Tinggi (Basa):** Dapat menyebabkan rasa pahit/soda, endapan kerak pada pipa, dan mengurangi efektivitas desinfeksi klorin.
            """
        },
        'Hardness': {
            'name': 'Hardness (Kekerasan Air)',
            'desc': 'Kekerasan air terutama disebabkan oleh kandungan ion kalsium (Ca²⁺) dan magnesium (Mg²⁺) yang terlarut. Air dengan konsentrasi tinggi disebut "air sadah".',
            'unit': 'mg/L (miligram per liter) atau ppm (parts per million)',
            'ideal_range': 'Umumnya < 150 mg/L (meskipun tidak ada batas kesehatan spesifik, lebih ke preferensi & teknis)',
            'impact': """
            * **Tinggi:** Menyebabkan pembentukan kerak pada pemanas air dan pipa, mengurangi efektivitas sabun (sulit berbusa), dan terkadang memengaruhi rasa.
            * **Rendah:** Air "lunak" biasanya lebih disukai, tetapi air yang sangat lunak bisa bersifat sedikit korosif.
            """
        },
        'Solids': {
            'name': 'Solids (Total Dissolved Solids - TDS)',
            'desc': 'TDS adalah jumlah total semua zat padat (mineral, garam, logam) yang terlarut dalam air.',
            'unit': 'ppm (parts per million) atau mg/L',
            'ideal_range': '< 500 - 600 ppm (Untuk rasa yang baik)',
            'impact': """
            * **Tinggi:** Memberikan rasa asin, pahit, atau metalik pada air; dapat menyebabkan korosi atau endapan; dan mungkin mengindikasikan adanya kontaminan.
            * **Rendah:** Biasanya tidak masalah, tetapi air yang sangat murni (TDS sangat rendah) mungkin terasa "hambar".
            """
        },
        'Chloramines': {
            'name': 'Chloramines',
            'desc': 'Chloramines adalah disinfektan yang dibentuk dengan mencampurkan klorin dan amonia. Digunakan sebagai alternatif klorin bebas untuk mendisinfeksi air minum publik karena lebih stabil dan tahan lama.',
            'unit': 'ppm (parts per million) atau mg/L',
            'ideal_range': '< 4 ppm (Batas standar EPA)',
            'impact': """
            * **Dalam Batas:** Efektif membunuh patogen dan mengurangi pembentukan produk sampingan desinfeksi tertentu (THMs).
            * **Berlebihan:** Dapat menyebabkan rasa dan bau klorin yang kuat; bisa berbahaya bagi pasien dialisis dan ikan akuarium.
            """
        },
        'Sulfate': {
            'name': 'Sulfate (Sulfat)',
            'desc': 'Sulfat adalah zat alami yang ditemukan di banyak sumber air. Bisa berasal dari pelapukan batuan atau aktivitas industri.',
            'unit': 'ppm (parts per million) atau mg/L',
            'ideal_range': '< 250 ppm (Standar sekunder untuk rasa)',
            'impact': """
            * **Tinggi:** Dapat memberikan rasa pahit pada air dan memiliki efek laksatif (pencahar), terutama bagi orang yang tidak terbiasa.
            """
        },
        'Conductivity': {
            'name': 'Conductivity (Daya Hantar Listrik)',
            'desc': 'Mengukur kemampuan air untuk menghantarkan arus listrik. Ini berbanding lurus dengan jumlah ion terlarut (TDS) dalam air.',
            'unit': 'μS/cm (microSiemens per centimeter)',
            'ideal_range': 'Bervariasi, tetapi umumnya < 1500 μS/cm (Sebagai indikator TDS)',
            'impact': """
            * **Tinggi:** Mengindikasikan tingginya TDS, yang dapat memengaruhi rasa dan potensi korosi/kerak. Bukan parameter kesehatan langsung, tetapi indikator kualitas yang baik.
            """
        },
        'Organic_carbon': {
            'name': 'Organic Carbon (Total Organic Carbon - TOC)',
            'desc': 'TOC mengukur jumlah total karbon yang terikat dalam senyawa organik di dalam air. Ini berasal dari bahan tanaman dan hewan yang membusuk atau polusi.',
            'unit': 'ppm (parts per million) atau mg/L',
            'ideal_range': '< 2 ppm (Untuk air permukaan), < 4 ppm (Untuk air tanah)',
            'impact': """
            * **Tinggi:** Menjadi "makanan" bagi mikroorganisme, berpotensi meningkatkan pertumbuhan bakteri; dapat bereaksi dengan disinfektan (seperti klorin) membentuk *Disinfection Byproducts* (DBPs) seperti Trihalomethanes (THMs) yang berbahaya.
            """
        },
        'Trihalomethanes': {
            'name': 'Trihalomethanes (THMs)',
            'desc': 'THMs adalah kelompok senyawa kimia yang terbentuk sebagai produk sampingan ketika klorin atau disinfektan berbasis klorin lainnya bereaksi dengan bahan organik alami (TOC) dalam air.',
            'unit': 'μg/L (mikrogram per liter) atau ppb (parts per billion)',
            'ideal_range': '< 80 μg/L (Batas standar EPA Total THMs)',
            'impact': """
            * **Tinggi:** Beberapa THMs (seperti kloroform) diklasifikasikan sebagai kemungkinan karsinogen (penyebab kanker) jika terpapar dalam jangka panjang. Pengendaliannya penting untuk kesehatan publik.
            """
        },
        'Turbidity': {
            'name': 'Turbidity (Kekeruhan)',
            'desc': 'Kekeruhan adalah ukuran kejernihan atau kekaburan air. Disebabkan oleh partikel tersuspensi seperti lumpur, pasir, tanah liat, atau mikroorganisme.',
            'unit': 'NTU (Nephelometric Turbidity Units)',
            'ideal_range': '< 1 NTU (Idealnya < 0.5 NTU untuk efektivitas desinfeksi)',
            'impact': """
            * **Tinggi:** Dapat melindungi mikroorganisme (bakteri, virus) dari efek disinfektan; mengganggu efektivitas desinfeksi; menyebabkan masalah estetika (air terlihat kotor); dan dapat mengindikasikan adanya patogen.
            """
        }
    }

    # --- Tampilkan Expander untuk Setiap Fitur ---
    for key, value in feature_details.items():
        with st.expander(f"**{value['name']}**"):
            st.markdown(f"**Deskripsi:** {value['desc']}")
            st.markdown(f"**Satuan Umum:** {value['unit']}")
            st.markdown(f"**Rentang Ideal/Standar:** {value['ideal_range']}")
            st.markdown(f"**Dampak / Mengapa Penting:** {value['impact']}")
def main():
    load_css() # Muat CSS kustom
    df_raw = load_data(DATA_FILE)
    X_train, X_test, y_train, y_test, df_clean, imputer = preprocess_data(df_raw)
    X = df_clean.drop('Potability', axis=1)

    # --- Latih/Muat Model ---
    with st.spinner("Mempersiapkan model..."):
        model, scaler, eval_data, loaded_imputer = train_and_save_model(X_train, y_train, X_test, y_test, imputer)

    # --- Sidebar ---
    with st.sidebar:
        # st.image(LOGO_URL, width=150) # Logo bisa ditaruh di sini atau di bawah
        st.title("💧 AquaCheck Pro")
        st.write("Analisis & Prediksi Kualitas Air")
        st.markdown("---")

        # --- Navigasi Sidebar Baru ---
        selected = option_menu(
            menu_title="Menu Navigasi",  # required
            options=["Dashboard", "Prediksi", "Evaluasi", "Tentang"],  # required
            icons=["bar-chart-line", "sliders", "clipboard-data", "book"],  # optional
            menu_icon="water",  # optional
            default_index=0,  # optional
            orientation="vertical", # "horizontal" atau "vertical"
            styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#0068C9"},
            }
        )
        # st.markdown("---")
        # st.subheader("Info Dataset")
        # st.metric("Jumlah Data", len(df_raw))
        # potable_pct = df_raw['Potability'].mean() * 100
        # st.metric("Layak Minum", f"{potable_pct:.1f}%")
        st.markdown("---")
        st.info(f"""
            Proyek Capstone | Kelompok 13 | S1 Sains Data
            """)


    # --- Tampilkan Halaman yang Dipilih ---
    if selected == "Dashboard":
        show_exploration_page(df_raw, X)
    elif selected == "Prediksi":
        # Halaman prediksi tidak lagi menggunakan sidebar untuk form
        show_prediction_page(X, model, scaler, loaded_imputer)
    elif selected == "Evaluasi":
        show_evaluation_page(eval_data)
    elif selected == "Tentang":
        show_methodology_page()


if __name__ == '__main__':
    main()