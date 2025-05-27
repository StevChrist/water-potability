# Prediksi Kelayakan Air Minum (Water Potability Prediction)

Proyek ini bertujuan untuk mengembangkan sebuah sistem berbasis _machine learning_ yang mampu memprediksi tingkat kelayakan air minum berdasarkan berbagai parameter kualitas fisik dan kimiawi. Hasil prediksi dari berbagai model ditampilkan melalui aplikasi web interaktif yang dibangun menggunakan Streamlit.

![Teks Alt untuk Gambar Anda](Poster.png)

## Dataset

Proyek ini menggunakan dataset "Water Quality and Potability" yang tersedia di Kaggle:

- **Link:** [Water Quality and Potability Dataset](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability?resource=download)

Dataset ini berisi pengukuran berbagai parameter kualitas air.

### Fitur (Input Model)

Berikut adalah parameter yang digunakan sebagai input untuk model prediksi:

1.  `ph`: Tingkat pH air.
2.  `Hardness`: Kesadahan air (konsentrasi mineral terlarut).
3.  `Solids`: Total padatan terlarut (TDS).
4.  `Chloramines`: Konsentrasi kloramin (desinfektan).
5.  `Sulfate`: Konsentrasi sulfat.
6.  `Conductivity`: Konduktivitas listrik air (terkait dengan TDS).
7.  `Organic_carbon`: Total karbon organik (TOC).
8.  `Trihalomethanes`: Konsentrasi Trihalometana (produk sampingan desinfeksi).
9.  `Turbidity`: Kekeruhan air.

### Target Variable

- `Potability`: Menunjukkan kelayakan air (1 untuk layak minum, 0 untuk tidak layak minum).

## Metodologi & Teknologi yang Digunakan

- **Model Machine Learning:**
  - Support Vector Machine (SVM)
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - XGBoost
- **Optimasi Hyperparameter:** Optuna
- **Framework Aplikasi Web:** Streamlit
- **Bahasa Pemrograman:** Python
- **Library Utama:**
  - Pandas (untuk manipulasi data)
  - Scikit-learn (untuk model machine learning dan pra-pemrosesan)
  - XGBoost (untuk model XGBoost)
  - Optuna (untuk tuning hyperparameter)
  - Streamlit (untuk membangun antarmuka pengguna)

## Referensi Ilmiah

1.  **Untuk mendukung pernyataan tentang pentingnya air minum dan penggunaan _machine learning_ dalam kualitas air:**

    - Ahmed, U., Mumtaz, R., Anwar, H., Shah, A. A., & Irfan, R. (2022). _Efficient Water Quality Prediction Using Supervised Machine Learning_. Water, _14_(1), 121. [https://doi.org/10.3390/w14010121](https://doi.org/10.3390/w14010121)
      - **Relevansi:** Artikel ini membahas penggunaan berbagai model _machine learning_ untuk memprediksi kualitas air, yang sejalan dengan tujuan proyek ini. Ini juga menyoroti pentingnya kualitas air untuk kesehatan masyarakat dan seringkali menggunakan dataset atau parameter yang serupa.

2.  **Untuk mendukung pembahasan mengenai parameter kualitas air secara umum:**
    - Gorde, S. P., & Jadhav, M. V. (2013). _Assessment of Water Quality Parameters: A Review_. International Journal of Engineering Research and Applications, _3_(6), 2029-2035.
      - **Relevansi:** Artikel ulasan ini membahas berbagai parameter kualitas air (fisik, kimia, biologi) yang penting untuk dievaluasi. Ini memberikan dasar ilmiah mengenai parameter-parameter yang digunakan dalam dataset dan analisis _machine learning_ untuk menentukan kelayakan air.

## Note

Hasil prediksi yang ditampilkan oleh aplikasi ini bersifat sebagai panduan awal dan tidak boleh dianggap sebagai keputusan final tanpa validasi laboratorium profesional. Meskipun model dirancang untuk membantu dengan akurasi tertentu, ada kemungkinan hasil prediksinya keliru atau tidak tepat.
