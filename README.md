# Optimasi Long Short Term Memory menggunakan Bayesian Optimization untuk Meramalkan Konsentrasi PM2.5 di Jakarta

## 📋 Daftar Isi
- [Ringkasan Proyek](#ringkasan-proyek)
- [Dataset](#dataset)
- [Metodologi](#metodologi)
- [Struktur Notebook](#struktur-notebook)
- [Hasil Utama](#hasil-utama)
- [Instalasi & Setup](#instalasi--setup)
- [Cara Menggunakan](#cara-menggunakan)
- [Perbandingan Model](#perbandingan-model)
- [Kesimpulan](#kesimpulan)

---

## 🎯 Ringkasan Proyek

Proyek ini mengembangkan model prediksi untuk konsentrasi PM2.5 (particulate matter) di Jakarta menggunakan arsitektur **Long Short Term Memory (LSTM)** yang dioptimalkan dengan **Bayesian Optimization**. 

**Tujuan utama:**
- Membangun model baseline LSTM untuk memprediksi konsentrasi PM2.5
- Mengoptimalkan hyperparameter menggunakan Bayesian Optimization
- Membandingkan performa Baseline LSTM dengan BO-LSTM
- Mengevaluasi signifikansi statistik perbedaan antar model menggunakan Diebold-Mariano test

---

## 📊 Dataset

### Sumber Data
- **Dataset:** Daily PM2.5 concentration data dari AQICN (Air Quality Index China Network)
- **Periode:** 1 Februari 2022 — 10 Oktober 2025
- **Total observasi:** 1,348 hari
- **Lokasi:** Jakarta, Indonesia

### Karakteristik Data

| Metrik | Nilai |
|--------|-------|
| **Total baris** | 1,348 |
| **Kolom** | Tanggal, Konsentrasi PM2.5 |
| **Missing values** | 24 (1.78%) |
| **Rentang PM2.5** | 13 – 187 µg/m³ |
| **Mean PM2.5** | ~62 µg/m³ |
| **Std Dev PM2.5** | ~27 µg/m³ |

### Penanganan Missing Values
Missing values ditangani menggunakan **Seasonal Mean Imputation**:
- Untuk setiap tanggal yang hilang (bulan dan hari yang sama), nilai diisi dengan rata-rata dari tanggal serupa di tahun-tahun lain
- Hasil imputasi tersimpan dalam `hasil_imputasi.csv`
- Semua 24 missing values berhasil diimputasi

---

## 🔬 Metodologi

### 1. **Eksplorasi Data (EDA)**
- Visualisasi time series untuk melihat pola temporal
- Analisis Autocorrelation Function (ACF) untuk mengidentifikasi seasonal patterns
- Deteksi garis putus-putus pada lag 365 dan 730 (siklus tahunan)

### 2. **Preprocessing Data**
- **Normalisasi Z-Score:** Mean = 0, Std = 1
- **Sliding Window:** Input: 365 hari sebelumnya → Output: PM2.5 hari berikutnya
- **Train-Val-Test Split (Walk-Forward Validation):**
  - Training: 730 hari (2 tahun)
  - Validation: 365 hari (1 tahun)  
  - Test: 30 hari per fold (expanding window)

### 3. **Arsitektur LSTM**

**Baseline LSTM (Konfigurasi default):**
```
Input Layer: (batch_size, 365, 1)
  ↓
LSTM Layer 1: 64 units, return_sequences=True
  ↓
LSTM Layer 2: 64 units, return_sequences=False
  ↓
Dropout: 0.0
  ↓
Dense Layer: 1 neuron (output prediksi)
```

**Optimasi Hyperparameter (Bayesian Optimization):**
- Units Layer 1: 32 – 256
- Units Layer 2: 32 – 256
- Dropout rate: 0.1 – 0.5
- Batch size: 16 – 128
- Learning rate: 0.0001 – 0.001

### 4. **Training Strategy (Two-Phase)**

**Phase 1: Train → Validation**
- Early Stopping (monitor: val_loss, patience: 10 epochs)
- ReduceLROnPlateau (factor: 0.5, patience: 5 epochs)
- Tujuan: menemukan epoch optimal

**Phase 2: Retrain (Train+Val) → Test**
- Retrain menggunakan epoch optimal dari Phase 1
- Predictive forecastingiterativ (rolling window)
- Evaluasi pada test set

### 5. **Bayesian Optimization**
- **Init points:** 10 (random exploration)
- **Iterations:** 20 (Bayesian-guided search)
- **Total evaluations:** 30 configurations
- **Objective:** Minimize average validation RMSE across 8 folds

### 6. **Evaluasi Statistik**
- **RMSE (Root Mean Squared Error)** pada train, val, test
- **Diebold-Mariano (DM) Test** untuk membandingkan akurasi prediksi antar model
- Significance level: α = 0.05 (one-tailed)

---

## 📁 Struktur Notebook

### **BAGIAN 1 — SETUP & KONFIGURASI**
- Konfigurasi Kaggle & GPU T4
- Suppress logging TensorFlow
- Import libraries
- Setup direktori output
- Random seed initialization

**Output:** Sistem siap untuk training

---

### **BAGIAN 2 — KONSTRUKSI DATA SIMULASI**
- Load data CSV dari AQICN
- Validasi format tanggal & konsentrasi
- Imputasi seasonal mean untuk missing values
- Output CSV:
  - `hasil_imputasi.csv` (24 rows)
  - `data_simulasi_lengkap.csv` (1348 rows)

**Output:** Dataset lengkap tanpa missing values (df_processed)

---

### **BAGIAN 3 — PROSES ANALISIS**

#### **3.1 Eksplorasi Pola Data**
- **3.1.1** Visualisasi deret waktu PM2.5
- **3.1.2** Analisis ACF (autocorrelation) hingga 1095 lags
- Output PNG: `time_series_grafik.png`, `acf_pm25.png`

#### **3.2 Pemodelan**
- **3.2.1** Fungsi utilitas: dataset creation & LSTM builder
- **3.2.2** Fungsi walk-forward validation (expanding window)
- **3.2.3** Evaluasi Baseline LSTM (8 folds)
- **3.2.4** Fungsi objektif Bayesian Optimization
- **3.2.5** Jalankan BO (30 iterations)
- **3.2.6** Evaluasi BO-LSTM (8 folds)

#### **3.3 Perbandingan Model**
- **3.3.1** Side-by-side RMSE comparison
- **3.3.2** Diebold-Mariano statistical test
- **3.3.3** Visualisasi (time series, boxplot error per split)

---

## 📈 Hasil Utama

### Performance Summary

| Metrik | Baseline LSTM | BO-LSTM | Improvement |
|--------|---------------|---------|-------------|
| **Val RMSE (mean ± std)** | 20.1875 ± 0.4379 | 20.1782 ± 0.3438 | +0.05% ✓ |
| **Test RMSE (mean ± std)** | 29.7464 ± 8.8589 | 26.7632 ± 7.1853 | **+10.03%** ✓ |
| **Train RMSE (mean ± std)** | 17.4333 ± 0.4854 | 17.5710 ± 0.2082 | -0.79% |
| **Avg Duration (s)** | 41.70 | 44.46 | - |
| **Total Duration (s)** | 333.64 | 355.65 | - |

### Best Configuration (BO Result)
```
Units LSTM Layer 1:  213
Units LSTM Layer 2:  96
Dropout rate:        0.5000
Batch size:          94
Learning rate:       0.001000
Best Val RMSE:       19.9895
```

### Diebold-Mariano Test Results
| Parameter | Nilai |
|-----------|-------|
| n (observasi) | 240 |
| DM* statistic | 0.1200 |
| t-critical (α=0.05) | 1.6513 |
| p-value | 0.4523 |
| **Keputusan** | **GAGAL TOLAK H₀** (tidak signifikan) |

**Interpretasi:** Meskipun BO-LSTM menunjukkan peningkatan test RMSE sebesar 10.03%, perbedaan tersebut **tidak signifikan secara statistik** pada level kepercayaan 95% berdasarkan Diebold-Mariano test.

---

## 🚀 Instalasi & Setup

### Requirements
```
Python          >= 3.10
TensorFlow      >= 2.19.0
Keras           >= 2.15.0
Pandas          >= 1.5.0
NumPy           >= 1.23.0
Scikit-learn    >= 1.3.0
Matplotlib      >= 3.7.0
Seaborn         >= 0.13.0
Statsmodels     >= 0.14.0
Bayesian-opt    >= 1.4.3
SciPy           >= 1.11.0
```

### Instalasi
```bash
# Clone atau download repository
git clone <repository-url>
cd <project-folder>

# Install dependencies
pip install -r requirements.txt

# Jika menggunakan Kaggle Notebook, GPU akan otomatis tersedia
```

### GPU Configuration (Optional)
Notebook sudah dikonfigurasi untuk GPU T4 Kaggle. Untuk lokal:
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

---

## 💻 Cara Menggunakan

### 1. **Setup Data**
Letakkan file CSV data di path yang sesuai. Update path di Bagian 1:
```python
DATA_PATH = '/path/to/your/data.csv'
```

### 2. **Jalankan Notebook Secara Sequential**
- **Bagian 1:** Konfigurasi (run all)
- **Bagian 2:** Konstruksi data (run all)
- **Bagian 3.1:** EDA (run all untuk visualisasi)
- **Bagian 3.2:** Modeling & BO (run all, estimasi waktu ~20 menit)
- **Bagian 3.3:** Evaluasi & visualisasi (run all)

### 3. **Output Files**
Semua output tersimpan dalam struktur direktori:
```
/kaggle/working/
├── konstruksi_data/
│   ├── hasil_imputasi.csv
│   └── data_simulasi_lengkap.csv
└── proses_analisis/
    ├── visualizations/
    │   ├── time_series_grafik.png
    │   ├── acf_pm25.png
    │   ├── timeseries_test.png
    │   ├── boxplot_training.png
    │   ├── boxplot_validation.png
    │   ├── boxplot_test.png
    │   └── fold1_phase1_val_loss.png
    ├── baseline/
    │   ├── summary/
    │   │   ├── fold_splits.csv
    │   │   ├── results_summary.csv
    │   │   ├── train_predictions_all_folds.csv
    │   │   ├── val_predictions_all_folds.csv
    │   │   └── test_predictions_all_folds.csv
    │   ├── walkforward/
    │   └── visualizations/
    ├── bo_best/
    │   ├── summary/
    │   │   ├── fold_splits.csv
    │   │   ├── results_summary.csv
    │   │   └── [train/val/test]_predictions_all_folds.csv
    │   └── visualizations/
    │       └── fold1_phase1_val_loss.png
    ├── bo_iterations/
    │   ├── detailed/
    │   │   ├── iteration_01_folds.csv
    │   │   ├── iteration_02_folds.csv
    │   │   └── ...
    │   │   └── iterations_summary.csv
    │   └── visualizations/
    └── statistics/
        ├── final_comparison.csv
        ├── step1_loss_differential.csv
        └── ringkasan_dm_test.csv
```

### 4. **Modifikasi Hyperparameter**
Untuk mengubah BO search space, edit cell di 3.2.5:
```python
pbounds = {
    'units1'       : (32, 256),      # Ubah range di sini
    'units2'       : (32, 256),
    'dropout_rate' : (0.1, 0.5),
    'batch_size'   : (16, 128),
    'learning_rate': (0.0001, 0.001)
}
```

---

## 🔍 Perbandingan Model

### Visualisasi Perbandingan

#### 1. **Time Series Prediksi (Test Set)**
Menunjukkan actual vs predicted untuk Baseline & BO-LSTM pada test set (240 hari).

#### 2. **Error Distribution Boxplot**
Tiga boxplot untuk Training, Validation, dan Test set:
- Baseline LSTM (blue): distribusi error baseline
- BO-LSTM (red): distribusi error hasil optimasi
- Garis horizontal di 0 menunjukkan error sempurna

#### 3. **Val-Loss Kurva (Fold 1)**
Menampilkan early stopping behavior di Phase 1:
- Kurat val loss turun hingga epoch optimal
- Vertical line hijau: best epoch
- Vertical line abu-abu: early stopping

### Key Findings

✅ **BO-LSTM unggul di test set:** 10.03% peningkatan RMSE
- Baseline: 29.75 µg/m³
- BO-LSTM: 26.76 µg/m³

⚠️ **Improvement tidak signifikan secara statistik**
- DM test p-value = 0.452 > 0.05
- Mungkin karena volatilitas test set tinggi (std 8.86 → 7.19)

🔄 **Validasi set konsisten untuk kedua model**
- Baseline: 20.19 µg/m³
- BO-LSTM: 20.18 µg/m³
- Selisih < 0.1 µg/m³

📊 **Regularisasi BO lebih baik:** dropout 0.5 mengurangi overfitting

---

## 🎓 Kesimpulan

### Main Takeaways

1. **Bayesian Optimization efektif untuk hyperparameter tuning LSTM**
   - Menemukan konfigurasi dengan val RMSE 19.9895 (lebih baik dari baseline 20.19)
   - Proses otomatis & sistematis dibanding manual tuning

2. **BO-LSTM memberikan peningkatan praktis pada test set**
   - Test RMSE turun ~3 µg/m³ (10.03%)
   - Lebih konsisten di berbagai folds (std 7.19 vs 8.86)

3. **Pentingnya statistical testing**
   - Diebold-Mariano test menunjukkan improvement tidak signifikan
   - Saran: kumpulkan lebih banyak test data atau gunakan ensemble methods

4. **Seasonal patterns terdeteksi jelas**
   - ACF menunjukkan strong autocorrelation di lag 365 & 730
   - Window size 365 hari tepat untuk capturemaximum seasonal dependencies

### Rekomendasi Lanjutan

🔮 **Pengembangan Masa Depan:**
- Tambahkan exogenous features (temperature, humidity, wind speed)
- Implementasikan ensemble (LSTM + GRU + Transformer)
- Gunakan attention mechanism untuk fokus pada lag penting
- Explore ARIMA-LSTM hybrid models
- Real-time API deployment untuk prediksi operasional

---

## 📚 Referensi

### Dataset
- [AQICN (Air Quality Index China Network)](https://aqicn.org/)

### Tools
- TensorFlow/Keras: Deep learning
- Scikit-learn: Preprocessing & metrics
- Bayesian-optimization: Hyperparameter tuning
- Statsmodels: Autocorrelation & statistical tests

---

*© 2025 PM2.5 Prediction Project | Jakarta Air Quality Analysis*
