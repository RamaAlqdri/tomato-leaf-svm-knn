# Tomato Leaf Disease Classification: SVM vs KNN

Project ini mereplikasi penelitian **"Comparison of Tomato Leaf Disease Classification Accuracy Using Support Vector Machine and K-Nearest Neighbor Methods"** untuk klasifikasi penyakit daun tomat menggunakan machine learning klasik.

## Ringkasan Jurnal

Jurnal membandingkan Support Vector Machine dan K-Nearest Neighbor pada 5 kelas daun tomat: Early Blight, Healthy, Late Blight, Mosaic Virus, dan Yellow Leaf Curl Virus. Dataset yang dilaporkan berjumlah 5000 gambar, 1000 gambar per kelas, dengan pembagian 800 training dan 200 testing per kelas. Evaluasi memakai 5-Fold, 10-Fold, dan 20-Fold Cross Validation. Hasil terbaik jurnal adalah SVM 10-Fold dengan accuracy, precision, dan recall sebesar 95.3%.

## Struktur Folder

```text
tomato-leaf-svm-knn/
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── notebooks/
├── reports/
│   ├── figures/
│   ├── analisis_jurnal.md
│   ├── dataset_analysis.md
│   └── results_summary.md
├── src/
│   ├── analyze_dataset.py
│   ├── prepare_dataset.py
│   ├── feature_extraction.py
│   ├── train_svm.py
│   ├── train_knn.py
│   ├── evaluate_models.py
│   ├── compare_results.py
│   ├── utils.py
│   └── config.py
├── models/
├── requirements.txt
├── README.md
└── main.py
```

## Menaruh Dataset Lokal

Default dataset berada di:

```python
DATASET_DIR = "data/raw"
```

Ubah nilai tersebut di `src/config.py` jika dataset Anda berada di lokasi lain. Struktur yang didukung fleksibel, termasuk:

```text
data/raw/train/Tomato___Early_blight/
data/raw/val/Tomato___Early_blight/
```

atau folder kelas langsung di dalam `data/raw`.

Folder kelas PlantVillage akan dipetakan ke label sederhana:

- `Tomato___Early_blight` -> `Early Blight`
- `Tomato___healthy` -> `Healthy`
- `Tomato___Late_blight` -> `Late Blight`
- `Tomato___Tomato_mosaic_virus` -> `Mosaic Virus`
- `Tomato___Tomato_Yellow_Leaf_Curl_Virus` -> `Yellow Leaf Curl Virus`

Kelas lain akan terbaca pada analisis dataset tetapi diabaikan pada training agar sesuai jurnal.

## Install Dependency

Gunakan Python 3.10 sampai 3.12. Jika `python3` sistem Anda terlalu baru dan dependency belum tersedia, pakai interpreter Python 3.10/3.11/3.12 lain yang tersedia di mesin.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Menjalankan Pipeline

Analisis dataset:

```bash
python main.py --analyze-dataset
```

Preprocessing dan split 80/20:

```bash
python main.py --prepare
```

Ekstraksi fitur:

```bash
python main.py --extract-features
```

Training SVM:

```bash
python main.py --train-svm
```

Training KNN:

```bash
python main.py --train-knn
```

Evaluasi test set:

```bash
python main.py --evaluate
```

Perbandingan hasil:

```bash
python main.py --compare
```

Menjalankan seluruh pipeline:

```bash
python main.py --all
```

## Output Utama

- `reports/analisis_jurnal.md`: ringkasan dan analisis jurnal.
- `reports/dataset_summary.csv`: ringkasan jumlah gambar, format, ukuran, dan corrupt image.
- `reports/dataset_analysis.md`: laporan analisis dataset lokal.
- `data/processed/dataset_metadata.csv`: metadata seluruh data target.
- `data/processed/train_metadata.csv`: split training.
- `data/processed/test_metadata.csv`: split testing.
- `data/features/features.csv`: vektor fitur numerik.
- `data/features/labels.csv`: label dan split untuk setiap fitur.
- `models/svm_model.pkl`: model SVM final.
- `models/knn_model.pkl`: model KNN final.
- `reports/svm_results.csv`: hasil cross validation SVM.
- `reports/knn_results.csv`: hasil cross validation KNN, termasuk eksperimen `k=3,5,7,9`.
- `reports/evaluation_svm.txt`: metrik test set SVM.
- `reports/evaluation_knn.txt`: metrik test set KNN.
- `reports/figures/confusion_matrix_svm.png`: confusion matrix SVM.
- `reports/figures/confusion_matrix_knn.png`: confusion matrix KNN.
- `reports/results_summary.md`: kesimpulan perbandingan SVM vs KNN.

## Kenapa Perlu Ekstraksi Fitur

SVM dan KNN adalah metode machine learning klasik yang menerima vektor numerik, bukan citra mentah sebagai representasi visual otomatis. Karena itu, citra daun dikonversi menjadi fitur eksplisit:

- Histogram warna RGB
- Histogram warna HSV
- Tekstur GLCM
- Tekstur LBP

Setelah fitur diekstraksi, `StandardScaler` digunakan dalam pipeline SVM dan KNN agar skala fitur lebih seimbang.

## Catatan Replikasi

Hasil dapat berbeda dari jurnal karena artikel tidak menjelaskan detail preprocessing dan ekstraksi fitur citra sebelum SVM/KNN. Project ini membuat langkah tersebut eksplisit agar eksperimen dapat dijalankan ulang, diaudit, dan dimodifikasi secara lokal.
