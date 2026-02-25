# 👁️ Deteksi Penyakit Retinal OCT dengan Transfer Learning (DenseNet121)

Proyek Deep Learning ini berfokus pada klasifikasi citra medis **Retinal Optical Coherence Tomography (OCT)** untuk mendiagnosis 8 kondisi anatomi dan penyakit mata yang berbeda. Model ini dievaluasi dan dikembangkan sebagai bagian dari submission spesialisasi Deep Learning, dengan pencapaian akurasi di atas **95%**.

## 📌 Deskripsi Proyek
Diagnosis citra medis menuntut tingkat presisi yang tinggi dan kemampuan model untuk mengekstraksi fitur tekstur mikroskopis. Proyek ini mengimplementasikan teknik **Transfer Learning** menggunakan arsitektur bawaan ImageNet yang digabungkan dengan lapisan klasifikasi kustom (*double-funnel architecture*) untuk mencegah *overfitting* sekaligus mempertahankan kemampuan generalisasi pada data yang kompleks.

## 📊 Dataset
Dataset yang digunakan berasal dari dataset klinis Kermany et al., yang terdiri dari lebih dari **84.000 citra OCT**.

Dataset dibagi secara dinamis ke dalam 3 proporsi utama dengan teknik *stratified split*:
* **Data Latih (Training):** 80%
* **Data Validasi (Validation):** 10%
* **Data Uji (Testing):** 10%

**8 Kelas Klasifikasi:**
1. AMD (Age-related Macular Degeneration)
2. CNV (Choroidal Neovascularization)
3. CSR (Central Serous Retinopathy)
4. DME (Diabetic Macular Edema)
5. DR (Diabetic Retinopathy)
6. DRUSEN
7. MH (Macular Hole)
8. NORMAL (Retina Sehat)

## 🧠 Arsitektur Model
Model ini dibangun di atas fondasi Keras (TensorFlow 3.x) dengan memisahkan tahap *Feature Extraction* dan *Fine-Tuning*.



* **Base Model (Feature Extractor):** `DenseNet121` (Bobot dikunci pada Tahap 1, dan dibuka pada Tahap 2 untuk *fine-tuning* dengan *Learning Rate* yang sangat kecil).
* **Custom Classifier (Otak Model):**
    * `GlobalAveragePooling2D`: Mengurangi dimensi spasial secara drastis untuk mencegah ledakan komputasi.
    * `Dense(128, ReLU)` -> `Dropout(0.5)`
    * `Dense(64, ReLU)` -> `Dropout(0.3)`
    * `Dense(8, Softmax)`: Lapisan probabilitas akhir.

## 📈 Performa dan Evaluasi
Pelatihan dilakukan dengan memanfaatkan *callbacks* `EarlyStopping` dan `ModelCheckpoint` untuk memastikan model menyimpan metrik terbaiknya.
* **Training Accuracy:** ~96%
* **Validation/Test Accuracy:** > 95%

## 🚀 Ekspor dan Deployment
Model tidak hanya dilatih, tetapi juga dipersiapkan untuk ekosistem *production* menggunakan beberapa format:
1.  **H5 / .keras:** Format kerja standar Keras.
2.  **SavedModel:** Digunakan untuk *backend deployment* (misal: via `TFSMLayer` atau TensorFlow Serving) untuk melayani *request* API.
3.  **TensorFlow.js (TFJS):** Model dikonversi ke dalam format Web/JSON agar dapat dieksekusi langsung di sisi klien (*browser*) tanpa membebani server backend.
