# 📊 Klasifikasi Dataset Citrus (UTS - Naive Bayes)

## Deskripsi
Proyek ini merupakan bagian dari Ujian Tengah Semester (UTS) yang bertujuan membangun model klasifikasi menggunakan algoritma **Naive Bayes** untuk menentukan 
apakah sebuah buah merupakan **jeruk (orange)** atau **anggur (grapefruit)** berdasarkan fitur-fitur numerik.

Dataset yang digunakan adalah `citrus.csv`, yang memuat data karakteristik buah dan label nama buah (`orange` atau `grape`).
Anda bisa mengakses dan download file tersebut melalui link ini :
[oranges-vs-grapefruit](https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit)

---

## Tahapan Pengerjaan

1. **Import Library**  
   Menggunakan `pandas`, `scikit-learn`.

2. **Load Dataset**  
   Dataset `citrus.csv` dibaca menggunakan pandas.

3. **Pra-pemrosesan Data**  
   Memisahkan fitur (`X`) dan label (`y`), memeriksa missing values.

4. **Split Dataset**  
   Data dibagi 80% untuk training dan 20% untuk testing.

5. **Pembuatan Model**  
   Menggunakan algoritma Naive Bayes (`GaussianNB`).

6. **Prediksi dan Evaluasi**  
   Model diuji menggunakan:
   - Confusion Matrix
   - Classification Report
   - Accuracy Score

7. **Output Hasil Prediksi**  
   Menampilkan hasil prediksi 10 data pertama dari dataset testing:
   - Kolom asli (label sebenarnya)
   - Kolom prediksi

---

## Cara Menjalankan

1. Pastikan Python sudah terinstall.
2. Install library yang dibutuhkan melalui terminal:
   pip install pandas scikit-learn
3. Jalankan script :
   python naive_bayes.py
