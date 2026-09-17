# Proyek Klasifikasi OCR Dokumen Asuransi

![Tim Digitalisasi DigiNsure](digitizing_team.png)

## 🏢 Latar Belakang & Studi Kasus
**DigiNsure Inc.** adalah perusahaan asuransi inovatif yang berfokus pada peningkatan efisiensi pemrosesan klaim dan interaksi layanan pelanggan. Inisiatif terbaru mereka adalah mendigitalkan seluruh dokumen historis klaim asuransi. Hal ini mencakup upaya untuk meningkatkan akurasi pelabelan identitas (ID) yang dipindai dari dokumen kertas, serta mengklasifikasikannya sebagai **ID Primer** atau **ID Sekunder**.

Untuk membantu DigiNsure dalam inisiatif ini, proyek ini menggunakan pendekatan *multi-modal learning* (pembelajaran multi-modal) untuk melatih model *Optical Character Recognition* (OCR). 

Agar klasifikasi lebih akurat, model ini dirancang unik dengan menggunakan dua buah input:
1. **Gambar dokumen yang dipindai** (berukuran 64x64 piksel).
2. **Kategori jenis asuransi** (*home, life, auto, health, atau other*).

Mengintegrasikan berbagai modalitas data yang berbeda (seperti gambar dan teks kategori) memungkinkan model untuk bekerja jauh lebih baik dalam skenario yang kompleks, sehingga mampu menangkap informasi yang lebih spesifik dan mendetail. Target klasifikasi (label) yang akan ditebak oleh model ini ada dua jenis: **primary_id** dan **secondary_id**, yang dianalisis dari setiap pasangan gambar dan jenis asuransinya.

---

## 📂 Struktur Proyek

Berikut adalah daftar file dan kegunaannya di dalam proyek ini:

* **`notebook.ipynb`**: File utama Notebook yang berisi seluruh alur kerja proyek (Eksplorasi data, pembuatan arsitektur jaringan saraf/Neural Network, dan *training* model).
* **`ocr_insurance_dataset.pkl`**: File dataset mentah berbentuk *pickle* yang memuat data tensor PyTorch (gambar dan label/tipe).
* **`project_utils.py`**: Skrip Python pendukung yang berisi *class* atau fungsi tambahan (seperti `ProjectDataset`) yang di-*import* ke dalam notebook.
* **`ocr_model.pth`**: File berisi bobot (*weights*) dari model yang sudah selesai dilatih dan disimpan.
* **`digitizing_team.png`**: Ilustrasi/banner dari studi kasus DigiNsure.

---

## 🏗️ Arsitektur Model

Model (`OCRModel`) dibangun dengan arsitektur dua cabang (*multi-modal*):

1. **Image Layer**: Menggunakan *Convolutional Neural Network* (CNN) dengan `Conv2d` dan `MaxPool2d` untuk mengekstrak fitur visual dari gambar teks berukuran 64x64.
2. **Type Layer**: Menggunakan *Fully Connected Layer* (`Linear`) untuk memproses jenis asuransi (berbasis *one-hot encoding*).

Kedua fitur ini kemudian digabungkan (*concatenated*) sebelum dilewatkan ke layer *Classifier* akhir untuk menghasilkan tebakan label (Primer atau Sekunder).

---

## 🚀 Cara Menjalankan Proyek

Proyek ini dapat dijalankan menggunakan **Google Colab** (Sangat disarankan) atau **Jupyter Notebook** di komputer lokal.

### Opsi A: Menggunakan Google Colab (Online / Tidak perlu install)
1. Buka [Google Colab](https://colab.research.google.com/).
2. Pilih menu **File > Upload notebook** dan unggah file `notebook.ipynb`.
3. Di panel sebelah kiri Colab, klik ikon **Folder (Files)**.
4. Unggah (upload) file-file pendukung berikut ke dalam *session storage* Colab:
   * `ocr_insurance_dataset.pkl`
   * `project_utils.py`
5. Jalankan *cell* kode dari atas ke bawah secara berurutan dengan menekan tombol **Play** atau menekan `Shift + Enter`.

### Opsi B: Menggunakan Jupyter Notebook (Lokal)
1. Pastikan Python dan Jupyter Notebook sudah terinstal di komputer Anda. Anda juga membutuhkan *library* pendukung:
   ```bash
   pip install torch torchvision numpy matplotlib
   ```
2. Buka terminal atau *command prompt* di dalam folder proyek ini (folder yang berisi semua file).
3. Ketik perintah `jupyter notebook` dan tekan Enter.
4. Browser akan terbuka. Klik file `notebook.ipynb`.
5. Pada menu atas, klik **Cell > Run All** untuk menjalankan seluruh kode dari atas ke bawah.

---
