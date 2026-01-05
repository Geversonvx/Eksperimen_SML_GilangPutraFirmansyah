## Preprocessing Dataset

Tahap ini merupakan eksperimen preprocessing data sebelum proses training model.

Langkah preprocessing:
1. Menghapus kolom ID
2. Memisahkan fitur dan target
3. Menangani missing value dengan median imputation
4. Normalisasi fitur menggunakan StandardScaler
5. Menyimpan dataset hasil preprocessing ke CSV

File `creditcard_preprocessed_full.csv` merupakan
hasil akhir preprocessing yang digunakan sebagai
dasar eksperimen selanjutnya.
## Automated Preprocessing

Preprocessing dataset dilakukan secara otomatis menggunakan script
`automate_gilangputrafirmansyah.py`.

Workflow GitHub Actions akan berjalan otomatis setiap terdapat perubahan
pada data mentah atau script preprocessing, dan menghasilkan dataset siap latih
di folder `CreditCardDefaultDataset_preprocessing`.
