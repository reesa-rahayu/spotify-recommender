# Laporan Proyek Machine Learning - Rahayu Kartika Sari

## 🎶 Domain proyek

![Song Picture](https://moises.ai/_next/image/?url=https%3A%2F%2Fstorage.googleapis.com%2Fmoises-cms%2Fhow_to_reading_sheet_music_image_338d99b137%2Fhow_to_reading_sheet_music_image_338d99b137.jpg&w=1920&q=75)

Musik adalah bagian integral dari kehidupan sehari-hari dan konsumsi musik secara digital meningkat pesat melalui layanan seperti Spotify. Dengan jutaan lagu tersedia, menemukan lagu yang sesuai dengan selera pengguna menjadi tantangan tersendiri.

Proyek ini bertujuan untuk membangun sistem rekomendasi lagu berdasarkan fitur-fitur musik dan popularitas menggunakan dataset Spotify dari Kaggle. Dengan memanfaatkan teknik exploratory data analysis (EDA) dan algoritma machine learning sederhana seperti K-Nearest Neighbors (KNN), proyek ini menyarankan lagu-lagu yang mirip atau populer sesuai dengan preferensi pengguna[[1]](https://doi.org/10.1016/j.eswa.2024.124473).

### 🎯Mengapa dan Bagaimana Masalah Ini Dapat Dipecahkan

Dalam era di mana personalisasi adalah kunci pengalaman pengguna, sistem rekomendasi memainkan peran penting dalam menyaring konten yang relevan dari lautan pilihan yang tersedia. Dengan menganalisis karakteristik musik seperti danceability, valence, tempo, dan energi, kita dapat memahami pola dan preferensi pengguna secara lebih baik [[2]](https://doi.org/10.1109/IIHC55949.2022.10060806).

Melalui pendekatan berbasis content-based filtering dan popularity-based ranking, sistem ini memberikan pengalaman rekomendasi yang lebih terarah dan personal, bahkan tanpa data historis pengguna. Model ini dibangun menggunakan fitur numerik dari lagu serta analisis korelasinya untuk meningkatkan akurasi dan relevansi rekomendasi.

## 🧠 Business Understanding

### ❓ Problem Statements

- Dengan jutaan lagu tersedia di platform seperti Spotify, pengguna sering kewalahan dalam menemukan lagu yang sesuai dengan preferensi mereka secara cepat dan akurat.
- Rekomendasi umum berbasis popularitas sering kali tidak relevan secara personal karena tidak mempertimbangkan kesamaan fitur musik antar lagu
- Sistem rekomendasi berbasis histori pengguna (collaborative filtering) membutuhkan data pengguna yang kadang tidak tersedia, membatasi kemampuannya dalam memberikan rekomendasi bagi pengguna baru (cold start problem).

### 🎯 Goals

- Mengembangkan sistem rekomendasi lagu berbasis konten (content-based) yang mampu menyarankan lagu-lagu serupa berdasarkan fitur musik, tanpa perlu histori pendengaran pengguna.
- Menggunakan algoritma K-Nearest Neighbors (KNN) dengan metrik cosine similarity untuk mengukur kedekatan antar lagu berdasarkan fitur seperti danceability, valence, tempo, energy, dll.
- Memberikan hasil rekomendasi yang personal dan relevan berdasarkan karakteristik lagu input.

### Solution statements

- Membangun sistem rekomendasi menggunakan KNN berbasis fitur musik
  - Dataset Spotify digunakan untuk mengekstraksi fitur-fitur numerik dari lagu seperti danceability, energy, valence, tempo, dan lain-lain.
  - Fitur-fitur ini dinormalisasi menggunakan MinMaxScaler, kemudian digunakan dalam model KNN untuk menghitung kemiripan antar lagu.
  - Sistem mengembalikan daftar lagu paling mirip dengan lagu input berdasarkan kemiripan vektor fitur.
- Melakukan exploratory data analysis (EDA) dengan visualisasi distribusi fitur numerik (histogram, boxplot), identifikasi outlier (misalnya pada durasi lagu), genre yang dominan, dan korelasi antar fitur serta analisis popularitas berdasarkan genre dan fitur lainnya
- Membersihkan dan Mempersiapkan Data
  - Menghapus duplikat, menangani missing value, dan memfilter lagu berdasarkan durasi yang masuk akal (45 detik hingga 10 menit)
  - Melakukan encoding pada fitur kategorikal (explicit, track_genre) dan normalisasi fitur numerik agar model KNN bekerja optimal
- Menambahkan Sistem Rekomendasi Berdasarkan Popularitas
  Selain rekomendasi berbasis fitur, sistem juga dapat menyarankan lagu berdasarkan tingkat popularitas sebagai baseline sederhana.

Dengan pendekatan ini, sistem rekomendasi yang dibangun dapat membantu pengguna menemukan lagu-lagu yang sesuai dengan selera mereka tanpa memerlukan histori pengguna, serta memberikan hasil yang cepat, efisien, dan relevan.

## 📑 Data Understanding

Dataset yang digunakan dalam proyek ini adalah [Spotify Track Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data). Dataset ini berisi lagu-lagu Spotify dari 125 genre musik yang berbeda. Setiap lagu memiliki beberapa fitur audio yang terkait dengannya.

### Karakteristik Dataset

- Jumlah baris (lagu): 114000
- Jumlah fitur/kolom: 21
- Jumlah genre unik: 125

### Kualitas Data

| Aspek               | Hasil                                                                                                                            |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Missing Values      | terdapat 1 baris data yang kosong di kolom `artists`, `album_name`, `track_name`                                                 |
| Duplikasi           | 450 duplikat. Namun, jika dilihat lagi berdasarkan `track_id` terdapat 24.259 duplikat dan 40391 duplikat duplikat `track_name`. |
| Outlier             | Durasi Lagu dengan durasi <45 detik dan >10 menit (sekitar 0.5%)                                                                 |
| Kolom Tidak Relevan | `Unnamed: 0` adalah index auto-generated, tidak digunakan dalam model                                                            |

### Variabel-variabel pada dataset

| Nama Kolom         | Deskripsi                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `Unnamed: 0`       | indeks default hasil ekspor dari tools seperti pandas to_csv(). Tidak memiliki makna informasi                            |
| `track_id`         | ID unik lagu di Spotify                                                                                                   |
| `artists`          | Nama artis yang membawakan lagu. Jika lebih dari satu, dipisahkan dengan tanda `;`                                        |
| `album_name`       | Nama album tempat lagu tersebut dirilis                                                                                   |
| `track_name`       | Nama lagu                                                                                                                 |
| `popularity`       | Popularitas lagu (0–100), dihitung berdasarkan jumlah pemutaran dan seberapa baru pemutaran tersebut terjadi              |
| `duration_ms`      | Durasi lagu dalam milidetik                                                                                               |
| `explicit`         | Apakah lagu mengandung lirik eksplisit (`true` = ya, `false` = tidak atau tidak diketahui)                                |
| `danceability`     | Seberapa cocok lagu untuk menari, dinilai dari 0.0 (tidak bisa ditarikan) hingga 1.0 (sangat bisa ditarikan)              |
| `energy`           | Intensitas dan aktivitas lagu (0.0 – 1.0). Lagu cepat dan keras cenderung memiliki nilai tinggi                           |
| `key`              | Nada dasar lagu dalam notasi kelas pitch (mis. 0 = C, 1 = C#/Db, 2 = D, dst.), -1 jika tidak terdeteksi                   |
| `loudness`         | Tingkat kekerasan suara lagu dalam desibel (dB)                                                                           |
| `mode`             | Modus lagu, 1 = mayor, 0 = minor                                                                                          |
| `speechiness`      | Menilai kemiripan lagu dengan ucapan. Semakin mendekati 1.0, semakin banyak unsur ucapan                                  |
| `acousticness`     | Seberapa akustik lagu tersebut. Nilai 1.0 menunjukkan kepercayaan tinggi bahwa lagu bersifat akustik                      |
| `instrumentalness` | Kemungkinan bahwa lagu tidak mengandung vokal. Semakin mendekati 1.0, semakin besar kemungkinan lagu adalah instrumental  |
| `liveness`         | Menilai kemungkinan lagu direkam secara langsung (live). Nilai di atas 0.8 menunjukkan kemungkinan besar live performance |
| `valence`          | Mengukur seberapa positif nuansa lagu, dari 0.0 (sedih, suram) hingga 1.0 (bahagia, ceria)                                |
| `tempo`            | Tempo lagu dalam beat per minute (BPM)                                                                                    |
| `time_signature`   | Tanda birama lagu, biasanya antara 3 dan 7 (misalnya 4/4 atau 3/4)                                                        |
| `track_genre`      | Genre musik dari lagu tersebut                                                                                            |

### Exploratory Data Analysis (EDA) & Visualisasi

Analisis eksploratif dilakukan untuk memahami pola, distribusi, dan hubungan antar fitur dalam dataset.
Dari EDA terdapat beberapa insight yang didapatkan,antara lain:

- Terdapat sebaran yang cukup luas pada durasi lagu yang ada dalam data, yaitu paling sedikit 1 milidetik hingga 1 jam lebih.
  ![Duration Box Plot](https://github.com/reesa-rahayu/spotify-recommender/blob/main/images/duration%20box%20plot.png?raw=true)
- lagu paling populer yang ada dalam data yaitu Unholy Unholy (feat. Kim Petras) by Sam Smith.
- Hanya ada sekitar 8% lagu yang berstatus explicit
- genre yang paling populer yaitu pop film
  ![Genre Popularity](https://github.com/reesa-rahayu/spotify-recommender/blob/main/images/genre_popularity.png?raw=true)
- Terdapat distribusi yang rata pada jumlah lagu per genre pada data yaitu 1000 data per genre.
  ![Genre Distribution](https://github.com/reesa-rahayu/spotify-recommender/blob/main/images/genre_distribution.png?raw=true)

## 🛠️ Data Preparation

Data Preparation dilakukan untuk memastikan bahwa data yang digunakan dalam pemodelan machine learning bersih, konsisten, dan siap untuk dianalisis. Proses ini penting untuk meningkatkan kinerja model dengan mengatasi masalah potensial dalam kualitas data.

### Proses Data Preparation:

Data preparation dilakukan untuk memastikan bahwa data yang digunakan dalam pemodelan sistem rekomendasi lagu bersih, konsisten, dan siap dianalisis. Proses ini sangat penting untuk meningkatkan akurasi dan kinerja model dengan mengatasi berbagai masalah pada kualitas data.

#### 1. 🧹 Data Cleaning

Tujuan: Membersihkan data dari duplikasi, nilai hilang, serta outlier untuk menghindari informasi yang menyesatkan model.

- **Menghapus fitur yang tidak digunakan**
  Kolom `Unnamed: 0` dihapus karena hanya berisi index numerik tanpa nilai informasi.

- **Duplicate Handling**

  - Tujuan: Menghapus lagu-lagu yang terduplikasi. Prioritas pertama adalah menghapus duplikat berdasarkan `track_id` dengan mempertahankan versi yang lebih populer. Selanjutnya, duplikat berdasarkan kombinasi `track_name` dan `artists` juga dihapus, dengan mempertahankan kemunculan pertama.
  - Langkah-langkah:

    1.  Menghapus duplikat berdasarkan `track_id`:

        - Data diurutkan berdasarkan popularitas menurun.
        - Duplikat berdasarkan `track_id` dihapus, mempertahankan yang pertama (paling populer).
        - Indeks data direset.
          Sisa akhir: 89741 data.

    2.  Menghapus duplikat berdasarkan kombinasi `track_name` dan `artists`:
        - Duplikat berdasarkan kombinasi `track_name` dan `artists` dihapus, mempertahankan yang pertama muncul.
        - Indeks data direset.
          Sisa akhir: 81344 data.

- **Missing Value Handling**

  - Tujuan: Menghapus seluruh baris yang memiliki nilai kosong.
  - Dilakukan dengan perintah `drop_na()`
    Sisa akhir: 81343 data.

- **Outlier Handling (Durasi Lagu)**

  - Ditemukan bahwa beberapa lagu memiliki durasi yang terlalu pendek atau terlalu panjang dibandingkan durasi lagu pada umumnya.
  - Oleh karena itu, dilakukan pemfilteran hanya untuk lagu dengan durasi **antara 45 detik (45.000 ms)** hingga **10 menit (600.000 ms)**.
  - Alasan: Lagu yang terlalu pendek atau terlalu panjang bisa merupakan noise atau konten non-musikal, dan bisa menyesatkan sistem rekomendasi.
  - Kolom fitur lain seperti `energy`, `valence`, dll. tidak difilter karena merupakan karakteristik unik yang justru penting dalam proses rekomendasi.
  - Total data setelah difilter: 80533 data.

- **Data Type Conversion**
  Kolom `explicit` yang awalnya bertipe boolean diubah menjadi numerik (0 = tidak eksplisit, 1 = eksplisit) agar bisa digunakan dalam model machine learning.

#### 2. 🔢 Normalization

Tujuan: Menstandarisasi skala fitur numerik agar model tidak bias terhadap fitur dengan rentang nilai yang lebih besar.

**Langkah yang Diambil**:

- Digunakan **MinMaxScaler** dari `sklearn.preprocessing` untuk menskalakan semua kolom numerik ke rentang 0–1

  ![Min max scaler equation](https://miro.medium.com/v2/resize:fit:888/1*ye1I00S61GqpR34ABZZFLQ.png)

- Fitur yang dinormalisasi meliputi `danceability`, `energy`, `loudness`, `tempo`, dan sebagainya.

Alasan Penggunaan:
Normalisasi penting terutama untuk algoritma berbasis jarak seperti KNN agar semua fitur memiliki kontribusi yang seimbang dalam perhitungan kemiripan antar lagu.

#### 3. 📉 Dimensionality Reduction (PCA)

- Tujuan: Mengurangi kompleksitas fitur tanpa kehilangan informasi penting.
- Langkah: Mengambil data yang diperlukan, kemudian menggunakan PCA untuk mengubah fitur dengan `n_components=0.95`

## 🤖 Modeling

Sistem rekomendasi lagu dibangun menggunakan pendekatan berbasis popularitas, kemiripan konten (KNN), clustering, dan gabungan (hybrid) untuk hasil yang lebih relevan dan fleksibel.

#### 📌 Model yang Digunakan

#### 1. 📈 Popularity-Based Recommendation

Rekomendasi berdasarkan tingkat popularitas lagu. Sistem ini menyarankan lagu-lagu yang lebih populer dari lagu yang diinput oleh pengguna.

- **Logika**:

  - Periksa popularitas lagu input.
  - Ambil lagu-lagu lain dengan popularitas lebih tinggi atau sama.
  - Urutkan berdasarkan nilai `popularity`, dan ambil top N.

- ✅ Kelebihan:

  - Sangat cepat dan sederhana.
  - Cocok untuk pengguna baru (cold-start).

- ⚠️ Kekurangan:
  - Tidak mempertimbangkan preferensi fitur atau genre pengguna.

**Hasil Rekomendasi**

Input lagu
|track_name| artists| popularity| track_genre| duration_ms|
|--|--|--|--|--|
|Blinding Lights | The Weeknd |91 |pop | 200040|

| Lagu                                  | Artists                     |
| ------------------------------------- | --------------------------- |
| Unholy (feat. Kim Petras)             | Sam Smith, Kim Petras       |
| Quevedo: Bzrp Music Sessions, Vol. 52 | Bizarrap, Quevedo           |
| La Bachata                            | Manuel Turizo               |
| I'm Good (Blue)                       | David Guetta, Bebe Rexha    |
| Me Porto Bonito                       | Bad Bunny, Chencho Corleone |

##### 2. 🧠 K-Nearest Neighbors / KNN (Content-Based Recommendation)

Sistem rekomendasi berbasis kemiripan konten menggunakan algoritma KNN dengan metrik cosine similarity. Setiap lagu direpresentasikan sebagai vektor fitur audio seperti `danceability`, `energy`, `valence`, `tempo`, dll.

- Langkah-langkah:
  - Gunakan fitur yang sudah direduksi (PCA).
  - Gunakan KNN berbasis cosine similarity untuk mencari lagu mirip.
  - Diberikan satu lagu sebagai input, model mencari lagu-lagu yang paling mirip berdasarkan fitur musik.
- ✅ Kelebihan:
  - Memberikan rekomendasi yang sangat personal dan relevan.
  - Tidak bergantung pada popularitas.
- ⚠️ Kekurangan:
  - Membutuhkan data fitur lengkap dan akurat.
  - Rentan terhadap overfitting pada data kecil.

**Hasil Rekomendasi**
Input lagu
|track_name| artists| popularity| track_genre| duration_ms|
|--|--|--|--|--|
|Blinding Lights | The Weeknd |91 |pop | 200040|

| Lagu                                       | Artists                           |
| ------------------------------------------ | --------------------------------- |
| Thinkin About                              | ShockOne, Lee Mvtthews            |
| Damage Each Other                          | Steve Brian, Danni Baylor         |
| Noise (feat. Donnis) - Rise At Night Remix | Bassnectar, Donnis, Rise At Night |
| Runaway (U & I)                            | Galantis                          |
| Sight Of Your Soul                         | Dirtyphonics, Sullivan King       |

##### 3. 🗃️ Clustering-Based Recommendations

Sistem rekomendasi ini mengelompokkan lagu berdasarkan fitur-fitur audionya menggunakan algoritma K-Means. Rekomendasi diberikan berdasarkan lagu lain dalam klaster yang sama dengan lagu input.

- Langkah-langkah:
  - Fitur-fitur lagu di-reduksi dimensinya menggunakan PCA.
  - Algoritma K-Means diterapkan untuk mengelompokkan lagu.
  - Diberikan lagu input, sistem merekomendasikan lagu lain dari klaster yang sama berdasarkan popularitas.
- ✅ Kelebihan:
  - Dapat menemukan rekomendasi yang beragam dalam satu kelompok musik.
  - Skalabilitas yang baik untuk dataset besar.
- ⚠️ Kekurangan:
  - Kualitas rekomendasi sangat bergantung pada hasil klastering.
  - Kurang personal dibandingkan content-based filtering.

**Hasil Rekomendasi**
Input lagu
|track_name| artists| popularity| track_genre| duration_ms|
|--|--|--|--|--|
|Blinding Lights | The Weeknd |91 |pop | 200040|

| Lagu                      | Artists                    |
| ------------------------- | -------------------------- |
| Unholy (feat. Kim Petras) | Sam Smith, Kim Petras      |
| Dandelions                | Ruth B.                    |
| Call Out My Name          | The Weeknd                 |
| Hold Me Closer            | Elton John, Britney Spears |
| Ghost                     | Justin Bieber              |

##### 4. 🔀 Hybrid Recommendation System

Model gabungan dari pendekatan popularitas dan content-based untuk mendapatkan keseimbangan antara lagu populer dan lagu mirip secara audio.

- Langkah-langkah:

  - Filter lagu sesuai dengan cluster yang sama dengan input lagu.
  - Ambil rekomendasi dari KNN dan Popularity-Based.
  - Normalisasi skor popularitas dan jarak KNN.
  - Gabungkan skor menggunakan popularity_weight (misalnya 0.2 = 20% dari popularitas, 80% dari KNN).
  - Urutkan dan ambil top N lagu unik.

- ✅ Kelebihan:

  - Menyeimbangkan relevansi konten dan popularitas.
  - Cocok untuk berbagai jenis pengguna.

- ⚠️ Kekurangan:
  - Lebih kompleks dan membutuhkan lebih banyak data.

**Hasil Rekomendasi**
Input lagu
|track_name| artists| popularity| track_genre| duration_ms|
|--|--|--|--|--|
|Blinding Lights | The Weeknd |91 |pop | 200040|

| Lagu              | Artists                   |
| ----------------- | ------------------------- |
| Secrets           | OneRepublic               |
| Runaway (U & I)   | Galantis                  |
| Mind Over Matter  | Young the Giant           |
| Damage Each Other | Steve Brian, Danni Baylor |
| Battleships       | Daughtry                  |

## ✅ Evaluation

Untuk mengevaluasi performa sistem rekomendasi, digunakan beberapa metrik umum dalam domain Information Retrieval dan Recommender System, khususnya untuk sistem berbasis Top-N recommendation.

### 🎯 Metrik Evaluasi yang Digunakan

| Metrik          | Penjelasan                                                                                         |
| --------------- | -------------------------------------------------------------------------------------------------- |
| **Precision@K** | Persentase lagu relevan di antara K lagu yang direkomendasikan                                     |
| **Recall@K**    | Persentase lagu relevan yang berhasil ditemukan dari seluruh lagu relevan                          |
| **F1-Score@K**  | Harmonik rata-rata dari Precision dan Recall                                                       |
| **MAP@K**       | Rata-rata dari precision@k pada posisi setiap lagu relevan di antara K rekomendasi                 |
| **NDCG@K**      | Mengukur kualitas urutan rekomendasi berdasarkan relevansi dan posisi (semakin awal, semakin baik) |

### 🧪 Metodologi Evaluasi
- Dataset dibagi menjadi:
  - 80% data pelatihan untuk membangun model
  - 20% data uji (simulasi) untuk mengevaluasi seberapa baik model merekomendasikan lagu yang mirip dan relevan
- Rekomendasi diberikan untuk beberapa lagu populer dan kurang populer sebagai input.
- Setiap model memberikan Top-5 rekomendasi.
- Diperiksa apakah lagu-lagu rekomendasi tersebut masuk dalam genre, mood, dan fitur mirip dengan ground truth lagu dari data uji.

### 📈 Hasil Evaluasi – Perbandingan Antar Model


**🎵 Kesimpulan**: Model rekomendasi memberikan hasil yang sesuai dan relevan. Hybrid model memberikan keseimbangan antara lagu populer dan lagu yang mirip secara audio.


## Referensi

[1] Pourmoazemi, N., & Maleki, S. (2024). _A music recommender system based on compact convolutional transformers_. Expert Systems with Applications, 255(A). https://doi.org/10.1016/j.eswa.2024.124473
[2] Bhowmick, A., et. al. (2022). _Song Recommendation System based on Mood Detection using Spotify's Web API_. 2022 International Interdisciplinary Humanitarian Conference for Sustainability (IIHC), 14(1), 125. https://doi.org/10.1109/IIHC55949.2022.10060806
