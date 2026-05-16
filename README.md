# Hand Sign Scuba

Aplikasi Python desktop sederhana untuk mendeteksi pose tangan `SCUBA_POSE` dari webcam menggunakan OpenCV + MediaPipe. Saat pose terdeteksi stabil, aplikasi menampilkan dua video meme scuba dengan audio.

Project ini tidak memakai link TikTok langsung. Video meme harus disimpan manual di folder `assets/videos/`.

## Struktur Folder

```text
hand-sign-scuba/
├─ main.py
├─ requirements.txt
├─ README.md
├─ config/
│  └─ scuba_pose.json
├─ assets/
│  ├─ reference/
│  │  └─ scuba_pose_reference.jpg
│  ├─ models/
│  │  └─ hand_landmarker.task
│  └─ videos/
│     ├─ scuba_1.mp4
│     └─ scuba_2.mp4
└─ src/
   ├─ hand_detector.py
   ├─ gesture.py
   └─ video_popup.py
```

Catatan: `config/scuba_pose.json` akan dibuat otomatis setelah kalibrasi dari webcam (`C`) atau foto referensi (`L`). Jika file belum ada, aplikasi tetap jalan.

Catatan MediaPipe: project ini menggunakan MediaPipe Tasks. File model `assets/models/hand_landmarker.task` diperlukan untuk deteksi tangan. Jika file belum ada, download dari halaman resmi MediaPipe Hand Landmarker lalu simpan dengan nama tersebut.

## Cara Install

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan virtual environment di Windows:

```bash
.venv\Scripts\activate
```

Aktifkan virtual environment di Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
python main.py
```

## Cara Menaruh Video Meme

Simpan dua video meme di:

```text
assets/videos/scuba_1.mp4
assets/videos/scuba_2.mp4
```

Jika file bernama `scuba_1.mp4.mp4`, rename menjadi `scuba_1.mp4`.

Jika file bernama `scuba_2.mp4.mp4`, rename menjadi `scuba_2.mp4`.

Jika video belum ada, aplikasi tidak crash, tetapi terminal akan menampilkan:

```text
Missing video file: assets/videos/scuba_1.mp4
Missing video file: assets/videos/scuba_2.mp4
```

## Cara Kalibrasi Pose Scuba dari Webcam

1. Jalankan:
   ```bash
   python main.py
   ```
2. Pastikan webcam menyala.
3. Buat pose tangan scuba-scuba di depan kamera.
4. Usahakan dua tangan terdeteksi dan landmark terlihat jelas.
5. Tekan `C`.
6. Pose akan tersimpan ke `config/scuba_pose.json`.
7. Ulangi pose yang sama. Jika distance `<= threshold`, gesture menjadi `SCUBA_POSE` dan video meme akan muncul.

## Kalibrasi dari Foto Referensi

Fitur ini membuat template `SCUBA_POSE` dari foto menggunakan MediaPipe, sehingga tidak harus kalibrasi langsung dari webcam.

1. Buat folder:
   ```text
   assets/reference/
   ```
2. Simpan foto pose scuba sebagai:
   ```text
   assets/reference/scuba_pose_reference.jpg
   ```
3. Jalankan:
   ```bash
   python main.py
   ```
4. Tekan `L` untuk membuat template dari foto.
5. Jika berhasil, terminal menampilkan:
   ```text
   Loaded scuba pose template from reference image.
   ```
6. Tirukan pose yang sama di webcam.
7. Jika pose cocok, video meme akan muncul.

Jika file foto tidak ditemukan, aplikasi tidak crash dan terminal menampilkan:

```text
Missing reference image: assets/reference/scuba_pose_reference.jpg
```

Jika MediaPipe tidak mendeteksi tangan di foto, terminal menampilkan:

```text
No hands detected in reference image. Try another photo with clearer hands.
```

## Tombol Keyboard

| Tombol | Fungsi |
|---|---|
| `q` | Keluar aplikasi |
| `c` | Kalibrasi/simpan pose dari webcam |
| `l` | Load/create template dari `assets/reference/scuba_pose_reference.jpg` |
| `r` | Reload template dari `config/scuba_pose.json` |
| `b` | Tampilkan dua video meme dengan audio |
| `h` | Sembunyikan semua video |
| `1` | Test tampilkan video 1 dengan audio |
| `2` | Test tampilkan video 2 dengan audio |
| `+` | Tambah threshold/toleransi matching |
| `-` | Kurangi threshold/toleransi matching |

Default threshold adalah `0.55`. Karena distance kecil berarti pose lebih mirip, threshold yang lebih besar berarti lebih toleran. Lihat nilai `Distance` dan `Threshold` di overlay webcam untuk tuning.

## Tampilan Overlay Webcam

Window utama berjudul `Hand Sign Scuba` dan menampilkan:

- FPS
- jumlah tangan terdeteksi
- status template: `Loaded` / `Not Found`
- template source: `None` / `Webcam` / `Reference Image`
- current gesture: `SCUBA_POSE` / `UNKNOWN`
- distance score
- threshold saat ini
- status video: `Hidden` / `Showing`
- last action message
- instruksi tombol penting

Landmark tangan digambar dengan titik merah dan garis putih. Label `Left` / `Right` juga ditampilkan jika tersedia dari MediaPipe.

## Cara Kerja Matching Pose

- Setiap tangan dinormalisasi dengan wrist sebagai anchor.
- Koordinat landmark disimpan relatif terhadap wrist.
- Skala memakai ukuran bounding box tangan atau jarak wrist ke `middle_finger_mcp`.
- Untuk dua tangan, posisi relatif antar center tangan ikut dihitung agar pose lebih akurat.
- Matching memakai rata-rata Euclidean distance antar landmark normalized.
- Jika ada dua tangan, urutan tangan yang tertukar tetap ditangani dengan mencoba dua kemungkinan matching dan mengambil distance paling kecil.
- Jika jumlah tangan tidak cocok dengan template, pose dianggap tidak cocok.

## Troubleshooting

### Webcam tidak terbuka

- Pastikan webcam tidak sedang dipakai aplikasi lain.
- Tutup Zoom/Meet/Teams atau aplikasi kamera lain.
- Coba restart VS Code atau komputer.

### Video tidak muncul

- Pastikan file ada di `assets/videos/scuba_1.mp4` dan `assets/videos/scuba_2.mp4`.
- Tekan `B` untuk test dua video manual.
- Tekan `1` atau `2` untuk test masing-masing video.

### Video tidak ada suara

- OpenCV sendiri tidak bisa memutar audio.
- Project ini memakai `ffpyplayer` untuk audio.
- Jalankan `pip install -r requirements.txt` agar `ffpyplayer` terinstall.
- Pastikan file mp4 memang memiliki audio.

### File video tidak ditemukan

- Pastikan nama file tepat.
- Jika file bernama `scuba_1.mp4.mp4`, rename menjadi `scuba_1.mp4`.
- Jika file bernama `scuba_2.mp4.mp4`, rename menjadi `scuba_2.mp4`.

### Pose tidak terdeteksi

- Pastikan dua tangan terlihat jelas di kamera.
- Tekan `C` lagi untuk kalibrasi ulang dari webcam.
- Atau simpan foto yang lebih jelas lalu tekan `L` untuk kalibrasi dari foto referensi.
- Naikkan toleransi dengan tombol `+` jika distance sering sedikit di atas threshold.
- Turunkan toleransi dengan tombol `-` jika video terlalu gampang aktif sendiri.

### Tekan L tetapi gagal

- Pastikan foto berada di `assets/reference/scuba_pose_reference.jpg`.
- Pastikan foto jelas dan kedua tangan terlihat.
- Jika hanya satu tangan terdeteksi, coba pakai foto lain dengan tangan lebih jelas.
- Jika pose dari foto terlalu sulit dideteksi, tekan `+` untuk menaikkan threshold.
- Alternatif paling akurat: kalibrasi ulang langsung dari webcam dengan tombol `C`.