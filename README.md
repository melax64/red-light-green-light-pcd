# Red Light, Green Light — Pengolahan Citra Digital (PCD) Murni

Aplikasi game interaktif **"Red Light, Green Light"** berbasis webcam yang dibangun **100% menggunakan teknik Pengolahan Citra Digital (PCD) klasik** dengan Python, OpenCV, dan NumPy — **tanpa Machine Learning / Deep Learning** (tanpa YOLO, MediaPipe, TensorFlow, PyTorch, maupun Haar Cascade).

---

## Fitur Utama

- **100% PCD Klasik:** Deteksi gerak murni berbasis *Grayscale*, *Gaussian Blur*, *Frame Differencing* (`cv2.absdiff`), *Binary Thresholding*, *Operasi Morfologi* (*Opening* & *Dilation*), dan *Contour Analysis*.
- **State Machine Lengkap:** `MENU` → `GREEN_LIGHT` → `WARNING` (1 detik) → `RED_LIGHT` (*grace period* 0.5 detik) → `WIN` / `GAME_OVER`.
- **3 Tingkat Kesulitan:** `Easy`, `Medium`, dan `Hard` dengan preset parameter PCD (ukuran kernel blur, threshold, morfologi, luas kontur minimum, serta batas gerak lampu merah) yang dapat diganti secara *real-time*.
- **Mode Debug PCD 4-Panel (`D`):** Visualisasi langsung 4 tahap pipeline PCD secara *split-view* untuk pembelajaran maupun demo presentasi.
- **Penyimpanan Bukti Otomatis:** Saat tereliminasi, frame yang telah ditandai kotak merah pada area bergerak otomatis disimpan ke folder `captures/`.
- **Efek Suara Opsional Aman:** Mendukung audio `.wav` via `pygame` di `assets/sounds/` dengan *fallback* aman tanpa error jika `pygame` atau file audio belum tersedia.

---

## Alur Pipeline Pengolahan Citra Digital (PCD)

Setiap frame webcam diproses melalui 5 tahap PCD klasik di `src/motion_detector.py`:

```text
Frame Webcam (BGR)
   │
   ▼  1. Pra-pemrosesan
[BGR → Grayscale] + [Gaussian Blur]          ← Mereduksi noise sensor kamera
   │
   ▼  2. Deteksi Gerak (Frame Differencing)
[cv2.absdiff(baseline, current)]
  • GREEN LIGHT : baseline = frame sebelumnya (consecutive differencing)
  • RED LIGHT   : baseline = reference frame yang dikunci setelah grace period 0.5 detik
   │
   ▼  3. Segmentasi Gerak
[Binary Thresholding (cv2.threshold)]        ← Memisahkan perubahan piksel nyata dari fluktuasi cahaya
   │
   ▼  4. Operasi Morfologi
[Opening (MORPH_OPEN)] → [Dilation]          ← Membuang bintik noise kecil & menyatukan area gerak
   │
   ▼  5. Analisis Kontur
[cv2.findContours] → Filter min_contour_area → Hitung Motion Score & Bounding Box
```

### Parameter PCD per Tingkat Kesulitan

| Parameter PCD | Easy (`1`) | Medium (`2`) | Hard (`3`) |
| :--- | :---: | :---: | :---: |
| Gaussian Blur Kernel | `21 × 21` | `15 × 15` | `11 × 11` |
| Binary Threshold (`0–255`) | `35` | `25` | `18` |
| Morph Opening Kernel | `7 × 7` | `5 × 5` | `3 × 3` |
| Morph Dilation Kernel | `11 × 11` | `9 × 9` | `7 × 7` |
| `min_contour_area` (px²) | `1500` | `800` | `400` |
| Batas Gerak Lampu Merah | `4000` | `2000` | `900` |

---

## Struktur Folder

```text
red-light-green-light-pcd/
├── src/
│   ├── __init__.py
│   ├── config.py            # Konstanta warna, ukuran jendela, durasi, & preset PCD Easy/Medium/Hard
│   ├── motion_detector.py   # Inti pipeline PCD 5 tahap + output citra tiap tahap untuk Debug Mode
│   ├── game_engine.py       # State machine, timer 45s, progress bar, audio fallback, & simpan bukti
│   ├── ui_renderer.py       # HUD modern OpenCV + tampilan Mode Debug PCD 4-panel
│   └── main.py              # Entry point webcam (cv2.VideoCapture) & penanganan input keyboard
├── assets/
│   └── sounds/              # (Opsional) File audio .wav: green_light, warning, red_light, eliminated, win
├── captures/                # Penyimpanan otomatis screenshot bukti saat ELIMINATED
├── tests/
│   └── test_motion_detector.py  # Unit test pipeline PCD & state machine berbasis citra sintetis NumPy
├── docs/                    # Dokumentasi tambahan / laporan praktikum
├── requirements.txt         # Daftar dependensi Python
└── README.md
```

---

## Cara Instalasi & Menjalankan

### 1. Prasyarat
- Python 3.9 atau lebih baru
- Webcam yang terhubung ke perangkat

### 2. Instalasi Dependensi
Disarankan menggunakan *virtual environment*:

```bash
# Buat & aktifkan virtual environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependensi
pip install -r requirements.txt
```

> **Catatan:** `pygame` bersifat opsional. Jika tidak diinstal atau file suara di `assets/sounds/` belum ada, game tetap berjalan normal tanpa suara.

### 3. Menjalankan Aplikasi
```bash
python src/main.py
```

Atau jalankan sebagai modul Python dengan argumen opsional:
```bash
python -m src.main --camera 0 --difficulty Medium
```

Tambahkan flag `--debug` jika ingin langsung masuk ke **Mode Debug PCD 4-panel**:
```bash
python src/main.py --debug
```

---

## Aturan Bermain

1. **GREEN LIGHT (Lampu Hijau):** Bergeraklah terus di depan kamera agar *Progress Bar* meningkat hingga **100%** sebelum batas waktu **45 detik** habis.
2. **WARNING (Lampu Kuning — 1 detik):** Peringatan singkat agar pemain bersiap berhenti bergerak.
3. **RED LIGHT (Lampu Merah):**
   - Pemain diberi *grace period* selama **0,5 detik** untuk benar-benar diam sebelum *reference frame* dikunci.
   - Jika setelah frame dikunci luas kontur gerakan melewati ambang sensitivitas, area yang bergerak akan ditandai kotak merah, screenshot bukti disimpan ke folder `captures/`, dan status berubah menjadi **ELIMINATED (Game Over)**.

---

## Kontrol Keyboard

| Tombol | Fungsi |
| :---: | :--- |
| `SPACE` | Mulai permainan / Restart setelah Menang atau Game Over |
| `D` | Toggle **Mode Debug PCD 4-Panel** (*Game+HUD*, *Grayscale+Blur*, *Frame Differencing*, *Threshold+Morfologi*) |
| `1` | Pilih tingkat kesulitan **Easy** |
| `2` | Pilih tingkat kesulitan **Medium** |
| `3` | Pilih tingkat kesulitan **Hard** |
| `Q` atau `ESC` | Keluar dari aplikasi |

---

## Menjalankan Unit Test

Unit test menggunakan citra sintetis NumPy untuk memverifikasi perilaku detektor gerak PCD (citra diam tidak terdeteksi, objek bergerak terdeteksi dengan *bounding box* akurat, noise kecil tersaring, serta alur *grace period* & penyimpanan bukti):

```bash
python -m unittest discover -s tests -v
```
