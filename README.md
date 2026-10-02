# Yapay Sinir Ağları – Uygulama 4

Fırat Üniversitesi, YMH451 Yapay Zeka dersi, 2. hafta laboratuvar uygulaması.
Keras ile yapay sinir ağı (YSA) modelleri kurdum ve bunları çapraz doğrulama ve ROC eğrisiyle değerlendirdim.

| Soru | Konu | Sonuç |
|---|---|---|
| 1 | Telefon fiyat sınıfı için kendi YSA modelim (64-32-16-4, Dropout, EarlyStopping) | Test başarımı %91.5 |
| 2 | 5-katlı çapraz doğrulama: verilen model ve benim modelim | %91.95 ± 0.93 / %92.50 ± 0.95 |
| 3 | `blue`, `fc`, `int_memory`, `ram`, `wifi` çıkarılarak tekrar | %31.8 (en büyük etki `ram`, korelasyonu 0.92) |
| 4 | Diyabet verisi için ikili YSA, eğitim/doğrulama grafikleri | Test başarımı %70.8 |
| 5 | Diyabet modelinin ROC eğrisi | AUC = 0.796 |

## Dosyalar

- `Uygulama4_YSA_SenaKose.ipynb`: çıktılarıyla birlikte not defteri; her sorunun başında kısa bir açıklama var
- `Uygulama4_YSA_SenaKose.py`: aynı kod, açıklamaları yorum satırı olarak içeren tek bir script
- `telefon_fiyat_degisimi.csv`: [Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) (2000 satır, 20 özellik)
- `diyabet.csv`: [Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (768 satır, 8 özellik)

## Çalıştırma

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab Uygulama4_YSA_SenaKose.ipynb   # ya da: python Uygulama4_YSA_SenaKose.py
```

Sonuçların tekrarlanabilmesi için rastgelelik tohumu `SEED = 42` olarak sabitlendi.
