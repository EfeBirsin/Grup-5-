# Sepsis Erken Uyarı ve Karar Destek Sistemi

## 📋 Proje Açıklaması

Bu proje, yapay zeka destekli gerçek zamanlı sepsis risk analizi yapan interaktif bir Streamlit dashboard'udur. Klinisyenlerin (doktor, hemşire) saniyeler içinde hasta riskini değerlendirmesini sağlar.

## 🎯 Özellikler

### 🔍 Risk Analizi
- **Gerçek zamanlı tahmin**: Hasta ID'si girilerek anında risk değerlendirmesi
- **Görsel gauge chart**: Renk kodlu risk seviyesi göstergesi
- **Risk faktörleri**: Model kararının açıklanabilirliği

### 📊 Dashboard Bileşenleri
1. **Risk Özeti**: Hasta bilgileri ve sepsis riski
2. **Model Açıklaması**: Risk faktörleri ve klinik gerekçeler
3. **Klinik Trend**: 24 saatlik vital bulgu takibi

### 🏥 Klinik Veriler
- **Demografik bilgiler**: Yaş, cinsiyet, yatış tipi
- **Vital bulgular**: Laktat, tansiyon, CRP, vb.
- **Trend analizi**: Zaman içindeki değişimler

## 🚀 Kurulum

### Gereksinimler
```bash
Python 3.8+
```

### Paket Kurulumu
```bash
pip install -r requirements.txt
```

### Çalıştırma
```bash
streamlit run app.py
```

## 📁 Dosya Yapısı

```
sepsis_uygulama/
├── app.py                          # Ana Streamlit uygulaması
├── requirements.txt                 # Python paketleri
├── random_forest_model.pkl         # Eğitilmiş Random Forest modeli
├── imputer.pkl                     # Veri ön işleme modeli
├── columns.pkl                     # Model sütun listesi
├── sepsis_eğitim_verisi.csv        # Eğitim verisi
└── README.md                       # Bu dosya
```

## 🧠 Model Detayları

### Model Türü
- **Algoritma**: Random Forest Classifier
- **Estimator Sayısı**: 100 ağaç
- **Criterion**: Gini
- **Class Weight**: Balanced

### Özellikler (11 adet)
- `age`: Yaş
- `heart_rate_mean`: Ortalama kalp atış hızı
- `sbp_mean`: Ortalama sistolik tansiyon
- `spo2_mean`: Ortalama oksijen satürasyonu
- `resp_rate_mean`: Ortalama solunum hızı
- `temp_c_mean`: Ortalama vücut sıcaklığı
- `lactate_mean`: Ortalama laktat seviyesi
- `platelets_mean`: Ortalama trombosit sayısı
- `creatinine_mean`: Ortalama kreatinin
- `bun_mean`: Ortalama BUN
- `gender_M`: Cinsiyet (erkek=1, kadın=0)

## 📈 Kullanım

### 1. Uygulamayı Başlatın
```bash
streamlit run app.py
```

### 2. Tarayıcıda Açın
```
http://localhost:8501
```

### 3. Hasta ID'si Girin
- **Gerçek hasta**: 10006, 10011, 10013, vb. (100 adet)
- **Demo hasta**: 100000, 99999, vb.

### 4. Analiz Edin
- "Analiz Et" butonuna tıklayın
- Risk seviyesini ve faktörleri inceleyin
- Trend grafiğini değerlendirin

## 🎨 Dashboard Bileşenleri

### Sol Sidebar - Kontrol Paneli
- Hasta ID girişi
- Analiz butonu

### Ana Panel - 3 Sütunlu Dashboard

#### Sütun 1: Risk Özeti
- Hasta bilgileri
- Sepsis riski yüzdesi
- Gauge chart (0-100%)

#### Sütun 2: Model Açıklaması
- Risk faktörleri listesi
- Klinik gerekçeler
- Model kararı açıklaması

#### Sütun 3: Klinik Trend
- 24 saatlik vital bulgu takibi
- Zaman içindeki değişimler
- Trend analizi

## 🔧 Teknik Detaylar

### Veri İşleme
- **Imputation**: Eksik değerler için SimpleImputer
- **Encoding**: Gender için one-hot encoding
- **Normalization**: Sayısal değerler için standartlaştırma

### Model Performansı
- **Accuracy**: Yüksek doğruluk oranı
- **Explainability**: Risk faktörleri açıklanabilir
- **Real-time**: Anında tahmin

### Güvenlik
- **Hata yönetimi**: Graceful degradation
- **Demo veri**: Bilinmeyen ID'ler için
- **Veri doğrulama**: Input validation

## 📊 Veri Seti

### Hasta Sayısı
- **Toplam kayıt**: 136
- **Benzersiz hasta**: 100
- **Demo hasta**: Sınırsız

### Örnek Hasta ID'leri
```
10006, 10011, 10013, 10017, 10019, 10026, 10027, 10029, 10032, 10033
```

## 🎯 Klinik Kullanım

### Risk Seviyeleri
- **Düşük Risk (0-40%)**: Yeşil
- **Orta Risk (40-70%)**: Turuncu
- **Yüksek Risk (70-100%)**: Kırmızı

### Klinik Karar Desteği
- **Risk faktörleri**: En önemli 3 faktör
- **Trend analizi**: Zaman içindeki değişim
- **Aksiyon önerileri**: Klinik müdahale rehberi

## 🔄 Güncellemeler

### v1.0.0 (Final)
- ✅ Streamlit dashboard entegrasyonu
- ✅ Random Forest model entegrasyonu
- ✅ Gauge chart ve trend grafikleri
- ✅ Risk faktörleri açıklaması
- ✅ Profesyonel UI/UX tasarımı
- ✅ Gerçek zamanlı tahmin
- ✅ Demo veri desteği

## 📞 Destek

Herhangi bir sorun veya öneri için lütfen iletişime geçin.

---

**Sepsis Erken Uyarı ve Karar Destek Sistemi v1.0.0**  
*Yapay Zeka Destekli Gerçek Zamanlı Risk Analizi* 