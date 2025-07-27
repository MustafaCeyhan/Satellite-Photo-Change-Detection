# 🛰️ Satellite Change Detection

Bu proje, uydu görüntüleri arasındaki değişiklikleri tespit etmek için geliştirilmiş bir Streamlit web uygulamasıdır. Farklı bilgisayarlı görü teknikleri ve yapay zeka modelleri kullanarak iki uydu görüntüsü arasındaki değişiklikleri analiz eder ve görselleştirir.

## ✨ Özellikler

### 🔍 Değişiklik Tespit Yöntemleri
- **SSIM (Structural Similarity Index)**: Yapısal benzerlik analizi
- **AbsDiff (Absolute Difference)**: Mutlak fark hesaplama
- **Canny Edge Detection**: Kenar tespit tabanlı değişiklik analizi
- **LLM (OpenAI GPT-4)**: Yapay zeka tabanlı görüntü analizi
- **Deep Learning Model**: Önceden eğitilmiş derin öğrenme modeli ([SenseEarth2020](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection))
- **Hibrit Yöntem**: Birden fazla yöntemin birleştirilmesi

### 🚀 Gelişmiş Özellikler
- **Otomatik Görüntü Hizalama**: ORB feature matching ve FLANN kullanarak
- **Ön İşleme**: CLAHE histogram eşitleme ve Gaussian filtreleme
- **Morfolojik Filtreleme**: Gürültü azaltma ve temizleme
- **Parametre Özelleştirme**: Tüm algoritmalar için ayarlanabilir parametreler
- **İnteraktif Arayüz**: Kullanıcı dostu Streamlit web arayüzü

## 📋 Gereksinimler

### Sistem Gereksinimleri
- **Python**: 3.9 (Önerilen)
- **İşletim Sistemi**: Windows, Linux, macOS

### Temel Kütüphaneler
```bash
streamlit
opencv-python
numpy
scikit-image
scikit-learn
pillow
openai
torch
matplotlib
```

### Ek Bağımlılıklar
```bash
# requirements.txt dosyası oluşturun:
streamlit>=1.28.0
opencv-python>=4.5.0
numpy==1.19.5
scikit-image>=0.19.0
scikit-learn>=1.0.0
Pillow>=8.0.0
openai>=1.0.0
torch>=1.12.0
matplotlib>=3.3.4
```

### ⚠️ NumPy Sürüm Uyarısı
Deep Learning modeli için **NumPy 1.19.5** sürümü zorunludur. Daha yeni sürümler model yükleme hatalarına neden olabilir.

### ⚠️ Matplotlib Kurulum Uyarısı
Eğer Conda kullanıyorsanız matplotlib kurulumunda sorun yaşayabilirsiniz. Bu durumda:

```bash
# Mevcut matplotlib kütüphanelerini kaldırın
pip uninstall matplotlib
conda remove matplotlib

# Conda-forge'dan belirli versiyonu yükleyin
conda install -c conda-forge matplotlib=3.3.4
```

## 🛠️ Kurulum

1. **Projeyi klonlayın:**
```bash
git clone [repository-url]
cd satellite-change-detection
```

2. **Sanal ortam oluşturun:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Gerekli paketleri yükleyin:**
```bash
pip install -r requirements.txt
```

   **NumPy Sürüm Kontrolü:**
```bash
# NumPy sürümünü kontrol edin
python -c "import numpy; print(numpy.__version__)"

# Eğer farklı bir sürüm yüklüyse:
pip uninstall numpy
pip install numpy==1.19.5
```

   **Matplotlib Sorunu Yaşıyorsanız (Conda kullanıcıları):**
```bash
# Matplotlib sorununu çözmek için:
pip uninstall matplotlib
conda remove matplotlib
conda install -c conda-forge matplotlib=3.3.4
```

4. **Deep Learning modeli için gerekli dosyaları indirin:**
   - SenseEarth2020 reposundan model dosyasını indirin: [https://github.com/LiheYoung/SenseEarth2020-ChangeDetection](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection)
   - `pspnet_hrnet_w40_39.37.pth` model dosyasını proje dizinine yerleştirin
   - Aynı repodan `models/` ve `utils/` klasörlerini kopyalayın

## 🚀 Kullanım

### Temel Kullanım
```bash
streamlit run app.py
```

### Web Arayüzü Kullanımı

1. **Görüntüleri Yükleyin:**
   - "BEFORE" görüntüsünü yükleyin (değişiklik öncesi)
   - "AFTER" görüntüsünü yükleyin (değişiklik sonrası)

2. **Yöntem Seçin:**
   - Tek veya birden fazla tespit yöntemi seçin
   - Hibrit yöntem için en az 2 yöntem seçin

3. **Parametreleri Ayarlayın:**
   - Sol panelden algoritma parametrelerini özelleştirin
   - Gelişmiş hizalama seçeneğini etkinleştirin

4. **Analizi Başlatın:**
   - "Run Change Detection" butonuna tıklayın
   - Sonuçları inceleyin

### OpenAI API Kullanımı

LLM yöntemini kullanmak için:

1. OpenAI API anahtarınızı edinin
2. Ortam değişkeni olarak ayarlayın:
```bash
export OPENAI_API_KEY="your-api-key-here"
```
3. Veya web arayüzünde doğrudan girin

## 📁 Proje Yapısı

```
satellite-change-detection/
├── app.py                 # Ana Streamlit uygulaması
├── models/               # Deep learning model modülleri
│   └── model_zoo.py
├── utils/                # Yardımcı fonksiyonlar
│   └── palette.py
├── requirements.txt      # Python bağımlılıkları
├── README.md            # Bu dosya
└── pspnet_hrnet_w40_39.37.pth  # Önceden eğitilmiş model
```

## 🔧 Algoritma Parametreleri

### SSIM Parametreleri
- **Threshold**: 0.1-0.8 (varsayılan: 0.3)
- **Window Size**: 7, 11, 15 (varsayılan: 11)
- **Min Area**: Minimum değişiklik alanı (varsayılan: 100)

### AbsDiff Parametreleri
- **Threshold**: 0-150 (0=Otsu, varsayılan: 60)
- **Min Area**: Minimum değişiklik alanı (varsayılan: 100)

### Canny Parametreleri
- **Low Threshold**: 10-100 (varsayılan: 50)
- **High Threshold**: 100-300 (varsayılan: 150)
- **Min Area**: Minimum değişiklik alanı (varsayılan: 50)

## 📊 Çıktılar

### Görsel Sonuçlar
- **BEFORE/AFTER**: Orijinal görüntüler
- **Değişiklik Maskesi**: Binary değişiklik haritası
- **Renklendirilmiş Sonuç**: Değişikliklerin üzerine çizilmiş görüntü
- **Hibrit Sonuç**: Birden fazla yöntemin birleşimi

### Metin Sonuçları (LLM)
- Detaylı değişiklik açıklaması
- Koordinat bilgileri
- Değişiklik türü analizi

### İstatistiksel Sonuçlar
- Değişiklik yüzdesi
- SSIM skoru
- Precision, Recall, F1-Score (ground truth ile)

## 🎯 Kullanım Alanları

- **Şehir Planlama**: Kentsel gelişim takibi
- **Çevre İzleme**: Orman kaybı, tarım alanı değişikleri
- **Afet Değerlendirme**: Deprem, sel hasarı tespiti
- **İnşaat Takibi**: Yeni yapıların tespit edilmesi
- **Araştırma**: Akademik çalışmalar için değişiklik analizi

## ⚠️ Dikkat Edilmesi Gerekenler

1. **Görüntü Kalitesi**: Yüksek çözünürlüklü görüntüler daha iyi sonuç verir
2. **Hizalama**: Görüntüler mümkün olduğunca benzer açıdan çekilmiş olmalı
3. **Işık Koşulları**: Farklı ışık koşulları yanlış pozitif sonuçlara yol açabilir
4. **API Limitleri**: OpenAI API kullanımında maliyet ve hız limitleri dikkate alın
5. **Sürüm Uyumluluğu**: 
   - NumPy 1.19.5 sürümü zorunludur (DL model uyumluluğu için)
   - Python 3.9 önerilir
   - Matplotlib 3.3.4 sürümü Conda ortamlarında sorun yaşamamak için

## 🔧 Gelişmiş Konfigürasyon

### Ortam Değişkenleri
```bash
export OPENAI_API_KEY="your-openai-api-key"
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Özel Model Yükleme
```python
# Kendi modelinizi yüklemek için:
@st.cache_resource
def load_custom_model():
    model = YourCustomModel()
    model.load_state_dict(torch.load('your_model.pth'))
    return model
```

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🆘 Sorun Giderme

### Yaygın Sorunlar

1. **Import Hataları:**
```bash
pip install --upgrade -r requirements.txt

# NumPy sürüm sorunu için:
pip install numpy==1.19.5
```

2. **CUDA Hataları:**
```bash
# CPU-only PyTorch yükleyin
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. **Model Dosyası Bulunamadı:**
   - `pspnet_hrnet_w40_39.37.pth` dosyasının proje dizininde olduğundan emin olun
   - Model dosyasını [SenseEarth2020 GitHub reposundan](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection) indirin
   - `models/` ve `utils/` klasörlerinin de aynı repodan kopyalandığından emin olun

4. **Deep Learning Model Yükleme Hatası:**
```bash
# NumPy sürüm uyumsuzluğu genellikle bu hataya neden olur
pip uninstall numpy
pip install numpy==1.19.5

# Torch versiyonunu da kontrol edin
pip install torch==1.12.0
```

5. **Matplotlib Import Hatası:**
```bash
# Conda kullanıcıları için özel çözüm:
pip uninstall matplotlib
conda remove matplotlib  
conda install -c conda-forge matplotlib=3.3.4
```

4. **OpenAI API Hataları:**
   - API anahtarınızın geçerli ve kredisi olduğundan emin olun
   - Rate limit hatalarında bekleyin ve tekrar deneyin

## 📚 Referanslar ve Kaynaklar

### Deep Learning Model
Bu projede kullanılan derin öğrenme modeli **SenseEarth2020-ChangeDetection** reposundan alınmıştır:
- **Repo**: [https://github.com/LiheYoung/SenseEarth2020-ChangeDetection](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection)
- **Model Dosyaları**: Eğitilmiş model ağırlıkları bu repodan indirilebilir
- **Kod Yapısı**: `models/` ve `utils/` klasörleri bu repodan adapte edilmiştir

### Kullanılan Algoritmalar
- **SSIM**: Wang, Z., et al. (2004). "Image quality assessment: from error visibility to structural similarity"
- **ORB Feature Matching**: Rublee, E., et al. (2011). "ORB: An efficient alternative to SIFT or SURF"
- **FLANN Matcher**: Muja, M. and Lowe, D.G. (2009). "Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration"

## 📞 İletişim

Sorularınız için:
- Issue oluşturun
- Pull request gönderin
- Proje maintainerları ile iletişime geçin

## 🙏 Teşekkürler

Bu proje aşağıdaki açık kaynak projelerden yararlanmıştır:
- OpenCV
- Streamlit
- scikit-image
- OpenAI API
- PyTorch