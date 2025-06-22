# Python ile Makine Öğrenmesi 🤖
![python-ile-makine-ogrenmesi-egitimi](https://github.com/user-attachments/assets/8dad93fd-48a0-4b9d-ba71-13236eaa8551)

Bu repository, Python programlama dili kullanarak makine öğrenmesi alanında temel kavramları öğrenmek ve pratik uygulamalar geliştirmek için hazırlanmış kapsamlı bir eğitim materyalidir.

## 📋 İçindekiler

- [Kurulum](#kurulum)
- [Proje Yapısı](#proje-yapısı)
- [Konu Başlıkları](#konu-başlıkları)
- [Kullanılan Kütüphaneler](#kullanılan-kütüphaneler)
- [Başlangıç](#başlangıç)
- [Projeler](#projeler)
- [Katkı Sağlama](#katkı-sağlama)
- [Lisans](#lisans)

## 🚀 Kurulum

### Gereksinimler
- Python 3.7 veya üzeri
- pip paket yöneticisi

### Adım 1: Repository'yi klonlayın
```bash
git clone https://github.com/jesusamaisa/PythonlaMakineOgrenmesi.git
cd PythonlaMakineOgrenmesi
```

### Adım 2: Sanal ortam oluşturun (önerilen)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate     # Windows
```

### Adım 3: Gerekli paketleri yükleyin
```bash
pip install -r requirements.txt
```

## 📁 Proje Yapısı

```
PythonlaMakineOgrenmesi/
│
├── data/                      # Veri setleri
│   ├── raw/                   # Ham veriler
│   ├── processed/             # İşlenmiş veriler
│   └── external/              # Dış kaynaklardan veriler
│
├── notebooks/                 # Jupyter Notebook dosyaları
│   ├── 01_temel_kavramlar/
│   ├── 02_veri_analizi/
│   ├── 03_supervised_learning/
│   ├── 04_unsupervised_learning/
│   └── 05_deep_learning/
│
├── src/                       # Kaynak kod dosyaları
│   ├── data_processing/
│   ├── models/
│   ├── utils/
│   └── visualization/
│
├── tests/                     # Test dosyaları
├── docs/                      # Dokümantasyon
├── requirements.txt           # Python paket bağımlılıkları
└── README.md                  # Bu dosya
```

## 📚 Konu Başlıkları

### 1. Temel Kavramlar
- Makine öğrenmesi nedir?
- Supervised, Unsupervised ve Reinforcement Learning
- Veri türleri ve özellikleri
- Model değerlendirme metrikleri

### 2. Veri Analizi ve Ön İşleme
- Pandas ile veri manipülasyonu
- Eksik veri yönetimi
- Veri görselleştirme (Matplotlib, Seaborn)
- Feature Engineering
- Veri normalizasyonu ve standardizasyonu

### 3. Supervised Learning (Denetimli Öğrenme)
- **Regresyon Algoritmaları:**
  - Linear Regression
  - Polynomial Regression
  - Ridge ve Lasso Regression
  - Random Forest Regression
  
- **Sınıflandırma Algoritmaları:**
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes

### 4. Unsupervised Learning (Denetimsiz Öğrenme)
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Principal Component Analysis (PCA)
- t-SNE

### 5. Model Değerlendirme ve Optimizasyon
- Cross Validation
- Grid Search
- Hyperparameter Tuning
- Confusion Matrix
- ROC Curve ve AUC

### 6. Deep Learning Temelleri
- Neural Networks
- TensorFlow/Keras ile basit modeller
- Convolutional Neural Networks (CNN) giriş
- Recurrent Neural Networks (RNN) giriş

## 🛠 Kullanılan Kütüphaneler

```python
# Veri işleme ve analiz
import pandas as pd
import numpy as np

# Görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Makine öğrenmesi
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Deep Learning
import tensorflow as tf
from tensorflow import keras
```

## 🎯 Başlangıç

1. **Başlangıç Seviyesi:** `notebooks/01_temel_kavramlar/` klasöründen başlayın
2. **Veri analizi:** `notebooks/02_veri_analizi/` ile devam edin
3. **İlk modelinizi oluşturun:** `notebooks/03_supervised_learning/linear_regression.ipynb`

### Örnek Kullanım

```python
# Basit bir sınıflandırma örneği
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Veri yükleme
data = pd.read_csv('data/processed/sample_dataset.csv')

# Özellik ve hedef değişken ayrımı
X = data.drop('target', axis=1)
y = data['target']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma ve eğitim
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahmin ve değerlendirme
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Doğruluğu: {accuracy:.2f}")
```

## 🎨 Projeler

### Proje 1: Ev Fiyat Tahmini
- **Veri seti:** Boston Housing Dataset
- **Algoritma:** Linear Regression, Random Forest
- **Amaç:** Ev özelliklerine göre fiyat tahmini

### Proje 2: İris Çiçeği Sınıflandırması
- **Veri seti:** Iris Dataset
- **Algoritma:** KNN, Decision Tree, SVM
- **Amaç:** Çiçek türü sınıflandırması

### Proje 3: Müşteri Segmentasyonu
- **Veri seti:** Mall Customer Dataset
- **Algoritma:** K-Means Clustering
- **Amaç:** Müşteri gruplarını belirleme

### Proje 4: El Yazısı Rakam Tanıma
- **Veri seti:** MNIST Dataset
- **Algoritma:** CNN (Convolutional Neural Network)
- **Amaç:** El yazısı rakamları tanıma

## 📖 Öğrenme Yol Haritası

1. **Hafta 1-2:** Python temelleri ve veri analizi
2. **Hafta 3-4:** Supervised learning algoritmaları
3. **Hafta 5-6:** Unsupervised learning ve clustering
4. **Hafta 7-8:** Model optimizasyonu ve değerlendirme
5. **Hafta 9-10:** Deep learning temelleri
6. **Hafta 11-12:** Proje geliştirme

## 🤝 Katkı Sağlama

Bu projeye katkı sağlamak isterseniz:

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📊 Performans Metrikleri

| Model | Veri Seti | Doğruluk | F1-Score |
|-------|-----------|---------|----------|
| Random Forest | Iris | 96.7% | 0.97 |
| SVM | Wine | 94.4% | 0.94 |
| Linear Regression | Boston Housing | R²: 0.89 | RMSE: 3.2 |

## 🔗 Faydalı Kaynaklar

- [Scikit-learn Dokümantasyonu](https://scikit-learn.org/stable/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

## 📝 Notlar

- Tüm notebook'lar detaylı açıklamalar içermektedir
- Her algoritma için teorik bilgi ve pratik uygulama sunulmaktadır
- Gerçek veri setleri kullanılarak örnekler verilmektedir

## 🏷 Sürüm Geçmişi

- **v1.0.0** - İlk sürüm yayınlandı
- **v1.1.0** - Deep learning bölümü eklendi
- **v1.2.0** - Yeni projeler ve veri setleri eklendi

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

## 👨‍💻 Yazar

**İsa Erkan** - [@jesusamaisa](https://github.com/jesusamaisa)

## 📞 İletişim

Sorularınız için:
- GitHub Issues açabilirsiniz
- Email: [email address](muhammedisaerkan@gmail.com)
- LinkedIn: [profile](https://www.linkedin.com/in/isa-erkan-b38698313/)

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!

**Birkaç kod örneği aşağıdadır**
![Ekran görüntüsü 2025-06-22 183011](https://github.com/user-attachments/assets/01cc8ac4-ec04-47c8-b7fc-2367fd29a890)

![Ekran görüntüsü 2025-06-22 182942](https://github.com/user-attachments/assets/636bf276-f50a-420b-9b8b-ae3de05ae333)
