# 🩺 SepsisGuard: Sepsis Riski Tahmin ve Klinik Destek Sistemi

## 🧑‍🤝‍🧑 Team & Product Name

**SepsisGuard AI**

## 📌 Information About Team and Product

### 👥 Team Members

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Title</th>
      <th>Socials</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <img src="https://avatars.githubusercontent.com/u/137696827?v=4" width="60" height="60" style="border-radius:50%" /><br>
        <strong>Efe Yarkın Birsin</strong>
      </td>
      <td>Scrum Master</td>
      <td>
        <a href="https://www.linkedin.com/in/efebirsin" target="_blank">🔗 LinkedIn</a>
      </td>
    </tr>
    <tr>
      <td>
        <img src="https://avatars.githubusercontent.com/u/100940828?v=4" width="60" height="60" style="border-radius:50%" /><br>
        <strong>Aslı Şemşimoğlu</strong>
      </td>
      <td>Developer</td>
      <td>
        <a href="https://www.linkedin.com/in/aslisemsimoglu" target="_blank">🔗 LinkedIn</a>
      </td>
    </tr>
    <tr>
      <td>
        <img src="https://avatars.githubusercontent.com/u/86380130?v=4" width="60" height="60" style="border-radius:50%" /><br>
        <strong>Melih Eren</strong>
      </td>
      <td>Developer</td>
      <td>
        <a href="https://www.linkedin.com/in/meliheren" target="_blank">🔗 LinkedIn</a>
      </td>
    </tr>
    <tr>
      <td>
        <img src="https://avatars.githubusercontent.com/u/202075982?v=4" width="60" height="60" style="border-radius:50%" /><br>
        <strong>Esra Çilesiz</strong>
      </td>
      <td>Developer</td>
      <td>
        <a href="https://www.linkedin.com/in/esra-cilesiz/" target="_blank">🔗 LinkedIn</a>
      </td>
    </tr>
    <tr>
      <td>
        <img src="https://avatars.githubusercontent.com/u/184748762?v=4" width="60" height="60" style="border-radius:50%" /><br>
        <strong>Beyza İrem Kaya</strong>
      </td>
      <td>Developer</td>
      <td>
        <a href="https://www.linkedin.com/in/beyza-irem-kaya-0141b9313/" target="_blank">🔗 LinkedIn</a>
      </td>
    </tr>
  </tbody>
</table>

---

## 🔍 Proje Tanımı

**SepsisGuard**, MIMIC-III Clinical Database üzerine inşa edilmiş, kritik durumdaki hastalarda *sepsis kaynaklı ölüm riskini erken tespit etmek* amacıyla geliştirilen bir yapay zekâ destekli tahmin ve karar destek sistemidir. Proje kapsamında hem makine öğrenmesi hem de derin öğrenme modelleri test edilerek, en güvenilir tahmin başarısını sağlayacak yapı hedeflenmektedir.

Web tabanlı arayüz sayesinde klinisyenler, yalnızca belirli biyobelirteçleri (örn. kalp atışı, kan basıncı, laktat düzeyi, solunum sayısı...) girerek, hastanın akut risk durumunu yüzdelik olarak öğrenebilecek; sistem ayrıca riskli verileri yorumlayarak önerilerde bulunabilecek bir klinik asistan olarak hizmet verecektir.

---

## 🎯 Amacımız

- Sepsis gibi ölümcül ancak belirtileri gizli ilerleyen bir sendromu, **erken evrede yüksek doğrulukla tahmin etmek**.
- Hastanelerdeki doktorlara ve yoğun bakım personeline, **gerçek zamanlı veri analizi ile destek vermek**.
- Klinik süreçlerde en kritik olanı sağlamak: **ZAMAN** kazandırmak.
- Sadece tahmin etmekle kalmayıp, **hangi parametrenin riskte ne kadar etkili olduğunu yorumlayan bir açıklayıcı yapay zekâ sistemi** tasarlamak.
- Son olarak, veriye dayalı akıllı yönlendirme ile "Hangi birime sevk edilmeli?", "Takip sıklığı ne olmalı?" gibi **yardımcı klinik kararları desteklemek**.

---

## 👨‍⚕️ Hedef Kitlemiz

- Hastanelerin yoğun bakım (ICU) ve acil servis (ER) birimlerinde çalışan **hekimler**, **hemşireler** ve **klinik destek personeli**.
- Sağlık bilişimi alanında çalışan **veri bilimciler**, **biyomedikal mühendisleri** ve **yapay zekâ geliştiricileri**.
- Tıp fakültelerinde veya sağlık bilimleri bölümlerinde, klinik karar destek sistemlerine ilgi duyan **araştırmacılar ve öğrenciler**.

---

## 🧑‍💻 Takımımız ve Vizyonumuz

Bizler, yapay zekânın sadece teknolojik değil, **insani bir faydaya dönüşmesi gerektiğine inanan bir mühendislik ve sağlık ekibiyiz**. 
Projemizin her satırı, bir hayatın daha zamanında kurtarılması için yazılıyor. Bu nedenle sadece model doğruluğu değil, sistemin anlaşılabilir, güvenilir ve kullanılabilir olması da bizim için aynı derecede önemlidir.

### 💡 Uzun vadeli hedeflerimiz:
- Projemizi açık kaynak olarak geliştirerek **genişletilebilir ve özelleştirilebilir** bir klinik karar destek sistemine dönüştürmek.
- Farklı hastalık sınıflandırmaları ve erken uyarı sistemlerine genişletilebilir bir altyapı sağlamak.
- Sağlıkta yapay zekâ uygulamalarına dair etik, teknik ve pratik bir referans noktası oluşturmak.

---

## 📊 Kullanım Senaryosu

- Kullanıcı, belirlenen 10–15 adet *klinik öznitelik* girer (örneğin: HR, MAP, Lactate...).
- Sistem, hastaya ait *Sepsis Risk Skoru*nu yüzdelik olarak hesaplar.
- Ekranın yanında yer alan bir **akıllı asistan** (chatbot):
  - "Risk %87 çünkü laktat 4.5 mmol/L ve MAP 50 mmHg."
  - "Bu hasta 2 saat içinde **yüksek ihtimalle şok** gelişimi gösterebilir. Hekim müdahalesi önerilir."
  - "Takip için yoğun bakım izlem protokolüne alınmalı." gibi geri bildirimler sunar.

---

## 📁 Kullanılan Veri Seti

- [MIMIC-III Clinical Database Demo v1.4 (PhysioNet)](https://doi.org/10.13026/C2HM2Q)
- Geliştiriciler: A. Johnson, T. Pollard, R. Mark (2019)
- Veritabanı; `CHARTEVENTS`, `LABEVENTS`, `ADMISSIONS`, `ICUSTAYS`, `DIAGNOSES_ICD`, `PATIENTS` gibi ilişkili tablolardan oluşturulmuştur.

---

## 🔗 Lisans ve Atıf

> Bu proje, PhysioNet'in açık verilerinden türetilmiştir. MIMIC-III kullanımına dair lisans ve atıf koşullarına uygun biçimde geliştirilmiştir.  
> Lütfen yaygın olarak şu atıf biçimini kullanınız:

> Johnson, A.E.W., et al. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3, 160035.

---

## 🚀 Başlamak İçin

> Kurulum talimatları, model eğitimi ve demo arayüzle ilgili belgeler için [Wiki](./wiki) sayfamızı ziyaret edin.  
> Örnek çalıştırmalar için `notebooks/` dizininden Jupyter dosyalarına ulaşabilirsiniz.

<details>
<summary><h2>Sprint 1</h2></summary>

## Sprint Notları:
-	Proje alanı belirlenip proje fikri oluşturuldu.
-	Görev dağılımı yapıldı, takım ismi bulundu
-	Proje ürünü hakkında genel fikirler github’a yazıldı
-	Toplantılar “Jitzi” veya “Google Meet” üzerinden yapıldı, gerekli durumlarda "Whatsapp” grubu üzerinden konuşmalar devam ettirildi.
-	Proje yönetimi için “Trello” kullanıldı.
-	Proje için “MIMIC-III demo 1.4” veri seti incelendi, veri setindeki her bir veri için rapor oluşturulup modele katkısı ölçüldü.
-	“Sepsis tahmini” yapılacak model oluşturulmaya ve geliştirilmeye başlandı.
-	Figma UI tasarımına başlanıldı.

## Sprint için Tamamlanması Beklenen Puan: 80 puan

## Tahmin Mantığı:

Toplamda 3 sprint olarak planlanan projemizin toplam puanı 300 olarak belirlenip ilk sprint'i için 80 puanlık bir hedef konulmuştur. Bu puan, fiili kodlama içerse de veri setini anlama, detaylı rapor oluşturma ve strateji geliştirme gibi bilişsel yük gerektiren görevleri kapsamaktadır. Projenin en önemli adımı olan "ne yapılacağını ve nasıl yapılacağını" netleştiren bu görevler, projenin temelini oluşturduğu için yüksek puanlanmıştır. Amaç, Sprint 2'ye "sadece kodlamaya odaklanabileceğimiz" temiz bir başlangıçla girmektir.
Sprint 2 ve Sprint 3, projenin en yoğun geliştirme fazlarını temsil etmektedir. Sprint 2, modellemenin; Sprint 3 ise arayüz geliştirmenin ağırlıklı olduğu dönemlerdir. Bu sprint'lerdeki görevler, yüksek derecede teknik karmaşıklık ve uygulama eforu gerektirdiğinden, her biri için hedeflenen puan 110 olarak belirlenmiştir.

## Daily Scrum: 
Günlük  iletişimin, kolaylık ve hız gibi artılarından ötürü  Whatsapp üzerinden yapılmasına karar verilmiştir. Günlük iletişim örnekleri pdf olarak tarafımızdan paylaşılmaktadır: [sprint1_daily](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint1/sprint1_daily%20(1).pdf)

## Sprint board update

![trello.png](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint1/trello.png)

## Ürün Durumu Çıktısı

![Ürün Çıktısı 1](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint1/%C3%9Cr%C3%BCn_Tasar%C4%B1m%C4%B1_Figma.jpg)
![Ürün Çıktısı 2](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint1/%C3%9Cr%C3%BCn_Tasar%C4%B1m%C4%B1_Figma_2.jpg)
![Ürün Çıktısı 3](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint1/%C3%9Cr%C3%BCn_Tasar%C4%B1m%C4%B1_Figma_3.jpg)

## Sprint Retrospektifi:

- İkinci sprintte modelin geliştirilmesine devam edilmesine karar verildi.
-	Modelin performansını ölçmek için bir başarı metriği belirlenmesine ve bu metrik üzerinden iyileştirme yapılmasına karar verildi.
-	Ürünün arayüzü ve kullanıcı deneyimi (UI/UX) tasarımı üzerine yapılacak toplantıların sıklaştırılmasına karar verildi.
-	Araştırma fazında elde edilen bulgular doğrultusunda modelin ilk versiyonunun oluşturulmasına başlanmasına karar verildi.

## Sprint Review:

**Sprint Hedefi:**  

Sprint 1'in ana hedefi, "Sepsis Tahmini" projesinin temelini atmak, kullanılacak MIMIC-III demo 1.4 veri setini derinlemesine analiz etmek, bu analiz sonucunda bir strateji geliştirmek ve modelin ilk altyapısını kurmaktı. Amaç, Sprint 2'ye temiz ve net bir başlangıç yapabilmek için tüm hazırlık ve araştırma adımlarını tamamlamaktı.

**Tamamlanan İşler ve Çıktılar:**

- Veri setindeki her bir klinik özniteliğin sepsis tahminine olası katkısı detaylı bir şekilde incelenmiş ve bu bulguları içeren kapsamlı bir rapor oluşturulmuştur.
- Araştırma fazında elde edilen bulgular doğrultusunda, sepsis tahmini yapacak modelin ilk iskeletini ve temel fonksiyonlarını içeren Python script'leri geliştirilmeye başlanmıştır.
- Trello board'u sprint takibi için aktif hale getirilmiş, proje hakkındaki genel fikirler ve hedefler GitHub reposuna işlenerek tüm takım için ortak bir anlayış zemini oluşturulmuştur.

**Alınan Kararlar ve Sonraki Adımlar:**

- Modelin performansını objektif olarak ölçmek ve iyileştirmeleri bu doğrultuda yapmak için Sprint 2'nin başında spesifik bir başarı metriği (örn: AUC-ROC, F1-Score, Precision-Recall) belirlenmesine karar verildi.
- Veri analiz raporu temel alınarak, Sprint 2'de modelin geliştirme, eğitim ve iyileştirme çalışmalarına odaklanılacaktır.
- Ürünün son kullanıcı için değerini ve kullanılabilirliğini artırmak amacıyla, kullanıcı arayüzü (UI) ve kullanıcı deneyimi (UX) üzerine yapılacak toplantıların sıklaştırılmasına karar verildi.

**Sprint Katılımcıları:**

Aslı Şemsimoğlu, Beyza İrem Kaya, Efe Birsin, Esra Çilesiz, Melih Eren

</details>

<details>
<summary><h2>Sprint 2</h2></summary>

## Sprint Notları:
-	Sprint 1'de oluşturulan strateji doğrultusunda basit bir "Sepsis Tahmini" modeli oluşturuldu. Modelinin geliştirilmesine ve iyileştirilmesine odaklanıldı.
-	Figma üzerinde tamamlanan tasarımlar, kullanıcı arayüzüne (UI) dönüştürülmeye başlandı.
-	Proje yönetimi "Trello" üzerinden, günlük iletişim ise "Whatsapp” grubu üzerinden devam ettirildi.
-	Sprint sonunda, ürünün mevcut durumunu ve işlevselliğini gösteren bir video kaydı oluşturuldu.
-	Hastanın sepsis olup olmadığına dair bağımlı değişken sütun oluşturuldu ve buna göre başarı metriği belirlendi.

## Sprint için Tamamlanması Beklenen Puan: 110 puan

## Tahmin Mantığı:

Toplamda 3 sprint olarak planlanan projemizin toplam puanı 300 olarak belirlenip 2. sprint'i için 110 puanlık bir hedef konulmuştur. Bu sprint, hem yapay zeka modelinin kodlanıp eğitilmesi hem de kullanıcı arayüzünün geliştirilmesi gibi yüksek teknik karmaşıklık ve uygulama eforu gerektiren görevleri içermektedir.

## Daily Scrum: 
Günlük iletişimin, kolaylık ve hız gibi artılarından ötürü Whatsapp üzerinden yapılmasına karar verilmiştir. [Sprint 2 - Daily Scrum](https://github.com/EfeBirsin/Grup-5-/tree/main/ProjectManagement/Sprint2/Daily%20Scrum%202)

## Sprint board update

![Trello 2](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint2/trello2.png)

## Ürün Durumu Çıktısı

![Ürün Çıktısı 1](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint2/%C3%9Cr%C3%BCn_Tasar%C4%B1m%C4%B1_Sprint2.jpg)
![Ürün Çıktısı 2](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint2/%C3%9Cr%C3%BCn_Tasar%C4%B1m%C4%B12_Sprint2.jpg)
![Ürün Çıktısı 3](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint2/SepsisGuardAI_%C3%B6n_izlenim.gif)

Ayrıca ürünün video kaydı için:
[SepsisGuardAI Video Kaydı](https://youtu.be/eKWXF5uF4bY)

## Modellerin Sınıflandırma Sonuçları

Oluşturulan model için: [Sepsis Modeli Tahmini](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint2/sepsis_model_deneme.ipynb)

<details>
<summary> Lojistik Regresyon </summary>

### Lojistik Regresyon Performansı

| Metrik | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Sınıf 0** | 0.76 | 0.84 | 0.80 | 19 |
| **Sınıf 1** | 0.50 | 0.38 | 0.43 | 8 |
| | | | | |
| **Accuracy** | | | **0.70** | **27** |
| **Macro Avg** | 0.63 | 0.61 | 0.61 | 27 |
| **Weighted Avg**| 0.68 | 0.70 | 0.69 | 27 |

**ROC AUC Skoru:** **`0.7632`**

</details>

<details>
<summary> Random Forest </summary>

### Random Forest Performansı

| Metrik | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Sınıf 0** | 0.75 | 0.95 | 0.84 | 19 |
| **Sınıf 1** | 0.67 | 0.25 | 0.36 | 8 |
| | | | | |
| **Accuracy** | | | **0.74** | **27** |
| **Macro Avg** | 0.71 | 0.60 | 0.60 | 27 |
| **Weighted Avg**| 0.73 | 0.74 | 0.70 | 27 |

**ROC AUC Skoru:** **`0.7204`**

</details>

<details>
<summary> XGBoost </summary>

### XGBoost Performansı

| Metrik | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Sınıf 0** | 0.83 | 0.79 | 0.81 | 19 |
| **Sınıf 1** | 0.56 | 0.62 | 0.59 | 8 |
| | | | | |
| **Accuracy** | | | **0.74** | **27** |
| **Macro Avg** | 0.69 | 0.71 | 0.70 | 27 |
| **Weighted Avg**| 0.75 | 0.74 | 0.74 | 27 |

</details>


## Sprint Retrospektifi:

- Modelin tahmin doğruluğu %90 ve üzerine çıkarılmaya çalışıldı.
- Modelin performansı arttırılması için veri ön işleme ve özellik mühendisliği üzerinde duruldu.
-	Son sprint olan Sprint 3 için görevler netleştirildi: Modelin son haline getirilmesi, arayüzün tamamlanması, tam entegrasyonun sağlanması ve projenin sunuma hazır hale getirilmesi.
-	Bağımlı değişkenin kategorik değerleri dengeli dağılmadığı için "Accuracy" başarı metriği yerine "Recall" ve "F1-Score" metriklerinin değerleri göz önünde bulundurulmaya karar verildi.

## Sprint Review:

**Sprint Hedefi:**  

Sprint 2'nin ana hedefi, Sprint 1'de atılan temeller üzerine "Sepsis Tahmini" modelini geliştirmek, eğitmek ve performansını optimize etmekti. Aynı zamanda, Figma'da tasarlanan arayüzünü geliştirmekti.

**Tamamlanan İşler ve Çıktılar:**

- Sepsis tahmin modeli, belirlenen başarı metrikleri doğrultusunda başarıyla eğitilmiş ve ilk versiyonu tamamlanmıştır.
- Figma tasarımları temel alınarak kullanıcı arayüzünün kodlaması büyük ölçüde tamamlanmıştır.
- Model ve arayüzün temel işlevlerini bir araya getiren bir prototip oluşturulmuştur.
- Sprint çıktısı olarak, ürünün mevcut çalışma durumunu gösteren bir video kaydı başarıyla oluşturulmuş ve paylaşılmıştır.
- Trello board'u sprint boyunca aktif olarak güncellenmiş ve tamamlanan görevler işaretlenmiştir.

**Alınan Kararlar ve Sonraki Adımlar:**

- Sprint 3'te model üzerinde iyileştirmeler yapılacak ve performans metrikleri son kez raporlanacaktır.
- Kullanıcı arayüzü ile model arasındaki entegrasyon tamamlanarak ürünün son kullanıcıya sunulacak hale getirilmesine karar verildi.
- Son sprintin ana odağı, projenin tamamlanması, test edilmesi ve sunuma hazır hale getirilmesi olacaktır.

**Sprint Katılımcıları:**

Aslı Şemsimoğlu, Beyza İrem Kaya, Efe Birsin, Esra Çilesiz, Melih Eren

</details>

<details>
<summary><h2>Sprint 3</h2></summary>

## Sprint Notları:
- Sprint 2'de iyileştirilmesi hedeflenen tahmin modeli, yapılan optimizasyonlarla son haline getirildi ve model olarak Random Forest seçildi.
- Kullanıcı arayüzünün kodlaması tamamlandı ve son kullanıcı deneyimine yönelik ince ayarlar yapıldı.
- Model ile kullanıcı arayüzü arasındaki entegrasyon tamamlanarak ürün tamamen işlevsel hale getirildi.
-	Toplantılar “Jitzi” veya “Google Meet” üzerinden yapıldı, gerekli durumlarda "Whatsapp” grubu üzerinden konuşmalar devam ettirildi.
-	Proje yönetimi için “Trello” kullanıldı.


## Sprint için Tamamlanması Beklenen Puan: 110 puan

## Tahmin Mantığı:

Toplamda 3 sprint olarak planlanan projemizin toplam puanı 300 olarak belirlenip 3. sprint'i için 110 puanlık bir hedef konulmuştur. Bu sprint, geliştirilen tüm bileşenlerin (model, arayüz) birleştirilmesi, uçtan uca testlerin yapılması, olası hataların giderilmesi ve projenin sunuma hazır hale getirilmesi gibi kritik tamamlama görevlerini içermektedir.

## Daily Scrum: 
Günlük iletişimin, kolaylık ve hız gibi artılarından ötürü Whatsapp üzerinden yapılmasına karar verilmiştir. [Sprint 3 - Daily Scrum](https://github.com/EfeBirsin/Grup-5-/tree/main/ProjectManagement/Sprint3/Daily%20Scrum%203)

## Sprint board update

![trello.png](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint3/trello3.png)

## Ürün Durumu Çıktısı

![Ürün Çıktısı 1](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint3/SepsisGuardAI-1.png)
![Ürün Çıktısı 2](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint3/SepsisGuardAI-2.png)
![Ürün Çıktısı 3](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint3/SepsisGuardAI-3.png)

## Sprint Retrospektifi:

- Sprint 2 retrospektifinde belirlenen model performansını artırma hedefi başarıyla gerçekleştirildi.
- Model ile arayüzün entegrasyonu planlandığı gibi tamamlandı.
- Projenin son halini sunan video ve sunum materyalleri hazırlandı.

## Sprint Review:

**Sprint Hedefi:**  

Sprint 3'ün ana hedefi, Sepsis Tahmin modelini ve kullanıcı arayüzünü nihai hale getirmek, bu iki bileşeni başarılı bir şekilde entegre etmek ve projeyi sunuma hazır, tamamen işlevsel bir ürün olarak tamamlamaktı.

**Tamamlanan İşler ve Çıktılar:**

- Son optimizasyonları yapılmış, performansı iyileştirilmiş ve final sürümü belirlenmiş yapay zeka modeli (Random Forest) teslim edildi.
- Kullanıcı arayüzü kodlandı, test edildi ve modelle entegre edildi.
- Trello üzerindeki tüm görevler "Tamamlandı" sütununa taşınarak proje yönetimi süreci sonlandırıldı.

**Sprint Katılımcıları:**

Aslı Şemsimoğlu, Beyza İrem Kaya, Efe Birsin, Esra Çilesiz, Melih Eren

</details>

