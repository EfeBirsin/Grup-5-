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
        <img src="https://avatars.githubusercontent.com/u/44444444" width="60" height="60" style="border-radius:50%" /><br>
        <strong>Esra Çilesiz</strong>
      </td>
      <td>Developer</td>
      <td>
        <a href="https://www.linkedin.com/in/esra-cilesiz/" target="_blank">🔗 LinkedIn</a>
      </td>
    </tr>
    <tr>
      <td>
        <img src="https://avatars.githubusercontent.com/u/55555555" width="60" height="60" style="border-radius:50%" /><br>
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
