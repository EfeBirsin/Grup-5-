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

Toplamda 3 sprint olarak planlanan projemizintoplam puanı 300 olarak belirlenip ilk sprint'i için 80 puanlık bir hedef konulmuştur. Bu puan, fiili kodlama içerse de veri setini anlama, detaylı rapor oluşturma ve strateji geliştirme gibi bilişsel yük gerektiren görevleri kapsamaktadır. Projenin en önemli adımı olan "ne yapılacağını ve nasıl yapılacağını" netleştiren bu görevler, projenin temelini oluşturduğu için yüksek puanlanmıştır. Amaç, Sprint 2'ye "sadece kodlamaya odaklanabileceğimiz" temiz bir başlangıçla girmektir.
Sprint 2 ve Sprint 3, projenin en yoğun geliştirme fazlarını temsil etmektedir. Sprint 2, modellemenin; Sprint 3 ise arayüz geliştirmenin ağırlıklı olduğu dönemlerdir. Bu sprint'lerdeki görevler, yüksek derecede teknik karmaşıklık ve uygulama eforu gerektirdiğinden, her biri için hedeflenen puan 110 olarak belirlenmiştir.

## Daily Scrum: 
Günlük  iletişimin, kolaylık ve hız gibi artılarından ötürü  Whatsapp üzerinden yapılmasına karar verilmiştir. Günlük iletişim örnekleri pdf olarak tarafımızdan paylaşılmaktadır: [sprint1_daily](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint1Documents/sprint1_daily%20(1).pdf)

## Sprint board update

![trello.png](https://raw.githubusercontent.com/EfeBirsin/Grup-5-/f75666b8a4f7b3be9eff96505825ce9f63a4d2bd/ProjectManagement/Sprint1Documents/trello.png)

## Ürün Durumu Çıktısı

![Ürün Çıktısı 1](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint1Documents/%C3%9Cr%C3%BCn_Tasar%C4%B1m%C4%B1_Figma.jpg?raw=true)
![Ürün Çıktısı 2](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint1Documents/%C3%9Cr%C3%BCn_Tasar%C4%B1m%C4%B1_Figma_2.jpg?raw=true)
![Ürün Çıktısı 3](https://github.com/EfeBirsin/Grup-5-/blob/main/ProjectManagement/Sprint1Documents/%C3%9Cr%C3%BCn_Tasar%C4%B1m%C4%B1_Figma_3.jpg?raw=true)

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
