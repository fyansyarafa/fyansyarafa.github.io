---
title: testa
description: >-
  Dahulu, dibutuhkan seorang ahli untuk menyelesaikan masalah dan menjawab
  berbagai pertanyaan-pertanyaan di berbagai bidang pekerjaan…
date: '2020-10-09T12:41:48.261Z'
categories: []
keywords: []
slug: /@fyansyarafa/machine-learning-dan-implementasinya-di-dunia-nyata-f49f7547af4c
---

Dahulu, dibutuhkan seorang ahli untuk menyelesaikan masalah dan menjawab berbagai pertanyaan-pertanyaan di berbagai bidang pekerjaan. Seorang ahli yang telah menguasai berbagai macam masalah yang secara spesifik di bidangnya masing-masing akan mempelajari masalah baru terkait dengan bidang yang ditekuninya. Mempelajari masalah, mengekstrak segala kemungkinan yang akan terjadi, memperhitungkannya, sehingga menghasilkan berbagai macam aturan untuk menyelesaikan dan menjawab pertanyaan terkait masalah yang sedang dikerjakan. Kemudian, ahli tersebut akan meminta seorang _programmer_ menerjemahkan aturan-aturan ke dalam program komputer. Tetapi, yang perlu diperhatikan adalah program komputer yang dihasilkan akan menyelesaikan masalah yang sangat spesifik terhadap hal-hal yang telah diinisialisasi sebelumnya tanpa adanya fleksibilitas yang besar. Ketika masalah menjadi lebih kompleks, aturan-aturan tersebut perlu ada yang ditambahkan/diperbaharui atau bahkan program tersebut akan dibuat dari awal. Pekerjaan-pekerjaan tersebut tentu akan memakan _cost_ yang sangat besar.

Lalu bagaimana suatu komputer untuk mengurangi _cost_ tersebut? Tentunya dibutuhkan suatu metode yang secara otomatis mempelajari berbagai macam relasi dari data yang disediakan serta secara otomatis pula menghasilkan aturan-aturan untuk menyelesaikan masalah. Dengan kata lain, dibutuhkan suatu kemampuan komputer untuk meniru kemampuan manusia untuk menyelesaikan suatu masalah. Manusia dapat menyelesaikan masalah berdasarkan pengalaman yang telah terjadi di masa lampau. Komputer atau _machine_ diharapkan untuk dapat meniru atau paling tidak mendekati kemampuan manusia tersebut. Hal inilah yang menjadi konsep utama dari _Machine Learning._

### Apa itu _machine learning_?

> Machine learning merupakan sub area ilmu komputer yang mempu memberikan komputer kemampuan untuk belajar tanpa diprogram secara eksplisit.

> Arthur Samuel, IBM researcher, 1959

Bagaimana komputer/mesin dapat belajar sendiri tanpa diprogram? Komputer dapat memiliki kemampuan tersebut dengan memberikan _experience_ kepada mesin sehingga dapat menghasilkan output yang diharapkan tanpa harus terus-menerus diprogram.

![](\images\img\1__07gR7X8__LE4FmkgjuYhCFA.png)


Machine learning sendiri merupakan bagian dari Artificial Intelligence yang dapat diartikan sebagai kemampuan yang dapat meniru kecerdasan alami yang ada pada makhluk hidup, terutama manusia. Kemampuan tersebut lebih lanjut dapat menyelesaikan berbagai bentuk masalah selayaknya manusia menyelesaikan masalahnya.

### Tipe-tipe machine learning

Machine learning dapat digunakan dalam berbagai kebutuhan/permasalahan. Berdasarkan dampak yang diharapkan oleh _user_ untuk menyelesaikan masalah dapat dikategorikan ke dalam berbagai tipe berikut.

#### Supervised Learning

![](\images\img\1__23VHNduP3__IK9__rBiVLZPA.png)

Tipe ini akan memetakan _input_ ke _output_ yang sesuai dengan label yang diinginkan. Dengan demikian kualitas hasil pembelajaran sangat bergantung pada kesesuaian antara _input_ dan _output_ yang dihasilkan. Oleh karena itu, peran user sangat dibutuhkan untuk memvalidasi _input_ dan _output_ yang dihasilkan. Atau dengan pengertian lain, pembelajaran dan pelatihan model dapat diterapkan ke bentuk dataset yang berlabel.

![Contoh dataset berlabel](\images\img\1__jSsviVLjXvWpvgl9MuKR3A.png)
Contoh dataset berlabel

Karena peran user dibutuhkan untuk validasi kesesuaian tersebut, _supervised learning_ juga sering disebut dengan pembelajaran terawasi. Tipe ini seringkali digunakan untuk permasalahan-permasalahan prediktif.

Dari hasil yang diharapkan, supervised learning dibedakan menjadi dua metode. Jika output yang dihasilkan berupa kelas-kelas tertentu (diskrit), maka metode yang digunakan adalah klasifikasi. Dan jika output yang dihasilkan berupa nilai-nilai kontinu, maka metode regresi adalah yang paling tepat untuk digunakan.

#### Unsupervised Learning

![](\images\img\1__0SkUsGp9ntltiYg4KJuPVQ.png)

Jika _supervised learning_ merupakan tipe _machine learning_ terhadap data berlabel, maka _unsupervised learning_ adalah kebalikannya. _Unsupervised learning_ bekerja pada data yang tidak berlabel/kelas target. Yang artinya user tidak dapat melakukan klasifikasi untuk memprediksi kelas-kelas tertentu sebagaimana _supervised learning_. Dengan demikian, tipe ini disebut juga dengan tipe pembelajaran tanpa panduan. Tipe ini hanya mengkategorikan sesuatu berdasarkan kedekatan, kemiripan, ataupun kriteria-kriteria lainnya. Tipe ini biasa digunakan untuk masalah klasterisasi, di mana output yang dihasilkan berupa pengelompokkan data ke dalam sejumalah klaster yang diinginkan. Selain masalah klasterisasi, _unsupervised learning_ juga digunakan dalam masalah reduksi dimensi, estimasi kepadatan, serta _market basket analysis_.

#### Reinforcement Learning

![](\images\img\1__u2NmeQ__NYwj8I3O2iWl6PA.png)

Tipe ini memungkinkan untuk mempelajari suatu tindakan dengan mempertimbangkan apa yang ada di lingkungan sekitar. Setiap aksi dapat menghasilkan akibat bagi lingkungan, serta lingkungan dapat menjadi umpan balik feedback untuk tindakan-tindakan/aksi berikutnya. Tipe ini banyak digunakan dalam perancangan game bots dan perancangan kecerdasan robot.

### Beberapa contoh penerapan machine learning

Pernakah kita menyadari bahwa segala yang ada di beranda YouTube atau pada berbagai sosial media kita sesuai dengan apa yang telah kita eksplor sebelumnya atau bahkan konten-konten pada beranda kita sesuai dengan yang kita harapkan? Yup, ini merupakan salah satu contoh penggunaan machine learning dalam aktivitas sehari-hari kita. Tidak hanya memberikan rekomendasi-rekombendasi konten tertentu, _machine learning_ dapat digunakan dalam berbagai permasalahan sesuai dengan beberapa tipe di atas.

Dalam bidang finansial, _machine learning_ dapat mengklasifikasikan mana customer yang berpotensi gagal bayar pada permasalahan kartu kredit. Dengan rekomendasi yang dihasilkan dari proses machine learning, user dapat memberikan kredit kepada customer yang memiliki resiko rendah.

Di bidang _marketing_, _machine learning_ dapat digunakan untuk melakukan klasterisasi segmentasi pasar berdasarkan atribut-atribut yang terdapat pada dataset yang dimiliki. Lebih lanjut, pihak _marketing_ akan memberikan penawaran-penawaran tertentu dengan lebih tepat sasaran ke klaster-klaster yang dibentuk.

Berbagai produk-produk modern telah dihasilkan dengan memanfaatkan _artificial intelligence_. AI dapat dilakukan untuk melukis/menggambar objek yang tidak pernah ada sebelumnya dengan Deep Dream, Generative Adversarial Network. AI dapat menjadi sutradara sebuah film: _Sunspring_. AI dapat men-_generate_ kode program: DeepCoder. Atau berbagai tools-tools cerdas lainnya seperti:

#### self driving car

Sesuai dengan namanya, mobil yang dapat berjalan tanpa harus dikendalikan oleh manusia.

![](\images\img\1__gzOkGrNVgr__4vd1sy37qMw.jpeg)

Mobil akan mengenali objek objek yang ada dihadapannya dan kemudian beradaptasi untuk melakukan berbagai macam aksi seperti melaju dengan kencang, sedang, melambat, belok ke kiri atau ke kanan, serta berhenti dengan sendirinya. Juga tentunya terintegrasi dengan peta digital.

#### smart home

Berbagai macam produk _smart home_ telah dihasilkan dengan memanfaatkan _artificial intelligence_. Seperti lampu yang dapat dikontrol pada smartphone ataupun tanpa dikontrol secara langsung. Ataupun pada cctv yang dapat mengenali objek-objek/aktivitas anomali dan melaporkannya secara otomatis kepada pemilik rumah sebagai notifikasi keamanan.

#### machine translation

Di era sekarang, hampir seluruh manusia di muka bumi pernah menggunakan mesin penerjemah untuk aktivitas hariannya. Mesin penerjemah digunakan untuk melakukan penerjemahan dari bahasa satu ke bahasa lainnya. Metode _Recurrent Neural Network_ pada _Deep Learning_ dapat digunakan untuk memproses aktivitas penerjemahan tersebut.

Tidak hanya yang disebutkan di atas. Metode metode _artificial intelligence_ juga banyak digunakan dalam produk-produk modern lainnya seperti virtual assistant: Cortana, Siri, atau Google Now.

Read on [medium](https://medium.com/@fyansyarafa/machine-learning-dan-implementasinya-di-dunia-nyata-f49f7547af4c)
