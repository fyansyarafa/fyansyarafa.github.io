---

title: "Twitter Hate Speech Classification"
data: 2021-05-11
tags: [python,  exploratory data analysis, classification]
header:
excerpt: ""
mathjax: "true"
toc: true
toc_sticky: false
header:
  teaser: '/images/twitterhate/teaser.jpeg'
---

<a href="https://colab.research.google.com/github/fyansyarafa/hate-speech-classification/blob/main/hate_speech_classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Klasifikasi Tweet Bermakna Ujaran Kebencian Berbahasa Indonesia**

Berdasarkan dataset https://raw.githubusercontent.com/ialfina/id-hatespeech-detection/master/IDHSD_RIO_unbalanced_713_2017.txt, akan dibuat model yang dapat mengklasifikasikan sebuah tweet ke dalam tweet berisikan ujaran kebencian atau tidak.

Model klasifikasi yang dibuat menggunakan model:


1.   Logistic Regression, dan
2.   K-Nearest Neighbour dengan jumlah tetangga = 5





```
import pandas as pd
import numpy as np

import requests
import io
import warnings
warnings.filterwarnings("ignore")

import nltk

response = requests.get('https://raw.githubusercontent.com/ialfina/id-hatespeech-detection/master/IDHSD_RIO_unbalanced_713_2017.txt')
data = io.StringIO(response.text)
```


```
df = pd.read_csv(data, sep ='\t')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Non_HS</td>
      <td>RT @spardaxyz: Fadli Zon Minta Mendagri Segera...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Non_HS</td>
      <td>RT @baguscondromowo: Mereka terus melukai aksi...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Non_HS</td>
      <td>Sylvi: bagaimana gurbernur melakukan kekerasan...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Non_HS</td>
      <td>Ahmad Dhani Tak Puas Debat Pilkada, Masalah Ja...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Non_HS</td>
      <td>RT @lisdaulay28: Waspada KTP palsu.....kawal P...</td>
    </tr>
  </tbody>
</table>
</div>




```
df['Label']=df['Label'].map({'HS' : 1, 'Non_HS': 0})
```


```
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>RT @spardaxyz: Fadli Zon Minta Mendagri Segera...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>RT @baguscondromowo: Mereka terus melukai aksi...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>Sylvi: bagaimana gurbernur melakukan kekerasan...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Ahmad Dhani Tak Puas Debat Pilkada, Masalah Ja...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>RT @lisdaulay28: Waspada KTP palsu.....kawal P...</td>
    </tr>
  </tbody>
</table>
</div>




```
import seaborn as sns
```


```
sns.countplot(df.Label);
```


![png](/images/twitterhate/hate_speech_classification_7_0.png)


# 2. PreProcessing (Data Cleaning)

## 2.1 Case Folding
Pada section ini, akan dilakukan text cleaning pada teksnya langsung agar dapat diproses pada fase training model. Tujuannya, untuk mengubah semua huruf ke dalam huruf kecil, menghilangkan angka, situs, username, dan tanda baca.


```
df2 = df.copy()
```


```
df2['Tweet'][0] = df2['Tweet'][0].lower()
```

### 2.1.1 Karakter ke LowerCase


```
for i in range(len(df2)):
  df2['Tweet'][i] = df2['Tweet'][i].lower()
```


```
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>rt @spardaxyz: fadli zon minta mendagri segera...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>rt @baguscondromowo: mereka terus melukai aksi...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>sylvi: bagaimana gurbernur melakukan kekerasan...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>ahmad dhani tak puas debat pilkada, masalah ja...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>rt @lisdaulay28: waspada ktp palsu.....kawal p...</td>
    </tr>
  </tbody>
</table>
</div>



### 2.1.2 Menghilangkan `http...`, `www...`, dan `@...`


```
df2['Tweet'] = df2['Tweet'].replace(r"http\S+", "", regex=True)
```


```
df2['Tweet'] = df2['Tweet'].replace(r'www([A-Za-z0-9_]+)', '', regex=True)
```


```
df2['Tweet'] = df2['Tweet'].replace(r'@([A-Za-z0-9_]+)', '', regex=True)
```


```
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>rt : fadli zon minta mendagri segera menonakti...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>rt : mereka terus melukai aksi dalam rangka me...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>sylvi: bagaimana gurbernur melakukan kekerasan...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>ahmad dhani tak puas debat pilkada, masalah ja...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>rt : waspada ktp palsu.....kawal pilkada</td>
    </tr>
  </tbody>
</table>
</div>



### 2.1.2 Hapus Angka


```
import re
```


```
testli = []
```


```
for i in range(len(df2)):
  testli.append(df2['Tweet'][i])
```


```
for i in range(len(df2)):
  df2['Tweet'][i] = re.sub(r"\d+","",testli[i])
```


```
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>rt : fadli zon minta mendagri segera menonakti...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>rt : mereka terus melukai aksi dalam rangka me...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>sylvi: bagaimana gurbernur melakukan kekerasan...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>ahmad dhani tak puas debat pilkada, masalah ja...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>rt : waspada ktp palsu.....kawal pilkada</td>
    </tr>
  </tbody>
</table>
</div>



### 2.1.3 Hapus Tanda Baca


```
import string
for i in range(len(df2)):
  df2['Tweet'][i] = df2['Tweet'][i].translate(str.maketrans("","",string.punctuation))
```


```
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>rt  fadli zon minta mendagri segera menonaktif...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>rt  mereka terus melukai aksi dalam rangka mem...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>sylvi bagaimana gurbernur melakukan kekerasan ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>ahmad dhani tak puas debat pilkada masalah jal...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>rt  waspada ktp palsukawal pilkada</td>
    </tr>
  </tbody>
</table>
</div>



## 2.2 Tokenizing
Merupakan proses pemisahan teks menjadi potongan-potongan kata di dalam suatu list untuk kemudian dianalisis. Analisis yang dapat digunakan yaitu untuk melihat kata-kata yang paling sering muncul dalam suatu teks/tweet.


```
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
```


```
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!





    True




```
for i in range(len(df2)):
  df2['Tweet'][i] = word_tokenize(df2['Tweet'][i])
```


```
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[rt, fadli, zon, minta, mendagri, segera, meno...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>[rt, mereka, terus, melukai, aksi, dalam, rang...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>[sylvi, bagaimana, gurbernur, melakukan, keker...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>[ahmad, dhani, tak, puas, debat, pilkada, masa...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>[rt, waspada, ktp, palsukawal, pilkada]</td>
    </tr>
  </tbody>
</table>
</div>



## 2.2 Filtering StopWords

Stopwords merupakan proses filtering untuk memisahkan kata-kata yang kurang penting dan dianggap sebagai kata umum dalam jumlah besar yang tidak memiliki makna.


```
pip install Sastrawi
```

    Requirement already satisfied: Sastrawi in /usr/local/lib/python3.6/dist-packages (1.0.1)



```
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
```


```
# stopwords pada lib sastrawi
factory = StopWordRemoverFactory()
stopwords_list = factory.get_stop_words()
print(stopwords_list)
```

    ['yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua', 'ia', 'seperti', 'jika', 'jika', 'sehingga', 'kembali', 'dan', 'tidak', 'ini', 'karena', 'kepada', 'oleh', 'saat', 'harus', 'sementara', 'setelah', 'belum', 'kami', 'sekitar', 'bagi', 'serta', 'di', 'dari', 'telah', 'sebagai', 'masih', 'hal', 'ketika', 'adalah', 'itu', 'dalam', 'bisa', 'bahwa', 'atau', 'hanya', 'kita', 'dengan', 'akan', 'juga', 'ada', 'mereka', 'sudah', 'saya', 'terhadap', 'secara', 'agar', 'lain', 'anda', 'begitu', 'mengapa', 'kenapa', 'yaitu', 'yakni', 'daripada', 'itulah', 'lagi', 'maka', 'tentang', 'demi', 'dimana', 'kemana', 'pula', 'sambil', 'sebelum', 'sesudah', 'supaya', 'guna', 'kah', 'pun', 'sampai', 'sedangkan', 'selagi', 'sementara', 'tetapi', 'apakah', 'kecuali', 'sebab', 'selain', 'seolah', 'seraya', 'seterusnya', 'tanpa', 'agak', 'boleh', 'dapat', 'dsb', 'dst', 'dll', 'dahulu', 'dulunya', 'anu', 'demikian', 'tapi', 'ingin', 'juga', 'nggak', 'mari', 'nanti', 'melainkan', 'oh', 'ok', 'seharusnya', 'sebetulnya', 'setiap', 'setidaknya', 'sesuatu', 'pasti', 'saja', 'toh', 'ya', 'walau', 'tolong', 'tentu', 'amat', 'apalagi', 'bagaimanapun']



```
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True




```
from nltk.corpus import stopwords
```


```
list_stop = stopwords.words('indonesian')
```


```
# menggabungkannya dengan stopwords bahasa indonesia dalam lib nltk corpus indonesian
myset = set(stopwords_list + list_stop)
myset.add('rt')
```


```
def stopwords_removal(words):
  return [word for word in words if word not in myset]
```


```
df2['Tweet'] = df2['Tweet'].apply(stopwords_removal)
```


```
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[fadli, zon, mendagri, menonaktifkan, ahok, gu...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>[melukai, aksi, rangka, memenjarakan, ahok, ah...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>[sylvi, gurbernur, kekerasan, perempuan, bukti...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>[ahmad, dhani, puas, debat, pilkada, jalan, be...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>[waspada, ktp, palsukawal, pilkada]</td>
    </tr>
  </tbody>
</table>
</div>



## 2.4 Stemming
Proses ini mengembalikan suatu kata berimbuhan ke bentuk dasarnya. Hal ini berguna untuk fleksibilitas suatu kata terhadap berbagai konteks kalimat dan tidak terpaku pada imbuhan-imbuhan tertentu.


```
pip install swifter
```


```
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
```


```
factory = StemmerFactory()
stemmer = factory.create_stemmer()
```


```
def stemmed_wrapper(term):
  return stemmer.stem(term)
```


```
term_dict = {}
```


```
for document in df2['Tweet']:
  for term in document:
    if term not in term_dict:
      term_dict[term] = ' '

print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])

print(term_dict)
print("------------------------")

def get_stemmed_term(document):
    return [term_dict[term] for term in document]

df2['Final Tweets Stemmed'] = df2['Tweet'].swifter.apply(get_stemmed_term)
print(df2['Final Tweets Stemmed'])
```


```
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Tweet</th>
      <th>Final Tweets Stemmed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[fadli, zon, mendagri, menonaktifkan, ahok, gu...</td>
      <td>[fadli, zon, mendagri, nonaktif, ahok, gubernu...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>[melukai, aksi, rangka, memenjarakan, ahok, ah...</td>
      <td>[luka, aksi, rangka, penjara, ahok, ahok, gaga...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>[sylvi, gurbernur, kekerasan, perempuan, bukti...</td>
      <td>[sylvi, gurbernur, keras, perempuan, bukti, fo...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>[ahmad, dhani, puas, debat, pilkada, jalan, be...</td>
      <td>[ahmad, dhani, puas, debat, pilkada, jalan, be...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>[waspada, ktp, palsukawal, pilkada]</td>
      <td>[waspada, ktp, palsukawal, pilkada]</td>
    </tr>
  </tbody>
</table>
</div>



# 3. Visualisasi Tweet Dominan Berdasarkan Label

## 3.1 Label 1: HS (HateSpeach)


```
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
```


```
import matplotlib.pyplot as plt
```


```
word = df2['Final Tweets Stemmed'][df2['Label']==1].astype('string')
word.head()
wordcloud = WordCloud(max_font_size=65, max_words=200, background_color="grey").generate_from_text(' '.join(word))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```


![png](/images/twitterhate/hate_speech_classification_57_0.png)


## 3.2 Label 0: Non_HS (Non_HateSpeach)


```
word = df2['Final Tweets Stemmed'][df2['Label']==0].astype('string')
word.head()
wordcloud = WordCloud(max_font_size=65, max_words=200, background_color="grey").generate_from_text(' '.join(word))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```


![png](/images/twitterhate/hate_speech_classification_59_0.png)


# 4. Model Klasifikasi

## 4.1 Text Vectorization Menggunakan TF-IDF

Karena model machine learning hanya dapat diproses dari fitur berupa angka, maka tweet-tweet tersebut akan dilakukan vektorisasi yang memetakan setiap kata dalam tweet ke dalam angka. Vektorisasi yang digunakan adalah TF-IDF.


```
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
```


```
df3 = df2.copy()
```


```
df3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Tweet</th>
      <th>Final Tweets Stemmed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[fadli, zon, mendagri, menonaktifkan, ahok, gu...</td>
      <td>[fadli, zon, mendagri, nonaktif, ahok, gubernu...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>[melukai, aksi, rangka, memenjarakan, ahok, ah...</td>
      <td>[luka, aksi, rangka, penjara, ahok, ahok, gaga...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>[sylvi, gurbernur, kekerasan, perempuan, bukti...</td>
      <td>[sylvi, gurbernur, keras, perempuan, bukti, fo...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>[ahmad, dhani, puas, debat, pilkada, jalan, be...</td>
      <td>[ahmad, dhani, puas, debat, pilkada, jalan, be...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>[waspada, ktp, palsukawal, pilkada]</td>
      <td>[waspada, ktp, palsukawal, pilkada]</td>
    </tr>
  </tbody>
</table>
</div>




```
# list ke string
def listToString(s):
  str1 = ' '
  return(str1.join(s))
```


```
df3['Clean Tweets'] = df3['Final Tweets Stemmed'].apply(listToString)
```


```
df3['Tweet'] = df['Tweet']
```


```
df3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Tweet</th>
      <th>Final Tweets Stemmed</th>
      <th>Clean Tweets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>RT @spardaxyz: Fadli Zon Minta Mendagri Segera...</td>
      <td>[fadli, zon, mendagri, nonaktif, ahok, gubernu...</td>
      <td>fadli zon mendagri nonaktif ahok gubernur dki</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>RT @baguscondromowo: Mereka terus melukai aksi...</td>
      <td>[luka, aksi, rangka, penjara, ahok, ahok, gaga...</td>
      <td>luka aksi rangka penjara ahok ahok gagal pilkada</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>Sylvi: bagaimana gurbernur melakukan kekerasan...</td>
      <td>[sylvi, gurbernur, keras, perempuan, bukti, fo...</td>
      <td>sylvi gurbernur keras perempuan bukti foto bar...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Ahmad Dhani Tak Puas Debat Pilkada, Masalah Ja...</td>
      <td>[ahmad, dhani, puas, debat, pilkada, jalan, be...</td>
      <td>ahmad dhani puas debat pilkada jalan bekas ungkap</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>RT @lisdaulay28: Waspada KTP palsu.....kawal P...</td>
      <td>[waspada, ktp, palsukawal, pilkada]</td>
      <td>waspada ktp palsukawal pilkada</td>
    </tr>
  </tbody>
</table>
</div>




```
X = df3['Clean Tweets']
```


```
y = df3['Label']
```


```
tfidf = TfidfVectorizer()
tfid_vector = tfidf.fit_transform(X)
```

## 4.2 Split Data Ke Training dan Test Data


```
from sklearn.model_selection import train_test_split
```


```
X_train, X_test, y_train, y_test = train_test_split(tfid_vector, y, test_size=0.33, random_state=42)
```

## 4.3 Mengguanakan Model Logistic Regression


```
from sklearn.linear_model import LogisticRegression
```


```
logreg = LogisticRegression()
```


```
logreg.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)



### 4.3.1 Prediksi dan Evaluasi


```
pred = logreg.predict(X_test)
```


```
from sklearn.metrics import classification_report as cr
print(cr(y_test,pred))
```

                  precision    recall  f1-score   support

               0       0.86      0.96      0.91       163
               1       0.89      0.64      0.75        73

        accuracy                           0.86       236
       macro avg       0.87      0.80      0.83       236
    weighted avg       0.87      0.86      0.86       236




```
from sklearn.metrics import confusion_matrix as cm
from sklearn import metrics
```


```
def plot_cm(y_true, predictions, figsize=(5,4)):
    cm = confusion_matrix(y_test, pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    annot_kws = {"ha": 'left',"va": 'bottom'}

    ae = sns.heatmap(cm, cmap= "YlGnBu", annot=annot,fmt='', ax=ax)
    bottom, top = ae.get_ylim()
    ae.set_ylim(bottom + 0.5, top - 0.5)
plot_cm(y_test, pred)
plt.tight_layout();
```


![png](/images/twitterhate/hate_speech_classification_84_0.png)


Karena terjadi *imbalance* label pada dataframe, maka metrics evaluasi yang baik digunakan adalah metric AUC.


```
# countplot untuk label dataframe
sns.countplot(df3.Label);
```


![png](/images/twitterhate/hate_speech_classification_86_0.png)



```
# Set Size and Style
plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('ggplot')

# Visualize ROC Curve
y_pred_logmodel_proba = logreg.predict_proba(X_test)[::,1]
fprlogmodel, tprlogmodel, _ = metrics.roc_curve(y_test,  y_pred_logmodel_proba)
auclogmodel = metrics.roc_auc_score(y_test, y_pred_logmodel_proba)
plt.plot(fprlogmodel,tprlogmodel,label="Logistic Regression, auc="+str(auclogmodel))
plt.title('ROC Curve - Logistic Regression')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc=4)
plt.show()
```


![png](/images/twitterhate/hate_speech_classification_87_0.png)


## 4.4 Menggunakan Model KNN


```
from sklearn.neighbors import KNeighborsClassifier
```


```
knn = KNeighborsClassifier(n_neighbors=5)
```

### 4.4.1 Prediksi dan Evaluasi


```
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
```


```
print(cr(y_test,knn_pred))
```

                  precision    recall  f1-score   support

               0       0.89      0.83      0.86       163
               1       0.67      0.78      0.72        73

        accuracy                           0.81       236
       macro avg       0.78      0.80      0.79       236
    weighted avg       0.82      0.81      0.82       236




```
def plot_cm(y_true, predictions, figsize=(5,4)):
    cm = confusion_matrix(y_test, knn_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    annot_kws = {"ha": 'left',"va": 'bottom'}

    ae = sns.heatmap(cm, cmap= "YlGnBu", annot=annot,fmt='', ax=ax)
    bottom, top = ae.get_ylim()
    ae.set_ylim(bottom + 0.5, top - 0.5)
plot_cm(y_test, knn_pred)
plt.tight_layout();
```


![png](/images/twitterhate/hate_speech_classification_94_0.png)



```
plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('ggplot')

y_pred_knn_proba = knn.predict_proba(X_test)[::,1]
fprknn, tprknn, _ = metrics.roc_curve(y_test,  y_pred_knn_proba)
aucknn = metrics.roc_auc_score(y_test, y_pred_knn_proba)
plt.plot(fprknn,tprknn,label="KNN, auc="+str(aucknn))
plt.title('ROC Curve - KNN')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc=4)
plt.show()
```


![png](/images/twitterhate/hate_speech_classification_95_0.png)


# 5. Menentukan Model Terbaik


```
plt.plot(fprlogmodel,tprlogmodel,label="Logistic Regression, auc="+str(auclogmodel))
plt.plot(fprknn,tprknn,label="KNN, auc="+str(aucknn))
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
```


![png](/images/twitterhate/hate_speech_classification_97_0.png)


Model Logistic Regression memberikan nilai yang lebih besar daripada KNN. Oleh karena itu, model terbaik yang dapat dijadikan sebagai model untuk prediksi tweet selanjutnya yaitu model dengan logistic regression.
