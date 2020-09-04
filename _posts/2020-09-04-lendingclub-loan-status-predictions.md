---
title: "Membuat Model Prediktif terhadap Peminjam Gagal Bayar atau Berhasil Membayar Kembali Pinjaman dengan Artificial Neural Network"
data: 2020-09-04
tags: [python,  exploratory data analysis, statistics, probability, deep learning, artificial neural network]
header:
excerpt: ""
mathjax: "true"
---



## The Data

Sumber data didapatkan dari: https://www.kaggle.com/wordsforthewise/lending-club


LendingClub adalah perusahaan pinjaman peer-to-peer Amerika , yang berkantor pusat di San Francisco, California . Ini adalah pemberi pinjaman peer-to-peer pertama yang mendaftarkan penawarannya sebagai sekuritas dengan Securities and Exchange Commission (SEC), dan menawarkan perdagangan pinjaman di pasar sekunder. LendingClub adalah platform pinjaman peer-to-peer terbesar di dunia. Perusahaan mengklaim bahwa pinjaman senilai $ 15,98 miliar telah berasal dari platformnya hingga 31 Desember 2015.

LendingClub memungkinkan peminjam untuk membuat pinjaman pribadi tanpa jaminan antara \$ 1.000 dan \$ 40.000. Jangka waktu pinjaman standar adalah tiga tahun. Investor dapat mencari dan menelusuri daftar pinjaman di situs LendingClub dan memilih pinjaman yang ingin mereka investasikan berdasarkan informasi yang diberikan tentang peminjam, jumlah pinjaman, nilai pinjaman, dan tujuan pinjaman. Investor menghasilkan uang dari bunga. LendingClub menghasilkan uang dengan membebankan peminjam biaya origination dan investor biaya layanan.

*Sumber: [Wikipedia](https://en.wikipedia.org/wiki/LendingClub)*






```python
from google.colab import drive
drive.mount('/content/drive')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly&response_type=code

    Enter your authorization code:
    ··········
    Mounted at /content/drive


### Our Goal

Dengan adanya data historis yang diekstrak dari informasi bahwa apakan peminjam dapat berstatus gagal bayar (charged-off) atau berhasil membayar pinjaman (fully paid), dapatkah kita membangun suatu model prediksi yang dapat memprediksi kemungkinan status peminjam setelah melakukan pinjaman?

Dengan cara ini, kedepannya kita dapat mengakses/melakukan screening terhadap customer baru berdasarkan data-data yang mereka berikan untuk memprediksi kemungkinan mereka bersatus gagal bayar (charged-off) atau tidak.


### Data Overview


Berikut adalah beberapa informasi deskripsi variable/feature column yang akan digunakan:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LoanStatNew</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>loan_amnt</td>
      <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>term</td>
      <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>int_rate</td>
      <td>Interest Rate on the loan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>installment</td>
      <td>The monthly payment owed by the borrower if the loan originates.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>grade</td>
      <td>LC assigned loan grade</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sub_grade</td>
      <td>LC assigned loan subgrade</td>
    </tr>
    <tr>
      <th>6</th>
      <td>emp_title</td>
      <td>The job title supplied by the Borrower when applying for the loan.*</td>
    </tr>
    <tr>
      <th>7</th>
      <td>emp_length</td>
      <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>home_ownership</td>
      <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
    </tr>
    <tr>
      <th>9</th>
      <td>annual_inc</td>
      <td>The self-reported annual income provided by the borrower during registration.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>verification_status</td>
      <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
    </tr>
    <tr>
      <th>11</th>
      <td>issue_d</td>
      <td>The month which the loan was funded</td>
    </tr>
    <tr>
      <th>12</th>
      <td>loan_status</td>
      <td>Current status of the loan</td>
    </tr>
    <tr>
      <th>13</th>
      <td>purpose</td>
      <td>A category provided by the borrower for the loan request.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>title</td>
      <td>The loan title provided by the borrower</td>
    </tr>
    <tr>
      <th>15</th>
      <td>zip_code</td>
      <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
    </tr>
    <tr>
      <th>16</th>
      <td>addr_state</td>
      <td>The state provided by the borrower in the loan application</td>
    </tr>
    <tr>
      <th>17</th>
      <td>dti</td>
      <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
    </tr>
    <tr>
      <th>18</th>
      <td>earliest_cr_line</td>
      <td>The month the borrower's earliest reported credit line was opened</td>
    </tr>
    <tr>
      <th>19</th>
      <td>open_acc</td>
      <td>The number of open credit lines in the borrower's credit file.</td>
    </tr>
    <tr>
      <th>20</th>
      <td>pub_rec</td>
      <td>Number of derogatory public records</td>
    </tr>
    <tr>
      <th>21</th>
      <td>revol_bal</td>
      <td>Total credit revolving balance</td>
    </tr>
    <tr>
      <th>22</th>
      <td>revol_util</td>
      <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
    </tr>
    <tr>
      <th>23</th>
      <td>total_acc</td>
      <td>The total number of credit lines currently in the borrower's credit file</td>
    </tr>
    <tr>
      <th>24</th>
      <td>initial_list_status</td>
      <td>The initial listing status of the loan. Possible values are – W, F</td>
    </tr>
    <tr>
      <th>25</th>
      <td>application_type</td>
      <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
    </tr>
    <tr>
      <th>26</th>
      <td>mort_acc</td>
      <td>Number of mortgage accounts.</td>
    </tr>
    <tr>
      <th>27</th>
      <td>pub_rec_bankruptcies</td>
      <td>Number of public record bankruptcies</td>
    </tr>
  </tbody>
</table>


## Starter Code




```python
import pandas as pd
```


```python
data_info = pd.read_csv('/content/drive/My Drive/Colab Notebooks/ANN/DATA/lending_club_info.csv',index_col='LoanStatNew')
```


```python
print(data_info.loc['revol_util']['Description'])
```

    Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.



```python
def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])
```


```python
feat_info('mort_acc')
```

    Number of mortgage accounts.


## Loading the data


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# might be needed depending on your version of Jupyter
%matplotlib inline
```


```python
df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/ANN/DATA/lending_club_loan_two.csv')
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 396030 entries, 0 to 396029
    Data columns (total 27 columns):
     #   Column                Non-Null Count   Dtype  
    ---  ------                --------------   -----  
     0   loan_amnt             396030 non-null  float64
     1   term                  396030 non-null  object
     2   int_rate              396030 non-null  float64
     3   installment           396030 non-null  float64
     4   grade                 396030 non-null  object
     5   sub_grade             396030 non-null  object
     6   emp_title             373103 non-null  object
     7   emp_length            377729 non-null  object
     8   home_ownership        396030 non-null  object
     9   annual_inc            396030 non-null  float64
     10  verification_status   396030 non-null  object
     11  issue_d               396030 non-null  object
     12  loan_status           396030 non-null  object
     13  purpose               396030 non-null  object
     14  title                 394275 non-null  object
     15  dti                   396030 non-null  float64
     16  earliest_cr_line      396030 non-null  object
     17  open_acc              396030 non-null  float64
     18  pub_rec               396030 non-null  float64
     19  revol_bal             396030 non-null  float64
     20  revol_util            395754 non-null  float64
     21  total_acc             396030 non-null  float64
     22  initial_list_status   396030 non-null  object
     23  application_type      396030 non-null  object
     24  mort_acc              358235 non-null  float64
     25  pub_rec_bankruptcies  395495 non-null  float64
     26  address               396030 non-null  object
    dtypes: float64(12), object(15)
    memory usage: 81.6+ MB



```python
df.shape
```




    (396030, 27)



Data tersebut berisikan 396030 buah baris dan juga terdapat 27 buah feature columns/variabel yang akan digunakan.

Preview data:


```python
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
      <th>loan_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>purpose</th>
      <th>title</th>
      <th>dti</th>
      <th>earliest_cr_line</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>initial_list_status</th>
      <th>application_type</th>
      <th>mort_acc</th>
      <th>pub_rec_bankruptcies</th>
      <th>address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10000.0</td>
      <td>36 months</td>
      <td>11.44</td>
      <td>329.48</td>
      <td>B</td>
      <td>B4</td>
      <td>Marketing</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>117000.0</td>
      <td>Not Verified</td>
      <td>Jan-2015</td>
      <td>Fully Paid</td>
      <td>vacation</td>
      <td>Vacation</td>
      <td>26.24</td>
      <td>Jun-1990</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>36369.0</td>
      <td>41.8</td>
      <td>25.0</td>
      <td>w</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0174 Michelle Gateway\r\nMendozaberg, OK 22690</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8000.0</td>
      <td>36 months</td>
      <td>11.99</td>
      <td>265.68</td>
      <td>B</td>
      <td>B5</td>
      <td>Credit analyst</td>
      <td>4 years</td>
      <td>MORTGAGE</td>
      <td>65000.0</td>
      <td>Not Verified</td>
      <td>Jan-2015</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Debt consolidation</td>
      <td>22.05</td>
      <td>Jul-2004</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>20131.0</td>
      <td>53.3</td>
      <td>27.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1076 Carney Fort Apt. 347\r\nLoganmouth, SD 05113</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15600.0</td>
      <td>36 months</td>
      <td>10.49</td>
      <td>506.97</td>
      <td>B</td>
      <td>B3</td>
      <td>Statistician</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>43057.0</td>
      <td>Source Verified</td>
      <td>Jan-2015</td>
      <td>Fully Paid</td>
      <td>credit_card</td>
      <td>Credit card refinancing</td>
      <td>12.79</td>
      <td>Aug-2007</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>11987.0</td>
      <td>92.2</td>
      <td>26.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>87025 Mark Dale Apt. 269\r\nNew Sabrina, WV 05113</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7200.0</td>
      <td>36 months</td>
      <td>6.49</td>
      <td>220.65</td>
      <td>A</td>
      <td>A2</td>
      <td>Client Advocate</td>
      <td>6 years</td>
      <td>RENT</td>
      <td>54000.0</td>
      <td>Not Verified</td>
      <td>Nov-2014</td>
      <td>Fully Paid</td>
      <td>credit_card</td>
      <td>Credit card refinancing</td>
      <td>2.60</td>
      <td>Sep-2006</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>5472.0</td>
      <td>21.5</td>
      <td>13.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>823 Reid Ford\r\nDelacruzside, MA 00813</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24375.0</td>
      <td>60 months</td>
      <td>17.27</td>
      <td>609.33</td>
      <td>C</td>
      <td>C5</td>
      <td>Destiny Management Inc.</td>
      <td>9 years</td>
      <td>MORTGAGE</td>
      <td>55000.0</td>
      <td>Verified</td>
      <td>Apr-2013</td>
      <td>Charged Off</td>
      <td>credit_card</td>
      <td>Credit Card Refinance</td>
      <td>33.95</td>
      <td>Mar-1999</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>24584.0</td>
      <td>69.8</td>
      <td>43.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>679 Luna Roads\r\nGreggshire, VA 11650</td>
    </tr>
  </tbody>
</table>
</div>



# Project Tasks

# Section 1: Exploratory Data Analysis

**TUJUAN UTAMA: Memahami variabel-variabel apa saja yang penting untuk membangun model prediksi, melihat *summary statistics*, dan visualisasi data.**

**TASK 1: Variabel `loan_status` akan dibuatkan model prediktif. Oleh karena itu, pertama-tama akan dibuatkan countplot berupa barchart untuk mengetahui jumlah customer di masing-masing status**


```python
sns.countplot(x='loan_status',data=df)
plt.title('Jumlah customer di masing-masing status pinjaman')
plt.xlabel('Loan Status');
```


![png](/images/lending/03_Keras_Project_Exercise_23_0.png)



```python
df['loan_status'].value_counts()
```




    Fully Paid     318357
    Charged Off     77673
    Name: loan_status, dtype: int64



Dapat diketahui bahwa customer dengan status Fully Paid lebih banyak ketimbang status Charged-Off. Fully paid memiliki jumlah customer lebih dari 300000 orang, sedangkan Charged-Off ada di angka kurang dari 8 ribuan customer.

**TASK: Mengeksplor variabel `loan_amnt`**


```python
df['loan_amnt'].describe()
```




    count    396030.000000
    mean      14113.888089
    std        8357.441341
    min         500.000000
    25%        8000.000000
    50%       12000.000000
    75%       20000.000000
    max       40000.000000
    Name: loan_amnt, dtype: float64




```python
plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],kde=False, bins=50)
plt.title('Jumlah customer untuk beberapa jumlah pinjaman')
plt.xlabel('Loan Amount')
plt.ylabel('count');

```


![png](/images/lending/03_Keras_Project_Exercise_28_0.png)


Terlihat bahwa pinjaman dengan amount 10000 dolar lebih banyak daripada jumlah pinjaman lainnya, dengan total customer yang meminjam nominal tersebut berada di sekitar 35000 orang.

Juga terdapat nominal nominal lain yang lebih sering dipinjam oleh customer. Ada sekitar 25000 orang meminjam sekitar 12 ribuan atau 14 ribuan. Kemudian ada nominal 20000 yang dipinjam oleh sekitar 20 ribu orang. Serta yang nominal terbesar yang sering dipinjam adalah  di nominal 34 ribuan.

Jika dari histogram tersebut dibagi menjadi 4 buah segmen berdasarkan jumlah customer yang meminjam dan jumlah nominal pinjaman, akan dihasilkan plot histogram berikut:


```python
plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],kde=False, bins=50)
plt.axvline(df['loan_amnt'].describe()['25%'], label='Q1',color='green')
plt.axvline(df['loan_amnt'].describe()['50%'], label='Q2',color='coral')
plt.axvline(df['loan_amnt'].describe()['75%'], label='Q3')
plt.title('Jumlah customer untuk beberapa jumlah pinjaman')
plt.xlabel('Loan Amount')
plt.ylabel('count');
plt.legend();
```


![png](/images/lending/03_Keras_Project_Exercise_30_0.png)


Terlihat bahwa:
* Kelompok pertama dengan range peminjaman paling rendah (\$500 - \$8.000), didominasi oleh peminjam yang meminjam dengan nominal sekitar $5.000.
* Kelompok kedua dengan range di atas \$8.000 sampai dengan maksimal \$12.000, berisikan nominal paling sering dipinjam dari keseluruhan kelompok, yaitu dengan nominal \$10.000.

* Kelompok ketiga dengan range di atas \$12.000 sampai maksimal \$20.000, peminjam lebih condong ke angka pinjaman \$15.000.
* Kelompok terakhir dengan range nominal tertinggi di atas \$20.000 sampai maksimal /$40.000, nampaknya peminjam di sekitar \$35.000 lebih mendominasi.



**TASK: Sekarang, saatnya mengeksplor korelasi antar variabel kontinu pada data tersebut**


```python
df.corr()
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
      <th>loan_amnt</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>mort_acc</th>
      <th>pub_rec_bankruptcies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loan_amnt</th>
      <td>1.000000</td>
      <td>0.168921</td>
      <td>0.953929</td>
      <td>0.336887</td>
      <td>0.016636</td>
      <td>0.198556</td>
      <td>-0.077779</td>
      <td>0.328320</td>
      <td>0.099911</td>
      <td>0.223886</td>
      <td>0.222315</td>
      <td>-0.106539</td>
    </tr>
    <tr>
      <th>int_rate</th>
      <td>0.168921</td>
      <td>1.000000</td>
      <td>0.162758</td>
      <td>-0.056771</td>
      <td>0.079038</td>
      <td>0.011649</td>
      <td>0.060986</td>
      <td>-0.011280</td>
      <td>0.293659</td>
      <td>-0.036404</td>
      <td>-0.082583</td>
      <td>0.057450</td>
    </tr>
    <tr>
      <th>installment</th>
      <td>0.953929</td>
      <td>0.162758</td>
      <td>1.000000</td>
      <td>0.330381</td>
      <td>0.015786</td>
      <td>0.188973</td>
      <td>-0.067892</td>
      <td>0.316455</td>
      <td>0.123915</td>
      <td>0.202430</td>
      <td>0.193694</td>
      <td>-0.098628</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>0.336887</td>
      <td>-0.056771</td>
      <td>0.330381</td>
      <td>1.000000</td>
      <td>-0.081685</td>
      <td>0.136150</td>
      <td>-0.013720</td>
      <td>0.299773</td>
      <td>0.027871</td>
      <td>0.193023</td>
      <td>0.236320</td>
      <td>-0.050162</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>0.016636</td>
      <td>0.079038</td>
      <td>0.015786</td>
      <td>-0.081685</td>
      <td>1.000000</td>
      <td>0.136181</td>
      <td>-0.017639</td>
      <td>0.063571</td>
      <td>0.088375</td>
      <td>0.102128</td>
      <td>-0.025439</td>
      <td>-0.014558</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>0.198556</td>
      <td>0.011649</td>
      <td>0.188973</td>
      <td>0.136150</td>
      <td>0.136181</td>
      <td>1.000000</td>
      <td>-0.018392</td>
      <td>0.221192</td>
      <td>-0.131420</td>
      <td>0.680728</td>
      <td>0.109205</td>
      <td>-0.027732</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>-0.077779</td>
      <td>0.060986</td>
      <td>-0.067892</td>
      <td>-0.013720</td>
      <td>-0.017639</td>
      <td>-0.018392</td>
      <td>1.000000</td>
      <td>-0.101664</td>
      <td>-0.075910</td>
      <td>0.019723</td>
      <td>0.011552</td>
      <td>0.699408</td>
    </tr>
    <tr>
      <th>revol_bal</th>
      <td>0.328320</td>
      <td>-0.011280</td>
      <td>0.316455</td>
      <td>0.299773</td>
      <td>0.063571</td>
      <td>0.221192</td>
      <td>-0.101664</td>
      <td>1.000000</td>
      <td>0.226346</td>
      <td>0.191616</td>
      <td>0.194925</td>
      <td>-0.124532</td>
    </tr>
    <tr>
      <th>revol_util</th>
      <td>0.099911</td>
      <td>0.293659</td>
      <td>0.123915</td>
      <td>0.027871</td>
      <td>0.088375</td>
      <td>-0.131420</td>
      <td>-0.075910</td>
      <td>0.226346</td>
      <td>1.000000</td>
      <td>-0.104273</td>
      <td>0.007514</td>
      <td>-0.086751</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>0.223886</td>
      <td>-0.036404</td>
      <td>0.202430</td>
      <td>0.193023</td>
      <td>0.102128</td>
      <td>0.680728</td>
      <td>0.019723</td>
      <td>0.191616</td>
      <td>-0.104273</td>
      <td>1.000000</td>
      <td>0.381072</td>
      <td>0.042035</td>
    </tr>
    <tr>
      <th>mort_acc</th>
      <td>0.222315</td>
      <td>-0.082583</td>
      <td>0.193694</td>
      <td>0.236320</td>
      <td>-0.025439</td>
      <td>0.109205</td>
      <td>0.011552</td>
      <td>0.194925</td>
      <td>0.007514</td>
      <td>0.381072</td>
      <td>1.000000</td>
      <td>0.027239</td>
    </tr>
    <tr>
      <th>pub_rec_bankruptcies</th>
      <td>-0.106539</td>
      <td>0.057450</td>
      <td>-0.098628</td>
      <td>-0.050162</td>
      <td>-0.014558</td>
      <td>-0.027732</td>
      <td>0.699408</td>
      <td>-0.124532</td>
      <td>-0.086751</td>
      <td>0.042035</td>
      <td>0.027239</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




**TASK: Agar lebih mudah dilihat, dibuatlah heatmap untuk memvisualisasikan tabel korelasi di atas**



```python
matrix = np.triu(df.corr())
plt.figure(figsize=(12,7))
sns.heatmap(df.corr(), annot=True, cmap='viridis', mask=matrix)
plt.title('Matriks korelasi untuk variabel-variabel numerik');
```


![png](/images/lending/03_Keras_Project_Exercise_35_0.png)


Dari heatmap korelasi tersebut, variabel yang paling dominan yang mempengaruhi `loan_amnt` adalah `installment`. Keduanya merberkorelasi dengan arah positif. Artinya bahwa peningkatan jumlah `loan_amnt` berbanding lurus dengan peningkatan cicilan bulanan (`installment`) atau sebaliknya.


**TASK: Karena loan_amnt dan installment berkorelasi kuat, maka kedua variabel ini akan dieksplor lebih jauh. Buat scatterplot dan deskripsi dari masing-masing variabel. Apakah kedua hubungan kedua variabel ini masuk akal?**

Deskripsi antar variabel:


```python
feat_info('installment')
```

    The monthly payment owed by the borrower if the loan originates.



```python
feat_info('loan_amnt')
```

    The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.


Scatterplot hubungan kedua variabel:


```python
sns.scatterplot(x='installment', y='loan_amnt', data=df)
plt.title('Hubungan cicilan (installment) terhadap jumlah loan amount')
plt.xlabel('Installment')
plt.ylabel('Loan Amount');
```


![png](/images/lending/03_Keras_Project_Exercise_42_0.png)


Terlihat, bahwa semakin besar `installment`, maka akan semakin besar total pinjaman pada variable `loan_amnt`.

Hal ini jelas akan terjadi. Sebagai contoh apabila jumlah pinjaman berada di nominal 40000 dollar, maka jumlah cicilan perbulan ada di atas 1400 dollar perbulannya. Hal ini juga berlaku sebaliknya, semakin kecil jumlah pinjaman, maka akan semakin kecil pula nomilan cicilan perbulannya.

Namun di beberapa kasus, nominal installment yang cenderung rendah juga dipakai untuk menyicil pinjaman yang cukup besar. Di mana hal tersebut berbeda dengan dominasi sebaran data point-nya. Sebagai contoh, cicilan bulanan/`installment` dengan nominal antara \$ 16,08 sampai maksimal \$ 200.0 memiliki rata-rata pinjaman di angka 4 ribuan dollar. Tetapi masih ada peminjam yang meminjam sampai dengan maksimal \$ 25.000 untuk range nominal cicilan tersebut.

Deskripsi statistik:


```python
df_temp = df[['installment', 'loan_amnt']].sort_values(by='installment')
```


```python
df_temp[df_temp['installment'] <= 200].describe().transpose()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>installment</th>
      <td>67069.0</td>
      <td>136.410867</td>
      <td>45.415306</td>
      <td>16.08</td>
      <td>101.62</td>
      <td>146.86</td>
      <td>173.31</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>loan_amnt</th>
      <td>67069.0</td>
      <td>4166.061071</td>
      <td>1511.218999</td>
      <td>500.00</td>
      <td>3000.00</td>
      <td>4450.00</td>
      <td>5000.00</td>
      <td>25000.0</td>
    </tr>
  </tbody>
</table>
</div>




**TASK: Membuat boxplot yang memvisualisasikan relasi variabel loan_status dan loan_amnt. (Variabel kategorikal dan variabel kontinu)**


```python
sns.boxplot(x='loan_status', y='loan_amnt', data=df)
plt.title('Relasi Loan Status terhadap Loan Amount');
plt.xlabel('Loan Status')
plt.ylabel('Loan Amount');
```


![png](/images/lending/03_Keras_Project_Exercise_47_0.png)


Nampak dari kedua boxplot tersebut per status loan mereka, sepertinya 50% dari jumlah customer dari kedua status tersebut hampir mirip. Tetapi, customer dengan status `Charged Off` sedikit lebih tinggi dibanding `Fully Paid`. Hal ini menunjukkan bahwa semakin besar jumlah pinjaman, maka pembayarannya juga menjadi semakin berat hingga bersatus `Charged Off`.

**TASK: Tabel summary statistics dari boxplot di atas.**


```python
df.groupby('loan_status')['loan_amnt'].describe()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>77673.0</td>
      <td>15126.300967</td>
      <td>8505.090557</td>
      <td>1000.0</td>
      <td>8525.0</td>
      <td>14000.0</td>
      <td>20000.0</td>
      <td>40000.0</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>318357.0</td>
      <td>13866.878771</td>
      <td>8302.319699</td>
      <td>500.0</td>
      <td>7500.0</td>
      <td>12000.0</td>
      <td>19225.0</td>
      <td>40000.0</td>
    </tr>
  </tbody>
</table>
</div>



Dari tabel di atas, dapat diketahui bahwa rata-rata nominal pinjaman `Charged Off` sedikit lebih tinggi daripada status `Fully Paid`. Ini juga dapat dilihat secara visualisasi pada boxplot sebelumnya.


**TASK: Eksplorasi variabel Grade dan SubGrade. Grade dan Subgrade apa saja yang ada pada data LandingClub ini?**

Deskripsi dan entri unik dari `grade`:


```python
feat_info('grade')
```

    LC assigned loan grade



```python
sorted(df['grade'].unique())
```




    ['A', 'B', 'C', 'D', 'E', 'F', 'G']



Deskripsi dan entri unik dari `sub_grade`:


```python
feat_info('sub_grade')
```

    LC assigned loan subgrade



```python
df['sub_grade'].unique()
```




    array(['B4', 'B5', 'B3', 'A2', 'C5', 'C3', 'A1', 'B2', 'C1', 'A5', 'E4',
           'A4', 'A3', 'D1', 'C2', 'B1', 'D3', 'D5', 'D2', 'E1', 'E2', 'E5',
           'F4', 'E3', 'D4', 'G1', 'F5', 'G2', 'C4', 'F1', 'F3', 'G5', 'G4',
           'F2', 'G3'], dtype=object)




**TASK: Melihat jumlah customer pada masing-masing grade dengan acuan loan_status dan performansinya.**

Maksud performansi: seberapa besar kesamaan jumlah customer yang gagal bayar dan yang tidak. Jika memiliki kesamaan jumlah customer, maka semakin buruk pula performansinya.


```python
grade_order = sorted(df['grade'].unique())
sns.countplot(x='grade',data=df, hue='loan_status', order=grade_order)
plt.title('Jumlah customer perGrade')
plt.xlabel('Grade');
```


![png](/images/lending/03_Keras_Project_Exercise_60_0.png)


Untuk melihat performansi masing masing grade, akan dilihat dari rasio antara jumlah customer yang berstatus `Charged Off` terhadap `Fully Paid ` perGradenya. Semakin kecil, semakin bagus:


```python
srt_ind_gr = sorted(df[df['loan_status']=='Charged Off']['grade'].value_counts().index)

grade_fully_paid = df[df['loan_status']=='Fully Paid']['grade'].value_counts()
grade_charged_off = df[df['loan_status']=='Charged Off']['grade'].value_counts()

grade_ratio = pd.DataFrame(index=srt_ind_gr, data=grade_charged_off/grade_fully_paid)
grade_ratio.columns=['ratio']

#plot
grade_ratio.plot(kind='bar')
plt.title('Rasio Charged Off terhadap Fully Paid perGrade')
plt.ylabel('Ratio')
plt.xlabel('Grade')
plt.xticks(rotation=360)
plt.legend().set_visible(False);
```


![png](/images/lending/03_Keras_Project_Exercise_62_0.png)



```python
print('Ratio A = ',round(grade_ratio.loc['A'].iloc[0],3))
```

    Ratio A =  0.067


Nampaknya grade A menempati urutan pertama terbaik untuk sektor performansi. Hal ini ditunjukkan dengan rasio paling kecil, yaitu sebesar 0.067.


**TASK: Sekarang, saatnya mengeksplorasi subgrade. Akan ditampilkan jumlah customer di masing-masing subgrade di semua loan_status, serta akan ditampilkan visualisasinya jika mengaju pada jenis-jenis loan_status.
Di visualisasi tersebut juga ditampilkan color palette yang menunjukkan performansi masing-masing subgrade.**


```python
plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade', data=df, order=subgrade_order, palette='coolwarm');

```


![png](/images/lending/03_Keras_Project_Exercise_66_0.png)


Secara detail persubgrade, semakin biru warna yang diberikan, maka semakin baik performanya. Hal tersebut mengacu pada rasio `Charged Off` terhadap `Fully Paid`. Jika didetailkan lagi dengan jenis-jenis loan_status:


```python
plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade', data=df, order=subgrade_order, palette='coolwarm', hue='loan_status');
```


![png](/images/lending/03_Keras_Project_Exercise_68_0.png)


Lebih detail, dapat dilihat bahwa performa dari subgroup dapat ditentukan dari besaran nilai rasio `Charged Off` terhadap `Fully Paid`. Semakin kecil rasionya, maka akan semakin baik performansinya. Hal ini juga berlaku sebaliknya.


**TASK: Tampaknya subgrade pada grade F dan G performansinya lebih buruk daripada pada grade lainnya.**


```python
f_and_g = df[(df['grade']== 'G') | (df['grade']=='F')]
plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade', data=df, order=subgrade_order, palette='coolwarm', hue='loan_status');
```


![png](/images/lending/03_Keras_Project_Exercise_71_0.png)


Sangat jelas terlihat bahwa pada subgroup-subgroup tersebut memiliki tingkat jumlah customer gagal bayar dan yang tidak memiki kesamaan yang tinggi dengan rasio kesamaan yang tinggi.



**TASK: Dalam pemrosesan yang lebih lanjut, akan dibuat kolom baru `loan_repaid` yang mengubah `loan_status` : "Fully Paid" menjadi 1 dan sisanya 0.**


```python
df['loan_repaid']=df['loan_status'].map({'Fully Paid' : 1, 'Charged Off': 0})
```


```python
df[['loan_repaid', 'loan_status']].head(10)
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
      <th>loan_repaid</th>
      <th>loan_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Charged Off</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>Fully Paid</td>
    </tr>
  </tbody>
</table>
</div>





**TASK: Sekarang, saatnya melihat korelasi loan_repaid dengan variabel-variabel numerik lainnya**


```python
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar');
```


![png](/images/lending/03_Keras_Project_Exercise_77_0.png)


Terlihat bahwa `int_rate` memiliki nilai korelasi tertinggi dengan arah negatif. Menunjukkan bahwa semakin kecil `int_rate`, maka akan semakin tinggi kemungkinan peminjaman dilunasi. Berlaku pula sebaliknya.


# Section 2: Data PreProcessing


**Section Goals: Remove atau fill data yang hilang, remove fitur/variabel yang tidak penting atau yang terduplikasi. Convert variabel kategorik ke dalam dummy variable agar diolah secara kuantitatif.**





```python
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
      <th>loan_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>purpose</th>
      <th>title</th>
      <th>dti</th>
      <th>earliest_cr_line</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>initial_list_status</th>
      <th>application_type</th>
      <th>mort_acc</th>
      <th>pub_rec_bankruptcies</th>
      <th>address</th>
      <th>loan_repaid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10000.0</td>
      <td>36 months</td>
      <td>11.44</td>
      <td>329.48</td>
      <td>B</td>
      <td>B4</td>
      <td>Marketing</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>117000.0</td>
      <td>Not Verified</td>
      <td>Jan-2015</td>
      <td>Fully Paid</td>
      <td>vacation</td>
      <td>Vacation</td>
      <td>26.24</td>
      <td>Jun-1990</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>36369.0</td>
      <td>41.8</td>
      <td>25.0</td>
      <td>w</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0174 Michelle Gateway\r\nMendozaberg, OK 22690</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8000.0</td>
      <td>36 months</td>
      <td>11.99</td>
      <td>265.68</td>
      <td>B</td>
      <td>B5</td>
      <td>Credit analyst</td>
      <td>4 years</td>
      <td>MORTGAGE</td>
      <td>65000.0</td>
      <td>Not Verified</td>
      <td>Jan-2015</td>
      <td>Fully Paid</td>
      <td>debt_consolidation</td>
      <td>Debt consolidation</td>
      <td>22.05</td>
      <td>Jul-2004</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>20131.0</td>
      <td>53.3</td>
      <td>27.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1076 Carney Fort Apt. 347\r\nLoganmouth, SD 05113</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15600.0</td>
      <td>36 months</td>
      <td>10.49</td>
      <td>506.97</td>
      <td>B</td>
      <td>B3</td>
      <td>Statistician</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>43057.0</td>
      <td>Source Verified</td>
      <td>Jan-2015</td>
      <td>Fully Paid</td>
      <td>credit_card</td>
      <td>Credit card refinancing</td>
      <td>12.79</td>
      <td>Aug-2007</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>11987.0</td>
      <td>92.2</td>
      <td>26.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>87025 Mark Dale Apt. 269\r\nNew Sabrina, WV 05113</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7200.0</td>
      <td>36 months</td>
      <td>6.49</td>
      <td>220.65</td>
      <td>A</td>
      <td>A2</td>
      <td>Client Advocate</td>
      <td>6 years</td>
      <td>RENT</td>
      <td>54000.0</td>
      <td>Not Verified</td>
      <td>Nov-2014</td>
      <td>Fully Paid</td>
      <td>credit_card</td>
      <td>Credit card refinancing</td>
      <td>2.60</td>
      <td>Sep-2006</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>5472.0</td>
      <td>21.5</td>
      <td>13.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>823 Reid Ford\r\nDelacruzside, MA 00813</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24375.0</td>
      <td>60 months</td>
      <td>17.27</td>
      <td>609.33</td>
      <td>C</td>
      <td>C5</td>
      <td>Destiny Management Inc.</td>
      <td>9 years</td>
      <td>MORTGAGE</td>
      <td>55000.0</td>
      <td>Verified</td>
      <td>Apr-2013</td>
      <td>Charged Off</td>
      <td>credit_card</td>
      <td>Credit Card Refinance</td>
      <td>33.95</td>
      <td>Mar-1999</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>24584.0</td>
      <td>69.8</td>
      <td>43.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>679 Luna Roads\r\nGreggshire, VA 11650</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# Missing Data

**Sekarang saatnya mengeksplor mising data di semua kolom. Juga, akan mengevaluasi beberapa kolom yang nantinya akang berpengaruh lebih dalam pembangunan model, entah dengan membiarkannya atau mengisi missing data tersebut**


```python
len(df)
```




    396030




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 396030 entries, 0 to 396029
    Data columns (total 28 columns):
     #   Column                Non-Null Count   Dtype  
    ---  ------                --------------   -----  
     0   loan_amnt             396030 non-null  float64
     1   term                  396030 non-null  object
     2   int_rate              396030 non-null  float64
     3   installment           396030 non-null  float64
     4   grade                 396030 non-null  object
     5   sub_grade             396030 non-null  object
     6   emp_title             373103 non-null  object
     7   emp_length            377729 non-null  object
     8   home_ownership        396030 non-null  object
     9   annual_inc            396030 non-null  float64
     10  verification_status   396030 non-null  object
     11  issue_d               396030 non-null  object
     12  loan_status           396030 non-null  object
     13  purpose               396030 non-null  object
     14  title                 394275 non-null  object
     15  dti                   396030 non-null  float64
     16  earliest_cr_line      396030 non-null  object
     17  open_acc              396030 non-null  float64
     18  pub_rec               396030 non-null  float64
     19  revol_bal             396030 non-null  float64
     20  revol_util            395754 non-null  float64
     21  total_acc             396030 non-null  float64
     22  initial_list_status   396030 non-null  object
     23  application_type      396030 non-null  object
     24  mort_acc              358235 non-null  float64
     25  pub_rec_bankruptcies  395495 non-null  float64
     26  address               396030 non-null  object
     27  loan_repaid           396030 non-null  int64  
    dtypes: float64(12), int64(1), object(15)
    memory usage: 84.6+ MB


**TASK: Berapa jumlah missing data di setiap kolom?**


```python
df.isnull().sum().sort_values(ascending=False)
```




    mort_acc                37795
    emp_title               22927
    emp_length              18301
    title                    1755
    pub_rec_bankruptcies      535
    revol_util                276
    loan_repaid                 0
    issue_d                     0
    term                        0
    int_rate                    0
    installment                 0
    grade                       0
    sub_grade                   0
    home_ownership              0
    annual_inc                  0
    verification_status         0
    purpose                     0
    loan_status                 0
    address                     0
    dti                         0
    earliest_cr_line            0
    open_acc                    0
    pub_rec                     0
    revol_bal                   0
    total_acc                   0
    initial_list_status         0
    application_type            0
    loan_amnt                   0
    dtype: int64





**TASK: Seberapa besar persentase missing data perkolom tersebut terhadap total jumlah data keseluruhan?**


```python
df.isnull().sum().sort_values(ascending=False)/len(df) * 100
```




    mort_acc                9.543469
    emp_title               5.789208
    emp_length              4.621115
    title                   0.443148
    pub_rec_bankruptcies    0.135091
    revol_util              0.069692
    loan_repaid             0.000000
    issue_d                 0.000000
    term                    0.000000
    int_rate                0.000000
    installment             0.000000
    grade                   0.000000
    sub_grade               0.000000
    home_ownership          0.000000
    annual_inc              0.000000
    verification_status     0.000000
    purpose                 0.000000
    loan_status             0.000000
    address                 0.000000
    dti                     0.000000
    earliest_cr_line        0.000000
    open_acc                0.000000
    pub_rec                 0.000000
    revol_bal               0.000000
    total_acc               0.000000
    initial_list_status     0.000000
    application_type        0.000000
    loan_amnt               0.000000
    dtype: float64



Jumlah missing data di kolom `mort_acc`, `emp_title`, `emp_length` masing-masing adalah 9.5%, 5.8%, 4.6% dari keseluruhan jumlah entri data. Sisanya, jumlah missing datanya di bawah 1% dari keseluruhan.



**TASK: Analisis variabel emp_title dan emp_length apakah dapat didrop atau tidak.**


```python
feat_info('emp_title')
```

    The job title supplied by the Borrower when applying for the loan.*



```python
feat_info('emp_length')
```

    Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.


**TASK: Ada berapa pekerjaan di kolom emp_title?**


```python
df['emp_title'].nunique()
```




    173105



**Melihat kemungkinan `emp_title` dapat dibuat menjadi `dummy variable` atau dapat dibuang ketika terdapat null values**


```python
df['emp_title'].value_counts()
```




    Teacher                     4389
    Manager                     4250
    Registered Nurse            1856
    RN                          1846
    Supervisor                  1830
                                ...
    LOWES HOME IMPROVEMENTS        1
    Paragon Software Systems       1
    Sr. Manager, Marketing         1
    Abraham House                  1
    cnc                            1
    Name: emp_title, Length: 173105, dtype: int64



Melihat jumlah dari unique values yang begitu besar, maka sangat tidak direkomendasikan untuk dijadikan dummy variable

**TASK: Karena terdapat banyak sekali jenis pekerjaan yang ada, maka variabel tersebut tidak dapat dijadikan dummy variable. Oleh karena itu, variabel ini akan didrop**


```python
df = df.drop('emp_title', axis=1)
```

**TASK: Buat countplot untuk variabel `emp_length`!**


```python
sorted(df['emp_length'].dropna().unique())
```




    ['1 year',
     '10+ years',
     '2 years',
     '3 years',
     '4 years',
     '5 years',
     '6 years',
     '7 years',
     '8 years',
     '9 years',
     '< 1 year']




```python
emp_length_order = ['< 1 year',
 '1 year',
 '2 years',
 '3 years',
 '4 years',
 '5 years',
 '6 years',
 '7 years',
 '8 years',
 '9 years',
 '10+ years'
 ]
```


```python
plt.figure(figsize=(12,4))
sns.countplot(x='emp_length', data=df, order=emp_length_order);
```


![png](/images/lending/03_Keras_Project_Exercise_102_0.png)


Dari barplot tersebut, dapat diketahui sebagian besar peminjam memiliki durasi bekerja selama 10 tahun lebih.

**TASK: Countplot yang memisahkan Fully Paid dan Charged Off**


```python
plt.figure(figsize=(12,4))
sns.countplot(x='emp_length', data=df, order=emp_length_order, hue='loan_status');
```


![png](/images/lending/03_Keras_Project_Exercise_105_0.png)


**Berapa probabilitas kemungkinan peminjam dengan beberapa durasi kerja di atas dapat berstatus charged-off?**


```python
emp_co = df[df['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']
emp_co
```




    emp_length
    1 year        5154
    10+ years    23215
    2 years       6924
    3 years       6182
    4 years       4608
    5 years       5092
    6 years       3943
    7 years       4055
    8 years       3829
    9 years       3070
    < 1 year      6563
    Name: loan_status, dtype: int64




```python
emp_fp = df[df['loan_status']=='Fully Paid'].groupby('emp_length').count()['loan_status']
emp_fp
```




    emp_length
    1 year        20728
    10+ years    102826
    2 years       28903
    3 years       25483
    4 years       19344
    5 years       21403
    6 years       16898
    7 years       16764
    8 years       15339
    9 years       12244
    < 1 year      25162
    Name: loan_status, dtype: int64




```python
print('Probabilitas charged off = ',emp_co/(emp_fp+emp_co))
```

    Probabilitas charged off =  emp_length
    1 year       0.199135
    10+ years    0.184186
    2 years      0.193262
    3 years      0.195231
    4 years      0.192385
    5 years      0.192187
    6 years      0.189194
    7 years      0.194774
    8 years      0.199760
    9 years      0.200470
    < 1 year     0.206872
    Name: loan_status, dtype: float64



```python
emp_len = emp_co/(emp_fp+emp_co)
emp_len.plot(kind= 'bar');
```


![png](/images/lending/03_Keras_Project_Exercise_110_0.png)


Melihat dari data yang dimiliki dan dari barplot di atas, peminjam memiliki probabilitas sekitar 20% dari mereka berstatus `Charged Off` dikarenakan mereka tidak melunasi pinjaman mereka.

Karena hampir tidak ada perbedaan probabilitas Charge Off di antara variable ini, maka fitur ini akan didrop.


```python
df = df.drop('emp_length', axis=1)
```

**TASK: Lihat kembali jumlah missing data**


```python
df.isnull().sum().sort_values(ascending=False)
```




    mort_acc                37795
    title                    1755
    pub_rec_bankruptcies      535
    revol_util                276
    loan_status                 0
    term                        0
    int_rate                    0
    installment                 0
    grade                       0
    sub_grade                   0
    home_ownership              0
    annual_inc                  0
    verification_status         0
    issue_d                     0
    loan_repaid                 0
    purpose                     0
    address                     0
    dti                         0
    earliest_cr_line            0
    open_acc                    0
    pub_rec                     0
    revol_bal                   0
    total_acc                   0
    initial_list_status         0
    application_type            0
    loan_amnt                   0
    dtype: int64



Masih terdapat missing values

**TASK: Apakah kolom purpose dan title merupakan kolom yang terduplikasi?**


```python
feat_info('purpose')
```

    A category provided by the borrower for the loan request.



```python
feat_info('title')
```

    The loan title provided by the borrower



```python
df['purpose'].value_counts()
```




    debt_consolidation    234507
    credit_card            83019
    home_improvement       24030
    other                  21185
    major_purchase          8790
    small_business          5701
    car                     4697
    medical                 4196
    moving                  2854
    vacation                2452
    house                   2201
    wedding                 1812
    renewable_energy         329
    educational              257
    Name: purpose, dtype: int64




```python
df['title'].value_counts()
```




    Debt consolidation                     152472
    Credit card refinancing                 51487
    Home improvement                        15264
    Other                                   12930
    Debt Consolidation                      11608
                                            ...  
    Credit Card Escape Loan                     1
    Daughter's home burned = total loss         1
    Chase                                       1
    livewire                                    1
    Get better loan                             1
    Name: title, Length: 48817, dtype: int64





**TASK: Nampaknya kolom title hanya sebuah deskripsi/subkategori dari kolom purpose. Drop kolom title:**


```python
df=df.drop('title', axis=1)
```


**TASK: Apa yang direpresentasikan oleh variabel mort_acc?**


```python
feat_info('mort_acc')
```

    Number of mortgage accounts.



```python
df['mort_acc'].value_counts()
```




    0.0     139777
    1.0      60416
    2.0      49948
    3.0      38049
    4.0      27887
    5.0      18194
    6.0      11069
    7.0       6052
    8.0       3121
    9.0       1656
    10.0       865
    11.0       479
    12.0       264
    13.0       146
    14.0       107
    15.0        61
    16.0        37
    17.0        22
    18.0        18
    19.0        15
    20.0        13
    24.0        10
    22.0         7
    21.0         4
    25.0         4
    27.0         3
    23.0         2
    32.0         2
    26.0         2
    31.0         2
    30.0         1
    28.0         1
    34.0         1
    Name: mort_acc, dtype: int64



Karena persentasi missing valuenya paling banyak, maka fitur ini tidak dapat dihapus begitu saja.


**TASK: Ada beberapa cara untuk mengisi missing data. Kita dapat menggunakan model linear sederhana, atau mengisinya dengan mean variabel. Karena mort_acc merupakan variabel yang bernilai numerik, maka akan lebih baik jika melihat korelasi dengan variabel numerik lain apakah memiliki hubungan yang signifikan atau tidak**


```python
df.corr()['mort_acc'].sort_values()
```




    int_rate               -0.082583
    dti                    -0.025439
    revol_util              0.007514
    pub_rec                 0.011552
    pub_rec_bankruptcies    0.027239
    loan_repaid             0.073111
    open_acc                0.109205
    installment             0.193694
    revol_bal               0.194925
    loan_amnt               0.222315
    annual_inc              0.236320
    total_acc               0.381072
    mort_acc                1.000000
    Name: mort_acc, dtype: float64




**TASK: Tampaknya total_acc berkorelasi paling tinggi kepada mort_acc. Akan dibuat pengelompokkan berdasarkan total_acc dan hitung mean untuk mort_acc per nilai total_acc.**


```python
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
total_acc_avg
```




    total_acc
    2.0      0.000000
    3.0      0.052023
    4.0      0.066743
    5.0      0.103289
    6.0      0.151293
               ...   
    124.0    1.000000
    129.0    1.000000
    135.0    3.000000
    150.0    2.000000
    151.0    0.000000
    Name: mort_acc, Length: 118, dtype: float64




```python
total_acc_avg[3.0]
```




    0.05202312138728324





**TASK: Isi missing data pada mort_acc berdasarkan nilai pada total_acc. Jika ada missing value pada mort_acc, maka akan diisi dengan mean yang berhubungan dengan total_acc dari series di atas.**

[Hint!](https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe)


```python
def fill_mort_acc(total_acc, mort_acc):
  if np.isnan(mort_acc):
    return total_acc_avg[total_acc]
  else:
    return mort_acc
```


```python
df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
```

Cek missing value:


```python
df['mort_acc'].isnull().sum()
```




    0




```python
df.isnull().sum()
```




    loan_amnt                 0
    term                      0
    int_rate                  0
    installment               0
    grade                     0
    sub_grade                 0
    home_ownership            0
    annual_inc                0
    verification_status       0
    issue_d                   0
    loan_status               0
    purpose                   0
    dti                       0
    earliest_cr_line          0
    open_acc                  0
    pub_rec                   0
    revol_bal                 0
    revol_util              276
    total_acc                 0
    initial_list_status       0
    application_type          0
    mort_acc                  0
    pub_rec_bankruptcies    535
    address                   0
    loan_repaid               0
    dtype: int64




```python
(df.isnull().sum().sort_values(ascending=False)/len(df) * 100)[:2]
```




    pub_rec_bankruptcies    0.135091
    revol_util              0.069692
    dtype: float64




**TASK: tampaknya masih ada missing value pada revol_util dan pub_rec_bankruptcies. Namun jumlah persentasinya kurang dari 0.5% dari keseluruhan data. Drop missing value:**


```python
df = df.dropna()
```


```python
df.isnull().sum()
```




    loan_amnt               0
    term                    0
    int_rate                0
    installment             0
    grade                   0
    sub_grade               0
    home_ownership          0
    annual_inc              0
    verification_status     0
    issue_d                 0
    loan_status             0
    purpose                 0
    dti                     0
    earliest_cr_line        0
    open_acc                0
    pub_rec                 0
    revol_bal               0
    revol_util              0
    total_acc               0
    initial_list_status     0
    application_type        0
    mort_acc                0
    pub_rec_bankruptcies    0
    address                 0
    loan_repaid             0
    dtype: int64



Tidak terdapat missing value lagi.

## Categorical Variables and Dummy Variables

**Sekarang saatnya berurusan dengan categorical variables dan mengubahnya ke dalam dummy variable untuk menkonversinya menjadi variabel kuantitatif.**

**TASK: Kolom apa saja yang bersifat kualitatif?**


```python
df.select_dtypes(['object']).columns
```




    Index(['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status',
           'issue_d', 'loan_status', 'purpose', 'earliest_cr_line',
           'initial_list_status', 'application_type', 'address'],
          dtype='object')






### term




```python
feat_info('term')
```

    The number of payments on the loan. Values are in months and can be either 36 or 60.



```python
df['term'].value_counts()
```




     36 months    301247
     60 months     93972
    Name: term, dtype: int64



Merapihkan item dari kolom ini (menghapus 'months'):


```python
df['term'] = df['term'].apply(lambda term: int(term[:3]))
```


```python
df['term'].value_counts()
```




    36    301247
    60     93972
    Name: term, dtype: int64



### grade

**TASK: Telah diketahui bahwa grade merupakan bagian dari sub_grade. Hapus kolom ini!.**


```python
df = df.drop('grade', axis=1)
```


**TASK: Konversi subgrade ke dalam dummy variables:**


```python
dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
df = pd.concat([df.drop('sub_grade', axis=1), dummies], axis=1)
```


```python
df.columns
```




    Index(['loan_amnt', 'term', 'int_rate', 'installment', 'home_ownership',
           'annual_inc', 'verification_status', 'issue_d', 'loan_status',
           'purpose', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec',
           'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
           'application_type', 'mort_acc', 'pub_rec_bankruptcies', 'address',
           'loan_repaid', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5',
           'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2',
           'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4',
           'G5'],
          dtype='object')



### verification_status, application_type,initial_list_status,purpose
**TASK: Konversi feature columns tersebut ke dummy variables:**


```python
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose'] ], drop_first=True)
df = pd.concat([df.drop(['verification_status', 'application_type','initial_list_status','purpose'] , axis=1), dummies], axis=1)
```


```python
df.columns
```




    Index(['loan_amnt', 'term', 'int_rate', 'installment', 'home_ownership',
           'annual_inc', 'issue_d', 'loan_status', 'dti', 'earliest_cr_line',
           'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
           'mort_acc', 'pub_rec_bankruptcies', 'address', 'loan_repaid', 'A2',
           'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4',
           'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1',
           'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5',
           'verification_status_Source Verified', 'verification_status_Verified',
           'application_type_INDIVIDUAL', 'application_type_JOINT',
           'initial_list_status_w', 'purpose_credit_card',
           'purpose_debt_consolidation', 'purpose_educational',
           'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase',
           'purpose_medical', 'purpose_moving', 'purpose_other',
           'purpose_renewable_energy', 'purpose_small_business',
           'purpose_vacation', 'purpose_wedding'],
          dtype='object')



### home_ownership
**TASK: Menampilkan item apa saja di fitur kolom ini**


```python
df['home_ownership'].value_counts()
```




    MORTGAGE    198022
    RENT        159395
    OWN          37660
    OTHER          110
    NONE            29
    ANY              3
    Name: home_ownership, dtype: int64





**TASK: Sebenarnya, other, none, dan any dapat dinyatakan ke ke dalam satu item. Oleh sebab itu, ketiga item tersebut akan digabung ke dalam other saja. Konversi kolom-kolom tersebut ke dummy variabel:**


```python
df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'],'OTHER')
```


```python
df['home_ownership'].value_counts()
```




    MORTGAGE    198022
    RENT        159395
    OWN          37660
    OTHER          142
    Name: home_ownership, dtype: int64




```python
dummies = pd.get_dummies(df['home_ownership'], drop_first=True)
df = pd.concat([df.drop('home_ownership', axis=1), dummies], axis=1)
```

### address



```python
df['address']
```




    0            0174 Michelle Gateway\r\nMendozaberg, OK 22690
    1         1076 Carney Fort Apt. 347\r\nLoganmouth, SD 05113
    2         87025 Mark Dale Apt. 269\r\nNew Sabrina, WV 05113
    3                   823 Reid Ford\r\nDelacruzside, MA 00813
    4                    679 Luna Roads\r\nGreggshire, VA 11650
                                    ...                        
    396025     12951 Williams Crossing\r\nJohnnyville, DC 30723
    396026    0114 Fowler Field Suite 028\r\nRachelborough, ...
    396027    953 Matthew Points Suite 414\r\nReedfort, NY 7...
    396028    7843 Blake Freeway Apt. 229\r\nNew Michael, FL...
    396029        787 Michelle Causeway\r\nBriannaton, AR 48052
    Name: address, Length: 395219, dtype: object



**TASK: Agar lebih mudah dilihat ketika dikonversi ke dalam dummy variable, alamat hanya akan direpresentasikan ke dalam zipcode**


```python
df['zipcode'] = df['address'].apply(lambda address:address[-5:])
```


```python
df['zipcode'].value_counts()
```




    70466    56880
    22690    56413
    30723    56402
    48052    55811
    00813    45725
    29597    45393
    05113    45300
    11650    11210
    93700    11126
    86630    10959
    Name: zipcode, dtype: int64



**TASK: Konversi ke dalam dummy variable**


```python
dummies = pd.get_dummies(df['zipcode'], drop_first=True)
df = pd.concat([df.drop('zipcode', axis=1), dummies], axis=1)
```


```python
df = df.drop('address', axis=1)
```

### issue_d


**TASK: Fitur ini dapat menyebabkan [data leakage](https://towardsdatascience.com/data-leakage-in-machine-learning-6161c167e8ba) karena fitur ini berisikan informasi tentang waktu pinjaman yang diberikan (pada data training). Pada saat model telah di-deploy, sebenarnya model akan menerima inputan pada waktu saat itu juga. Serta informasi tentang masa mendatang tidak akan tersedia pada model. Oleh karena itu, kolom ini akan dihapus.**


```python
feat_info('issue_d')
```

    The month which the loan was funded



```python
df = df.drop('issue_d', axis=1)
```

### earliest_cr_line
**TASK: Ekstrak tahun pada fitur ini.**


```python
df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
```


```python
df['earliest_cr_line'].value_counts()
```




    2000    29302
    2001    29031
    1999    26444
    2002    25849
    2003    23623
            ...  
    1951        3
    1950        3
    1953        2
    1948        1
    1944        1
    Name: earliest_cr_line, Length: 65, dtype: int64



## Train Test Split

**TASK: Import train_test_split dari sklearn.**


```python
from sklearn.model_selection import train_test_split
```

**TASK: Karena loan_status telah direpresentasikan ke dalam loan_repaid, maka hapus kolom tersebut:**


```python
df = df.drop('loan_status', axis=1)
```

**TASK: Memisahkan fitur dengan targetnya**


```python
X = df.drop('loan_repaid', axis=1).values
y = df['loan_repaid'].values
```

**TASK: train/test split dengan test_size=0.2 dan a random_state = 101 yang akan memisahkan data training dan testing. Ukuran data testing adalah 20% dari total data. Sisaya 80% untuk data training.**


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
```

## Normalisasi Data

**TASK: Karena inputan harus memiliki ukuran range yang sama untuk setiap fitur kolomnya, maka data X_train dan X_test harus dinormalisasi. Akan dilakukan MinMaxScaler yang diimport dari sklean.preprocessing**


```python
from sklearn.preprocessing import MinMaxScaler
```


```python
scaler = MinMaxScaler()
```


```python
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

# Creating the Model

**TASK: Import Libraries**


```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
```

**TASK: Build model Sequential dan Compile**


```python
X_train.shape
```




    (316175, 78)




```python
# CODE HERE
model = Sequential()

model.add(Dense(78, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(39, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(19, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

**TASK: Model difit kepada data training, dengan 25 epochs yang diaplikasikan pula dengan early stopping untuk mencegah overfitting. Karena ukuran data yang sangat besar, maka akan dipecah menjadi beberapa batch dengan ukuran perbatch sebesar 256.**


```python
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
```


```python
model.fit(x=X_train, y=y_train, epochs=25, batch_size=256, validation_data=(X_test, y_test), callbacks=[early_stop])
```

# Section 3: Evaluasi perfoma model.




```python
loss = pd.DataFrame(model.history.history)[['loss','val_loss']]
loss.head()
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
      <th>loss</th>
      <th>val_loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.315454</td>
      <td>0.265228</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.269634</td>
      <td>0.263968</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.266143</td>
      <td>0.263972</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.264149</td>
      <td>0.263520</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.263682</td>
      <td>0.263007</td>
    </tr>
  </tbody>
</table>
</div>




```python
loss.plot();
```


![png](/images/lending/03_Keras_Project_Exercise_202_0.png)


Tampak terjadi sedikit overfitting karena loss training lebih rendah daripada loss pada training.

**TASK: Menampilkan classification report dan confusion matrix**


```python
from sklearn.metrics import classification_report, confusion_matrix
```


```python
predictions = model.predict_classes(X_test)
```

    WARNING:tensorflow:From <ipython-input-169-bc83193b8b59>:1: Sequential.predict_classes (from tensorflow.python.keras.engine.sequential) is deprecated and will be removed after 2021-01-01.
    Instructions for updating:
    Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).



```python
print(classification_report(y_test, predictions))
```

                  precision    recall  f1-score   support

               0       1.00      0.43      0.60     15658
               1       0.88      1.00      0.93     63386

        accuracy                           0.89     79044
       macro avg       0.94      0.71      0.77     79044
    weighted avg       0.90      0.89      0.87     79044



Secara akurasi, model berada di angka 89%. Tetapi ...


```python
df['loan_repaid'].value_counts()
```




    1    317696
    0     77523
    Name: loan_repaid, dtype: int64




```python
sns.countplot(x='loan_repaid', data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f5afc48d160>




![png](/images/lending/03_Keras_Project_Exercise_210_1.png)


Sebenarnya, terdapat ketidakseimbangan data dari masing-masing kategori.

Oleh karena itu, akurasi tidak dapat digunakan sebagai acuan untuk menyatakan keakuratan. Sebagai gantinya, f1-score akan digunakan.

Pada kelas fully paid, model memprediksi benar sebesar 93%. Sedangkan charged off terprediksi benar sebesar 60%. Sehingga jika ada peminjam baru terprediksi charged-off, masih ada kemungkinan 40% model memprediksi salah.

Sehingga dapat dikatakan model masih memiliki kelemahan dalam memprediksi kelas Charged-Off karena memiliki tingkat prediksi yang tidak terlalu tinggi dan error yang besar (40%).


```python
print(confusion_matrix(y_test, predictions))
```

    [[ 6721  8937]
     [    3 63383]]


**TASK: Contoh penggunaan model. Ada customer baru mengajukan pinjaman, apakah LendingClub akan memeberikan pinjaman?**


```python
import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer
```




    loan_amnt      25000.00
    term              60.00
    int_rate          18.24
    installment      638.11
    annual_inc     61665.00
                     ...   
    30723              1.00
    48052              0.00
    70466              0.00
    86630              0.00
    93700              0.00
    Name: 305323, Length: 78, dtype: float64




```python
new_customer.shape
```




    (78,)



Sebelumnya, harus dinormalisasi terlebih dahulu:


```python
new_customer = scaler.transform(new_customer.values.reshape(1,78))
new_customer.shape
```




    (1, 78)




```python
model.predict_classes(new_customer)
```




    array([[1]], dtype=int32)



Berdasarkan hasil  di atas yang memprediksi bahwa customer tersebut diduga akan memenuhi/membayar kembali pinjaman (kategori 1), maka LendingClub akan memberikan pinjaman kepada customer tersebut.

**TASK: Karena contoh new_customer tersebut diambil dari data, sekarang akan dibandingkan dengan hasil yang sebenarnya.**


```python
df.iloc[random_ind]['loan_repaid']
```




    1.0



Karena hasilnya sama dengan hasil prediksi, maka berarti model mempredisi dengan benar untuk contoh kasus new_customer tersebut.


```python

```
