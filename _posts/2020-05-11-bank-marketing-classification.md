---

title: "Bank Marketing Classification"
data: 2020-09-14
tags: [python,  exploratory data analysis, classification]
header:
excerpt: ""
mathjax: "true"
toc: true
toc_sticky: false
header:
  teaser: '/images/malaria/malaria_m.jpg'
---


<a href="https://colab.research.google.com/github/fyansyarafa/bank-marketing-classification/blob/main/bank_marketing_classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Tujuan

Menghasilkan model klasifikasi prediktif terhadap deposit nasabah berupa 'yes' atau 'no' berdasarkan fitur-fitur independen pada dataset menggukan beberapa metode klasifikasi machine learning (Decision Tree, Random Forest, Logistic Regression dan KNN). Kemudian, akan dipilih beberapa model terbaik berdasarkan beberapa metrics evaluasi seperti elemen-elemen yang ada pada classfification report, confusion matrix, dan ROC curve.

# Import Library dan Data


```
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

```


```
df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Projects/Telkom Digital Talent Incubator/Bank Marketing Classification/bank-marketing dataset/bank.csv')
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
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2343</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1042</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>45</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1467</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>1270</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1389</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2476</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>579</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>admin.</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>184</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>673</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>



# Exploratory Data Analysis

Mengecek missing values pada data dan tipe data yang tidak sesuai (jika ada):


```
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11162 entries, 0 to 11161
    Data columns (total 17 columns):
     #   Column     Non-Null Count  Dtype
    ---  ------     --------------  -----
     0   age        11162 non-null  int64
     1   job        11162 non-null  object
     2   marital    11162 non-null  object
     3   education  11162 non-null  object
     4   default    11162 non-null  object
     5   balance    11162 non-null  int64
     6   housing    11162 non-null  object
     7   loan       11162 non-null  object
     8   contact    11162 non-null  object
     9   day        11162 non-null  int64
     10  month      11162 non-null  object
     11  duration   11162 non-null  int64
     12  campaign   11162 non-null  int64
     13  pdays      11162 non-null  int64
     14  previous   11162 non-null  int64
     15  poutcome   11162 non-null  object
     16  deposit    11162 non-null  object
    dtypes: int64(7), object(10)
    memory usage: 1.4+ MB


Missing values:


```
df.isnull().sum()
```




    age          0
    job          0
    marital      0
    education    0
    default      0
    balance      0
    housing      0
    loan         0
    contact      0
    day          0
    month        0
    duration     0
    campaign     0
    pdays        0
    previous     0
    poutcome     0
    deposit      0
    dtype: int64



## Deposit

Melihat kondisi fitur target deposit, seperti proporsi kategori deposit:


```

# countplot
splot=sns.countplot(df.deposit)
for p in splot.patches:
  splot.annotate(format(p.get_height(), '.1f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center',
                   xytext = (0, -12),
                   textcoords = 'offset points')
plt.title('Jumlah Masing-Masing Kategori Deposit')

# pie chart
dict = {
    'yes' : len(df[df.deposit == 'yes']['deposit']),
    'no' : len(df[df.deposit == 'no']['deposit'])
}
ser_deposit = pd.Series(dict)
ser_deposit

pie, ax = plt.subplots(figsize=[10,6])
labels = ser_deposit.keys()
plt.pie(x=ser_deposit, autopct="%.1f%%", pctdistance=0.5, startangle=90)
plt.title("Persentase Kategori Deposit", fontsize=14)
plt.legend(labels=labels);

# subplot

```


![png](bank_marketing_classification_files/bank_marketing_classification_13_0.png)



![png](bank_marketing_classification_files/bank_marketing_classification_13_1.png)


Sepertinya proporsi kategori no pada fitur deposit sedikit lebih tinggi dibanding yes.

## Hubungan antar variabel numerik


```
matrix = np.triu(df[['age','balance','duration','campaign','pdays','previous']].corr())
plt.figure(figsize=(12,7))
sns.heatmap(df[['age','balance','duration','campaign','pdays','previous']].corr(), annot=True, cmap='viridis', mask=matrix)
plt.title('Matriks Korelasi untuk Variabel Numerik');
```


![png](bank_marketing_classification_files/bank_marketing_classification_16_0.png)


Hanya terdapat satu hubungan yang menonjol antara `pdays` dan `previous`, dengan nilai korelasi 0.51 dengan arah positif.

## Age


```
plt.figure(figsize=(10,4))
sns.distplot(df.age,bins=20)
plt.xticks(ticks=list(np.arange(0,100,5)))
#plt.axvline(df['age'].mean(), label='mean = {}'.format(round(df['age'].mean()),2),color='r')
plt.axvline(df['age'].median(), label='Average = {}'.format(round(df['age'].median()),2),color='y')
#plt.axvline(df['age'].mode()[0], label='mode = {}'.format(df['age'].mode()[0]),color='g')
plt.title('Distribusi Usia Nasabah')
plt.legend();
```


![png](bank_marketing_classification_files/bank_marketing_classification_19_0.png)


Nilai *average* dihasilkan dari nilai *median* fitur `age`, karena sepertinya terdapat *outlier* sehingga mempengaruhi nilai *mean*.


```
age_deposit_yes = df['deposit'] == 'yes'
age_deposit_yes = df.loc[age_deposit_yes]
age_deposit_yes.head()
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
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2343</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1042</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>45</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1467</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>1270</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1389</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2476</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>579</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>admin.</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>184</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>673</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>




```
age_deposit_no = df['deposit'] == 'no'
age_deposit_no = df.loc[age_deposit_no]
age_deposit_no.head()
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
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5289</th>
      <td>57</td>
      <td>retired</td>
      <td>single</td>
      <td>primary</td>
      <td>no</td>
      <td>604</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>19</td>
      <td>nov</td>
      <td>187</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5290</th>
      <td>45</td>
      <td>admin.</td>
      <td>divorced</td>
      <td>secondary</td>
      <td>no</td>
      <td>0</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>17</td>
      <td>nov</td>
      <td>102</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5291</th>
      <td>48</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>238</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>2</td>
      <td>jun</td>
      <td>118</td>
      <td>2</td>
      <td>81</td>
      <td>1</td>
      <td>success</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5292</th>
      <td>34</td>
      <td>admin.</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>673</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>29</td>
      <td>jan</td>
      <td>89</td>
      <td>1</td>
      <td>260</td>
      <td>2</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5293</th>
      <td>37</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>7944</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>21</td>
      <td>nov</td>
      <td>102</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```
plt.figure(figsize=(10,4))
plt.xticks(ticks=list(np.arange(0,100,5)))
sns.distplot(age_deposit_yes.age, bins=20,label="deposit = 'yes'")
sns.distplot(age_deposit_no.age, bins = 20,label="deposit = 'no'")
plt.axvline(age_deposit_yes.age.median(),label='Deposit yes average = {}'.format(round(age_deposit_yes.age.median(),3)))
plt.axvline(age_deposit_no.age.median(),label='Deposit no average = {}'.format(round(age_deposit_no.age.median(),3)),c='coral')
plt.title("Distribusi Usia Berdasarkan Kategori Deposito ('yes' | 'no')")
plt.legend();
```


![png](bank_marketing_classification_files/bank_marketing_classification_23_0.png)



```
sns.boxplot(data=df, x = 'deposit', y = 'age', hue='deposit')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fcfdcc1c2e8>




![png](bank_marketing_classification_files/bank_marketing_classification_24_1.png)



```
deleted = {'age', 'balance', 'duration', 'campaign','pdays','previous'}
list2 = [ele for ele in df.columns if ele not in deleted]

```


```
age_to_cat = df[list2]
age_to_cat['age'] = df['age']
age_to_cat.head()
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
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>poutcome</th>
      <th>deposit</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>unknown</td>
      <td>yes</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>unknown</td>
      <td>yes</td>
      <td>56</td>
    </tr>
    <tr>
      <th>2</th>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>unknown</td>
      <td>yes</td>
      <td>41</td>
    </tr>
    <tr>
      <th>3</th>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>unknown</td>
      <td>yes</td>
      <td>55</td>
    </tr>
    <tr>
      <th>4</th>
      <td>admin.</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>unknown</td>
      <td>yes</td>
      <td>54</td>
    </tr>
  </tbody>
</table>
</div>



### Age vs variabel-variabel kategorik


```
 # age vs categorical variables
 plt.figure(figsize=(23,35))
 for i in range(len(age_to_cat.columns)-1):
   plt.subplot(7,2,i+1)
   sns.boxplot(data=age_to_cat, x=age_to_cat.columns[i], y = 'age')
  # plt.axhline(age_to_cat[age_to_cat.columns[i]]['age'].mean(), label ='Mean = {}'.format(round(age_to_cat[age_to_cat.columns[i]]['age'].mean(),3)))
   plt.title(age_to_cat.columns[i].upper()+' vs AGE')
plt.tight_layout()
```


![png](bank_marketing_classification_files/bank_marketing_classification_28_0.png)


## Balance


```
plt.figure(figsize=(10,4))
sns.distplot(df.balance,bins=20)
#plt.xticks(ticks=list(np.arange(0,100,5)))
#plt.axvline(df['age'].mean(), label='mean = {}'.format(round(df['age'].mean()),2),color='r')
plt.axvline(df['balance'].median(), label='Average = {}'.format(round(df['balance'].median()),2),color='y')
#plt.axvline(df['age'].mode()[0], label='mode = {}'.format(df['age'].mode()[0]),color='g')
plt.title('Distribusi Balance Nasabah')
plt.legend();
```


![png](bank_marketing_classification_files/bank_marketing_classification_30_0.png)



```
balance_deposit_yes = df['deposit'] == 'yes'
balance_deposit_yes = df.loc[balance_deposit_yes]
balance_deposit_yes.head()
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
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2343</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1042</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>45</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1467</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>1270</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1389</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2476</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>579</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>admin.</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>184</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>673</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>




```
balance_deposit_no = df['deposit'] == 'no'
balance_deposit_no = df.loc[balance_deposit_no]
balance_deposit_no.head()
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
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5289</th>
      <td>57</td>
      <td>retired</td>
      <td>single</td>
      <td>primary</td>
      <td>no</td>
      <td>604</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>19</td>
      <td>nov</td>
      <td>187</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5290</th>
      <td>45</td>
      <td>admin.</td>
      <td>divorced</td>
      <td>secondary</td>
      <td>no</td>
      <td>0</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>17</td>
      <td>nov</td>
      <td>102</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5291</th>
      <td>48</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>238</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>2</td>
      <td>jun</td>
      <td>118</td>
      <td>2</td>
      <td>81</td>
      <td>1</td>
      <td>success</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5292</th>
      <td>34</td>
      <td>admin.</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>673</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>29</td>
      <td>jan</td>
      <td>89</td>
      <td>1</td>
      <td>260</td>
      <td>2</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5293</th>
      <td>37</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>7944</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>21</td>
      <td>nov</td>
      <td>102</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```
plt.figure(figsize=(10,4))
#plt.xticks(ticks=list(np.arange(0,100,5)))
sns.distplot(balance_deposit_yes.balance, bins=20,label="deposit = 'yes'")
sns.distplot(balance_deposit_no.balance, bins = 20,label="deposit = 'no'")
plt.axvline(balance_deposit_yes.balance.median(),label='Deposit yes balance average = {}'.format(round(balance_deposit_yes.balance.median(),3)))
plt.axvline(balance_deposit_no.balance.median(),label='Deposit no balance average = {}'.format(round(balance_deposit_no.balance.median(),3)),c='coral')
plt.title("Distribusi Balance Berdasarkan Kategori Deposito ('yes' | 'no')")
plt.legend();
```


![png](bank_marketing_classification_files/bank_marketing_classification_33_0.png)



```
sns.boxplot(data=df, x = 'deposit', y = 'balance', hue='deposit');
```


![png](bank_marketing_classification_files/bank_marketing_classification_34_0.png)


### Balance vs Variabel-Variabel Kategorik


```
deleted_balance = {'age', 'balance', 'duration', 'campaign','pdays','previous'}
list3 = [ele for ele in df.columns if ele not in deleted_balance]

balance_to_cat = df[list3]
balance_to_cat['balance'] = df['balance']

 # balance vs categorical variables
plt.figure(figsize=(23,35))
for i in range(len(balance_to_cat.columns)-1):
  plt.subplot(7,2,i+1)
  sns.boxplot(data=balance_to_cat, x=balance_to_cat.columns[i], y = 'balance')
  # plt.axhline(age_to_cat[age_to_cat.columns[i]]['age'].mean(), label ='Mean = {}'.format(round(age_to_cat[age_to_cat.columns[i]]['age'].mean(),3)))
  plt.title(balance_to_cat.columns[i].upper()+' vs BALANCE')
plt.tight_layout()
```


![png](bank_marketing_classification_files/bank_marketing_classification_36_0.png)


## Duration


```
plt.figure(figsize=(10,4))
sns.distplot(df.duration,bins=20)
#plt.xticks(ticks=list(np.arange(0,100,5)))
#plt.axvline(df['age'].mean(), label='mean = {}'.format(round(df['age'].mean()),2),color='r')
plt.axvline(df['duration'].median(), label='Average = {}'.format(round(df['duration'].median()),2),color='y')
#plt.axvline(df['age'].mode()[0], label='mode = {}'.format(df['age'].mode()[0]),color='g')
plt.title('Distribusi Duration Nasabah')
plt.legend();
```


![png](bank_marketing_classification_files/bank_marketing_classification_38_0.png)



```
duration_deposit_yes = df['deposit'] == 'yes'
duration_deposit_yes = df.loc[duration_deposit_yes]
duration_deposit_no = df['deposit'] == 'no'
duration_deposit_no = df.loc[duration_deposit_no]

plt.figure(figsize=(10,4))
#plt.xticks(ticks=list(np.arange(0,100,5)))
sns.distplot(duration_deposit_yes.duration, bins=20,label="deposit = 'yes'")
sns.distplot(duration_deposit_no.duration, bins = 20,label="deposit = 'no'")
plt.axvline(duration_deposit_yes.duration.median(),label='Deposit yes duration average = {}'.format(round(duration_deposit_yes.duration.median(),3)))
plt.axvline(duration_deposit_no.duration.median(),label='Deposit no duration average = {}'.format(round(duration_deposit_no.duration.median(),3)),c='coral')
plt.title("Distribusi Duration Berdasarkan Kategori Deposito ('yes' | 'no')")
plt.legend();


```


![png](bank_marketing_classification_files/bank_marketing_classification_39_0.png)



```
sns.boxplot(data=df, x = 'deposit', y = 'duration', hue='deposit');
```


![png](bank_marketing_classification_files/bank_marketing_classification_40_0.png)


### Duration vs Variabel-Variabel Kategorik


```
deleted_duration = {'age', 'balance', 'duration', 'campaign','pdays','previous'}
list4 = [ele for ele in df.columns if ele not in deleted_duration]

duration_to_cat = df[list4]
duration_to_cat['duration'] = df['duration']

 # duration vs categorical variables
plt.figure(figsize=(23,35))
for i in range(len(duration_to_cat.columns)-1):
  plt.subplot(7,2,i+1)
  sns.boxplot(data=duration_to_cat, x=duration_to_cat.columns[i], y = 'duration')
  # plt.axhline(age_to_cat[age_to_cat.columns[i]]['age'].mean(), label ='Mean = {}'.format(round(age_to_cat[age_to_cat.columns[i]]['age'].mean(),3)))
  plt.title(duration_to_cat.columns[i].upper()+' vs DURATION')
plt.tight_layout()
```


![png](bank_marketing_classification_files/bank_marketing_classification_42_0.png)


## Pdays


```
plt.figure(figsize=(10,4))
sns.distplot(df.pdays,bins=20)
#plt.xticks(ticks=list(np.arange(0,100,5)))
#plt.axvline(df['age'].mean(), label='mean = {}'.format(round(df['age'].mean()),2),color='r')
plt.axvline(df['pdays'].median(), label='Average = {}'.format(round(df['pdays'].median()),2),color='y')
#plt.axvline(df['age'].mode()[0], label='mode = {}'.format(df['age'].mode()[0]),color='g')
plt.title('Distribusi pdays Nasabah')
plt.legend();
```


![png](bank_marketing_classification_files/bank_marketing_classification_44_0.png)



```
df.pdays.max()
```




    854




```
pdays_deposit_yes = df['deposit'] == 'yes'
pdays_deposit_yes = df.loc[pdays_deposit_yes]
pdays_deposit_no = df['deposit'] == 'no'
pdays_deposit_no = df.loc[pdays_deposit_no]

plt.figure(figsize=(10,4))
#plt.xticks(ticks=list(np.arange(0,100,5)))
sns.distplot(pdays_deposit_yes.pdays, bins=20,label="deposit = 'yes'")
sns.distplot(pdays_deposit_no.pdays, bins = 20,label="deposit = 'no'")
plt.axvline(pdays_deposit_yes.pdays.median(),label='Deposit yes pdays average = {}'.format(round(pdays_deposit_yes.pdays.median(),3)))
plt.axvline(pdays_deposit_no.pdays.median(),label='Deposit no duration average = {}'.format(round(pdays_deposit_no.pdays.median(),3)),c='coral')
plt.title("Distribusi pdays Berdasarkan Kategori Deposito ('yes' | 'no')")
plt.legend();


```


![png](bank_marketing_classification_files/bank_marketing_classification_46_0.png)



```
sns.boxplot(data=df, x = 'deposit', y = 'pdays', hue='deposit');
```


![png](bank_marketing_classification_files/bank_marketing_classification_47_0.png)


### Pdays vs Variabel-Variabel Kategorik


```
deleted_pdays = {'age', 'balance', 'duration', 'campaign','pdays','previous'}
list5 = [ele for ele in df.columns if ele not in deleted_pdays]

pdays_to_cat = df[list5]
pdays_to_cat['pdays'] = df['pdays']

 # pdays vs categorical variables
plt.figure(figsize=(23,35))
for i in range(len(pdays_to_cat.columns)-1):
  plt.subplot(7,2,i+1)
  sns.boxplot(data=pdays_to_cat, x=pdays_to_cat.columns[i], y = 'pdays')
  # plt.axhline(age_to_cat[age_to_cat.columns[i]]['age'].mean(), label ='Mean = {}'.format(round(age_to_cat[age_to_cat.columns[i]]['age'].mean(),3)))
  plt.title(pdays_to_cat.columns[i].upper()+' vs PDAYS')
plt.tight_layout()
```


![png](bank_marketing_classification_files/bank_marketing_classification_49_0.png)


## Previous


```
plt.figure(figsize=(10,4))
sns.distplot(df.previous,bins=20)
#plt.xticks(ticks=list(np.arange(0,100,5)))
#plt.axvline(df['age'].mean(), label='mean = {}'.format(round(df['age'].mean()),2),color='r')
plt.axvline(df['previous'].median(), label='Average = {}'.format(round(df['previous'].median()),2),color='y')
#plt.axvline(df['age'].mode()[0], label='mode = {}'.format(df['age'].mode()[0]),color='g')
plt.title('Distribusi previous Nasabah')
plt.legend();
```


![png](bank_marketing_classification_files/bank_marketing_classification_51_0.png)



```
previous_deposit_yes = df['deposit'] == 'yes'
previous_deposit_yes = df.loc[previous_deposit_yes]
previous_deposit_no = df['deposit'] == 'no'
previous_deposit_no = df.loc[previous_deposit_no]

plt.figure(figsize=(10,4))
#plt.xticks(ticks=list(np.arange(0,100,5)))
sns.distplot(previous_deposit_yes.previous, bins=20,label="deposit = 'yes'")
sns.distplot(previous_deposit_no.previous, bins = 20,label="deposit = 'no'")
plt.axvline(previous_deposit_yes.previous.median(),label='Deposit yes previous average = {}'.format(round(previous_deposit_yes.previous.median(),3)))
plt.axvline(previous_deposit_no.previous.median(),label='Deposit no previous average = {}'.format(round(previous_deposit_no.previous.median(),3)),c='coral')
plt.title("Distribusi previous Berdasarkan Kategori Deposito ('yes' | 'no')")
plt.legend();


```


![png](bank_marketing_classification_files/bank_marketing_classification_52_0.png)



```
sns.boxplot(data=df, x = 'deposit', y = 'previous', hue='deposit');
```


![png](bank_marketing_classification_files/bank_marketing_classification_53_0.png)


### Previous vs Variabel-Variabel Kategorik


```
deleted_previous = {'age', 'balance', 'duration', 'campaign','pdays','previous'}
list6 = [ele for ele in df.columns if ele not in deleted_pdays]

pdays_to_cat = df[list5]
pdays_to_cat['pdays'] = df['pdays']

 # Previous vs categorical variables
plt.figure(figsize=(23,35))
for i in range(len(pdays_to_cat.columns)-1):
  plt.subplot(7,2,i+1)
  sns.boxplot(data=pdays_to_cat, x=pdays_to_cat.columns[i], y = 'pdays')
  # plt.axhline(age_to_cat[age_to_cat.columns[i]]['age'].mean(), label ='Mean = {}'.format(round(age_to_cat[age_to_cat.columns[i]]['age'].mean(),3)))
  plt.title(pdays_to_cat.columns[i].upper()+' vs PDAYS')
plt.tight_layout()
```


![png](bank_marketing_classification_files/bank_marketing_classification_55_0.png)


## Campaign


```
plt.figure(figsize=(10,4))
sns.distplot(df.campaign,bins=20)
#plt.xticks(ticks=list(np.arange(0,100,5)))
#plt.axvline(df['age'].mean(), label='mean = {}'.format(round(df['age'].mean()),2),color='r')
plt.axvline(df['campaign'].median(), label='Average = {}'.format(round(df['campaign'].median()),2),color='y')
#plt.axvline(df['age'].mode()[0], label='mode = {}'.format(df['age'].mode()[0]),color='g')
plt.title('Distribusi campaign Nasabah')
plt.legend();
```


![png](bank_marketing_classification_files/bank_marketing_classification_57_0.png)



```
campaign_deposit_yes = df['deposit'] == 'yes'
campaign_deposit_yes = df.loc[campaign_deposit_yes]
campaign_deposit_no = df['deposit'] == 'no'
campaign_deposit_no = df.loc[campaign_deposit_no]

plt.figure(figsize=(10,4))
#plt.xticks(ticks=list(np.arange(0,100,5)))
sns.distplot(campaign_deposit_yes.campaign, bins=20,label="deposit = 'yes'")
sns.distplot(campaign_deposit_no.campaign, bins = 20,label="deposit = 'no'")
plt.axvline(campaign_deposit_yes.campaign.median(),label='Deposit yes campaign average = {}'.format(round(campaign_deposit_yes.campaign.median(),3)))
plt.axvline(campaign_deposit_no.campaign.median(),label='Deposit no campaign average = {}'.format(round(campaign_deposit_no.campaign.median(),3)),c='coral')
plt.title("Distribusi campaign Berdasarkan Kategori Deposito ('yes' | 'no')")
plt.legend();


```


![png](bank_marketing_classification_files/bank_marketing_classification_58_0.png)



```
sns.boxplot(data=df, x = 'deposit', y = 'campaign', hue='deposit');
```


![png](bank_marketing_classification_files/bank_marketing_classification_59_0.png)


### Campaign Variabel-Variabel Kategorik


```
deleted_campaign = {'age', 'balance', 'duration', 'campaign','pdays','previous'}
list7 = [ele for ele in df.columns if ele not in deleted_campaign]

campaign_to_cat = df[list7]
campaign_to_cat['campaign'] = df['campaign']

 # age vs categorical variables
plt.figure(figsize=(23,35))
for i in range(len(campaign_to_cat.columns)-1):
  plt.subplot(7,2,i+1)
  sns.boxplot(data=campaign_to_cat, x=campaign_to_cat.columns[i], y = 'campaign')
  # plt.axhline(age_to_cat[age_to_cat.columns[i]]['age'].mean(), label ='Mean = {}'.format(round(age_to_cat[age_to_cat.columns[i]]['age'].mean(),3)))
  plt.title(campaign_to_cat.columns[i].upper()+' vs PDAYS')
plt.tight_layout()
```


![png](bank_marketing_classification_files/bank_marketing_classification_61_0.png)


# Features Engineering
Akan dilakukan pemilihan fitur-fitur yang akan dijadikan sebagai parameter dalam membangun model. Dilakukan dengan mendeteksi adayana outlier antar variabel numerik. Setelah itu, outlier tersebut akan disisihkan.


```
df.columns
```




    Index(['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
           'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
           'previous', 'poutcome', 'deposit'],
          dtype='object')




```
deleted_out = {'day'}
list8 = [ele for ele in df.columns if ele not in deleted_out]
```


```
df_out = df[list8]
#df_out['campaign'] = df['campaign']
df_out.columns
```




    Index(['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
           'loan', 'contact', 'month', 'duration', 'campaign', 'pdays', 'previous',
           'poutcome', 'deposit'],
          dtype='object')




```
sns.boxplot(df['age']);
```


![png](bank_marketing_classification_files/bank_marketing_classification_66_0.png)



```
df_new = df.copy()
```

### Age


```
sns.boxplot(df_new['age']);
```


![png](bank_marketing_classification_files/bank_marketing_classification_69_0.png)



```
df_new["age"] = np.where(df_new["age"] <df_new['age'].quantile(0.10), df_new['age'].quantile(0.10),df_new['age'])
df_new["age"] = np.where(df_new["age"] >df_new['age'].quantile(0.90), df_new['age'].quantile(0.90),df_new['age'])
print(df_new['age'].skew())
```

    0.4123928651128603



```
sns.boxplot(df_new['age']);
```


![png](bank_marketing_classification_files/bank_marketing_classification_71_0.png)


### Balance


```
sns.boxplot(df_new['balance']);
```


![png](bank_marketing_classification_files/bank_marketing_classification_73_0.png)



```
df_new["balance"] = np.where(df_new["balance"] <df_new['balance'].quantile(0.10), df_new['balance'].quantile(0.10),df_new['balance'])
df_new["balance"] = np.where(df_new["balance"] >df_new['balance'].quantile(0.90), df_new['balance'].quantile(0.90),df_new['balance'])
print(df_new['balance'].skew())
```

    1.142247974036395



```
sns.boxplot(df_new['balance']);
```


![png](bank_marketing_classification_files/bank_marketing_classification_75_0.png)


### Duration


```
sns.boxplot(df_new['duration']);
```


![png](bank_marketing_classification_files/bank_marketing_classification_77_0.png)



```
df_new["duration"] = np.where(df_new["duration"] <df_new['duration'].quantile(0.10), df_new['duration'].quantile(0.10),df_new['duration'])
df_new["duration"] = np.where(df_new["duration"] >df_new['duration'].quantile(0.90), df_new['duration'].quantile(0.90),df_new['duration'])
print(df_new['duration'].skew())
```

    0.8506119184251896



```
sns.boxplot(df_new['duration']);
```


![png](bank_marketing_classification_files/bank_marketing_classification_79_0.png)


### Pdays


```
sns.boxplot(df_new['pdays']);
```


![png](bank_marketing_classification_files/bank_marketing_classification_81_0.png)



```
df_new["pdays"] = np.where(df_new["pdays"] <df_new['pdays'].quantile(0.10), df_new['pdays'].quantile(0.10),df_new['pdays'])
df_new["pdays"] = np.where(df_new["pdays"] >df_new['pdays'].quantile(0.90), df_new['pdays'].quantile(0.90),df_new['pdays'])
print(df_new['pdays'].skew())
```

    1.4344713357254215



```
sns.boxplot(df_new['pdays']);
```


![png](bank_marketing_classification_files/bank_marketing_classification_83_0.png)


### Previous


```
sns.boxplot(df_new['previous']);
```


![png](bank_marketing_classification_files/bank_marketing_classification_85_0.png)



```
df_new["previous"] = np.where(df_new["previous"] <df_new['previous'].quantile(0.10), df_new['previous'].quantile(0.10),df_new['previous'])
df_new["previous"] = np.where(df_new["previous"] >df_new['previous'].quantile(0.90), df_new['previous'].quantile(0.90),df_new['previous'])
print(df_new['previous'].skew())
```

    1.635296161472558



```
sns.boxplot(df_new['previous']);
```


![png](bank_marketing_classification_files/bank_marketing_classification_87_0.png)


### Campaign


```
sns.boxplot(df_new['campaign']);
```


![png](bank_marketing_classification_files/bank_marketing_classification_89_0.png)



```
df_new["campaign"] = np.where(df_new["campaign"] <df_new['campaign'].quantile(0.10), df_new['campaign'].quantile(0.10),df_new['campaign'])
df_new["campaign"] = np.where(df_new["campaign"] >df_new['campaign'].quantile(0.90), df_new['campaign'].quantile(0.90),df_new['campaign'])
print(df_new['campaign'].skew())
```

    0.9826361461567283



```
sns.boxplot(df_new['campaign']);
```


![png](bank_marketing_classification_files/bank_marketing_classification_91_0.png)


## Drop Kolom/Fitur

`pdays`, dan `previous` didrop karena masih terdapat *outlier* pada fitur tersebut dan juga miliki hubungan korelasi yang lemah dengan fitur-fitur numerik yang lain. `month` dan `day` didrop karena berpotensi menghasilkan masalah *data leakage* pada saat membagi data training dan testing sehingga menimbulkan *overfitting*.


```
df_new.drop(['pdays','previous','month','day'], axis=1,inplace=True)
```


```
df_new.columns
```




    Index(['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
           'loan', 'contact', 'duration', 'campaign', 'poutcome', 'deposit'],
          dtype='object')



## OneHotEncoder pada Variabel-Variabel Kategorik

Agar dapat diproses ke dalam machine learning, variabel-variabel kategorik harus diubah ke dalam bentuk numerik.


```
df_new2 = df_new.copy()
```

### job


```
# Import Module
from sklearn.preprocessing import OneHotEncoder

# Encoder
encoder = OneHotEncoder(sparse=False)

# Encode Categorical Data
df_encoded = pd.DataFrame(encoder.fit_transform(df_new2[['job', 'marital', 'education', 'default','housing','loan','contact','poutcome']]))
df_encoded.columns = encoder.get_feature_names(['job', 'marital', 'education', 'default','housing','loan','contact','poutcome'])

# Replace Categotical Data with Encoded Data
df_new2.drop(['job', 'marital', 'education', 'default','housing','loan','contact','poutcome'] ,axis=1, inplace=True)
df_encoded= pd.concat([df_new2, df_encoded], axis=1)

# Show Encoded Dataframe
df_encoded
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
      <th>age</th>
      <th>balance</th>
      <th>duration</th>
      <th>campaign</th>
      <th>deposit</th>
      <th>job_admin.</th>
      <th>job_blue-collar</th>
      <th>job_entrepreneur</th>
      <th>job_housemaid</th>
      <th>job_management</th>
      <th>job_retired</th>
      <th>job_self-employed</th>
      <th>job_services</th>
      <th>job_student</th>
      <th>job_technician</th>
      <th>job_unemployed</th>
      <th>job_unknown</th>
      <th>marital_divorced</th>
      <th>marital_married</th>
      <th>marital_single</th>
      <th>education_primary</th>
      <th>education_secondary</th>
      <th>education_tertiary</th>
      <th>education_unknown</th>
      <th>default_no</th>
      <th>default_yes</th>
      <th>housing_no</th>
      <th>housing_yes</th>
      <th>loan_no</th>
      <th>loan_yes</th>
      <th>contact_cellular</th>
      <th>contact_telephone</th>
      <th>contact_unknown</th>
      <th>poutcome_failure</th>
      <th>poutcome_other</th>
      <th>poutcome_success</th>
      <th>poutcome_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58.0</td>
      <td>2343.0</td>
      <td>838.0</td>
      <td>1.0</td>
      <td>yes</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56.0</td>
      <td>45.0</td>
      <td>838.0</td>
      <td>1.0</td>
      <td>yes</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41.0</td>
      <td>1270.0</td>
      <td>838.0</td>
      <td>1.0</td>
      <td>yes</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55.0</td>
      <td>2476.0</td>
      <td>579.0</td>
      <td>1.0</td>
      <td>yes</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54.0</td>
      <td>184.0</td>
      <td>673.0</td>
      <td>2.0</td>
      <td>yes</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11157</th>
      <td>33.0</td>
      <td>1.0</td>
      <td>257.0</td>
      <td>1.0</td>
      <td>no</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11158</th>
      <td>39.0</td>
      <td>733.0</td>
      <td>83.0</td>
      <td>4.0</td>
      <td>no</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>32.0</td>
      <td>29.0</td>
      <td>156.0</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11160</th>
      <td>43.0</td>
      <td>0.0</td>
      <td>77.0</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11161</th>
      <td>34.0</td>
      <td>0.0</td>
      <td>628.0</td>
      <td>1.0</td>
      <td>no</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>11162 rows × 37 columns</p>
</div>




```
dep = df_new2['deposit'].map({'yes':1, 'no':0})
```


```
df_encoded.drop('deposit',axis=1)
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
      <th>age</th>
      <th>balance</th>
      <th>duration</th>
      <th>campaign</th>
      <th>job_admin.</th>
      <th>job_blue-collar</th>
      <th>job_entrepreneur</th>
      <th>job_housemaid</th>
      <th>job_management</th>
      <th>job_retired</th>
      <th>job_self-employed</th>
      <th>job_services</th>
      <th>job_student</th>
      <th>job_technician</th>
      <th>job_unemployed</th>
      <th>job_unknown</th>
      <th>marital_divorced</th>
      <th>marital_married</th>
      <th>marital_single</th>
      <th>education_primary</th>
      <th>education_secondary</th>
      <th>education_tertiary</th>
      <th>education_unknown</th>
      <th>default_no</th>
      <th>default_yes</th>
      <th>housing_no</th>
      <th>housing_yes</th>
      <th>loan_no</th>
      <th>loan_yes</th>
      <th>contact_cellular</th>
      <th>contact_telephone</th>
      <th>contact_unknown</th>
      <th>poutcome_failure</th>
      <th>poutcome_other</th>
      <th>poutcome_success</th>
      <th>poutcome_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58.0</td>
      <td>2343.0</td>
      <td>838.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56.0</td>
      <td>45.0</td>
      <td>838.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41.0</td>
      <td>1270.0</td>
      <td>838.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55.0</td>
      <td>2476.0</td>
      <td>579.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54.0</td>
      <td>184.0</td>
      <td>673.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11157</th>
      <td>33.0</td>
      <td>1.0</td>
      <td>257.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11158</th>
      <td>39.0</td>
      <td>733.0</td>
      <td>83.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>32.0</td>
      <td>29.0</td>
      <td>156.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11160</th>
      <td>43.0</td>
      <td>0.0</td>
      <td>77.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11161</th>
      <td>34.0</td>
      <td>0.0</td>
      <td>628.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>11162 rows × 36 columns</p>
</div>




```
df_encoded['deposit'] = dep
```


```
df_encoded
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
      <th>age</th>
      <th>balance</th>
      <th>duration</th>
      <th>campaign</th>
      <th>deposit</th>
      <th>job_admin.</th>
      <th>job_blue-collar</th>
      <th>job_entrepreneur</th>
      <th>job_housemaid</th>
      <th>job_management</th>
      <th>job_retired</th>
      <th>job_self-employed</th>
      <th>job_services</th>
      <th>job_student</th>
      <th>job_technician</th>
      <th>job_unemployed</th>
      <th>job_unknown</th>
      <th>marital_divorced</th>
      <th>marital_married</th>
      <th>marital_single</th>
      <th>education_primary</th>
      <th>education_secondary</th>
      <th>education_tertiary</th>
      <th>education_unknown</th>
      <th>default_no</th>
      <th>default_yes</th>
      <th>housing_no</th>
      <th>housing_yes</th>
      <th>loan_no</th>
      <th>loan_yes</th>
      <th>contact_cellular</th>
      <th>contact_telephone</th>
      <th>contact_unknown</th>
      <th>poutcome_failure</th>
      <th>poutcome_other</th>
      <th>poutcome_success</th>
      <th>poutcome_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58.0</td>
      <td>2343.0</td>
      <td>838.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56.0</td>
      <td>45.0</td>
      <td>838.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41.0</td>
      <td>1270.0</td>
      <td>838.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55.0</td>
      <td>2476.0</td>
      <td>579.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54.0</td>
      <td>184.0</td>
      <td>673.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11157</th>
      <td>33.0</td>
      <td>1.0</td>
      <td>257.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11158</th>
      <td>39.0</td>
      <td>733.0</td>
      <td>83.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>32.0</td>
      <td>29.0</td>
      <td>156.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11160</th>
      <td>43.0</td>
      <td>0.0</td>
      <td>77.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11161</th>
      <td>34.0</td>
      <td>0.0</td>
      <td>628.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>11162 rows × 37 columns</p>
</div>



# Train Test Split


```
from sklearn.model_selection import train_test_split
X = df_encoded.drop('deposit',axis=1)
y = df_encoded['deposit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
```

# Decision Tree

## Training Model


```
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')



## Prediksi X_test


```
predictions = dtree.predict(X_test)
```


```
pd.DataFrame({
    'y_true' : y_test,
    'y_pred' : predictions
})
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
      <th>y_true</th>
      <th>y_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8096</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8180</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1387</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3047</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4670</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9252</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3197</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4335</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5417</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9367</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3349 rows × 2 columns</p>
</div>



## Evaluasi Model


```
from sklearn.metrics import classification_report,confusion_matrix
```

### Classification Report


```
print(classification_report(y_test,predictions))
```

                  precision    recall  f1-score   support

               0       0.75      0.75      0.75      1761
               1       0.72      0.72      0.72      1588

        accuracy                           0.74      3349
       macro avg       0.74      0.74      0.74      3349
    weighted avg       0.74      0.74      0.74      3349



Akurasi baik dipakai ketika proporsi fitur target (deposit) sama. Akan tetapi, di dataset tidak sama. Oleh karena itu metrics yang akan digunakan untuk menyatakan sebagai parameter seberapa bagus model yang dihasilkan adalah f1-score. F1-score akan digunakan sebagai parameter utama untuk menentukan keakuratan model yang dibuat ke metode-metode berukutnya. F1-score merupakan harmonic mean antara precision dan recall. Semakin mendekati angka 1, maka model yang dihasilkan untuk memprediksi suatu kelas juga makin bagus.

Decision tree lebih baik 0.02 dalam memprediksi kelas 0 dibanding dengan pada kelas 1. Pada kelas 0, f1-score = 0.75. Kelas 1 sebesar 0.73.

### Confusion Matrix


```
sns.heatmap(confusion_matrix(y_test,predictions),annot=True);
```


![png](bank_marketing_classification_files/bank_marketing_classification_118_0.png)


Dengan tuple unpacking, ekstrak tn, fp, fn, dan tp:


```
tn, fp, fn, tp = confusion_matrix(y_test,predictions).ravel()

print('False Positive = ', fp)
print('False Negative = ', fn)
print('True Positive = ', tp)
print('True Negative = ', tn)
```

    False Positive =  436
    False Negative =  441
    True Positive =  1147
    True Negative =  1325


Confusion matrix di atas menunjukkan bahwa 436 item berada pada error type 1 (False Positif) yang menandakan bahwa model menghasilkan prediksi kelas 0 (no) yang sebenarnya adalah kelas 1 (yes).

Kemudian sebanyak 441 item berada pada error type 2 (False Negative). Menunjukkan bahwa model menghasilkan klasifikasi ke kelas 0 (no), yang sebenarnya adalah kelas 1 (yes).

Sisaya untuk nilai nilai TP dan TN, merupakan hasil-hasil prediksi benar. Model memprediksi deposit yes yang sebenarnya deposit yes, atau model memprediksi deposit no, tetapi sebenarnya memang deposit no.

### ROC Curve

ROC curve menunjukkan trade-off antara sensivity (True Positive Rate) dan specivity (1-False Positive Rate). Semakin kurva melengkung ke arah kiri atas, maka semakin baik pula model yang dihasilkan. Hasilnya, akan meninggalkan *area under curve* (auc) yang akan semakin besar. Berlaku pula sebaliknya jika menghasilkan model yang buruk.


```
# Import Visualization Package
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set Size and Style
plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('ggplot')

# Visualize ROC Curve
y_pred_dtc_proba = dtree.predict_proba(X_test)[::,1]
fprdtc, tprdtc, _ = metrics.roc_curve(y_test,  y_pred_dtc_proba)
aucdtc = metrics.roc_auc_score(y_test, y_pred_dtc_proba)
plt.plot(fprdtc,tprdtc,label="Decision Tree, auc="+str(aucdtc))
plt.title('ROC Curve - Decision Tree')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc=4)
plt.show()
```


![png](bank_marketing_classification_files/bank_marketing_classification_124_0.png)


# Random Forest

## Training Model


```
from sklearn.ensemble import RandomForestClassifier
```

Random Forest merupakan metode *machine learning* yang berupa gabungan dari banyak *decision tree*. Akan di-*set* sebanyak 600 buah *decision tree* dalam *classifier* ini.


```
rfc = RandomForestClassifier(600)
rfc.fit(X_train,y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=600,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)



## Prediksi X_test


```
predictions = rfc.predict(X_test)
```


```
pd.DataFrame({
    'y_true' : y_test,
    'y_pred' : predictions
})
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
      <th>y_true</th>
      <th>y_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8096</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8180</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1387</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3047</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4670</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9252</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3197</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4335</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5417</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9367</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3349 rows × 2 columns</p>
</div>



## Evaluasi Model

### Classification Report


```
print(classification_report(y_test,predictions))
```

                  precision    recall  f1-score   support

               0       0.84      0.79      0.81      1761
               1       0.78      0.83      0.81      1588

        accuracy                           0.81      3349
       macro avg       0.81      0.81      0.81      3349
    weighted avg       0.81      0.81      0.81      3349



### Confusion Matrix


```
sns.heatmap(confusion_matrix(y_test,predictions),annot=True);
```


![png](bank_marketing_classification_files/bank_marketing_classification_137_0.png)



```
tn, fp, fn, tp = confusion_matrix(y_test,predictions).ravel()

print('False Positive = ', fp)
print('False Negative = ', fn)
print('True Positive = ', tp)
print('True Negative = ', tn)
```

    False Positive =  366
    False Negative =  268
    True Positive =  1320
    True Negative =  1395


### ROC Curve


```
# Set Size and Style
plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('ggplot')

# Visualize ROC Curve
y_pred_rfc_proba = rfc.predict_proba(X_test)[::,1]
fprrfc, tprrfc, _ = metrics.roc_curve(y_test,  y_pred_rfc_proba)
aucrfc = metrics.roc_auc_score(y_test, y_pred_rfc_proba)
plt.plot(fprrfc,tprrfc,label="Random Forest, auc="+str(aucrfc))
plt.title('ROC Curve - Random Forest')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc=4)
plt.show()
```


![png](bank_marketing_classification_files/bank_marketing_classification_140_0.png)


# Logistic Regression

## Training Model


```
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)



## Prediksi X_test


```
predictions = logmodel.predict(X_test)
pd.DataFrame({
    'y_true' : y_test,
    'y_pred' : predictions
})
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
      <th>y_true</th>
      <th>y_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8096</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8180</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1387</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3047</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4670</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9252</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3197</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4335</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5417</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9367</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3349 rows × 2 columns</p>
</div>



## Evaluasi Model

### Classification Report


```
print(classification_report(y_test,predictions))
```

                  precision    recall  f1-score   support

               0       0.81      0.84      0.82      1761
               1       0.82      0.77      0.80      1588

        accuracy                           0.81      3349
       macro avg       0.81      0.81      0.81      3349
    weighted avg       0.81      0.81      0.81      3349



### Confusion Matrix


```
sns.heatmap(confusion_matrix(y_test,predictions),annot=True);
```


![png](bank_marketing_classification_files/bank_marketing_classification_150_0.png)



```
tn, fp, fn, tp = confusion_matrix(y_test,predictions).ravel()

print('False Positive = ', fp)
print('False Negative = ', fn)
print('True Positive = ', tp)
print('True Negative = ', tn)
```

    False Positive =  275
    False Negative =  358
    True Positive =  1230
    True Negative =  1486


### ROC Curve


```
# Set Size and Style
plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('ggplot')

# Visualize ROC Curve
y_pred_logmodel_proba = logmodel.predict_proba(X_test)[::,1]
fprlogmodel, tprlogmodel, _ = metrics.roc_curve(y_test,  y_pred_logmodel_proba)
auclogmodel = metrics.roc_auc_score(y_test, y_pred_logmodel_proba)
plt.plot(fprlogmodel,tprlogmodel,label="Logistic Regression, auc="+str(auclogmodel))
plt.title('ROC Curve - Logistic Regression')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc=4)
plt.show()
```


![png](bank_marketing_classification_files/bank_marketing_classification_153_0.png)


# K Nearest Neighbors

## Standarisasi Variabel

Karena pada KNN, sebaran data pada fitur akan dipetakan dalam satu dimensi yang sama, maka semua fitur yang digunakan akan dilakukan standarisasi dengan angka maksimum dan minimum yang sama untuk setiap fitur.


```
from sklearn.preprocessing import StandardScaler
```


```
scaler = StandardScaler()
```


```
scaler.fit(df_encoded.drop('deposit',axis=1))
```




    StandardScaler(copy=True, with_mean=True, with_std=True)




```
scaled_features = scaler.transform(df_encoded.drop('deposit',axis=1))
```


```
df_knn1 = df_encoded.drop('deposit', axis=1)
df_knn1.columns
```




    Index(['age', 'balance', 'duration', 'campaign', 'job_admin.',
           'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
           'job_management', 'job_retired', 'job_self-employed', 'job_services',
           'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
           'marital_divorced', 'marital_married', 'marital_single',
           'education_primary', 'education_secondary', 'education_tertiary',
           'education_unknown', 'default_no', 'default_yes', 'housing_no',
           'housing_yes', 'loan_no', 'loan_yes', 'contact_cellular',
           'contact_telephone', 'contact_unknown', 'poutcome_failure',
           'poutcome_other', 'poutcome_success', 'poutcome_unknown'],
          dtype='object')




```
df_feat = pd.DataFrame(scaled_features,columns=df_knn1.columns)
df_feat.head()
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
      <th>age</th>
      <th>balance</th>
      <th>duration</th>
      <th>campaign</th>
      <th>job_admin.</th>
      <th>job_blue-collar</th>
      <th>job_entrepreneur</th>
      <th>job_housemaid</th>
      <th>job_management</th>
      <th>job_retired</th>
      <th>job_self-employed</th>
      <th>job_services</th>
      <th>job_student</th>
      <th>job_technician</th>
      <th>job_unemployed</th>
      <th>job_unknown</th>
      <th>marital_divorced</th>
      <th>marital_married</th>
      <th>marital_single</th>
      <th>education_primary</th>
      <th>education_secondary</th>
      <th>education_tertiary</th>
      <th>education_unknown</th>
      <th>default_no</th>
      <th>default_yes</th>
      <th>housing_no</th>
      <th>housing_yes</th>
      <th>loan_no</th>
      <th>loan_yes</th>
      <th>contact_cellular</th>
      <th>contact_telephone</th>
      <th>contact_unknown</th>
      <th>poutcome_failure</th>
      <th>poutcome_other</th>
      <th>poutcome_success</th>
      <th>poutcome_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.729355</td>
      <td>0.950535</td>
      <td>1.997251</td>
      <td>-0.864963</td>
      <td>2.714280</td>
      <td>-0.459229</td>
      <td>-0.173997</td>
      <td>-0.158636</td>
      <td>-0.546362</td>
      <td>-0.273721</td>
      <td>-0.194036</td>
      <td>-0.300242</td>
      <td>-0.182557</td>
      <td>-0.441818</td>
      <td>-0.18177</td>
      <td>-0.079441</td>
      <td>-0.361962</td>
      <td>0.870355</td>
      <td>-0.678403</td>
      <td>-0.394014</td>
      <td>1.018994</td>
      <td>-0.702598</td>
      <td>-0.215873</td>
      <td>0.123617</td>
      <td>-0.123617</td>
      <td>-1.055280</td>
      <td>1.055280</td>
      <td>0.387923</td>
      <td>-0.387923</td>
      <td>-1.605479</td>
      <td>-0.272963</td>
      <td>1.938527</td>
      <td>-0.35159</td>
      <td>-0.224814</td>
      <td>-0.325782</td>
      <td>0.583626</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.528570</td>
      <td>-0.833229</td>
      <td>1.997251</td>
      <td>-0.864963</td>
      <td>2.714280</td>
      <td>-0.459229</td>
      <td>-0.173997</td>
      <td>-0.158636</td>
      <td>-0.546362</td>
      <td>-0.273721</td>
      <td>-0.194036</td>
      <td>-0.300242</td>
      <td>-0.182557</td>
      <td>-0.441818</td>
      <td>-0.18177</td>
      <td>-0.079441</td>
      <td>-0.361962</td>
      <td>0.870355</td>
      <td>-0.678403</td>
      <td>-0.394014</td>
      <td>1.018994</td>
      <td>-0.702598</td>
      <td>-0.215873</td>
      <td>0.123617</td>
      <td>-0.123617</td>
      <td>0.947616</td>
      <td>-0.947616</td>
      <td>0.387923</td>
      <td>-0.387923</td>
      <td>-1.605479</td>
      <td>-0.272963</td>
      <td>1.938527</td>
      <td>-0.35159</td>
      <td>-0.224814</td>
      <td>-0.325782</td>
      <td>0.583626</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.022683</td>
      <td>0.117646</td>
      <td>1.997251</td>
      <td>-0.864963</td>
      <td>-0.368422</td>
      <td>-0.459229</td>
      <td>-0.173997</td>
      <td>-0.158636</td>
      <td>-0.546362</td>
      <td>-0.273721</td>
      <td>-0.194036</td>
      <td>-0.300242</td>
      <td>-0.182557</td>
      <td>2.263377</td>
      <td>-0.18177</td>
      <td>-0.079441</td>
      <td>-0.361962</td>
      <td>0.870355</td>
      <td>-0.678403</td>
      <td>-0.394014</td>
      <td>1.018994</td>
      <td>-0.702598</td>
      <td>-0.215873</td>
      <td>0.123617</td>
      <td>-0.123617</td>
      <td>-1.055280</td>
      <td>1.055280</td>
      <td>0.387923</td>
      <td>-0.387923</td>
      <td>-1.605479</td>
      <td>-0.272963</td>
      <td>1.938527</td>
      <td>-0.35159</td>
      <td>-0.224814</td>
      <td>-0.325782</td>
      <td>0.583626</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.428178</td>
      <td>1.053773</td>
      <td>0.956415</td>
      <td>-0.864963</td>
      <td>-0.368422</td>
      <td>-0.459229</td>
      <td>-0.173997</td>
      <td>-0.158636</td>
      <td>-0.546362</td>
      <td>-0.273721</td>
      <td>-0.194036</td>
      <td>3.330642</td>
      <td>-0.182557</td>
      <td>-0.441818</td>
      <td>-0.18177</td>
      <td>-0.079441</td>
      <td>-0.361962</td>
      <td>0.870355</td>
      <td>-0.678403</td>
      <td>-0.394014</td>
      <td>1.018994</td>
      <td>-0.702598</td>
      <td>-0.215873</td>
      <td>0.123617</td>
      <td>-0.123617</td>
      <td>-1.055280</td>
      <td>1.055280</td>
      <td>0.387923</td>
      <td>-0.387923</td>
      <td>-1.605479</td>
      <td>-0.272963</td>
      <td>1.938527</td>
      <td>-0.35159</td>
      <td>-0.224814</td>
      <td>-0.325782</td>
      <td>0.583626</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.327785</td>
      <td>-0.725334</td>
      <td>1.334170</td>
      <td>-0.119943</td>
      <td>2.714280</td>
      <td>-0.459229</td>
      <td>-0.173997</td>
      <td>-0.158636</td>
      <td>-0.546362</td>
      <td>-0.273721</td>
      <td>-0.194036</td>
      <td>-0.300242</td>
      <td>-0.182557</td>
      <td>-0.441818</td>
      <td>-0.18177</td>
      <td>-0.079441</td>
      <td>-0.361962</td>
      <td>0.870355</td>
      <td>-0.678403</td>
      <td>-0.394014</td>
      <td>-0.981360</td>
      <td>1.423289</td>
      <td>-0.215873</td>
      <td>0.123617</td>
      <td>-0.123617</td>
      <td>0.947616</td>
      <td>-0.947616</td>
      <td>0.387923</td>
      <td>-0.387923</td>
      <td>-1.605479</td>
      <td>-0.272963</td>
      <td>1.938527</td>
      <td>-0.35159</td>
      <td>-0.224814</td>
      <td>-0.325782</td>
      <td>0.583626</td>
    </tr>
  </tbody>
</table>
</div>



## Train Test Split


```
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df_encoded['deposit'],
                                                    test_size=0.30)
```

## Training Model


```
from sklearn.neighbors import KNeighborsClassifier
```


```
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                         weights='uniform')



## Prediksi X_test


```
predictions = knn.predict(X_test)
pd.DataFrame({
    'y_true' : y_test,
    'y_pred' : predictions
})
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
      <th>y_true</th>
      <th>y_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>854</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5777</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1513</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2387</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8249</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1517</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10097</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6278</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5868</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4568</th>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3349 rows × 2 columns</p>
</div>



## Evaluasi Model

### Classification Report


```
print(classification_report(y_test,predictions))
```

                  precision    recall  f1-score   support

               0       0.74      0.79      0.76      1760
               1       0.75      0.70      0.72      1589

        accuracy                           0.74      3349
       macro avg       0.74      0.74      0.74      3349
    weighted avg       0.74      0.74      0.74      3349



### Confusion Matrix


```
sns.heatmap(confusion_matrix(y_test,predictions),annot=True);
```


![png](bank_marketing_classification_files/bank_marketing_classification_174_0.png)



```
tn, fp, fn, tp = confusion_matrix(y_test,predictions).ravel()

print('False Positive = ', fp)
print('False Negative = ', fn)
print('True Positive = ', tp)
print('True Negative = ', tn)
```

    False Positive =  378
    False Negative =  478
    True Positive =  1111
    True Negative =  1382


### ROC Curve


```
# Set Size and Style
plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('ggplot')

# Visualize ROC Curve
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


![png](bank_marketing_classification_files/bank_marketing_classification_177_0.png)


# Membandingkan Model


```
# Comparing ROC Curve
plt.plot(fprdtc,tprdtc,label="Decision Tree, auc="+str(aucdtc))
plt.plot(fprrfc,tprrfc,label="Random Forest, auc="+str(aucrfc))
plt.plot(fprlogmodel,tprlogmodel,label="Logistic Regression, auc="+str(auclogmodel))
plt.plot(fprknn,tprknn,label="KNN, auc="+str(aucknn))
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
```


![png](bank_marketing_classification_files/bank_marketing_classification_179_0.png)


Model terbaik adalah model yang dihasilkan dengan Random Forest dan Logistic Regression. Keduanya berada di angka auc 0.8891 dan 0.8877 yang dapat di lihat pada ROC curve di atas. Begitu pula yang ada pada metrics evaluasi sebelumnya (classification report dan confusion matrix)


```

```
