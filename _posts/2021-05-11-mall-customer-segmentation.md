---

title: "Mall Customers Segmentation"
data: 2021-05-11
tags: [python,  exploratory data analysis, clustering]
header:
excerpt: ""
mathjax: "true"
toc: true
toc_sticky: false
header:
  teaser: '/images/mall/teaser.png'
---
<a href="https://colab.research.google.com/github/fyansyarafa/mall-customer-segmentation/blob/main/Mall%20Customer%20Segmentation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

#Import Libraries & Data


```
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```


```
bins = 15
```


```
df = pd.read_csv("https://raw.githubusercontent.com/fyansyarafa/mall-customer-segmentation/main/Mall_Customers.csv", error_bad_lines=False)
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
      <th>CustomerID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Male</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Female</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Female</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Female</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



# Pra-proses Data
Hanya akan dilakukan pengecekkan missing values pada dataset.


```
df.isnull().sum()
```




    CustomerID                0
    Gender                    0
    Age                       0
    Annual Income (k$)        0
    Spending Score (1-100)    0
    dtype: int64



# Exploratory Data Analysis


```
sns.countplot(data=df, x='Gender')
plt.title('Jumlah Pengunjung Mall Berdasarkan Gender');
```


![png](/images/mall/Mall%20Customer%20Segmentation_8_0.png)


Pengunjung mall sepertinya didominasi oleh wanita. Dengan proporsi seperti yang ditunjukkan dalam pie chart di bawah ini:


```
dict = {
    'Male' : len(df[df.Gender == 'Male']['Gender']),
    'Female' : len(df[df.Gender == 'Female']['Gender'])
}

ser_gender = pd.Series(dict)

pie, ax = plt.subplots(figsize=[10,6])
labels = ser_gender.keys()
plt.pie(x=ser_gender, autopct="%.1f%%", pctdistance=0.5, startangle=90)
plt.title("Proporsi Pengunjung Mall Berdasarkan Jenis Kelamin")
plt.legend(labels=labels);
```


![png](/images/mall/Mall%20Customer%20Segmentation_10_0.png)


Akan dilihat secara lebih detail untuk masing-masing tipe gender. Insight apa saja yang akan didapatkan?


```
# mask untuk gender
gender_male = df.Gender == 'Male'
gender_female = df.Gender == 'Female'
```


```
# dataframe baru berdasarkan mask di atas
df_male = df.loc[gender_male]
df_female = df.loc[gender_female]
```


```
df_male.head()
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
      <th>CustomerID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Male</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Male</td>
      <td>64</td>
      <td>19</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>Male</td>
      <td>67</td>
      <td>19</td>
      <td>14</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>Male</td>
      <td>37</td>
      <td>20</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>



## Gender = 'Male'
Distribusi `Age`, `Annual Income`, dan `Spending Score` untuk `Gender == 'Male'`:


```
cont_feats = list(df.columns)[2:]
```


```
plt.figure(figsize=(20,4))
for i in range(len(cont_feats)):
  plt.subplot(1, 3, i+1)
  plt.title("Distribusi Fitur "+cont_feats[i])
  sns.distplot(df_male[cont_feats[i]], bins=bins)
  plt.grid(True)
plt.tight_layout()
```


![png](/images/mall/Mall%20Customer%20Segmentation_17_0.png)


Dalam histogram:


```
plt.figure(figsize=(20,4))
for i in range(len(cont_feats)):
  plt.subplot(1, 3, i+1)
  plt.title("Histogram Fitur "+cont_feats[i])
  sns.distplot(df_male[cont_feats[i]],kde=False,bins=bins)
  plt.grid(True)
  plt.ylabel('Jumlah')
plt.tight_layout()
```


![png](/images/mall/Mall%20Customer%20Segmentation_19_0.png)


Apabila dibandingkan dengan keseluruhan pengunjung, maka distribusinya adalah sebagai berikut.


```
plt.figure(figsize=(20,4))
for i in range(len(cont_feats)):
  plt.subplot(1, 3, i+1)
  plt.title("Distribusi Fitur "+cont_feats[i])
  sns.distplot(df[cont_feats[i]], label='Seluruh Pengunjung',bins=bins)
  sns.distplot(df_male[cont_feats[i]], label='Male',bins=15)
  plt.grid(True)
  plt.legend();
plt.tight_layout()

```


![png](/images/mall/Mall%20Customer%20Segmentation_21_0.png)


Dalam histogram:


```
plt.figure(figsize=(20,4))
for i in range(len(cont_feats)):
  plt.subplot(1, 3, i+1)
  plt.title("Histogram Fitur "+cont_feats[i])
  sns.distplot(df[cont_feats[i]], label='Seluruh Pengunjung',kde=False,bins=bins)
  sns.distplot(df_male[cont_feats[i]], label='Male',kde=False,bins=bins)
  plt.grid(True)
  plt.ylabel('Jumlah')
  plt.legend();
plt.tight_layout()
```


![png](/images/mall/Mall%20Customer%20Segmentation_23_0.png)


Pada Gender = Male, distribusi usianya banyak berada pada usia 30an awal, 20an awal, serta pada usia mendekati 50 tahun. Dan cukup mendominasi pada usia 60 tahunan akhir dari keseluruhan pengunjung di usia tersebut.

Income pengunjung mall dari mulai dari angka sekitar $ 20k, terus menanjak distribusinya hingga usia early 80. Walupun sempat menurun pada rentang usia 20-40 tahun.

Spending score gender male hampir berimbang di segala rentang score. Walaupun terdapat minim sekali pada rentang 20 sampai sekitar 35.



## Gender = 'Female'
Distribusi `Age`, `Annual Income`, dan `Spending Score` untuk `Gender == 'Female'`:


```
plt.figure(figsize=(20,4))
for i in range(len(cont_feats)):
  plt.subplot(1, 3, i+1)
  plt.title("Distribusi Fitur "+cont_feats[i])
  sns.distplot(df_female[cont_feats[i]], label='Female',bins=bins)
  plt.grid(True)
  plt.legend();
plt.tight_layout()

```


![png](/images/mall/Mall%20Customer%20Segmentation_26_0.png)


Dalam histogram:


```
plt.figure(figsize=(20,4))
for i in range(len(cont_feats)):
  plt.subplot(1, 3, i+1)
  plt.title("Histogram Fitur "+cont_feats[i])
  sns.distplot(df_female[cont_feats[i]], label='Female',kde=False,bins=bins)
  plt.grid(True)
  plt.legend();
  plt.ylabel('Jumlah')
plt.tight_layout()

```


![png](/images/mall/Mall%20Customer%20Segmentation_28_0.png)


Apabila dibandingkan dengan keseluruhan pengunjung, maka distribusinya adalah sebagai berikut.


```
plt.figure(figsize=(20,4))
for i in range(len(cont_feats)):
  plt.subplot(1, 3, i+1)
  plt.title("Distribusi Fitur "+cont_feats[i])
  sns.distplot(df[cont_feats[i]], label='Seluruh Pengunjung',bins=bins)
  plt.grid(True)
  sns.distplot(df_female[cont_feats[i]], label='Female')
  plt.legend();
plt.tight_layout()

```


![png](/images/mall/Mall%20Customer%20Segmentation_30_0.png)


Dalam bentuk histogram:


```
plt.figure(figsize=(20,4))
for i in range(len(cont_feats)):
  plt.subplot(1, 3, i+1)
  plt.title("Histogram Fitur "+cont_feats[i])
  sns.distplot(df[cont_feats[i]], label='Seluruh Pengunjung',kde=False,bins=bins)
  plt.grid(True)
  sns.distplot(df_female[cont_feats[i]], label='Female',kde=False)
  plt.legend()
  plt.ylabel('Jumlah')
plt.tight_layout()

```


![png](/images/mall/Mall%20Customer%20Segmentation_32_0.png)


Dari distribusi dan histogram di atas, nampaknya gender female sangat mendominasi di segala aspek dari keseluruhan pengunjung mall. Mulai dari rentang usia, income, serta spending score.

## Male vs Female


```
plt.figure(figsize=(20,4))
for i in range(len(cont_feats)):
  plt.subplot(1, 3, i+1)
  plt.title("Distribusi Fitur "+cont_feats[i])
  sns.distplot(df_male[cont_feats[i]], label='Male',bins=bins)
  sns.distplot(df_female[cont_feats[i]], label='Female', bins=15)
  plt.grid(True)
  plt.legend();
plt.tight_layout()

```


![png](/images/mall/Mall%20Customer%20Segmentation_35_0.png)


Dalam Histogram:


```
plt.figure(figsize=(20,4))
for i in range(len(cont_feats)):
  plt.subplot(1, 3, i+1)
  plt.title("Histogram Fitur "+cont_feats[i])
  sns.distplot(df_male[cont_feats[i]], label='Male',kde=False,bins=bins)
  sns.distplot(df_female[cont_feats[i]], label='Female',kde=False, bins=15)
  plt.ylabel('Jumlah')
  plt.grid(True)
  plt.legend();
plt.tight_layout()

```


![png](/images/mall/Mall%20Customer%20Segmentation_37_0.png)


# Klasterisasi Mall Customers menggunakan K-Means

## Menyiapkan Data (Features Engineering)


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
      <th>CustomerID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Male</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Female</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Female</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Female</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```
df_for_clustering = df.iloc[:, [3, 4]]
df_for_clustering.head()
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
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



Standarisasi *values* dengan mean 0 dan standar deviasi 1.


```
from sklearn.preprocessing import StandardScaler
```


```
scaler = StandardScaler()

column_names = df_for_clustering.columns.tolist()
df_for_clustering[column_names] = scaler.fit_transform(df_for_clustering[column_names])
df_for_clustering.head()


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
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.738999</td>
      <td>-0.434801</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.738999</td>
      <td>1.195704</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.700830</td>
      <td>-1.715913</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.700830</td>
      <td>1.040418</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.662660</td>
      <td>-0.395980</td>
    </tr>
  </tbody>
</table>
</div>



## Mencari Jumlah Klaster Terbaik


### Dengan Elbow Method


```
array_for_clustering = df_for_clustering.to_numpy()


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(array_for_clustering)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss,'o')
plt.plot(range(1, 11), wcss)
plt.xticks(ticks=np.arange(0,10,1))
plt.title('Dengan Elbow Method')
plt.xlabel('Jumlah k Klaster')
plt.ylabel('WCSS')
plt.grid()
plt.show()
```


![png](/images/mall/Mall%20Customer%20Segmentation_47_0.png)


Pemilihan jumlah k klaster dengan elbow method menunjukkan hubungan antara jumlah k klaster dengan WCSS (within cluster sum of squares). WCSS, mendefenisikan jarak sum of squares antara setiap klaster dengan centroidnya. WCSS akan terus menurun tajam hingga pada titik ideal, kurvanya mulai melambat penurunannya.

Dari k = 1 terus menurun dengan tajam hingga akhirnya penurunannya melambat pada k = 3, tetapi melambat lagi lagi pada k = 5.

Jadi yang berpotensi sebagai k ideal adalah 3 atau 5.

Karena jumlah klaster yang paling ideal hanya satu, maka dilanjutkan dengan silhoutte method. Nilai koefisien silhoutte paling tinggi, maka itulah jumlah k yang paling ideal.



### Dengan Silhoutte Method


```
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(array_for_clustering)
    label = kmeans.labels_
    sil_coeff = silhouette_score(array_for_clustering, label, metric='euclidean')
    print('For n_clusters={}, The Silhouette Coefficient is {}'.format(n_cluster, sil_coeff))
```

    For n_clusters=2, The Silhouette Coefficient is 0.28640637225274423
    For n_clusters=3, The Silhouette Coefficient is 0.46658474419000145
    For n_clusters=4, The Silhouette Coefficient is 0.4939069237513199
    For n_clusters=5, The Silhouette Coefficient is 0.5546571631111091
    For n_clusters=6, The Silhouette Coefficient is 0.5398800926790663
    For n_clusters=7, The Silhouette Coefficient is 0.5263454490712252
    For n_clusters=8, The Silhouette Coefficient is 0.4541279523637649
    For n_clusters=9, The Silhouette Coefficient is 0.4527137182637434
    For n_clusters=10, The Silhouette Coefficient is 0.4408047020350327


## Memodelkan K-Means

### Training K-Means

Dengan k klaster = 5:


```
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster = kmeans.fit_predict(array_for_clustering)
```

### Visualisasi Klaster


```
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df_for_clustering)
plt.scatter(array_for_clustering[cluster == 0, 0], array_for_clustering[cluster == 0, 1], s = 50, label = 'Cluster 1')
plt.scatter(array_for_clustering[cluster == 1, 0], array_for_clustering[cluster == 1, 1], s = 50, label = 'Cluster 2')
plt.scatter(array_for_clustering[cluster == 2, 0], array_for_clustering[cluster == 2, 1], s = 50, label = 'Cluster 3')
plt.scatter(array_for_clustering[cluster == 3, 0], array_for_clustering[cluster == 3, 1], s = 50, label = 'Cluster 4')
plt.scatter(array_for_clustering[cluster == 4, 0], array_for_clustering[cluster == 4, 1], s = 50, label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=200,marker='s', alpha=0.7, label='Centroids')
plt.title('Segmen Pengunjung Mall')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(bbox_to_anchor=(1.0, 1.03));
```


![png](/images/mall/Mall%20Customer%20Segmentation_56_0.png)



```
df_cluster_1 = pd.DataFrame(data=scaler.inverse_transform(array_for_clustering[cluster == 0]),columns=['Annual Income (k$)', 'Spending Score (1-100)'])
df_cluster_2 = pd.DataFrame(data=scaler.inverse_transform(array_for_clustering[cluster == 1]),columns=['Annual Income (k$)', 'Spending Score (1-100)'])
df_cluster_3 = pd.DataFrame(data=scaler.inverse_transform(array_for_clustering[cluster == 2]),columns=['Annual Income (k$)', 'Spending Score (1-100)'])
df_cluster_4 = pd.DataFrame(data=scaler.inverse_transform(array_for_clustering[cluster == 3]),columns=['Annual Income (k$)', 'Spending Score (1-100)'])
df_cluster_5 = pd.DataFrame(data=scaler.inverse_transform(array_for_clustering[cluster == 4]),columns=['Annual Income (k$)', 'Spending Score (1-100)'])
```


```

```

# Analisis Klaster


```
print(df_cluster_1.shape)
print(df_cluster_2.shape)
print(df_cluster_3.shape)
print(df_cluster_4.shape)
print(df_cluster_5.shape)
```

    (35, 2)
    (22, 2)
    (81, 2)
    (39, 2)
    (23, 2)



```
sns.scatterplot(data=df_cluster_1, x='Annual Income (k$)',y='Spending Score (1-100)',label='Cluster 1')
sns.scatterplot(data=df_cluster_2, x='Annual Income (k$)',y='Spending Score (1-100)',label='Cluster 2')
sns.scatterplot(data=df_cluster_3, x='Annual Income (k$)',y='Spending Score (1-100)',label='Cluster 3')
sns.scatterplot(data=df_cluster_4, x='Annual Income (k$)',y='Spending Score (1-100)',label='Cluster 4')
sns.scatterplot(data=df_cluster_5, x='Annual Income (k$)',y='Spending Score (1-100)',label='Cluster 5')
plt.title('Segmen Pengunjung Mall')
plt.legend(bbox_to_anchor=(1.3, 1.03));
```


![png](/images/mall/Mall%20Customer%20Segmentation_61_0.png)


## Profiling Klaster

### Cluster 1


```
print('Deskripsi Cluster 1')
df_cluster_1.describe().transpose()
```

    Deskripsi Cluster 1





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
      <th>Annual Income (k$)</th>
      <td>35.0</td>
      <td>88.200000</td>
      <td>16.399067</td>
      <td>70.0</td>
      <td>77.5</td>
      <td>85.0</td>
      <td>97.5</td>
      <td>137.0</td>
    </tr>
    <tr>
      <th>Spending Score (1-100)</th>
      <td>35.0</td>
      <td>17.114286</td>
      <td>9.952154</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>16.0</td>
      <td>23.5</td>
      <td>39.0</td>
    </tr>
  </tbody>
</table>
</div>




```
plt.figure(figsize=(20,4))
for i in range(len(cont_feats[1:])):
  plt.subplot(1, 2, i+1)
  plt.title("Distribusi Fitur "+cont_feats[1:][i])
  sns.distplot(df_cluster_1[cont_feats[1:][i]],bins=bins)
  plt.grid(True)
 # plt.legend();
plt.tight_layout()
```


![png](/images/mall/Mall%20Customer%20Segmentation_65_0.png)


**Cluster 1**, Annual Income secara average berada di angka $ 85k, dan spending score banyak berada pada angka 17.11. Artinya, pada cluster ini memiliki pemasukkan yang tinggi, dan dengan pengeluaran yang kecil. Terbukti dengan angka spending score yang rendah. Dapat disebut sebagai cluster dengan para customer yang sangat **hemat**.

### Cluster 2


```
print('Deskripsi Cluster 2')
df_cluster_2.describe().transpose()
```

    Deskripsi Cluster 2





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
      <th>Annual Income (k$)</th>
      <td>22.0</td>
      <td>25.727273</td>
      <td>7.566731</td>
      <td>15.0</td>
      <td>19.25</td>
      <td>24.5</td>
      <td>32.25</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>Spending Score (1-100)</th>
      <td>22.0</td>
      <td>79.363636</td>
      <td>10.504174</td>
      <td>61.0</td>
      <td>73.00</td>
      <td>77.0</td>
      <td>85.75</td>
      <td>99.0</td>
    </tr>
  </tbody>
</table>
</div>




```
plt.figure(figsize=(20,4))
for i in range(len(cont_feats[1:])):
  plt.subplot(1, 2, i+1)
  plt.title("Distribusi Fitur "+cont_feats[1:][i])
  sns.distplot(df_cluster_2[cont_feats[1:][i]],bins=bins)
  plt.grid(True)
 # plt.legend();
plt.tight_layout()
```


![png](/images/mall/Mall%20Customer%20Segmentation_69_0.png)


**Cluster 2**, Annual Income secara average berada di angka sekitar $ 25k, dan spending score didominasi pada angka 73. Artinya, pada cluster ini memiliki pemasukkan terbilang rendah, dan dengan pengeluaran yang besar. Cluster ini dapat disebut sebagai cluster paling **boros**.

### Cluster 3


```
print('Deskripsi Cluster 3')
df_cluster_3.describe().transpose()
```

    Deskripsi Cluster 3





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
      <th>Annual Income (k$)</th>
      <td>81.0</td>
      <td>55.296296</td>
      <td>8.988109</td>
      <td>39.0</td>
      <td>48.0</td>
      <td>54.0</td>
      <td>62.0</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>Spending Score (1-100)</th>
      <td>81.0</td>
      <td>49.518519</td>
      <td>6.530909</td>
      <td>34.0</td>
      <td>44.0</td>
      <td>50.0</td>
      <td>55.0</td>
      <td>61.0</td>
    </tr>
  </tbody>
</table>
</div>




```
plt.figure(figsize=(20,4))
for i in range(len(cont_feats[1:])):
  plt.subplot(1, 2, i+1)
  plt.title("Distribusi Fitur "+cont_feats[1:][i])
  sns.distplot(df_cluster_3[cont_feats[1:][i]],bins=bins)
  plt.grid(True)
 # plt.legend();
plt.tight_layout()
```


![png](/images/mall/Mall%20Customer%20Segmentation_73_0.png)


**Cluster 3**, Annual Income secara average berada di angka sekitar $ 55k, dan spending score secara rata-rata berada pada angka 49. Artinya, pada cluster ini memiliki pemasukkan menengah, dan dengan pengeluaran menengah pula. Cluster ini dapat disebut sebagai cluster yang **normal**.

### Cluster 4


```
print('Deskripsi Cluster 4')
df_cluster_4.describe().transpose()
```

    Deskripsi Cluster 4





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
      <th>Annual Income (k$)</th>
      <td>39.0</td>
      <td>86.538462</td>
      <td>16.312485</td>
      <td>69.0</td>
      <td>75.5</td>
      <td>79.0</td>
      <td>95.0</td>
      <td>137.0</td>
    </tr>
    <tr>
      <th>Spending Score (1-100)</th>
      <td>39.0</td>
      <td>82.128205</td>
      <td>9.364489</td>
      <td>63.0</td>
      <td>74.5</td>
      <td>83.0</td>
      <td>90.0</td>
      <td>97.0</td>
    </tr>
  </tbody>
</table>
</div>




```
plt.figure(figsize=(20,4))
for i in range(len(cont_feats[1:])):
  plt.subplot(1, 2, i+1)
  plt.title("Distribusi Fitur "+cont_feats[1:][i])
  sns.distplot(df_cluster_4[cont_feats[1:][i]],bins=bins)
  plt.grid(True)
 # plt.legend();
plt.tight_layout()
```


![png](/images/mall/Mall%20Customer%20Segmentation_77_0.png)


**Cluster 4**, Annual Income secara average berada di angka sekitar $ 79k, dan spending score secara rata-rata berada pada angka sekitar 82. Artinya, pada cluster ini memiliki pemasukkan tinggi, dan dengan pengeluaran tinggi pula. Cluster ini juga dapat dinamakan **normal tinggi**.

### Cluster 5


```
print('Deskripsi Cluster 5')
df_cluster_5.describe().transpose()
```

    Deskripsi Cluster 5





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
      <th>Annual Income (k$)</th>
      <td>23.0</td>
      <td>26.304348</td>
      <td>7.893811</td>
      <td>15.0</td>
      <td>19.5</td>
      <td>25.0</td>
      <td>33.0</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>Spending Score (1-100)</th>
      <td>23.0</td>
      <td>20.913043</td>
      <td>13.017167</td>
      <td>3.0</td>
      <td>9.5</td>
      <td>17.0</td>
      <td>33.5</td>
      <td>40.0</td>
    </tr>
  </tbody>
</table>
</div>




```
plt.figure(figsize=(20,4))
for i in range(len(cont_feats[1:])):
  plt.subplot(1, 2, i+1)
  plt.title("Distribusi Fitur "+cont_feats[1:][i])
  sns.distplot(df_cluster_5[cont_feats[1:][i]],bins=bins)
  plt.grid(True)
 # plt.legend();
plt.tight_layout()
```


![png](/images/mall/Mall%20Customer%20Segmentation_81_0.png)


**Cluster 5**, Annual Income secara average berada di angka sekitar $ 25k, dan spending score secara rata-rata berada pada angka 20.91. Artinya, pada cluster ini memiliki pemasukkan rendah, dan dengan pengeluaran yang terbilang rendah pula. Cluster ini juga dapat dinamakan **normal rendah**.

## Nama-Nama Klaster Final
Dari klasterisasi dan profiling masing-masing klaster di atas, di dapatkan nama-nama klaster yang lebih representatif dengan deskripsi yang sudah dijabarkan sebelumnya.

Klaster-klaster tersebut adalah


1.   Hemat
2.   Boros
3. Normal
4. Normal Tinggi
5. Normal Rendah

Jika divisualisasikan sebarannya:




```
sns.scatterplot(data=df_cluster_1, x='Annual Income (k$)',y='Spending Score (1-100)',label='Hemat')
sns.scatterplot(data=df_cluster_2, x='Annual Income (k$)',y='Spending Score (1-100)',label='Boros')
sns.scatterplot(data=df_cluster_3, x='Annual Income (k$)',y='Spending Score (1-100)',label='Normal')
sns.scatterplot(data=df_cluster_4, x='Annual Income (k$)',y='Spending Score (1-100)',label='Normal Tinggi')
sns.scatterplot(data=df_cluster_5, x='Annual Income (k$)',y='Spending Score (1-100)',label='Normal Rendah')
plt.title('Segmen Pengunjung Mall')
plt.legend(bbox_to_anchor=(1., 1.03));
```


![png](/images/mall/Mall%20Customer%20Segmentation_84_0.png)


Mohon disertakan *feedback*-nya.
