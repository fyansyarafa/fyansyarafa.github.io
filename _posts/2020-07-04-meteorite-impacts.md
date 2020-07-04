---
title: "Meteorite Impacts"
data: 2020-07-04
tags: [python,  exploratory data analysis, statistics, probability]
header:
excerpt: "Memprediksi apakah ada kemungkinan dalam 1000 tahun, sebuah meteor besar berdiameter >= 1km akan menghantam bumi?"
mathjax: "true"
---


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Meteorite_Landings.csv")
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 45716 entries, 0 to 45715
    Data columns (total 10 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   name         45716 non-null  object
     1   id           45716 non-null  int64  
     2   nametype     45716 non-null  object
     3   recclass     45716 non-null  object
     4   mass         45585 non-null  float64
     5   fall         45716 non-null  object
     6   year         45428 non-null  float64
     7   reclat       38401 non-null  float64
     8   reclong      38401 non-null  float64
     9   GeoLocation  38401 non-null  object
    dtypes: float64(4), int64(1), object(5)
    memory usage: 3.5+ MB



```python
sns.heatmap(df.isnull())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18a3cc0e888>




![png](\images\aste\output_2_1.png)


Terdapat *null values* pada kolom mass, year, reclat, reclong, dan Geolocation


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
      <th>name</th>
      <th>id</th>
      <th>nametype</th>
      <th>recclass</th>
      <th>mass</th>
      <th>fall</th>
      <th>year</th>
      <th>reclat</th>
      <th>reclong</th>
      <th>GeoLocation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aachen</td>
      <td>1</td>
      <td>Valid</td>
      <td>L5</td>
      <td>21.0</td>
      <td>Fell</td>
      <td>1880.0</td>
      <td>50.77500</td>
      <td>6.08333</td>
      <td>(50.775000, 6.083330)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aarhus</td>
      <td>2</td>
      <td>Valid</td>
      <td>H6</td>
      <td>720.0</td>
      <td>Fell</td>
      <td>1951.0</td>
      <td>56.18333</td>
      <td>10.23333</td>
      <td>(56.183330, 10.233330)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Abee</td>
      <td>6</td>
      <td>Valid</td>
      <td>EH4</td>
      <td>107000.0</td>
      <td>Fell</td>
      <td>1952.0</td>
      <td>54.21667</td>
      <td>-113.00000</td>
      <td>(54.216670, -113.000000)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Acapulco</td>
      <td>10</td>
      <td>Valid</td>
      <td>Acapulcoite</td>
      <td>1914.0</td>
      <td>Fell</td>
      <td>1976.0</td>
      <td>16.88333</td>
      <td>-99.90000</td>
      <td>(16.883330, -99.900000)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Achiras</td>
      <td>370</td>
      <td>Valid</td>
      <td>L6</td>
      <td>780.0</td>
      <td>Fell</td>
      <td>1902.0</td>
      <td>-33.16667</td>
      <td>-64.95000</td>
      <td>(-33.166670, -64.950000)</td>
    </tr>
  </tbody>
</table>
</div>



## Investigate and explore Data


```python
# menyingkirkan null values pada kolom yang ingin dieksplor (mass, year)
df2 = df.dropna(subset=['mass', 'year'])
```


```python
df2 =df2[df2['mass'] > 0]
```


```python
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 45292 entries, 0 to 45715
    Data columns (total 10 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   name         45292 non-null  object
     1   id           45292 non-null  int64  
     2   nametype     45292 non-null  object
     3   recclass     45292 non-null  object
     4   mass         45292 non-null  float64
     5   fall         45292 non-null  object
     6   year         45292 non-null  float64
     7   reclat       38097 non-null  float64
     8   reclong      38097 non-null  float64
     9   GeoLocation  38097 non-null  object
    dtypes: float64(4), int64(1), object(5)
    memory usage: 3.8+ MB



```python
#sns.set(rc={'figure.figsize':(7,7)})
sns.pairplot(df2[['mass', 'year', 'reclat', 'reclong']])

```




    <seaborn.axisgrid.PairGrid at 0x18a3ff50948>




![png](/images/aste/output_9_1.png)


Konversi mass ke log agar lebih mudah dilihat distribusinya


```python
year = df2['year']
masses = df2['mass']
logmass = np.log(masses)
a=sns.distplot(logmass, bins=100, kde=False);
```


![png](/images/aste/output_11_0.png)



```python
fig, axes = plt.subplots(ncols=2,figsize=(10,4))
fig.suptitle("Perbedaaan sebelum dan sesudah logmass")
axes[0].hist(df['mass'])
axes[0].set_title("Sebelum")
axes[1].hist(logmass,bins=100)
axes[1].set_title("Sesudah");
```

    C:\Users\abulu\anaconda3\lib\site-packages\numpy\lib\histograms.py:839: RuntimeWarning: invalid value encountered in greater_equal
      keep = (tmp_a >= first_edge)
    C:\Users\abulu\anaconda3\lib\site-packages\numpy\lib\histograms.py:840: RuntimeWarning: invalid value encountered in less_equal
      keep &= (tmp_a <= last_edge)



![png](/images/aste/output_12_1.png)


### Peristiwa berdasarkan tahun (1980-2020)


```python
#membatasi event berdasarkan tahun
counts = year.value_counts()
sns.scatterplot(x=counts.index, y=counts)
plt.xlim(1980, 2020)
plt.ylabel("")
```




    Text(0, 0.5, '')




![png](/images/aste/output_14_1.png)



```python
counts.describe()
```




    count     254.000000
    mean      178.314961
    std       514.449561
    min         1.000000
    25%         3.000000
    50%        11.000000
    75%        23.000000
    max      3322.000000
    Name: year, dtype: float64



### Melakukan aproksimasi distribusi


```python
from scipy.stats import norm, skewnorm, lognorm
ms = np.linspace(-5,20,100)

mean, std = logmass.mean(), np.std(logmass)
sns.distplot(logmass, bins=100, label="Data")

#normal
pdf_norm = norm.pdf(ms, mean, std)
plt.plot(ms, pdf_norm, label='Normal Dist')

#skewnorm
p_skewnorm = skewnorm.fit(logmass)
pdf_skewnorm = skewnorm.pdf(ms, *p_skewnorm)
plt.plot(ms, pdf_skewnorm,label='Skewnorm Dist')

#lognorm
p_lognorm = lognorm.fit(logmass)
pdf_lognorm = lognorm.pdf(ms, *p_lognorm)
plt.plot(ms, pdf_lognorm, label="Lognorm Dist")

plt.legend();
```


![png](/images/aste/output_17_0.png)


Skewnorm dan Lognorm lebih mendekati data daripada normal dist. Tetapi, akan diproses dengan pendekatan lognorm dist


```python
# konversi diameter 1 km ke pendekatan logmass
mass_of_doom = np.log((4/3 * np.pi * 500**3 * 1600 * 1000))
print("Mass setelah konversi = ",mass_of_doom)
```

    Mass setelah konversi =  34.36175044077777



```python
ms2 = np.linspace(-5, 40, 200)
plt.plot(ms2, lognorm.logsf(ms2, *p_lognorm))
plt.axvline(mass_of_doom, ls="--")
plt.xlabel("log mass")
plt.ylabel("log probability")
plt.title("Log probability asteroid dengan mass yang diberikan");
```


![png](/images/aste/output_20_0.png)


## Menghitung Probabilitas

Menghitung probabilitas asteroid dengan diameter lebih dari 1 km dalam jangka waktu 1000 tahun ke depan, dengan asumsi pengamatan hanya dengan 20% dari asteroid impact yang sebenarnya (termasuk yang tidak terkover oleh dataset)


```python
prob_mass = lognorm.logcdf(mass_of_doom, *p_lognorm)
monitorized = 0.2
num_years = 1000
num_events = num_years * counts.max() / monitorized #20% dari worst case peristiwa asteroid impact (peristiwa terbanyak) dalam seribu tahun
prob = 1-np.exp(num_events * prob_mass)
print(f"Prob a > 1 km asteroid impacts dalam {num_years} tahun ke depan adalah {prob * 100:.2f}%")
```

    Prob a > 1 km asteroid impacts dalam 1000 tahun ke depan adalah 0.67%
