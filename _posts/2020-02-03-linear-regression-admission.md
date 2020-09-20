---

title: "Linear Regression Admission"
data: 2020-02-03
tags: [python, linear regression, exploratory data analysis, machine learning, template]
header:
excerpt: "Linear Regression Exercise Template"
mathjax: "true"
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df = pd.read_csv('Admission_Predict.csv')
```


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
      <th>Serial No.</th>
      <th>GRE Score</th>
      <th>TOEFL Score</th>
      <th>University Rating</th>
      <th>SOP</th>
      <th>LOR</th>
      <th>CGPA</th>
      <th>Research</th>
      <th>Chance of Admit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>337</td>
      <td>118</td>
      <td>4</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>9.65</td>
      <td>1</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>324</td>
      <td>107</td>
      <td>4</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>8.87</td>
      <td>1</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>316</td>
      <td>104</td>
      <td>3</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>8.00</td>
      <td>1</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>322</td>
      <td>110</td>
      <td>3</td>
      <td>3.5</td>
      <td>2.5</td>
      <td>8.67</td>
      <td>1</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>314</td>
      <td>103</td>
      <td>2</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>8.21</td>
      <td>0</td>
      <td>0.65</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 400 entries, 0 to 399
    Data columns (total 9 columns):
    Serial No.           400 non-null int64
    GRE Score            400 non-null int64
    TOEFL Score          400 non-null int64
    University Rating    400 non-null int64
    SOP                  400 non-null float64
    LOR                  400 non-null float64
    CGPA                 400 non-null float64
    Research             400 non-null int64
    Chance of Admit      400 non-null float64
    dtypes: float64(4), int64(5)
    memory usage: 28.2 KB



```python
df.describe()
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
      <th>Serial No.</th>
      <th>GRE Score</th>
      <th>TOEFL Score</th>
      <th>University Rating</th>
      <th>SOP</th>
      <th>LOR</th>
      <th>CGPA</th>
      <th>Research</th>
      <th>Chance of Admit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>200.500000</td>
      <td>316.807500</td>
      <td>107.410000</td>
      <td>3.087500</td>
      <td>3.400000</td>
      <td>3.452500</td>
      <td>8.598925</td>
      <td>0.547500</td>
      <td>0.724350</td>
    </tr>
    <tr>
      <th>std</th>
      <td>115.614301</td>
      <td>11.473646</td>
      <td>6.069514</td>
      <td>1.143728</td>
      <td>1.006869</td>
      <td>0.898478</td>
      <td>0.596317</td>
      <td>0.498362</td>
      <td>0.142609</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>290.000000</td>
      <td>92.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>6.800000</td>
      <td>0.000000</td>
      <td>0.340000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>100.750000</td>
      <td>308.000000</td>
      <td>103.000000</td>
      <td>2.000000</td>
      <td>2.500000</td>
      <td>3.000000</td>
      <td>8.170000</td>
      <td>0.000000</td>
      <td>0.640000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>200.500000</td>
      <td>317.000000</td>
      <td>107.000000</td>
      <td>3.000000</td>
      <td>3.500000</td>
      <td>3.500000</td>
      <td>8.610000</td>
      <td>1.000000</td>
      <td>0.730000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>300.250000</td>
      <td>325.000000</td>
      <td>112.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>9.062500</td>
      <td>1.000000</td>
      <td>0.830000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>400.000000</td>
      <td>340.000000</td>
      <td>120.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>9.920000</td>
      <td>1.000000</td>
      <td>0.970000</td>
    </tr>
  </tbody>
</table>
</div>




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
      <th>Serial No.</th>
      <th>GRE Score</th>
      <th>TOEFL Score</th>
      <th>University Rating</th>
      <th>SOP</th>
      <th>LOR</th>
      <th>CGPA</th>
      <th>Research</th>
      <th>Chance of Admit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>337</td>
      <td>118</td>
      <td>4</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>9.65</td>
      <td>1</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>324</td>
      <td>107</td>
      <td>4</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>8.87</td>
      <td>1</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>316</td>
      <td>104</td>
      <td>3</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>8.00</td>
      <td>1</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>322</td>
      <td>110</td>
      <td>3</td>
      <td>3.5</td>
      <td>2.5</td>
      <td>8.67</td>
      <td>1</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>314</td>
      <td>103</td>
      <td>2</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>8.21</td>
      <td>0</td>
      <td>0.65</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
           'LOR ', 'CGPA', 'Research', 'Chance of Admit '],
          dtype='object')




```python
X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research', ]]
y = df['Chance of Admit ']
```


```python
X
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
      <th>GRE Score</th>
      <th>TOEFL Score</th>
      <th>University Rating</th>
      <th>SOP</th>
      <th>LOR</th>
      <th>CGPA</th>
      <th>Research</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>337</td>
      <td>118</td>
      <td>4</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>9.65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>324</td>
      <td>107</td>
      <td>4</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>8.87</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>316</td>
      <td>104</td>
      <td>3</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>8.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>322</td>
      <td>110</td>
      <td>3</td>
      <td>3.5</td>
      <td>2.5</td>
      <td>8.67</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>314</td>
      <td>103</td>
      <td>2</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>8.21</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>395</th>
      <td>324</td>
      <td>110</td>
      <td>3</td>
      <td>3.5</td>
      <td>3.5</td>
      <td>9.04</td>
      <td>1</td>
    </tr>
    <tr>
      <th>396</th>
      <td>325</td>
      <td>107</td>
      <td>3</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>9.11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>397</th>
      <td>330</td>
      <td>116</td>
      <td>4</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>9.45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>398</th>
      <td>312</td>
      <td>103</td>
      <td>3</td>
      <td>3.5</td>
      <td>4.0</td>
      <td>8.78</td>
      <td>0</td>
    </tr>
    <tr>
      <th>399</th>
      <td>333</td>
      <td>117</td>
      <td>4</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>9.66</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 7 columns</p>
</div>




```python
sns.set()
sns.set_style('darkgrid')
sns.pairplot(df[['GRE Score', 'TOEFL Score','University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research', 'Chance of Admit ']])
```




    <seaborn.axisgrid.PairGrid at 0x225c69d5860>




![png](/images/output_9_1.png)



```python
sns.distplot(df['Chance of Admit '])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x225c6733be0>




![png](/images/output_10_1.png)



```python
ae = sns.heatmap(df[['GRE Score', 'TOEFL Score','University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research', 'Chance of Admit ']].corr(),annot=True)


bottom, top = ae.get_ylim()
ae.set_ylim(bottom + 0.5, top - 0.5)
```




    (8.0, 0.0)




![png](/images/output_11_1.png)



```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
```


```python
#import sklearn for linear regression
from sklearn.linear_model import LinearRegression
```


```python
lm = LinearRegression()
```


```python
lm.fit(X_train,y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
lm.intercept_
```




    -1.400807751325246




```python
lm.coef_
```




    array([ 0.0023919 ,  0.00308291,  0.01130127, -0.00185234,  0.01878345,
            0.10911333,  0.00969571])




```python
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['coef'])
```


```python
cdf
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
      <th>coef</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRE Score</th>
      <td>0.002392</td>
    </tr>
    <tr>
      <th>TOEFL Score</th>
      <td>0.003083</td>
    </tr>
    <tr>
      <th>University Rating</th>
      <td>0.011301</td>
    </tr>
    <tr>
      <th>SOP</th>
      <td>-0.001852</td>
    </tr>
    <tr>
      <th>LOR</th>
      <td>0.018783</td>
    </tr>
    <tr>
      <th>CGPA</th>
      <td>0.109113</td>
    </tr>
    <tr>
      <th>Research</th>
      <td>0.009696</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_pred = lm.predict(X_test)
```


```python
y_test
```




    38     0.52
    387    0.53
    270    0.72
    181    0.71
    195    0.78
           ...
    32     0.91
    261    0.71
    218    0.84
    70     0.94
    198    0.70
    Name: Chance of Admit , Length: 132, dtype: float64




```python
y_pred
```




    array([0.50230492, 0.62566912, 0.63574864, 0.64225778, 0.6855059 ,
           0.69702514, 0.80869952, 0.48953549, 0.86106051, 0.81055544,
           0.7936463 , 0.69282816, 0.89176355, 0.98649085, 0.57691989,
           0.7511946 , 0.69818895, 0.66350935, 0.50058284, 0.64962916,
           0.56625023, 0.65748297, 0.76681879, 0.61100155, 0.4629894 ,
           0.80518   , 0.76152748, 0.71687004, 0.51904997, 0.81977998,
           0.72521272, 0.68225313, 0.5446814 , 0.89117383, 0.51957982,
           0.57390105, 0.67269071, 0.58293808, 0.71698482, 0.71439432,
           0.62751974, 0.85893827, 0.89854122, 0.75921005, 0.71360867,
           0.95852053, 0.6260447 , 0.73067369, 0.54733203, 0.69772191,
           0.59705433, 0.90890356, 0.78694594, 0.52535155, 0.74086589,
           0.742621  , 0.73574015, 0.61101331, 0.63885736, 0.53936848,
           0.51471195, 0.69583975, 0.56347772, 0.53960502, 0.72538656,
           0.59887931, 0.67913592, 0.64082323, 0.51591045, 0.66482107,
           0.49604521, 0.79960065, 0.80279891, 0.74197876, 0.6539744 ,
           0.96976253, 0.63498999, 0.8680921 , 0.6500095 , 0.65123062,
           0.65444408, 0.80763143, 0.49920805, 0.88480745, 0.85783461,
           0.72463918, 0.68947392, 0.61708719, 0.77562293, 0.56311896,
           0.65703485, 0.59332779, 0.85944737, 0.88356262, 0.52293313,
           0.61977665, 0.55387268, 0.80786278, 0.63516298, 0.76213771,
           0.58756783, 0.57945467, 0.7089873 , 0.72687596, 0.60504254,
           0.85396974, 0.45904501, 0.81063567, 0.91790067, 0.85443197,
           0.71942681, 0.69085325, 0.64803786, 0.77601242, 0.96358245,
           0.74335157, 0.64286729, 0.67536081, 0.81881879, 0.72332998,
           0.89211447, 0.85779512, 0.72786642, 0.86108163, 0.64809366,
           0.71584801, 0.86796778, 0.93097254, 0.65136906, 0.80712048,
           0.95979663, 0.69361507])




```python
tb = pd.DataFrame(y_test)
```


```python
tb['pred'] = y_pred

tb
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
      <th>Chance of Admit</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38</th>
      <td>0.52</td>
      <td>0.502305</td>
    </tr>
    <tr>
      <th>387</th>
      <td>0.53</td>
      <td>0.625669</td>
    </tr>
    <tr>
      <th>270</th>
      <td>0.72</td>
      <td>0.635749</td>
    </tr>
    <tr>
      <th>181</th>
      <td>0.71</td>
      <td>0.642258</td>
    </tr>
    <tr>
      <th>195</th>
      <td>0.78</td>
      <td>0.685506</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.91</td>
      <td>0.930973</td>
    </tr>
    <tr>
      <th>261</th>
      <td>0.71</td>
      <td>0.651369</td>
    </tr>
    <tr>
      <th>218</th>
      <td>0.84</td>
      <td>0.807120</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.94</td>
      <td>0.959797</td>
    </tr>
    <tr>
      <th>198</th>
      <td>0.70</td>
      <td>0.693615</td>
    </tr>
  </tbody>
</table>
<p>132 rows × 2 columns</p>
</div>




```python
plt.scatter(y_test,y_pred)
```




    <matplotlib.collections.PathCollection at 0x225cc0c80f0>




![png](/images/output_26_1.png)



```python
sns.distplot((y_test-y_pred))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x225cc521f28>




![png](/images/output_27_1.png)



```python
from sklearn import metrics
print(metrics.mean_absolute_error(y_test,y_pred))
print(metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
```

    0.04725499639197657
    0.0047063181924947356
    0.06860261068279207



```python
sns.lmplot(x='Chance of Admit ',y='pred',data=tb)
```




    <seaborn.axisgrid.FacetGrid at 0x225ccdaf080>




![png](/images/output_29_1.png)



```python
tb
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
      <th>Chance of Admit</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38</th>
      <td>0.52</td>
      <td>0.502305</td>
    </tr>
    <tr>
      <th>387</th>
      <td>0.53</td>
      <td>0.625669</td>
    </tr>
    <tr>
      <th>270</th>
      <td>0.72</td>
      <td>0.635749</td>
    </tr>
    <tr>
      <th>181</th>
      <td>0.71</td>
      <td>0.642258</td>
    </tr>
    <tr>
      <th>195</th>
      <td>0.78</td>
      <td>0.685506</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.91</td>
      <td>0.930973</td>
    </tr>
    <tr>
      <th>261</th>
      <td>0.71</td>
      <td>0.651369</td>
    </tr>
    <tr>
      <th>218</th>
      <td>0.84</td>
      <td>0.807120</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.94</td>
      <td>0.959797</td>
    </tr>
    <tr>
      <th>198</th>
      <td>0.70</td>
      <td>0.693615</td>
    </tr>
  </tbody>
</table>
<p>132 rows × 2 columns</p>
</div>




```python
y_test
```




    38     0.52
    387    0.53
    270    0.72
    181    0.71
    195    0.78
           ...
    32     0.91
    261    0.71
    218    0.84
    70     0.94
    198    0.70
    Name: Chance of Admit , Length: 132, dtype: float64




```python
y_pred
```




    array([0.50230492, 0.62566912, 0.63574864, 0.64225778, 0.6855059 ,
           0.69702514, 0.80869952, 0.48953549, 0.86106051, 0.81055544,
           0.7936463 , 0.69282816, 0.89176355, 0.98649085, 0.57691989,
           0.7511946 , 0.69818895, 0.66350935, 0.50058284, 0.64962916,
           0.56625023, 0.65748297, 0.76681879, 0.61100155, 0.4629894 ,
           0.80518   , 0.76152748, 0.71687004, 0.51904997, 0.81977998,
           0.72521272, 0.68225313, 0.5446814 , 0.89117383, 0.51957982,
           0.57390105, 0.67269071, 0.58293808, 0.71698482, 0.71439432,
           0.62751974, 0.85893827, 0.89854122, 0.75921005, 0.71360867,
           0.95852053, 0.6260447 , 0.73067369, 0.54733203, 0.69772191,
           0.59705433, 0.90890356, 0.78694594, 0.52535155, 0.74086589,
           0.742621  , 0.73574015, 0.61101331, 0.63885736, 0.53936848,
           0.51471195, 0.69583975, 0.56347772, 0.53960502, 0.72538656,
           0.59887931, 0.67913592, 0.64082323, 0.51591045, 0.66482107,
           0.49604521, 0.79960065, 0.80279891, 0.74197876, 0.6539744 ,
           0.96976253, 0.63498999, 0.8680921 , 0.6500095 , 0.65123062,
           0.65444408, 0.80763143, 0.49920805, 0.88480745, 0.85783461,
           0.72463918, 0.68947392, 0.61708719, 0.77562293, 0.56311896,
           0.65703485, 0.59332779, 0.85944737, 0.88356262, 0.52293313,
           0.61977665, 0.55387268, 0.80786278, 0.63516298, 0.76213771,
           0.58756783, 0.57945467, 0.7089873 , 0.72687596, 0.60504254,
           0.85396974, 0.45904501, 0.81063567, 0.91790067, 0.85443197,
           0.71942681, 0.69085325, 0.64803786, 0.77601242, 0.96358245,
           0.74335157, 0.64286729, 0.67536081, 0.81881879, 0.72332998,
           0.89211447, 0.85779512, 0.72786642, 0.86108163, 0.64809366,
           0.71584801, 0.86796778, 0.93097254, 0.65136906, 0.80712048,
           0.95979663, 0.69361507])




```python
df
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
      <th>Serial No.</th>
      <th>GRE Score</th>
      <th>TOEFL Score</th>
      <th>University Rating</th>
      <th>SOP</th>
      <th>LOR</th>
      <th>CGPA</th>
      <th>Research</th>
      <th>Chance of Admit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>337</td>
      <td>118</td>
      <td>4</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>9.65</td>
      <td>1</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>324</td>
      <td>107</td>
      <td>4</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>8.87</td>
      <td>1</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>316</td>
      <td>104</td>
      <td>3</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>8.00</td>
      <td>1</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>322</td>
      <td>110</td>
      <td>3</td>
      <td>3.5</td>
      <td>2.5</td>
      <td>8.67</td>
      <td>1</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>314</td>
      <td>103</td>
      <td>2</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>8.21</td>
      <td>0</td>
      <td>0.65</td>
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
    </tr>
    <tr>
      <th>395</th>
      <td>396</td>
      <td>324</td>
      <td>110</td>
      <td>3</td>
      <td>3.5</td>
      <td>3.5</td>
      <td>9.04</td>
      <td>1</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>396</th>
      <td>397</td>
      <td>325</td>
      <td>107</td>
      <td>3</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>9.11</td>
      <td>1</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>397</th>
      <td>398</td>
      <td>330</td>
      <td>116</td>
      <td>4</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>9.45</td>
      <td>1</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>398</th>
      <td>399</td>
      <td>312</td>
      <td>103</td>
      <td>3</td>
      <td>3.5</td>
      <td>4.0</td>
      <td>8.78</td>
      <td>0</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>399</th>
      <td>400</td>
      <td>333</td>
      <td>117</td>
      <td>4</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>9.66</td>
      <td>1</td>
      <td>0.95</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 9 columns</p>
</div>




```python

```
