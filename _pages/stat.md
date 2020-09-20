---
layout: archive
permalink: /statnotebook/

author_profile: false
toc: true
search: true
---




```python

```

Exploring your data - **normally visually** - to gain insight into **relationships** in the data and to catch **data peculiarities** before impaction formal analysis

## Datasets and Data Prep

### Load the Data


```python
#importing libraries
import numpy as np
import pickle
import pandas as pd
filename = "Resources/Datasets/original/load.csv"
```

#### Manually


```python
cols = None
data = []
with open(filename) as f:
    for line in f.readlines():
        vals = line.replace("\n", "").split(",")
        if cols is None:
            cols = vals
        else:
            data.append([float(x) for x in vals])
d0 = pd.DataFrame(data, columns = cols)
print(d0.dtypes)
d0.head()
```

    A    float64
    B    float64
    C    float64
    D    float64
    E    float64
    dtype: object





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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.276</td>
      <td>21.400</td>
      <td>63.957</td>
      <td>216.204</td>
      <td>528.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.002</td>
      <td>21.950</td>
      <td>61.697</td>
      <td>204.484</td>
      <td>514.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.114</td>
      <td>22.454</td>
      <td>63.522</td>
      <td>205.608</td>
      <td>514.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.133</td>
      <td>22.494</td>
      <td>61.590</td>
      <td>206.565</td>
      <td>501.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.845</td>
      <td>21.654</td>
      <td>63.729</td>
      <td>201.289</td>
      <td>532.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Using Numpy


```python
d1 = np.loadtxt(filename, skiprows=1, delimiter=",")
print(d1.dtype)
print(d1[:5, :])
```

    float64
    [[  1.276  21.4    63.957 216.204 528.   ]
     [  1.002  21.95   61.697 204.484 514.   ]
     [  1.114  22.454  63.522 205.608 514.   ]
     [  1.133  22.494  61.59  206.565 501.   ]
     [  0.845  21.654  63.729 201.289 532.   ]]



```python
d2 = np.genfromtxt(
    filename,
    delimiter = ",",
    names = True,
    dtype = None
)
print(d2.dtype)
print(d2[:5])
```

    [('A', '<f8'), ('B', '<f8'), ('C', '<f8'), ('D', '<f8'), ('E', '<i4')]
    [(1.276, 21.4  , 63.957, 216.204, 528)
     (1.002, 21.95 , 61.697, 204.484, 514)
     (1.114, 22.454, 63.522, 205.608, 514)
     (1.133, 22.494, 61.59 , 206.565, 501)
     (0.845, 21.654, 63.729, 201.289, 532)]


#### Using pandas


```python
d3 = pd.read_csv(filename)
print(d3.dtypes)
d3.head()
```

    A    float64
    B    float64
    C    float64
    D    float64
    E      int64
    dtype: object





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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.276</td>
      <td>21.400</td>
      <td>63.957</td>
      <td>216.204</td>
      <td>528</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.002</td>
      <td>21.950</td>
      <td>61.697</td>
      <td>204.484</td>
      <td>514</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.114</td>
      <td>22.454</td>
      <td>63.522</td>
      <td>205.608</td>
      <td>514</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.133</td>
      <td>22.494</td>
      <td>61.590</td>
      <td>206.565</td>
      <td>501</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.845</td>
      <td>21.654</td>
      <td>63.729</td>
      <td>201.289</td>
      <td>532</td>
    </tr>
  </tbody>
</table>
</div>



#### Using pickle


```python
with open("Resources/Datasets/original/load_pickle.pickle","rb") as f:
    d4 = pickle.load(f)

print(d4.dtypes)
d4.head()

```

    A    float64
    B    float64
    C    float64
    D    float64
    E      int32
    dtype: object





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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.276405</td>
      <td>21.400157</td>
      <td>63.957476</td>
      <td>216.204466</td>
      <td>528</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.002272</td>
      <td>21.950088</td>
      <td>61.697286</td>
      <td>204.483906</td>
      <td>514</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.114404</td>
      <td>22.454274</td>
      <td>63.522075</td>
      <td>205.608375</td>
      <td>514</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.133367</td>
      <td>22.494079</td>
      <td>61.589683</td>
      <td>206.565339</td>
      <td>501</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.844701</td>
      <td>21.653619</td>
      <td>63.728872</td>
      <td>201.289175</td>
      <td>532</td>
    </tr>
  </tbody>
</table>
</div>



### Dataset Preparation

#### Preparing a dataset

- how does the dataset handle invalid values?
- what do we want to do with null values?
- do we want to summarise, group or filter the data?


```python
df = pd.read_csv("Resources/Datasets/Diabetes/Diabetes.csv")
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB



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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.fillna(0)
```


```python
#filtering columns
df2 = df[['Glucose', 'BMI', 'Age', 'Outcome']]
```


```python
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
      <th>Glucose</th>
      <th>BMI</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>148</td>
      <td>33.6</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85</td>
      <td>26.6</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>183</td>
      <td>23.3</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>89</td>
      <td>28.1</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>137</td>
      <td>43.1</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.describe()
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
      <th>Glucose</th>
      <th>BMI</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>120.894531</td>
      <td>31.992578</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>31.972618</td>
      <td>7.884160</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>99.000000</td>
      <td>27.300000</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>117.000000</td>
      <td>32.000000</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>140.250000</td>
      <td>36.600000</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>199.000000</td>
      <td>67.100000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



We dont want zero value at BMI entries, because it is not obvious at real world


```python
df2.columns
```




    Index(['Glucose', 'BMI', 'Age', 'Outcome'], dtype='object')



excluding zero values of BMI entries, and put them to df3 dataframe


```python
df3 = df2.loc[~(df2[df2.columns[:-1]] == 0).any(axis = 1)]
```


```python
df3
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
      <th>Glucose</th>
      <th>BMI</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>148</td>
      <td>33.6</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85</td>
      <td>26.6</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>183</td>
      <td>23.3</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>89</td>
      <td>28.1</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>137</td>
      <td>43.1</td>
      <td>33</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>763</th>
      <td>101</td>
      <td>32.9</td>
      <td>63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>764</th>
      <td>122</td>
      <td>36.8</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>765</th>
      <td>121</td>
      <td>26.2</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>126</td>
      <td>30.1</td>
      <td>47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>767</th>
      <td>93</td>
      <td>30.4</td>
      <td>23</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>752 rows × 4 columns</p>
</div>




```python
df3.describe()

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
      <th>Glucose</th>
      <th>BMI</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>752.000000</td>
      <td>752.000000</td>
      <td>752.000000</td>
      <td>752.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>121.941489</td>
      <td>32.454654</td>
      <td>33.312500</td>
      <td>0.351064</td>
    </tr>
    <tr>
      <th>std</th>
      <td>30.601198</td>
      <td>6.928926</td>
      <td>11.709395</td>
      <td>0.477621</td>
    </tr>
    <tr>
      <th>min</th>
      <td>44.000000</td>
      <td>18.200000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>99.750000</td>
      <td>27.500000</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>117.000000</td>
      <td>32.300000</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>141.000000</td>
      <td>36.600000</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>199.000000</td>
      <td>67.100000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 752 entries, 0 to 767
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   Glucose  752 non-null    int64  
     1   BMI      752 non-null    float64
     2   Age      752 non-null    int64  
     3   Outcome  752 non-null    int64  
    dtypes: float64(1), int64(3)
    memory usage: 29.4 KB



```python

```

##### Aggregating data


```python
#by mean
df3.groupby("Outcome").mean()
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
      <th>Glucose</th>
      <th>BMI</th>
      <th>Age</th>
    </tr>
    <tr>
      <th>Outcome</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>110.825820</td>
      <td>30.876434</td>
      <td>31.309426</td>
    </tr>
    <tr>
      <th>1</th>
      <td>142.488636</td>
      <td>35.371970</td>
      <td>37.015152</td>
    </tr>
  </tbody>
</table>
</div>




```python
#by multiple aggregation functions
df3.groupby("Outcome").agg({
    "Glucose" : "mean",
    "BMI" : "mean",
    "Age" : "sum"
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
      <th>Glucose</th>
      <th>BMI</th>
      <th>Age</th>
    </tr>
    <tr>
      <th>Outcome</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>110.825820</td>
      <td>30.876434</td>
      <td>15279</td>
    </tr>
    <tr>
      <th>1</th>
      <td>142.488636</td>
      <td>35.371970</td>
      <td>9772</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3.groupby("Outcome").agg([
    "mean",
    "median"
])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Glucose</th>
      <th colspan="2" halign="left">BMI</th>
      <th colspan="2" halign="left">Age</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>Outcome</th>
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
      <th>0</th>
      <td>110.825820</td>
      <td>107.5</td>
      <td>30.876434</td>
      <td>30.10</td>
      <td>31.309426</td>
      <td>27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>142.488636</td>
      <td>140.5</td>
      <td>35.371970</td>
      <td>34.25</td>
      <td>37.015152</td>
      <td>36</td>
    </tr>
  </tbody>
</table>
</div>




```python
positive = df3.loc[df3["Outcome"] == 1]
negative = df3.loc[df3["Outcome"] == 0]

print(positive.shape, negative.shape)
```

    (264, 4) (488, 4)



```python
#saving the data
df3.to_csv("cleanDiabetes.csv", index= False)
```

## Dealing with Outliers

### Outliers
Sometimes our data is not nice enough to simply a ``Nan`` or zero value to make it easy to tell what we should remove.
Sometimes our data has outliers in it. So, lets look at some strategies to identifying these points


```python
import matplotlib.pyplot as plt

```


```python
d1 = np.loadtxt("Resources/Datasets/Outliers/outlier_1d.txt")
d2 = np.loadtxt("Resources/Datasets/Outliers/outlier_2d.txt")
d3 = np.loadtxt("Resources/Datasets/Outliers/outlier_curve.txt")
print(d1.shape, d2.shape)

plt.scatter(
    x = d1,
    y = np.random.normal(
        7,
        0.2,
        size = d1.size
    ),
    s=1,
    alpha = 0.5
)
plt.scatter(
    d2[:, 0],
    d2[:, 1]
)
plt.show()
plt.plot(
    d3[:, 0],
    d3[:, 1]
)
```

    (1010,) (1010, 2)



![png](/images/stat_files/stat_40_1.png)





    [<matplotlib.lines.Line2D at 0x1f07c973608>]




![png](/images/stat_files/stat_40_3.png)



```python
print(d2)
```

    [[12.486 19.387]
     [ 8.184 23.854]
     [12.195 14.544]
     ...
     [ 8.06  20.962]
     [ 8.805 17.617]
     [ 9.798 16.666]]


### Basics
The most basic and most-common way of manually doing outlier pruning on data distributions is to:
1. Model your data as some analytic distribution
2. Find all points bellow a certain probability
3. Remove them
4. Refit the distributions, and potentially run again from step 1


```python
mean, std = np.mean(d1), np.std(d1)
z_score = np.abs((d1 - mean)/std)
threshold = 3
good = z_score < threshold

print(f"Rejection {(~good).sum()} points")
from scipy.stats import norm
print(f"z-score of 3 corresponds to a prob of {100*2*norm.sf(threshold):0.2f}%")
visual_scatter = np.random.normal(size = d1.size)
plt.scatter(d1[good], visual_scatter[good], s=1, label="Good", color="#4CAF50")
plt.scatter(d1[~good], visual_scatter[~good], s=2, label="Bad", color="#F44336")
plt.legend();
```

    Rejection 5 points
    z-score of 3 corresponds to a prob of 0.27%



![png](/images/stat_files/stat_43_1.png)



```python
d1.size
```




    1010




```python
from scipy.stats import multivariate_normal as mn

mean, cov = np.mean(d2, axis=0), np.cov(d2.T)
good = mn(mean, cov).pdf(d2) > 0.01 / 100

plt.scatter(d2[good, 0], d2[good, 1], s=2, label="Good", color="#4CAF50")
plt.scatter(d2[~good, 0], d2[~good, 1], s=8, label="Bad", color="#F44336")
plt.legend();
```


![png](/images/stat_files/stat_45_0.png)


So, how do we pick what our threshold should be? Visual inspection is actually hard to beat. You can make an argument for relating the number to the number of samples you have or how much of the data you are willing to cut, but be warned that too much rejection is going to eat away at your actual data sample and bias your results.

### Outliers in curve fitting

If you don't have a distribution but instead have data with uncertainties, you can do similar things. To take a real world example, in an [old paper of mine](https://arxiv.org/abs/1603.09438), we have some value of xs, ys and error (wavelength, flux and flux error) and want to subtract the smooth background. We wanted to do this with a simple polynomial fit, but unfortunately the data had several emission lines and cosmic ray impacts in it (visible as spikes) which biased our poly fitting and so we had to remove them.

What we did is fit a polynomial to it, remove all points more than three standard deviations from polynomial from consideration and loop until all points are within three standard deviations. In the example below, for simplicity the data is normalised so that all errors are one.


```python
xs, ys = d3.T
p = np.polyfit(xs, ys,deg=5)
ps = np.polyval(p, xs)
plt.plot(xs, ys, ".", label="Data", ms=1)
plt.plot(xs, ps, label="Bad poly fit")
plt.legend();
```


![png](/images/stat_files/stat_48_0.png)



```python
x, y = xs.copy(), ys.copy()
for i in range(7):
    p = np.polyfit(x, y, deg=5)
    ps = np.polyval(p, x)
    good = y - ps < 3  # only remove positive outliers

    x_bad, y_bad = x[~good], y[~good]
    x, y = x[good], y[good]

    plt.plot(x, y, ".", label="Used Data", ms=1)
    plt.plot(x, np.polyval(p, x), label=f"Poly fit {i}")
    plt.plot(x_bad, y_bad, ".", label="Not used Data", ms=5, c="r")
    plt.legend()
    plt.show()

    if (~good).sum() == 0:
        break
```


![png](/images/stat_files/stat_49_0.png)



![png](/images/stat_files/stat_49_1.png)



![png](/images/stat_files/stat_49_2.png)


[learn polynomial](https://id.wikipedia.org/wiki/Polinomial)

### Automating it

Blessed `sklearn` to the rescue. Check out [the main page](https://scikit-learn.org/stable/modules/outlier_detection.html) which lists a ton of ways you can do outlier detection. I think LOF (Local Outlier Finder) is great - it uses the distance from one point to its closest twenty neighbours to figure out point density and removes those in low density regions.



```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.005)
good = lof.fit_predict(d2) == 1
plt.scatter(d2[good, 0], d2[good, 1], s=2, label="Good", color="#4CAF50")
plt.scatter(d2[~good, 0], d2[~good, 1], s=8, label="Bad", color="#F44336")
plt.legend();
```


![png](/images/stat_files/stat_52_0.png)


## 1D Histogram

Let's load some 1D data and get some insight into it!

To keep this section nice and simple, the data generated is something I've thrown together. It's not just an analytic function from `scipy` - that would make it too easy - but I've made sure its not pathological.

Let's start with the imports to make sure we have everything right at the beginning. If this errors, `pip install` whichever dependency you don't have. If you have issues (especially on windows machines with numpy) try using `conda install`. For example, `conda install numpy`.

Also, if you want to get the same dark theme in your notebook as me, check out [jupyter-themes](https://github.com/dunovank/jupyter-themes).


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
```


```python
d1 = np.loadtxt("Resources/Datasets/1DVis/example_1.txt")
d2 = np.loadtxt("Resources/Datasets/1DVis/example_2.txt")
print(d1.shape, d2.shape)
```

    (500,) (500,)


### Histogram Plots

* Normal plots
* Density and bins
* Customising styles


```python
plt.hist(d1, label="D1")
plt.hist(d2, label="D2")
plt.ylabel("Count")
plt.xlabel("x")
plt.legend()
```




    <matplotlib.legend.Legend at 0x1e4ad9c4dc8>




![png](/images/stat_files/stat_57_1.png)



```python
bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()), 50)
counts1, _,_ = plt.hist(d1, bins = bins, label="D1")
plt.hist(d2, bins=bins, label= "D2")
plt.legend()
plt.ylabel("Count")
plt.xlabel("x")

print("Panjang = ",min(d1.min(), d2.min()), max(d1.max(), d2.max()))
print("Jumlah titik = ", 50)
```

    Panjang =  0.568 23.307
    Jumlah titik =  50



![png](/images/stat_files/stat_58_1.png)



```python
bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()), 50)
counts1, _,_ = plt.hist(d1, bins = bins, label="D1", density=True)
plt.hist(d2, bins=bins, label= "D2", density=True)
plt.legend()
plt.ylabel("Probability")
plt.xlabel("x")

print("Panjang = ",min(d1.min(), d2.min()), max(d1.max(), d2.max()))
print("Jumlah titik = ", 50)
```

    Panjang =  0.568 23.307
    Jumlah titik =  50



![png](/images/stat_files/stat_59_1.png)



```python
bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()), 50)
plt.hist([d1,d2], bins = bins, label="Stacked", density=True, alpha=0.5, histtype='barstacked')
plt.hist(d1, bins = bins, label="D1", density=True, histtype='step', lw=1)
plt.hist(d2, bins=bins, label= "D2", density=True, ls=":", histtype="step")
plt.legend()
plt.ylabel("Probability")
plt.xlabel("x")

print("Panjang = ",min(d1.min(), d2.min()), max(d1.max(), d2.max()))
print("Jumlah titik = ", 50)
```

    Panjang =  0.568 23.307
    Jumlah titik =  50



![png](/images/stat_files/stat_60_1.png)



```python
bins =50
plt.hist([d1,d2], bins = bins, label="Stacked", density=True, alpha=0.5, histtype='barstacked')
plt.hist(d1, bins = bins, label="D1", density=True, histtype='step', lw=1)
plt.hist(d2, bins=bins, label= "D2", density=True, ls=":", histtype="step")
plt.legend()
plt.ylabel("Probability")
plt.xlabel("x")


```




    Text(0.5, 0, 'x')




![png](/images/stat_files/stat_61_1.png)


### Bee Swarm Plots


```python
dataset = pd.DataFrame({
    "value" : np.concatenate((d1, d2)),
    "type" : np.concatenate((np.ones(d1.shape), np.zeros(d2.shape)))
})
dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   value   1000 non-null   float64
     1   type    1000 non-null   float64
    dtypes: float64(2)
    memory usage: 15.8 KB



```python
sb.swarmplot(dataset["value"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e4ae5f4208>




![png](/images/stat_files/stat_64_1.png)



```python
sb.swarmplot(x="type", y="value", data=dataset, size=2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e4ab46a988>




![png](/images/stat_files/stat_65_1.png)


### Box Plots


```python
sb.boxplot(x="type", y='value', data=dataset, whis=2.0)
sb.swarmplot(x="type", y='value', data=dataset, size=2, color='k', alpha=0.3)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e4af2ab408>




![png](/images/stat_files/stat_67_1.png)


### Violin Plots


```python
sb.violinplot(x="type", y='value', data=dataset)
sb.swarmplot(x="type", y='value', data=dataset, size=2, color='k', alpha=0.3)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e4af28c0c8>




![png](/images/stat_files/stat_69_1.png)



```python
sb.violinplot(x="type", y='value', data=dataset, inner="quartile", bw=0.2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e4af60cd88>




![png](/images/stat_files/stat_70_1.png)




### Empirical Cumulative Distribution Functions


When you form a histogram the fact you have to bin data means that the looks can change significantly when you change bin sizing. And each bin has statistical uncertainty. You can get past that using a CDF. Its harder - visually - to see features in the PDF when looking at the CDF, however its generally more useful when you are trying to do quantitative comparisons between multiple distributions. We'll get on that later.



```python
sd1 = np.sort(d1)
sd2 = np.sort(d2)
cdf = np.linspace(1/d1.size, 1, d1.size)

plt.plot(sd1, cdf, label="D1 CDF")
plt.plot(sd2, cdf, label="D2 CDF")
plt.hist(d1, histtype='step', density=True, alpha=0.3)
plt.hist(d2, histtype='step', density=True, alpha=0.3)
plt.xlabel("CDF")
plt.ylabel("Probability of Sorted Data")
plt.legend();
```


![png](/images/stat_files/stat_72_0.png)


### Describe


```python
df = pd.DataFrame({
    "Data1" : d1,
    "Data2" : d2
})
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
      <th>Data1</th>
      <th>Data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500.000000</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13.511172</td>
      <td>7.390714</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.296363</td>
      <td>3.589993</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.640000</td>
      <td>0.568000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>12.572000</td>
      <td>5.164750</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.389500</td>
      <td>6.531000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>14.291250</td>
      <td>9.664500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>19.262000</td>
      <td>23.307000</td>
    </tr>
  </tbody>
</table>
</div>



## Higher Dimensional Distributions

### ND Scatter Matrix


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
```


```python
df_original = pd.read_csv("Resources/Datasets/Higher Dimensional Distributions/Diabetes.csv")
df_original.head()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols = [c for c in df_original.columns if c not in ["Pregnancies", "Outcome"]]
df = df_original.copy()
df[cols] = df[cols].replace({
    0:np.NaN
})
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148.0</td>
      <td>72.0</td>
      <td>35.0</td>
      <td>NaN</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85.0</td>
      <td>66.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183.0</td>
      <td>64.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89.0</td>
      <td>66.0</td>
      <td>23.0</td>
      <td>94.0</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137.0</td>
      <td>40.0</td>
      <td>35.0</td>
      <td>168.0</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
cols
```




    ['Glucose',
     'BloodPressure',
     'SkinThickness',
     'Insulin',
     'BMI',
     'DiabetesPedigreeFunction',
     'Age']




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   763 non-null    float64
     2   BloodPressure             733 non-null    float64
     3   SkinThickness             541 non-null    float64
     4   Insulin                   394 non-null    float64
     5   BMI                       757 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    float64
     8   Outcome                   768 non-null    int64  
    dtypes: float64(7), int64(2)
    memory usage: 54.1 KB



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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>763.000000</td>
      <td>733.000000</td>
      <td>541.000000</td>
      <td>394.000000</td>
      <td>757.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>121.686763</td>
      <td>72.405184</td>
      <td>29.153420</td>
      <td>155.548223</td>
      <td>32.457464</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>30.535641</td>
      <td>12.382158</td>
      <td>10.476982</td>
      <td>118.775855</td>
      <td>6.924988</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>44.000000</td>
      <td>24.000000</td>
      <td>7.000000</td>
      <td>14.000000</td>
      <td>18.200000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>64.000000</td>
      <td>22.000000</td>
      <td>76.250000</td>
      <td>27.500000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>29.000000</td>
      <td>125.000000</td>
      <td>32.300000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>141.000000</td>
      <td>80.000000</td>
      <td>36.000000</td>
      <td>190.000000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.plotting.scatter_matrix(df, figsize=(7,7));
```


![png](/images/stat_files/stat_83_0.png)



```python
df2=df.dropna()
colors = df2["Outcome"].map(lambda x: "#44d9ff" if x else "#f95b4a")
pd.plotting.scatter_matrix(df2, figsize=(7,7), color=colors);

```


![png](/images/stat_files/stat_84_0.png)



```python
sb.pairplot(df2, hue="Outcome")
```




    <seaborn.axisgrid.PairGrid at 0x1e4ba732a48>




![png](/images/stat_files/stat_85_1.png)


### ND Correlation


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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Pregnancies</th>
      <td>1.000000</td>
      <td>0.128135</td>
      <td>0.214178</td>
      <td>0.100239</td>
      <td>0.082171</td>
      <td>0.021719</td>
      <td>-0.033523</td>
      <td>0.544341</td>
      <td>0.221898</td>
    </tr>
    <tr>
      <th>Glucose</th>
      <td>0.128135</td>
      <td>1.000000</td>
      <td>0.223192</td>
      <td>0.228043</td>
      <td>0.581186</td>
      <td>0.232771</td>
      <td>0.137246</td>
      <td>0.267136</td>
      <td>0.494650</td>
    </tr>
    <tr>
      <th>BloodPressure</th>
      <td>0.214178</td>
      <td>0.223192</td>
      <td>1.000000</td>
      <td>0.226839</td>
      <td>0.098272</td>
      <td>0.289230</td>
      <td>-0.002805</td>
      <td>0.330107</td>
      <td>0.170589</td>
    </tr>
    <tr>
      <th>SkinThickness</th>
      <td>0.100239</td>
      <td>0.228043</td>
      <td>0.226839</td>
      <td>1.000000</td>
      <td>0.184888</td>
      <td>0.648214</td>
      <td>0.115016</td>
      <td>0.166816</td>
      <td>0.259491</td>
    </tr>
    <tr>
      <th>Insulin</th>
      <td>0.082171</td>
      <td>0.581186</td>
      <td>0.098272</td>
      <td>0.184888</td>
      <td>1.000000</td>
      <td>0.228050</td>
      <td>0.130395</td>
      <td>0.220261</td>
      <td>0.303454</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>0.021719</td>
      <td>0.232771</td>
      <td>0.289230</td>
      <td>0.648214</td>
      <td>0.228050</td>
      <td>1.000000</td>
      <td>0.155382</td>
      <td>0.025841</td>
      <td>0.313680</td>
    </tr>
    <tr>
      <th>DiabetesPedigreeFunction</th>
      <td>-0.033523</td>
      <td>0.137246</td>
      <td>-0.002805</td>
      <td>0.115016</td>
      <td>0.130395</td>
      <td>0.155382</td>
      <td>1.000000</td>
      <td>0.033561</td>
      <td>0.173844</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.544341</td>
      <td>0.267136</td>
      <td>0.330107</td>
      <td>0.166816</td>
      <td>0.220261</td>
      <td>0.025841</td>
      <td>0.033561</td>
      <td>1.000000</td>
      <td>0.238356</td>
    </tr>
    <tr>
      <th>Outcome</th>
      <td>0.221898</td>
      <td>0.494650</td>
      <td>0.170589</td>
      <td>0.259491</td>
      <td>0.303454</td>
      <td>0.313680</td>
      <td>0.173844</td>
      <td>0.238356</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['Pregnancies', 'Insulin']].corr()
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
      <th>Pregnancies</th>
      <th>Insulin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Pregnancies</th>
      <td>1.000000</td>
      <td>0.082171</td>
    </tr>
    <tr>
      <th>Insulin</th>
      <td>0.082171</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sb.heatmap(df.corr(), annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e4bbb71288>




![png](/images/stat_files/stat_89_1.png)



```python
sb.heatmap(df.corr(), annot=True, cmap='viridis', fmt='0.2f');
```


![png](/images/stat_files/stat_90_0.png)


And you can see this is a symmetric matrix too. But it immedietly allows us to point out the most correlated and anti-correlated attributes. Some might just be common sense - Pregnancies v Age for example - but some might give us real insight into the data.

### 2D Histograms

For the rest of this section, we're going to use a different dataset which has more data in it.

Useful when you have a *lot* of data. [See here for the API](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist2d.html)


```python
df2 = pd.read_csv("Resources/Datasets/Higher Dimensional Distributions/height_weight.csv")
df2.info()
df2.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4231 entries, 0 to 4230
    Data columns (total 3 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   sex     4231 non-null   int64  
     1   height  4231 non-null   float64
     2   weight  4231 non-null   float64
    dtypes: float64(2), int64(1)
    memory usage: 99.3 KB





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
      <th>sex</th>
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4231.000000</td>
      <td>4231.000000</td>
      <td>4231.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.540061</td>
      <td>66.903607</td>
      <td>174.095122</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.498451</td>
      <td>4.313004</td>
      <td>38.896171</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>55.400000</td>
      <td>96.590000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>63.730000</td>
      <td>144.315000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>66.630000</td>
      <td>170.100000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>69.970000</td>
      <td>198.660000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000000</td>
      <td>79.610000</td>
      <td>298.440000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.hist2d(df2['height'], df2['weight'], bins=20)
plt.xlabel('height')
plt.ylabel('weight')
```




    Text(0, 0.5, 'weight')




![png](/images/stat_files/stat_94_1.png)


### Contour Plot

to gain more correlation of the data


```python
hist, x_edge, y_edge = np.histogram2d(df2['height'], df2['weight'], bins=20)
x_center = 0.5 * (x_edge[1:] + x_edge[:-1])
y_center = 0.5 * (y_edge[1:] + y_edge[:-1])

plt.contour(x_center, y_center, hist, levels=4)
plt.xlabel('height')
plt.ylabel('weight')
```




    Text(0, 0.5, 'weight')




![png](/images/stat_files/stat_96_1.png)


Ouch, looks like its just as noisy with the contour plot! In general, for 2D histograms and contour plots, have a lot of data. We simply don't have enough data to get smooth results!



### KDE Plots

If only we could smooth the data ourselves. [Seaborn to the rescue!](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)


```python
sb.kdeplot(df2['height'], df2['weight'], cmap='viridis', bw=(2, 20))
plt.hist2d(df2['height'], df2['weight'], bins=20, cmap='magma', alpha=0.3);
```


![png](/images/stat_files/stat_99_0.png)



```python
sb.kdeplot(df2['height'], df2['weight'], cmap='magma', shade=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e4c191dec8>




![png](/images/stat_files/stat_100_1.png)


### ND Scatter Probability - Practical


#### In Defense of Simplicity


```python

m = df2["sex"] == 1
plt.scatter(df2.loc[m, "height"], df2.loc[m, "weight"], c="#16c6f7", s=1, label="Male")
plt.scatter(df2.loc[~m, "height"], df2.loc[~m, "weight"], c="#ff8b87", s=1, label="Female")
plt.ylabel("Height")
plt.xlabel("Weight")
plt.legend(loc=2);
```


![png](/images/stat_files/stat_103_0.png)



```python

```


```python
df2['sex'].values
```




    array([1, 1, 2, ..., 2, 2, 2], dtype=int64)



#### Treating points with probability

Using the library ChainConsumer ([examples here](https://samreay.github.io/ChainConsumer/examples/index.html)). I wrote it, as I deal with MCMC chains and posterior samples for hours every day and needed better tools to analyse them.

ChainConsumer is not a standard anaconda package, so you'll need to run the below to install it.

`pip install chainconsumer`

Note that without LaTeX for fancy labels, set `usetex=False` in the `configure` method. I've done this below for your version of the code, its not in the video.


```python
params = ['height', 'weight']
male = df2.loc[m, params].values
female = df2.loc[~m, params].values
male.shape
```




    (1946, 2)




```python
from chainconsumer import ChainConsumer
c = ChainConsumer()
c.add_chain(male, parameters=params, name='Male', kde=1.0, color='b')
c.add_chain(female, parameters=params, name='Male', kde=1.0, color='r')
c.configure(contour_labels='confidence', usetex=False)
c.plotter.plot(figsize=2.0)
```




![png](/images/stat_files/stat_108_0.png)




![png](/images/stat_files/stat_108_1.png)



```python
c.plotter.plot_summary(figsize=2.0)
```




![png](/images/stat_files/stat_109_0.png)




![png](/images/stat_files/stat_109_1.png)


Remember, when you're visualising, start simple and add complexity as the data seems to indicate. No point wasting your time getting in real deep when its not needed.

Oh and you might have noticed we've talked about a lot of plots, but not pie charts. Never use pie charts. Never.

## Summary

1. Know what you're looking for
2. Take a moment to **prepare** your dataset
3. Develop a feel for your data using **plots**
    - Understand the **distributions**
    - Understand the **relationships** between attributes


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

# Characterising

1. Why bother?

    Numbers matter.<br>
    Every **pretty plot** should be accompanied by **quantification** in a **presentation** or **report**.<br>
    It also saves space

2. Characterising Distributions
3. Multivariate Distributions


```python

```

## Characterising 1D


```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
```


```python
data = np.loadtxt("Resources/Datasets/Characterising/dataset.txt")
plt.hist(data,bins=30);
```


![png](/images/stat_files/stat_160_0.png)


Okay, so the first thing we can do is try and get some metrics for the center of the distribution. Much of this section may not be ground-breaking for those watching, so let's power through it.

### Central Tendency

#### Mean

If we have a set of points N points denoted $x_i$, the mean is defined as

$$ \frac{1}{N} \sum_{i=1}^N x_i $$

A way to manually compute the mean is given by:


```python
def get_mean(xs):
    summed = 0
    for x in xs:
        summed += x
    return summed / len(xs)
print(get_mean([3,5,3,5]))
```

    4.0


But in this modern age we shouldn't have to write the function ourself. We can use `np.mean`. If we want datapoints to have different weights, we can use `np.average` instead. (For example, dice rolling might only record the value and number of times, not each individual roll).


```python
mean = np.mean(data)
print(mean, data.mean(), np.average(data))
```

    7.68805056 7.68805056 7.68805056


#### Median

Sort all your data and take out the middle element. Thats your median. `[1,3,5,7,7]` has a median of `5`. Here's how we can manually find the median:


```python
def get_median(xs):
    mid = len(xs) // 2
    if len(xs) % 2 == 1:
        return sorted(xs)[mid]
    else:
        return 0.5 * np.sum(sorted(xs)[mid - 1:mid + 1])

print(get_median([3,4,5,63,2,6,3,3]))
```

    3.5



```python
median = np.median(data)
print(median)
```

    6.7325



```python
outlier = np.insert(data, 0, 5000)
plt.hist(data, label="Data", bins=50);
plt.axvline(np.mean(data), ls="--", label="Mean Data")
plt.axvline(np.median(data), ls=":", label="Median Data")
plt.axvline(np.mean(outlier), c='r', ls="--", label="Mean Outlier", alpha=0.7)
plt.axvline(np.median(outlier), c='r', ls=":", label="Median Outlier", alpha=0.7)
plt.legend()
plt.xlim(0,20);
```


![png](/images/stat_files/stat_172_0.png)


#### Mode

Another outlier insensitive method, it returns to us the value which is most common. **This works for discrete distributions only... sort of.** If you have a continuous distribution, you will need to bin the data first. For example, the mode of `[1,7,2,5,3,3,8,3,2]` is `3`, because three shows up more than any other number. Here is a manual mode function:


```python
def get_mode(xs):
    values, counts = np.unique(xs, return_counts = True)
    max_count_index = np.argmax(counts)
    return values[max_count_index]
print(get_mode([2,5,3,2,3,5,6]))
```

    2



```python
mode = st.mode(data)
print(mode)
```

    ModeResult(mode=array([5.519]), count=array([9]))



```python
hist, edges = np.histogram(data, bins=100)
edge_centers = 0.5 * (edges[1:] + edges[:-1])
mode = edge_centers[hist.argmax()]
print(mode)
```

    5.223165



```python
kde = st.gaussian_kde(data)
xvals = np.linspace(data.min(), data.max(), 1000)
yvals = kde(xvals)
mode = xvals[yvals.argmax()]

plt.hist(data, bins=1000, density=True, label="Data hist", histtype="step")
plt.plot(xvals, yvals, label="KDE")
plt.axvline(mode, label="Mode", c='yellow')
plt.legend();
```


![png](/images/stat_files/stat_177_0.png)



```python
#using seaborn
sb.distplot(data, bins=100)
plt.axvline(mean, label="Mean", ls="--", c='#f9ee4a')
plt.axvline(median, label="Median", ls="-", c='#44d9ff')
plt.axvline(mode, label="Mode", ls=":", c='#f95b4a')
plt.legend();
```


![png](/images/stat_files/stat_178_0.png)



```python

```

#### Comparison


```python
plt.hist(data, bins=100, label="Data", alpha=0.5)
plt.axvline(mean, label="Mean", ls="--", c='#f9ee4a')
plt.axvline(median, label="Median", ls="-", c='#44d9ff')
plt.axvline(mode, label="Mode", ls=":", c='#f95b4a')
plt.legend();
```


![png](/images/stat_files/stat_181_0.png)


### Measures of width and balance

* Variance
* Standard deviation
* Skewness
* Kurtosis

#### Variance

The variance of a distrbution is a measure of how much it spreads out around the mean. A touch more formally, its the expected value of the squared deviation from the mean. Even more formally, it is given by

$$ Var = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2, $$

where $\mu$ is the mean of the dataset $x$, as described in the previous section. Note there is a fine point about whether you should divide by $N$ or $N-1$. Here is a manual way of calculating it:


```python

def get_variance(xs):
    mean = np.mean(xs)
    summed = 0
    for x in xs:
        summed += (x - mean)**2
    return summed / (len(xs) - 1)
print(get_variance([1,2,3,4,5]))
```

    2.5



```python
variance = np.var([1,2,3,4,5], ddof = 1) #using N-1
variance
```




    2.5




```python
np.var(data)
```




    13.136575622563686



#### Standart Deviation


```python
std = np.std(data)
print(std, std**2)
```

    3.6244414221454435 13.136575622563685


#### Our powers combined

Welcome to the Gaussian approximation! Also known as a normal approximation. Check it out:


```python
xs = np.linspace(data.min(), data.max(), 100)
ys = st.norm.pdf(xs, loc=mean, scale=std)

plt.hist(data, bins=50, density=True, histtype="step", label="Data")
plt.plot(xs, ys, label="Normal approximation")
plt.legend()
plt.ylabel("Probability");
```


![png](/images/stat_files/stat_189_0.png)



```python
from scipy.stats import norm
from scipy.stats import skewnorm
sb.distplot(
    data,
    kde=False,
    rug = False,
    bins=50,
    fit_kws={
        'label' : 'Normal approximation'
    },
    hist_kws={
        'label' : 'Data',
        'histtype' : 'step'


    },
    fit=norm

)

plt.xlabel('data(x)')
plt.ylabel('Probability')

plt.legend()


```




    <matplotlib.legend.Legend at 0x299899d1188>




![png](/images/stat_files/stat_190_1.png)


Dari pendekatan normal tersebut, kita tidak mendekatinya dengan sempurna. tetapi kita dapat memperoleh lebih dari setengah informasi dari data dengan pendekatan tersebut

Its not *too* bad, but its not the best thing either. It seems like our data isn't perfectly symmetrical, so lets quantify how asymmetrical it is.

#### Skewness

In this section I might drop the word "moment" a few times. There are some standardised ways of quantifying "moments". The first moment is zero by definition. The second is variance. The third is skewness, which is often defined as $\gamma_1$.

$$ \gamma_1 = \frac{\kappa_3}{\kappa_2^{3/2}} = \frac{E[(x-\mu)^3]}{E[(x-\mu)^2]^{3/2}} $$


```python
def get_skewness(xs):
    mean = np.mean(xs)
    var = np.var(xs)
    summed = 0
    for x in xs:
        summed += (x-mean)**3
    return (summed / (len(xs))) / (var**1.5)
print(get_skewness([1,2,3,4,5]))


```

    0.0



```python
skewness = st.skew(data)
print(skewness, get_skewness(data))
```

    0.7408773663373577 0.7408773663373582


Let's update our normal approximation to a skewed normal approximation and see how it looks, just for fun.


```python
xs = np.linspace(data.min(), data.max(), 100)
ys1 = st.norm.pdf(xs, loc=mean, scale=std)
ys2 = st.skewnorm.pdf(xs, skewness, loc=mean, scale=std)

plt.hist(data, bins=50, density=True, histtype="step", label="Data")
plt.plot(xs, ys1, label="Normal approximation")
plt.plot(xs, ys2, label="Skewnormal approximation")
plt.legend()
plt.ylabel("Probability");
```


![png](/images/stat_files/stat_196_0.png)



```python
from scipy.stats import skew
```


```python
xs = np.linspace(data.min(), data.max(), 1000)
skewness = st.skew(data)
ys1 = st.norm.pdf(xs, loc=mean, scale=std)
ys2 = st.skewnorm.pdf(xs, skewness, loc=mean, scale=std)

sb.distplot(data,label="Data")
plt.plot(xs, ys1, label="Normal approximation")
plt.plot(xs, ys2, label="Skewnormal approximation")
plt.legend()


```




    <matplotlib.legend.Legend at 0x2998ade7f48>




![png](/images/stat_files/stat_198_1.png)


Oh no, this doesn't look better? Where did we go wrong?

A skew normal cant just be given the mean and standard deviation of a normal and expected to work. The skewness modifies the mean and standard deviation. You need to actually fit.


```python
xs = np.linspace(data.min(), data.max(), 100)
ys1 = st.norm.pdf(xs, loc=mean, scale=std)
ps = st.skewnorm.fit(data)
ys2 = st.skewnorm.pdf(xs, *ps)

plt.hist(data, bins=50, density=True, histtype="step", label="Data")
plt.plot(xs, ys1, label="Normal approximation")
plt.plot(xs, ys2, label="Skewnormal approximation")
plt.legend()
plt.ylabel("Probability");
```


![png](/images/stat_files/stat_200_0.png)



```python
sb.distplot(data,label="Data", fit=st.skewnorm)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x29988211b48>




![png](/images/stat_files/stat_201_1.png)


#### Kurtosis

The next moment, and the last one we'll consider is kurtosis. It has a similar definition, and is often represented as $\kappa$ or $\gamma_2$:

$$ \kappa = \frac{E[(x-\mu)^4]}{E[(x-\mu)^2]^{4/2}} $$


```python
def get_kurtosis(xs):
    mean = np.mean(xs)
    var = np.var(xs)
    summed = 0
    for x in xs:
        summed += (x - mean)**4
    return (summed / (len(xs))) / (var ** 2)
print(get_kurtosis([1,2,3,4,5]))
```

    1.7



```python
kurtosis = st.kurtosis(data)
print(kurtosis, get_kurtosis(data))
```

    0.5517538497309498 3.551753849730955



```python
kurtosis = st.kurtosis(data, fisher=False)
print(kurtosis, get_kurtosis(data))
```

    3.5517538497309498 3.551753849730955


`fisher` hey? So this is just a normalisation thing and because there are multiple definitions of kurtosis. With `fisher=False`, a normal distrubtion has a kurtosis of 3. With `fisher=True`, scipy subtracts 3 from the result so that a normal distribution would have a kurtosis of 0. Lots of things are compared to normal distributions, so having all the moments be 0 for them is handy. If you're curious about this, the difference is between "kurtosis" and "excess kurtosis", will help which will hopefully help when googling!

### When analytics fail

At the moment we've been coming up with ways to quantify our data distribution such that we could try and reconstruct something approximately the same using various analytic distributions, like the normal distribution. So what happens if that isn't going to be good enough?

#### Percentiles

What if we - instead of using a mean or other such numbers - simply turned our long data vector down into a few points representing different percentiles? We could essentially reconstruct our data distribution to an aribtrary accuracy and never worry about analytic functions.

percentiles range = 0 - 100%


```python
ps = np.linspace(0, 100, 10)
x_p = np.percentile(data, ps)

xs = np.sort(data)
ys = np.linspace(0, 1, len(data))

plt.plot(xs, ys * 100, label="ECDF")
plt.plot(x_p, ps, label="Percentiles", marker=".", ms=10)
plt.legend()
plt.ylabel("Percentile");
```


![png](/images/stat_files/stat_208_0.png)



```python
ps = 100 * st.norm.cdf(np.linspace(-4, 4, 50))
x_p = np.percentile(data, ps)

xs = np.sort(data)
ys = np.linspace(0, 1, len(data))

plt.plot(xs, ys * 100, label="ECDF")
plt.plot(x_p, ps, label="Percentiles", marker=".", ms=10)
plt.legend()
plt.ylabel("Percentile");
```


![png](/images/stat_files/stat_209_0.png)



```python
ps = 100 * st.norm.cdf(np.linspace(-3, 3, 50))
ps = np.concatenate(([0], ps, [100]))  # There is a bug in the insert way of doing it, this is better
x_p = np.percentile(data, ps)

xs = np.sort(data)
ys = np.linspace(0, 1, len(data))

plt.plot(xs, ys * 100, label="ECDF")
plt.plot(x_p, ps, label="Percentiles", marker=".", ms=10)
plt.legend()
plt.ylabel("Percentile");
```


![png](/images/stat_files/stat_210_0.png)



```python
from scipy.interpolate import interp1d

n = int(1e6)
u = np.random.uniform(siz e=n)
samp_percentile_1 = interp1d(ps / 100, x_p)(u)

_, bins, _ = plt.hist(data, bins=50, density=True, alpha=0.3, label="Data")
plt.hist(samp_percentile_1, bins=bins, density=True, histtype="step", label="Percentiles")
plt.ylabel("Probability")
plt.legend();
```


![png](/images/stat_files/stat_211_0.png)


Look at how nice those tails are now! And you can see that if we increased the number of samples in our second percentiles to around a hundred and we could very accurately describe our 1D distribution. And 100 data points are much faster to transfer than thousands of them.


```python

```

## Characterising ND



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

- characterize all the columns by themselves

    So if you have an n dimensional data set you can then summarize in one day dimensions once you've done that obviously you then need to go through and try and summarize the relationships between each of the different columns. This is actually quite difficult because there are really no good summary statistics.



### Covariance

We've talked about variance (the average square deviation from the mean). Covariance is, as you've guessed, similar. Let's say we have a data vector, $x^a$, which has $i$ points... so $x_i^a$ is the first element of the data vector, from the previous section we'd have that:

$$ Var^{a,a} = \frac{1}{N-1} \sum_{i=1}^N (x_i^a - \mu^a)(x_i^a - \mu^a), $$

This should look like the last section, except I've stuck $a$ in a few places. Another way of stating this is that this is covariance of vector $x^a$ with itself. Notice there are two sets of brackets, both use data vector $x^a$. Covariance is what you get when you change one of the letters. Like this:

$$ Var^{a,b} = \frac{1}{N-1} \sum_{i=1}^N (x_i^a - \mu^a)(x_i^b - \mu^b), $$

Easy! All we've done is now one set in the brackets iterates over a different data vector. The goal is to do this for each different vector you have to form a matrix. If we had only two vectors, our matrix is this:

$$ Cov = \begin{pmatrix} Var^{a,a} & Var^{a,b} \\ Var^{b,a} & Var^{b,b} \\ \end{pmatrix} $$

Notice how this is symmetric. $Var^{a,b} = Var^{b,a}$. And the diagonals are just the variance for each data vector. The off-diagonals are measure of the joint spread between the two. If the concept still isn't perfect, don't worry, the examples will clear everything up.

We can calculate the covariance using either `np.cov` ([doco here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html)) or `pd.DataFrame.cov` ([doco here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.cov.html)).


```python
dataset = pd.read_csv("Resources/Datasets/CharacterisingND/height_weight.csv")[['height', 'weight']]
dataset.head()
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
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>71.74</td>
      <td>259.88</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71.00</td>
      <td>186.73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>63.83</td>
      <td>172.17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>67.74</td>
      <td>174.66</td>
    </tr>
    <tr>
      <th>4</th>
      <td>67.28</td>
      <td>169.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
covariance = np.cov(dataset, rowvar=False)
print(covariance)
```

    [[  18.60200779   78.50218098]
     [  78.50218098 1512.91208783]]



```python
covariance = dataset.cov()
covariance
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
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>height</th>
      <td>18.602008</td>
      <td>78.502181</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>78.502181</td>
      <td>1512.912088</td>
    </tr>
  </tbody>
</table>
</div>



### Correlation

Correlation and covariance are easily linked. If we take that 2D covariance matrix from above, which is written in terms of variance, we can rewrite it in terms of standard deviation $\sigma$, as $Var = \sigma^2$.

$$ Cov = \begin{pmatrix} \sigma^2_{a,a} & \sigma^2_{a,b} \\ \sigma^2_{b,a} & \sigma^2_{b,b} \\ \end{pmatrix} $$

Great. And here is the correlation matrix:

$$ Corr = \begin{pmatrix} \sigma^2_{a,a}/\sigma^2_{a,a} & \sigma^2_{a,b}/(\sigma_{a,a}\sigma_{b,b}) \\ \sigma^2_{b,a}/(\sigma_{a,a}\sigma_{b,b}) & \sigma^2_{b,b}/\sigma^2_{b,b} \\ \end{pmatrix} $$

Which is the same as

$$ Corr = \begin{pmatrix} 1 & \rho_{a,b} \\ \rho_{b,a} & 1 \\ \end{pmatrix}, $$

where $\rho_{a,b} = \sigma^2_{a,b}/(\sigma_{a,a}\sigma_{b,b})$. Another way to think about this is that

$$ Corr_{a,b} = \frac{Cov_{a,b}}{\sigma_a \sigma_b} $$

It is the joint variability normalised by the variability of each independent variable.

But this is *still too mathy for me*. Let's just go back to the code. We can calculate a correlation matrix using `np.corrcoef` ([doco here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html)) or `pd.DataFrame.corr` ([doco here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html))


```python
corr = np.corrcoef(dataset.T)
corr
```




    array([[1.        , 0.46794517],
           [0.46794517, 1.        ]])




```python
dataset.corr()
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
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>height</th>
      <td>1.000000</td>
      <td>0.467945</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>0.467945</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sb
sb.heatmap(dataset.corr(), annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1acdf42ec48>




![png](/images/stat_files/stat_224_1.png)



```python
sb.lmplot(data=dataset, y='weight', x='height')
```




    <seaborn.axisgrid.FacetGrid at 0x1acdf706388>




![png](/images/stat_files/stat_225_1.png)


So, what does this mean? In as simple as I can put it, the fact we have a positive number for the height-weight correlation means that, *on average*, a taller person probably weights more (than a shorter person). A shorter person, *on average* probably weighs less than a tall person.

If the number was negative, it would mean the opposite - a tall person would normally weigh less than a short person. One is as far as you can get, and if our $0.468$ was $1$, it would mean that a tall person would *always* weight more than a short person. The bigger the number (in absolute terms), the more likely you are to find that correlation.

Here are some other examples:

* **Age vs number of pregnancies**: Positive correlation
* **Temperature in Celcius vs Temperature in Kelvin**: Total positive correlation ($1.0$)
* **Amount of cigarettes smoked vs Life expectance**: Negative correlation
* **Height vs Comfort on plane seats**: Negative correlation
* **Number of units purchased vs Cost of individual unit**: Hopefully a negative correlation!

Take two things and ask yourself, if one goes up, do I expect the other to go, down or not change?

That is correlation. And now you can quantify it.

And, given we did this in the EDA section, you can also make plots of it and explore it visually.

## Summary

1. start with the basics - **mean** and **standard deviation**
2. Use summary stats to **approximate** your dataset
3. **Correlation** and 1D statistics can go a long way

# Probability

1. Probability distributions
2. Using distributions
3. Nonparametric statistics
4. Sampling Distributions

## Probability Refresher

1. Probabilities range from **0 (impossible)** to **1 (certain)**
2. Probabilities **sum to 1**
3. A distribution relates outcome **x** with probability **p(x)**
4. A mass function is for when you have **discrete** outcomes

Pada discrete outcomes,
- menggunakan probability mass function.
- dapat dengan langsung menanyakan probabilitas untuk masing masing titik

Sedangkan pada continuous outcomes,
- menggunakan probability distribution.
- harus dengan interval, misalnya berapa probabilitas antara titik 0.2 dan 0.4


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
```


```python
xs = np.linspace(-5, 10, 200)
ks = np.arange(50) #ruang sampel
```


```python
ks
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])



### Discrete PMFs

Binomial distribution, Poisson distribution


```python
pmf_binom = st.binom.pmf(ks, 50, 0.25)
plt.bar(x=ks, height=pmf_binom, label = "Binomial Example (dice)", alpha=0.8)

pmf_poisson = st.poisson.pmf(ks, 30)
plt.bar(x=ks, height=pmf_poisson, label = "Poisson Example (car crash)", alpha=0.8)
plt.legend();

lemparan_dadu = 12
percobaan_leparan = 50

car_crash = 30
pengamatan_car_crash = 30
print(f"Peluang dari {lemparan_dadu} kali dari {percobaan_leparan} kali lemparan dadu adalah ",
      st.binom.pmf(lemparan_dadu, percobaan_leparan, 0.25))
print(f"Peluang dari {car_crash} kali dari {pengamatan_car_crash} rata-rata(rate) pengamatan kecelakaan mobil adalah ",
      st.poisson.pmf(car_crash, pengamatan_car_crash))
```

    Peluang dari 12 kali dari 50 kali lemparan dadu adalah  0.12936760901135547
    Peluang dari 30 kali dari 30 rata-rata(rate) pengamatan kecelakaan mobil adalah  0.07263452647159181



![png](/images/stat_files/stat_244_1.png)


### Continuous Distributions

Uniform, normal, exponential, student-t, log-normal, skew-normal


```python
pdf_uniform = st.uniform.pdf(xs, -4, 10)
plt.plot(xs, pdf_uniform, label="Uniform(-4,6)")

pdf_normal = st.norm.pdf(xs, 5, 2)
plt.plot(xs, pdf_normal, label="Normal(5,2)")

pdf_explonential = st.expon.pdf(xs, loc=-2, scale=2)
plt.plot(xs, pdf_explonential, label="Exponential(0.5)")

pdf_studentt = st.t.pdf(xs, 1)
plt.plot(xs, pdf_studentt, label="Student-t(1)")

pdf_lognorm = st.lognorm.pdf(xs, 1)
plt.plot(xs, pdf_lognorm, label="Lognorm(1)")

pdf_skewnorm = st.skewnorm.pdf(xs, -6)
plt.plot(xs, pdf_skewnorm, label="Skewnorm(5)")


plt.legend()
plt.ylabel("Prob")
plt.xlabel("x")
```




    Text(0.5, 0, 'x')




![png](/images/stat_files/stat_246_1.png)



```python
plt.plot(
    xs,
    st.t.pdf(
        xs,
        1,
        loc=4,
        scale=2
    ),
    label='in build'
)

plt.plot(
    xs,
    st.t.pdf(
        (xs-4)/2,
        1,
        loc=0,
        scale=1
    ),
    label='manually'
)

plt.legend();
```


![png](/images/stat_files/stat_247_0.png)


## Using Distributions

### Function

<img src="Resources/Image/function.PNG">

### Generalisatons

<img src="Resources/Image/generalisation.PNG">

## Nonparametric Statistics

### The Limits of Analytics

<img src="Resources/Image/limits.PNG">

#### Empirical PDF functions

How to calculate the PDF, CDF and SF from empirical data.


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
```


```python
xs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
      5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
ys = [0.2, 0.165, 0.167, 0.166, 0.154, 0.134, 0.117,
      0.108, 0.092, 0.06, 0.031, 0.028, 0.048, 0.077,
      0.103, 0.119, 0.119, 0.103, 0.074, 0.038, 0.003]

plt.scatter(xs, ys)
plt.xlabel("x")
plt.ylabel("Observed PDF");
```


![png](/images/stat_files/stat_251_0.png)



```python
x = np.linspace(min(xs), max(xs), 1000)
y1 = interp1d(xs, ys)(x)
y2 = interp1d(xs, ys, kind='nearest')(x)
y3 = interp1d(xs, ys, kind='quadratic')(x)
y4 = interp1d(xs, ys, kind='cubic')(x)

from scipy.interpolate import splev, splrep
y5 = splev(x, splrep(xs, ys))

plt.scatter(xs, ys, s= 30, label="Data", c="k")
plt.plot(x, y1, label= "Linear (default)")
plt.plot(x, y2, label= "Nearest", alpha=0.7) #stay away
plt.plot(x, y3, label= "Quadratic", ls=':')
plt.plot(x, y4, label= "Cubic", ls='-')
plt.plot(x, y5, label= "Spline", ls='-', alpha =1)
plt.legend();
```


![png](/images/stat_files/stat_252_0.png)



```python
interp1d(xs, ys)(2.3)
```




    array(0.142)



Using the `interp1d` we can now find a probability value for any `x` value.

How can we calculate the CDF and the probability we would find a value between two bounds? Using `scipy.integrate`. Scipy to the rescue once again!

We have many options:

* `scipy.integrate.trapz` for low accuracy but high speed. Accuracy scales as `O(h)`
* `scipy.integrate.simps` for medium accuracy and pretty high speed. Accuracy scales as `O(h^2)`
* `scipy.integrate.quad` for high accuracy and low sped. Arbitrary accuracy.

There are a few more functions, look them up if you're curious.


```python
from scipy.integrate import simps

def get_prob(xs, ys, a, b): # a,b: bounds
    x_vals = np.linspace(a, b , 1000)
    y_vals = interp1d(xs, ys, kind='quadratic')(x_vals)
    return simps(y_vals, x=x_vals)

def get_cdf(xs, ys, v):
    return get_prob(xs, ys, min(xs), v) # from min(xs) to value v

def get_sf(xs, ys, v):
    return 1 - get_cdf(xs, ys, v) # opposite of cdf

print(get_prob(xs, ys, 0, 10))
print("need to be normalized, because max of prob = 1")
```

    1.0012951939288757
    need to be normalized, because max of prob = 1



```python
# WITH NOrmalization
from scipy.integrate import simps

def get_prob(xs, ys, a, b, resolution = 1000): # a,b: bounds
    x_norm = np.linspace(min(xs), max(xs), resolution)
    y_norm = interp1d(xs, ys, kind='quadratic')(x_norm)
    normalization = simps(y_norm, x= x_norm)
    x_vals = np.linspace(a, b , resolution)
    y_vals = interp1d(xs, ys, kind='quadratic')(x_vals)
    return simps(y_vals, x=x_vals)/ normalization

def get_cdf(xs, ys, v):
    return get_prob(xs, ys, min(xs),  v) # from min(xs) to value v

def get_sf(xs, ys, v):
    return 1 - get_cdf(xs, ys, v) # opposite of cdf

print(get_prob(xs, ys, 0, 10))

```

    1.0



```python

```


```python
v1, v2 = 6, 9.3
area = get_prob(xs, ys, v1, v2)

plt.scatter(xs, ys, s= 30, label="Data", color = 'k')
plt.plot(x, y3, ls="-", label="Interpolation")
plt.fill_between(x, 0, y3, where=(x>=v1) & (x<=v2), alpha = 0.2)
plt.annotate(f"p = {area:.3f}", (7, 0.05))
plt.legend();
```


![png](/images/stat_files/stat_258_0.png)



```python
x_new = np.linspace(min(xs), max(xs), 100)
cdf_new = [get_cdf(xs, ys, i) for i in x_new]
cheap_cdf = y3.cumsum() / y3.sum() #rectangle rule, faster

plt.plot(x_new, cdf_new, label= "Interpolated CDF", ls='--')
plt.plot(x, cheap_cdf, label="Super cheap CDF for specific cases")
plt.ylabel("CDF")
plt.xlabel('x')
plt.legend();
```

    C:\Users\abulu\anaconda3\lib\site-packages\scipy\integrate\quadrature.py:376: RuntimeWarning: invalid value encountered in true_divide
      h0divh1 = h0 / h1
    C:\Users\abulu\anaconda3\lib\site-packages\scipy\integrate\quadrature.py:378: RuntimeWarning: invalid value encountered in true_divide
      y[slice1]*hsum*hsum/hprod +



![png](/images/stat_files/stat_259_1.png)



```python

```

# Sampling Distributions

<img src="Resources/Image/sampling.PNG">

<img src="Resources/Image/clt.PNG">

## Sampling Distributions - Practical



Good tool to have. Of course, the first way is the easiest

### Using Scipy

`rvs` is all you need


```python
from scipy.stats import norm, uniform
from scipy.integrate import simps
import numpy as np
import matplotlib.pyplot as plt
```


```python
plt.hist(norm.rvs(loc = 10, scale=2, size= 1000), bins= 50);
```


![png](/images/stat_files/stat_265_0.png)



```python
samples = np.ceil(uniform.rvs(loc = 0, scale=6, size=(1000,3))).sum(axis=1)
plt.hist(samples, bins=30);
```


![png](/images/stat_files/stat_266_0.png)


### Rejection Sampling

Let us say we don't have a nice easy analytic distribution, and that we cannot use one to approximate our distribution. We can brute force our sampling by sampling the uniform distribution and just throwing away points. It works like this:

1. Sample a uniform `x` value
2. Sample a uniform `y` value from `0` to the maximum probability in your PDF
3. If $y > p(x)$, throw out the point.

Easier to see in practise. Lets try and sample from the *unnormalised* distribution $p(x) = \sin(x^2) + 1$ from $0 \rightarrow 4$


```python
def pdf(x):
    return np.sin(x**2) + 1

xs = np.linspace(0, 4, 200)
ps = pdf(xs)
plt.plot(xs, ps)
plt.fill_between(xs, 0, ps, alpha=0.1)
plt.xlim(0, 4)
plt.ylim(0, 2);
```


![png](/images/stat_files/stat_268_0.png)



```python
n = 100
random_x = uniform.rvs(loc= 0, scale= 4, size=n)
random_y = uniform.rvs(loc= 0, scale= 2, size=n)

plt.scatter(random_x, random_y)
plt.plot(xs, ps, c="k")
plt.fill_between(xs, 0, ps, color="k", alpha=0.1)
plt.xlim(0,4)
plt.ylim(0,2)
```




    (0, 2)




![png](/images/stat_files/stat_269_1.png)



```python
passed = random_y <= pdf(random_x)

plt.scatter(
    random_x[passed],
    random_y[passed]
)
plt.scatter(
    random_x[~passed],
    random_y[~passed],
    marker = "x",
    s = 30,
    alpha=0.5
)
plt.plot(xs, ps, c="k")
plt.fill_between(xs, 0, ps, color="k", alpha=0.1)
plt.xlim(0,4)
plt.ylim(0,2)
```




    (0, 2)




![png](/images/stat_files/stat_270_1.png)



```python
n2 = 100000
x_test = uniform.rvs(scale = 4 , size = n2)
x_final = x_test[uniform.rvs(scale=2, size= n2) <= pdf(x_test)]

print(len(x_final))
from scipy.integrate import simps
plt.hist(x_final, density=True, histtype='step', label='sampled dist')
plt.plot(xs, ps / simps(ps, x=xs), c='k', label='empirical PDF')
plt.legend(loc=2);
```

    59527



![png](/images/stat_files/stat_271_1.png)


### Inversion Sampling

This is harder to conceptually understand. But the way I can put it the simplest is that we know that, for all PDFs, the CDF is going to go from 0 to 1. If we can uniformly sample the CDF from 0 to 1, can we invert our function so that we can recover the $x$ value that gives the sampled CDF value? For some functions, yes. For some, no, the math isn't solvable.

The function above is tricky, so let's find a function that's simpler to sample.

Let's say our new PDF is $p(x) = 3 x^2$ from $0 \rightarrow 1$. As opposed to the previous function, this one is normalised.

So, we can find the CDF via

$$ CDF(x) = \int_0^x p(x^\prime) dx^\prime = x^3 $$

Once we have the CDF, we want to invert it. That is, at the moment, we have an $x$ value and get a $y$ value - the CDF. We want to give it a CDF and get the $x$ value:

$$ y = x^3 \rightarrow x = y^3 $$

Which means $y = x^{1/3}$. Or to put the CDF syntax back in, $x = CDF^{1/3}$


```python
def pdf(x):
    return 3 * x**2
def cdf(x):
    return x**3
def icdf(cdf):
    return cdf**(1/3)
```


```python
xs = np.linspace(0,1,100)
pdfs = pdf(xs)
cdfs = cdf(xs)
n = 2000
u_samps = uniform.rvs(size=n)
x_samps = icdf(u_samps)
fig, axes = plt.subplots(ncols=2, figsize=(10,4))
axes[0].plot(xs, pdfs, color="k", label="PDF")
axes[0].hist(x_samps, density=True, histtype="step", label="Sampled dist", bins=50)
axes[1].plot(xs, cdfs, color="k", label="CDF")
axes[1].hlines(u_samps, 0, x_samps, linewidth=0.1, alpha=0.3)
axes[1].vlines(x_samps, 0, u_samps, linewidth=0.1, alpha=0.3)
axes[0].legend(), axes[1].legend()
axes[1].set_xlim(0, 1), axes[1].set_ylim(0, 1);
axes[0].set_xlim(0, 1), axes[0].set_ylim(0, 3);
```


![png](/images/stat_files/stat_274_0.png)



```python
from scipy.interpolate import interp1d

def pdf(x):
    return np.sin(x**2) + 1
xs = np.linspace(0, 4, 10000)
pdfs = pdf(xs)
cdfs = pdfs.cumsum() / pdfs.sum()  # Dangerous

u_samps = uniform.rvs(size=4000)
x_samps = interp1d(cdfs, xs)(u_samps)

fig, axes = plt.subplots(ncols=2, figsize=(10,4))
axes[0].hist(x_samps, density=True, histtype="step", label="Sampled dist", bins=50)
axes[0].plot(xs, pdfs/4.747, color="k", label="Analytic PDF")
axes[0].legend(loc=3), axes[0].set_xlim(0, 4)
axes[1].plot(xs, cdfs, color="k", label="Numeric CDF")
axes[1].hlines(u_samps, 0, x_samps, linewidth=0.1, alpha=0.3)
axes[1].vlines(x_samps, 0, u_samps, linewidth=0.1, alpha=0.3)
axes[1].legend(loc=2), axes[1].set_xlim(0, 4), axes[1].set_ylim(0, 1);
```


![png](/images/stat_files/stat_275_0.png)


## Central Limit Theorem

Let's use a highly non-Gaussian distribution to illustrate the CLT.

To restate from the slides, the CLT asserts that the distribution you get from collection the mean of many samples of an underlying distribution, approaches a normal distribution as you keep getting more and more samples.


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, skewnorm

```


```python
def get_data(n):
    data = np.concatenate((expon.rvs(scale=1, size=n//2), skewnorm.rvs(5, loc=3, size=n//2)))
    np.random.shuffle(data)
    return data
plt.hist(get_data(2000), bins=50);
```


![png](/images/stat_files/stat_278_0.png)


Definitely not a normal distribution. So, `get_data` represents some way of sampling an underlying distribution. Normally in experiments, you can't ask for infinite amounts of data, you have to gather it, and that takes time, effort and money. Let's say we can only get 10 data points, and let's calculate the mean of those ten.


```python
d10 = get_data(10)
print(d10.mean())
```

    2.732196534900356


Right, but if we re-run the above a few times, we'll see that the answer changes a lot with each run!

The magic is in the fact that the amount the answer changes by is not random. Well, between runs, it is, but I mean that it has certain statistical properties. Let's see what happens if we manage to get 10 data points 1000 times.


```python
means = [get_data(100).mean() for i in range (1000)]
plt.hist(means, bins =50 )
print(np.std(means))
```

    0.07915320510427529



![png](/images/stat_files/stat_282_1.png)


The distribution of means is becoming more and more like a normal distribution. The peak of the distribution is approaching the true mean of the population, and the width of this distribution is a function of both the width of the population and the number of samples we used. If we used 100 data points, we would expect less scatter between samples, right?

Let's test.


```python
num_samps = [10, 50, 100, 500, 1000, 5000, 10000]
stds = []
for n in num_samps:
    stds.append(np.std([get_data(n).mean() for i in range(1000)]))
plt.plot(num_samps, stds, 'o', label="Obs scatter")
plt.plot(num_samps, 1 / np.sqrt(num_samps), label="Rando function", alpha=0.5)
plt.legend();
```


![png](/images/stat_files/stat_284_0.png)


Surprise, the rando function isn't arbitrary, its $1/\sqrt{N}$.

The distribution of means has standard deviation proportional to the underlying distribution divided by the root of the number of samples.

Or another way to say this, if you have $N$ samples, the mean of your samples is distributed as per a normal around the true mean, with standard deviation $\sigma/\sqrt{N}$.

Or *another* way of saying this, is that if you go from $N_1$ data points to $N_2$ data points, you can determine the mean $\sqrt{N_2/N_1}$ more accurately. 4 times as many samples doesn't give 4 times more accuracy, only double the accuracy.


```python
# recap
plt.hist([get_data(100).mean() for i in range(1000)], bins=25);
plt.xlim(0.5,3);
```


![png](/images/stat_files/stat_286_0.png)


### Recap

1. Distribution of sample means approaches a normal
2. The width is determined by the number of points use to compute each sample mean

*****

If you have $N$ samples, the mean of your samples is distributed as per a normal around the true mean, with standard deviation $\sigma/\sqrt{N}$.

Or *another* way of saying this, is that if you go from $N_1$ data points to $N_2$ data points, you can determine the mean $\sqrt{N_2/N_1}$ more accurately. 4 times as many samples doesn't give 4 times more accuracy, only double the accuracy.


```python
n = 1000
data = get_data(n)
sample_mean = np.mean(data)
uncert_mean = np.std(data) / np.sqrt(n)
print(f"We have determined the mean of population to be {sample_mean:.2f} +- {uncert_mean:.2f}")
```

    We have determined the mean of population to be 2.40 +- 0.05


By convention, adding a $\pm$ represents the uncertainty to $1\sigma$ for a normal distribution. To refresh, $1\sigma$ is notation for 1 standard deviation away from the mean.


```python
from scipy.stats import norm
xs = np.linspace(sample_mean - 0.2, sample_mean + 0.2, 100)
ys = norm.pdf(xs, sample_mean, uncert_mean)
plt.plot(xs, ys)
plt.xlabel("Pop mean")
plt.ylabel("Probability");
```


![png](/images/stat_files/stat_290_0.png)


## Summary

1. Familiarise yourself with common distributions
2. Data may not always look like a common distribution
3. **Try things. Half Science, half art**

# Hypothesis Testing

1. Basic Tests
2. Comparing Distributions
3. Nonparametric Tests

## What is Hypothesis Testing?

The ability to **ask** and **quatitatively answer** questions.
More formally, if you formulate two hyphotheses, **how confidently** can you point to the **true one**

### Examples

1. Tom is winning a lot at dice. Are his dice **loaded**?
2. Does the vehicle meet **emission standards**?
3. Is the evidence of **election fraud**?
4. Does an incoming patient has **diabetes**?
5. What is the chance that a **giant asteroid** will hit the planet in the next thousand years?


## Practical Example

### Loaded Dice

Tommy seems to be winning a lot of games recently. You are, in fact, *highly suspicious* of his treasured dice. So you've been recording the result of every role, and they are in `loaded_500.txt`.

Let's try and answer the simplest question we can: Is *Tommy* rolling too many sixes?

Let's answer the question rigorously.

1. Visualise the data. Make sure we understand it.
2. Reduce or quantify the data
3. Pose our hypothesis (and visualise)
4. Calculate


```python
import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("Resources/Datasets/loadedDie/loaded_500.txt")
```

#### Visualizing


```python
unique, counts = np.unique(data, return_counts = True)
print(unique, counts)
plt.hist(data, bins= 50);
```

    [1. 2. 3. 4. 5. 6.] [70 84 86 74 88 98]



![png](/images/stat_files/stat_297_1.png)


#### Reduce or quantify data

We don't need each individual roll. We really just want the total number of sixes rolled and the total number of rolls.


```python
num_sixes = (data == 6).sum()
num_total = data.size
print(num_sixes, num_total)
```

    98 500


#### Pose our Hypothesis

We have two outcomes when we roll a dice. We roll a six, or we roll something else. With a fair die (our null hypothesis), $p(6) = \frac{1}{6}$. As this is discrete with two options, we are looking at the *binomial* distribution.

What is the chance we roll 98 *or more* sixes with a fair die?


```python
from scipy.stats import binom
n = np.arange(num_total)
prob_n = binom.pmf(n, num_total, 1/6)
plt.plot(n, prob_n, label = "Prob num")
plt.axvline(num_total / 6, ls="--", lw=1, label="Mean num")
plt.axvline(num_sixes, ls=":", color="#ff7272", label="Obs num")
plt.xlabel(f"Num sixes rolled out of {num_total} rolls")
plt.ylabel("Probability")
plt.legend();
```


![png](/images/stat_files/stat_301_0.png)



```python
# using survival function
d = binom(num_total, 1 / 6)
plt.plot(n, d.sf(n))
plt.axvline(num_sixes, ls="--")
sf = d.sf(num_sixes)
plt.axhline(sf, ls="--")
plt.xlabel("Num Sixes")
plt.ylabel("SF")
print(f"Only {sf * 100:.1f}% of the time with a fair dice you'd roll this many or more sixes.")
plt.legend();
```

    No handles with labels found to put in legend.


    Only 3.7% of the time with a fair dice you'd roll this many or more sixes.



![png](/images/stat_files/stat_302_2.png)


## Meteorite Impacts - Practical Example

NASA has helpfully provided a dataset which lists recorded meteorite impacts - get it in `"Meteorite_Landings.csv"`. Can we utilise this dataset to predict the chance that, within 1000 years, a high-impact meteor will strike the planet?

Let's define high-impact as an asteroid greater than 1km in diameter.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Resources/Datasets/2_Meteorites/Meteorite_Landings.csv")
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



### Investigate and explore Data

Now, from the fact that we have different numbers of non-null objects in the previous info, we should make sure the columns we want to work with all have sensible values.


```python
df2 = df.dropna(subset=["mass", 'year'] )
df2 = df2[df2['mass'] > 0]
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
pd.plotting.scatter_matrix(df[['mass','year','reclat','reclong']], figsize=(7,7));
```


![png](/images/stat_files/stat_308_0.png)



```python
import seaborn as sb
```


```python
sb.pairplot(df[['mass','year','reclat','reclong']]);
```


![png](/images/stat_files/stat_310_0.png)


An important thing to note is we can make out the continents in the scatter. A real analysis would have to look at detection efficiency properly (something like the percentage of meteorites we successfully observe as a function of location on the planet), but we'll just keep in mind that this dataset only captures some meteorites and only in populated areas.

Now, mass is a positive value that spans many orders of magnitude, so it'll probably be easier to look at log mass instead of mass itself.


```python
year = df2['year']
masses = df2['mass']
logmass = np.log(masses)
sb.distplot(logmass, bins=100, kde=False);
```


![png](/images/stat_files/stat_312_0.png)


Yup, this is far easier to quantify than the mass distribution. As to the years, obviously there is an issue here - our technology has changed a lot over the last few thousand years, and so our detection efficiency should also take this into account. However, we don't have the data to make that analysis in this dataset. So what we can do is take our peak year as a pseudo-worst case.


```python
counts = year.value_counts()
sb.scatterplot(counts.index, counts)
#sb.scatter(counts.index, counts)
plt.xlim(1980, 2020);
plt.ylabel("");

```


![png](/images/stat_files/stat_314_0.png)


So, lets quantify our log-mass distribution from before. We can start and see if a normal works well.


```python
from scipy.stats import norm, skewnorm, lognorm
ms = np.linspace(-5, 20, 100)

mean, std = logmass.mean(), np.std(logmass)
pdf_norm = norm.pdf(ms, mean, std)
sb.distplot(logmass, bins=100, label='Data')
plt.plot(ms, pdf_norm, label='Normal Dist',ls='--')
plt.legend();
```


![png](/images/stat_files/stat_316_0.png)


Not the best... Lets try both a skew-norm and a log-norm.


```python
p_lognorm = lognorm.fit(logmass)
pdf_logmass = lognorm.pdf(ms, *p_lognorm)

p_skewnorm = skewnorm.fit(logmass)
pdf_skewnorm = skewnorm.pdf(ms, *p_skewnorm)

sb.distplot(logmass, bins=100, label='Data')
plt.plot(ms, pdf_logmass,label='Logmass')
plt.plot(ms, pdf_skewnorm,label='Skewnorm')
plt.plot(ms, pdf_norm, label='Normal',ls='--')
plt.xlabel('Logmass')
plt.ylabel("probability")
plt.legend();
```


![png](/images/stat_files/stat_318_0.png)


So either the log-norm or skew-norm looks like an adequate fit to the data. We want to extrapolate this distribution out to a mass we're concerned about - the mass of a 1km diameter meteor.


```python
mass_of_doom = np.log((4/3) * np.pi * 500**3 * 1600 * 1000)  # Just using a spherical approximation and some avg density
mass_of_doom #volume with convertion
```




    34.36175044077777



So where does this value lie on our distribution. We'll go with the lognorm for now.


```python
ms2 = np.linspace(-5, 40, 200)
plt.plot(ms2, lognorm.logsf(ms2, *p_lognorm))
plt.axvline(mass_of_doom, ls="--")
plt.xlabel("log mass")
plt.ylabel("log probability")
plt.title("Log probability of asteroid being over given mass");
```


![png](/images/stat_files/stat_322_0.png)


So we have here the probability of an asteroid being above a certain mass when it hits Earth. But to answer the question "What is the probability that one or more asteroids of high mass strike Earth in 1000 years?" we need to factor in the actual time component. Assume that in the next 1000 years, we predict to have $N$ impacts.

$$P(>=1\  {\rm highmass}) = 1 - P(0\ {\rm highmass}) = 1 - P(N\ {\rm not\_highmass}) = 1 - P({\rm not\_highmass})^N$$

Imagine a similar question: Prob of getting no sixes in 5 rolls. Well its $$\frac{5}{6}\times\frac{5}{6}\times\frac{5}{6}\times\frac{5}{6}\times\frac{5}{6} = \left(\frac{5}{6}\right)^5 $$
The prob of getting one or more sixes is then $1 - (5/6)^5$.

So to give a number, we need to calculate $N$ from the yearly rate, number of years, and our detection efficiency and use that with the probability that any given impact is not high mass.


```python
prob_small_mass = lognorm.logcdf(mass_of_doom, *p_lognorm)
frac_sky_monitored = 0.2 #pengamatan real
num_years = 1000
num_events = num_years * counts.max() / frac_sky_monitored # 20 persen dari 1000 tahun
prob_bad = 1 - np.exp(num_events * prob_small_mass)
print(f"Prob a > 1 km asteroid impacts with {num_years} years is {prob_bad * 100:.2f}%")
```

    Prob a > 1 km asteroid impacts with 1000 years is 0.67%


## Election Scenario - Practical Example

* Candidate A won the state by an average of 4% points, however they lost District 29 to candidate B by 22%, making it a significant outlier.
* The final results for the distrct are 39% Candidate A, 61% Candidate B.
* You are tasked to investigate this to determine if it is worth a thorough follow-up.
* You call a a hundred members of the general public who reported as voting. 48 out of the 100 reported voting for Candidate A.
* What do you report?


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
```

But what question are we actually wanting to answer? Maybe this is a good one:

*What is the chance that more than Candidate A got more votes than reported?*

Or to put this another way:
* Null Hypothesis - 39% of votes went to Candidate A and 61% to Candidate B
* Alternative Hypothesis - >39% of votes went to Candidate A and <61% to Candidate B


```python
sigma = np.sqrt(0.39 * 0.61 / 100)
reported = 0.39
sample = 0.48

xs = np.linspace(0, 1, 500)
ps = norm.pdf(xs, sample, sigma)
plt.plot(xs, ps, label="Underlying sample probability")
plt.axvline(reported, ls="--", label="Reported Proportion")
plt.fill_between(xs, ps, 0, alpha=0.2, where=xs>=reported, label="prob")
plt.legend(bbox_to_anchor=(1.6, 1));
```


![png](/images/stat_files/stat_328_0.png)


All the above plot tells us is that things might indeed look fishy. But lets look into the CDF for specifics and then quantify the probability.


```python
prob_more = norm.sf(reported, sample, sigma)
print(f"There is a {100 * prob_more:.1f}% chance that Candidate A would have received more votes")
```

    There is a 96.7% chance that Candidate A would have received more votes


If we want to phrase this in terms of z-scores for a one-tail test and checking if we have a p-value of < 0.05


```python
z_score = (sample - reported) / sigma
z_score_needed_for_significance = norm.ppf(0.95)
print(f"{z_score:.3f} is larger than {z_score_needed_for_significance:.3f}, so we are significant")
print(f"Have p-value {norm.sf(sample, reported, sigma):.3f}")
```

    1.845 is larger than 1.645, so we are significant
    Have p-value 0.033


So what does this mean? We should probably tell our supervisors that this is a significant result (p < 0.05) and warrants a deeper investigation.

Follow up: What if instead of asking *What is the chance that more than Candidate A got more votes than reported?*, we instead asked *What is the chance that more than Candidate A got a significantly different amount of votes than reported?*

Ie we now move from the one-tail case to the two-tailed? Let's continue assuming someone has asked if it meets $p = 0.05$ criteria


```python
xs = np.linspace(0, 1, 500)
ps = norm.pdf(xs, reported, sigma)

plt.plot(xs, ps, label="Proportion uncert")
plt.axvline(sample, ls="--", label="Sample")
limits = norm.ppf([0.025, 0.975], reported, sigma)
plt.fill_between(xs, ps, 0, alpha=0.2, where=(xs<limits[0])|(xs>limits[1]), label="Significant")
plt.legend(loc=2, bbox_to_anchor=(1, 1))
plt.xlim(0.2, 0.65);
```


![png](/images/stat_files/stat_334_0.png)


So it looks like it's not $p<0.05$ significant for the two-tailed case. Or more formally:


```python
# Using z-scores
z_score = (sample - reported) / sigma
z_score_needed_for_significance = norm.ppf(0.975)
print(f"{z_score:.3f} is less than {z_score_needed_for_significance:.3f}, so we aren't significant")

# Using p-values
p_value_two_tailed = 2 * norm.sf(z_score)
print(f"{p_value_two_tailed:.3f} is > 0.05")
```

    1.845 is less than 1.960, so we aren't significant
    0.065 is > 0.05


## Pearson's $\chi^2$ Test

Let's revisit Tommy, the cheating bastard he is. Last time we tried to answer the question *"Is Tommy rolling too many sixes?"*.

Now, time to ask a different question. What if we're not worried about just the number of sixes, but what to ask *"Is the distribution of rolls we get consistent with a fair die?"*

The Pearson's $\chi^2$ test for rolling a die gives

$$ \chi^2 = \sum_{i=1}^{6} \frac{(C_i - E_i)^2}{E_i} $$

But what does this $\chi^2$ value mean? We can convert it to a probability given the $\chi^2$ distribution, with 5 degrees of freedom (six sides - 1)


```python

```

Comparing numbers and thresholds like we've just done and compare distributions


```python
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("Resources/Datasets/loadedDie/loaded_500.txt")
unique, counts = np.unique(data, return_counts=True)
plt.hist(data, bins=50);
```


![png](/images/stat_files/stat_340_0.png)



```python
data.size
expected = data.size/6
chi2_val = np.sum((counts - expected)**2/ expected)
print(chi2_val)
```

    6.112



```python
counts, unique
```




    (array([70, 84, 86, 74, 88, 98], dtype=int64), array([1., 2., 3., 4., 5., 6.]))




```python
from scipy.stats import chi2
chi2s = np.linspace(0, 15, 500)
prob = chi2.pdf(chi2s, 5)

plt.plot(chi2s, prob, label="Distribution")
plt.axvline(chi2_val, label="$\chi2$", ls="--")
plt.fill_between(chi2s, prob, 0, where=(chi2s>=chi2_val), alpha=0.1)
plt.legend();
print(f"Our p-value is {chi2.sf(chi2_val, 5):.3f}")
```

    C:\Users\abulu\anaconda3\lib\importlib\_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    C:\Users\abulu\anaconda3\lib\importlib\_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    C:\Users\abulu\anaconda3\lib\importlib\_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)


    Our p-value is 0.295



![png](/images/stat_files/stat_343_2.png)



```python
from scipy.stats import chisquare
chisq, p = chisquare(counts, expected)
print(f"We have a chi2 of {chisq:.2f} with a p-value of {p:.3f}")
```

    We have a chi2 of 6.11 with a p-value of 0.295


This leads to an interesting point. In the first test with only number of sixes, we had significance. In the last test, we only care about the occurence of sixes. In this example, our statistical power was spread out over many faces. Both tests are valid. Be careful about finding many ways to test a hypothesis - this is called "significance hunting".

In general, the more specific your "question", the more powerful a test you can use.

Also, whilst in this example we used a one-sided distribution (which is asking the question if our observed distrubition is *too discrepant* from the underlying), we could also use a two-sided distribution, which also tests to see if our observed distribution is *too similar* from the underlying. For example, imagine rolling a dice 600 times and getting exactly. 100 of each number. And then getting that result when you do it again and again. With random numbers, there is such a thing as being too perfect.


```python

```


```python

```

# Conclusion


```python

```


```python

```
