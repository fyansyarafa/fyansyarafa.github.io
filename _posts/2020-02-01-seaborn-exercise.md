---
permalink: /projects/:title
title: "Seaborn Excercise"
data: 2020-02-01
tags: [python, matplotlib, data visualization, seaborn]
header:
excerpt: "Seaborn Portofolio"
mathjax: "true"
---

Time to practice your new seaborn skills! Try to recreate the plots below (don't worry about color schemes, just the plot itself.

## The Data

We will be working with a famous titanic data set for these exercises. Later on in the Machine Learning section of the course, we will revisit this data, and use it to predict survival rates of passengers. For now, we'll just focus on the visualization of the data with seaborn:


```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
sns.set_style('whitegrid')
```


```python
titanic = sns.load_dataset('titanic')
```


```python
titanic.head()
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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



# Exercises

** Recreate the plots below using the titanic dataframe. There are very few hints since most of the plots can be done with just one or two lines of code and a hint would basically give away the solution. Keep careful attention to the x and y labels for hints.**

** *Note! In order to not lose the plot image, make sure you don't code in the cell that is directly above the plot, there is an extra cell above that one which won't overwrite that plot!* **


```python
sns.jointplot(x='fare',y='age',data=titanic)
```

    C:\Users\User\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    C:\Users\User\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "





    <seaborn.axisgrid.JointGrid at 0x1854f6e3c88>




![png](/images/seaweb_files/seaweb_7_2.png)



```python

```




    <seaborn.axisgrid.JointGrid at 0x11d0389e8>




![png](/images/seaweb_files/seaweb_8_1.png)



```python
sns.set_style(style='whitegrid')
sns.distplot(titanic['fare'],kde=False,color='red')
```

    C:\Users\User\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "





    <matplotlib.axes._subplots.AxesSubplot at 0x1854fbd2d30>




![png](/images/seaweb_files/seaweb_9_2.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x11fc5ca90>




![png](/images/seaweb_files/seaweb_10_1.png)



```python
sns.boxplot(x='class',y='age',data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1854fc5d550>




![png](/images/seaweb_files/seaweb_11_1.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x11f23da90>




![png](/images/seaweb_files/seaweb_12_1.png)



```python

sns.swarmplot(x='class',y='age',data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1855022c208>




![png](/images/seaweb_files/seaweb_13_1.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x11f215320>




![png](/images/seaweb_files/seaweb_14_1.png)



```python
sns.countplot(x='sex',data=titanic)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18551268160>




![png](/images/seaweb_files/seaweb_15_1.png)



```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x11f207ef0>




![png](/images/seaweb_files/seaweb_16_1.png)



```python
titanic.head(3)
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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
ticorr=titanic.corr()
```


```python

```




    <matplotlib.text.Text at 0x11d72da58>




![png](/images/seaweb_files/seaweb_19_1.png)



```python
sns.heatmap(ticorr,annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1855135db70>




![png](/images/seaweb_files/seaweb_20_1.png)



```python
import matplotlib.pyplot as plt
g=sns.FacetGrid(titanic,col='sex')
g.map(plt.hist,'age')
```




    <seaborn.axisgrid.FacetGrid at 0x1855145c2b0>




![png](/images/seaweb_files/seaweb_21_1.png)



```python

```




    <seaborn.axisgrid.FacetGrid at 0x11d81c240>




![png](/images/seaweb_files/seaweb_22_1.png)


# Great Job!

### That is it for now! We'll see a lot more of seaborn practice problems in the machine learning section!
