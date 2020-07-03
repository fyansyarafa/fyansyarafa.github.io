---
title: "Pandas Data Visualization Exercise"
data: 2020-02-01
tags: [python, pandas]
header:
excerpt: "Pandas Portofolio"
mathjax: "true"
---

# Pandas Data Visualization Exercise

This is just a quick exercise for you to review the various plots we showed earlier. Use **df3** to replicate the following plots.


```python
import pandas as pd
import matplotlib.pyplot as plt
df3 = pd.read_csv('df3')
%matplotlib inline
```


```python
df3.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 4 columns):
    a    500 non-null float64
    b    500 non-null float64
    c    500 non-null float64
    d    500 non-null float64
    dtypes: float64(4)
    memory usage: 15.7 KB



```python
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.336272</td>
      <td>0.325011</td>
      <td>0.001020</td>
      <td>0.401402</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.980265</td>
      <td>0.831835</td>
      <td>0.772288</td>
      <td>0.076485</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.480387</td>
      <td>0.686839</td>
      <td>0.000575</td>
      <td>0.746758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.502106</td>
      <td>0.305142</td>
      <td>0.768608</td>
      <td>0.654685</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.856602</td>
      <td>0.171448</td>
      <td>0.157971</td>
      <td>0.321231</td>
    </tr>
  </tbody>
</table>
</div>



** Recreate this scatter plot of b vs a. Note the color and size of the points. Also note the figure size. See if you can figure out how to stretch it in a similar fashion. Remeber back to your matplotlib lecture...**


```python
df3.plot.scatter(x='a',y='b')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1176a7da0>




![png](/images/output_5_1.png)



```python
df3.plot.scatter(x='a',y='b',figsize=(12.1,3),c='r',s=50,edgecolor='black')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22dcac18ef0>




![png](/images/output_6_1.png)


** Create a histogram of the 'a' column.**


```python
import seaborn as sns
sns.set_style('darkgrid')
df3['a'].plot.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22dcb90a898>




![png](/images/output_8_1.png)


** These plots are okay, but they don't look very polished. Use style sheets to set the style to 'ggplot' and redo the histogram from above. Also figure out how to add more bins to it.***


```python
plt.style.use('ggplot')
```


```python
import seaborn as sns
sns.set_style('darkgrid')
df3['a'].plot.hist(bins=20,alpha=0.5)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22dcba5a6d8>




![png](/images/output_11_1.png)


** Create a boxplot comparing the a and b columns.**


```python
df3[['a','b']].plot.box()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22dcd0f3d30>




![png](/images/output_13_1.png)


** Create a kde plot of the 'd' column **


```python
df3['d'].plot.kde()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22dcd1619b0>




![png](/images/output_15_1.png)


** Figure out how to increase the linewidth and make the linestyle dashed. (Note: You would usually not dash a kde plot line)**


```python
df3['d'].plot.density(ls='--',lw=3)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22dcd290908>




![png](/images/output_17_1.png)


** Create an area plot of all the columns for just the rows up to 30. (hint: use .ix).**


```python
baru=df3.ix[0:30]
```

    C:\Users\User\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      """Entry point for launching an IPython kernel.



```python
baru.plot.area(alpha=0.5,stacked=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22dcd46b898>




![png](/images/output_20_1.png)



```python

```

## Bonus Challenge!
Note, you may find this really hard, reference the solutions if you can't figure it out!
** Notice how the legend in our previous figure overlapped some of actual diagram. Can you figure out how to display the legend outside of the plot as shown below?**

** Try searching Google for a good stackoverflow link on this topic. If you can't find it on your own - [use this one for a hint.](http://stackoverflow.com/questions/23556153/how-to-put-legend-outside-the-plot-with-pandas)**


```python
#f = plt.figure()
baru.plot.area(alpha=0.5,stacked=True)
plt.legend(loc='best',bbox_to_anchor=(1.0,0.5))
```


![png](/images/output_23_0.png)


# Great Job!
