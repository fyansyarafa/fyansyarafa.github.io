---
title: "Childbird in Vienna General Hospital"
data: 2020-09-25
tags: []
header:
excerpt: "Trying to analyze the effect of midwives do handwashing procedure to reduce proportion of deaths of childbirths."
mathjax: "true"
toc: true
toc_sticky: false

---
## 1. Meet Dr. Ignaz Semmelweis
<p><img style="float: left;margin:5px 20px 5px 1px" src="https://assets.datacamp.com/production/project_20/img/ignaz_semmelweis_1860.jpeg"></p>
<!--
<img style="float: left;margin:5px 20px 5px 1px" src="https://assets.datacamp.com/production/project_20/datasets/ignaz_semmelweis_1860.jpeg">
-->
<p>This is Dr. Ignaz Semmelweis, a Hungarian physician born in 1818 and active at the Vienna General Hospital. If Dr. Semmelweis looks troubled it's probably because he's thinking about <em>childbed fever</em>: A deadly disease affecting women that just have given birth. He is thinking about it because in the early 1840s at the Vienna General Hospital as many as 10% of the women giving birth die from it. He is thinking about it because he knows the cause of childbed fever: It's the contaminated hands of the doctors delivering the babies. And they won't listen to him and <em>wash their hands</em>!</p>
<p>In this notebook, we're going to reanalyze the data that made Semmelweis discover the importance of <em>handwashing</em>. Let's start by looking at the data that made Semmelweis realize that something was wrong with the procedures at Vienna General Hospital.</p>


```python
# importing modules
import pandas as pd

# Read datasets/yearly_deaths_by_clinic.csv into yearly
yearly = pd.read_csv('datasets/yearly_deaths_by_clinic.csv')

# Print out yearly
yearly
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
      <th>year</th>
      <th>births</th>
      <th>deaths</th>
      <th>clinic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1841</td>
      <td>3036</td>
      <td>237</td>
      <td>clinic 1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1842</td>
      <td>3287</td>
      <td>518</td>
      <td>clinic 1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1843</td>
      <td>3060</td>
      <td>274</td>
      <td>clinic 1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1844</td>
      <td>3157</td>
      <td>260</td>
      <td>clinic 1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1845</td>
      <td>3492</td>
      <td>241</td>
      <td>clinic 1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1846</td>
      <td>4010</td>
      <td>459</td>
      <td>clinic 1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1841</td>
      <td>2442</td>
      <td>86</td>
      <td>clinic 2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1842</td>
      <td>2659</td>
      <td>202</td>
      <td>clinic 2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1843</td>
      <td>2739</td>
      <td>164</td>
      <td>clinic 2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1844</td>
      <td>2956</td>
      <td>68</td>
      <td>clinic 2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1845</td>
      <td>3241</td>
      <td>66</td>
      <td>clinic 2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1846</td>
      <td>3754</td>
      <td>105</td>
      <td>clinic 2</td>
    </tr>
  </tbody>
</table>
</div>



## 2. The alarming number of deaths
<p>The table above shows the number of women giving birth at the two clinics at the Vienna General Hospital for the years 1841 to 1846. You'll notice that giving birth was very dangerous; an <em>alarming</em> number of women died as the result of childbirth, most of them from childbed fever.</p>
<p>We see this more clearly if we look at the <em>proportion of deaths</em> out of the number of women giving birth. Let's zoom in on the proportion of deaths at Clinic 1.</p>


```python
# Calculate proportion of deaths per no. births
yearly['proportion_deaths'] = yearly['deaths'] / yearly['births']

# Extract clinic 1 data into yearly1 and clinic 2 data into yearly2
yearly1 = yearly[yearly['clinic'] == 'clinic 1']
yearly2 = yearly[yearly['clinic'] == 'clinic 2']

# Print out yearly1
yearly1
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
      <th>year</th>
      <th>births</th>
      <th>deaths</th>
      <th>clinic</th>
      <th>proportion_deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1841</td>
      <td>3036</td>
      <td>237</td>
      <td>clinic 1</td>
      <td>0.078063</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1842</td>
      <td>3287</td>
      <td>518</td>
      <td>clinic 1</td>
      <td>0.157591</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1843</td>
      <td>3060</td>
      <td>274</td>
      <td>clinic 1</td>
      <td>0.089542</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1844</td>
      <td>3157</td>
      <td>260</td>
      <td>clinic 1</td>
      <td>0.082357</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1845</td>
      <td>3492</td>
      <td>241</td>
      <td>clinic 1</td>
      <td>0.069015</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1846</td>
      <td>4010</td>
      <td>459</td>
      <td>clinic 1</td>
      <td>0.114464</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Death at the clinics
<p>If we now plot the proportion of deaths at both clinic 1 and clinic 2  we'll see a curious pattern…</p>


```python
# This makes plots appear in the notebook
%matplotlib inline

# Plot yearly proportion of deaths at the two clinics
ax = yearly1.plot(x='year', y='proportion_deaths', label='clinic 1')
yearly2.plot(x='year', y='proportion_deaths', label='clinic 2',ax=ax);
```


![png](/images/Vienna/notebook_files/notebook_5_0.png)


## 4. The handwashing begins
<p>Why is the proportion of deaths constantly so much higher in Clinic 1? Semmelweis saw the same pattern and was puzzled and distressed. The only difference between the clinics was that many medical students served at Clinic 1, while mostly midwife students served at Clinic 2. While the midwives only tended to the women giving birth, the medical students also spent time in the autopsy rooms examining corpses. </p>
<p>Semmelweis started to suspect that something on the corpses, spread from the hands of the medical students, caused childbed fever. So in a desperate attempt to stop the high mortality rates, he decreed: <em>Wash your hands!</em> This was an unorthodox and controversial request, nobody in Vienna knew about bacteria at this point in time. </p>
<p>Let's load in monthly data from Clinic 1 to see if the handwashing had any effect.</p>


```python
# Read datasets/monthly_deaths.csv into monthly
monthly = pd.read_csv('datasets/monthly_deaths.csv')
monthly['date'] = pd.to_datetime(monthly['date'])
# Calculate proportion of deaths per no. births
monthly["proportion_deaths"] = monthly['deaths'] / monthly['births']

# Print out the first rows in monthly
monthly.iloc[0]
```




    date                 1841-01-01 00:00:00
    births                               254
    deaths                                37
    proportion_deaths               0.145669
    Name: 0, dtype: object




```python
monthly.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 98 entries, 0 to 97
    Data columns (total 4 columns):
    date                 98 non-null datetime64[ns]
    births               98 non-null int64
    deaths               98 non-null int64
    proportion_deaths    98 non-null float64
    dtypes: datetime64[ns](1), float64(1), int64(2)
    memory usage: 3.1 KB


## 5. The effect of handwashing
<p>With the data loaded we can now look at the proportion of deaths over time. In the plot below we haven't marked where obligatory handwashing started, but it reduced the proportion of deaths to such a degree that you should be able to spot it!</p>


```python
# Plot monthly proportion of deaths
import matplotlib.pyplot as plt
ax = monthly.plot(x = 'date', y='proportion_deaths')
plt.ylabel('Proportion deaths')
```




    <matplotlib.text.Text at 0x7f8c1fc80048>




![png](/images/Vienna/notebook_files/notebook_10_1.png)


## 6. The effect of handwashing highlighted
<p>Starting from the summer of 1847 the proportion of deaths is drastically reduced and, yes, this was when Semmelweis made handwashing obligatory. </p>
<p>The effect of handwashing is made even more clear if we highlight this in the graph.</p>


```python
# Date when handwashing was made mandatory
import pandas as pd
handwashing_start = pd.to_datetime('1847-06-01')

# Split monthly into before and after handwashing_start
before_washing = monthly[monthly['date'] < handwashing_start]
after_washing = monthly[monthly['date'] >= handwashing_start]

# Plot monthly proportion of deaths before and after handwashing

ax = before_washing.plot(x = 'date', y= 'proportion_deaths', label='before_washing')
after_washing.plot(x = 'date', y= 'proportion_deaths', label='after_washing', ax=ax)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f8c1f9c15c0>




![png](/images/Vienna/notebook_files/notebook_12_1.png)


## 7. More handwashing, fewer deaths?
<p>Again, the graph shows that handwashing had a huge effect. How much did it reduce the monthly proportion of deaths on average?</p>


```python
# Difference in mean monthly proportion of deaths due to handwashing

before_proportion = before_washing["proportion_deaths"]
after_proportion = after_washing["proportion_deaths"]
mean_diff = after_proportion.mean() - before_proportion.mean()
mean_diff
```




    -0.08395660751183336




```python
before_proportion.mean()
```




    0.10504998260908789



## 8. A Bootstrap analysis of Semmelweis handwashing data
<p>It reduced the proportion of deaths by around 8 percentage points! From 10% on average to just 2% (which is still a high number by modern standards). </p>
<p>To get a feeling for the uncertainty around how much handwashing reduces mortalities we could look at a confidence interval (here calculated using the bootstrap method).</p>


```python
# A bootstrap analysis of the reduction of deaths due to handwashing
boot_mean_diff = []
for i in range(3000):
    boot_before = before_proportion.sample(frac=1, replace=True)
    boot_after = after_proportion.sample(frac=1, replace=True)
    boot_mean_diff.append( boot_after.mean() - boot_before.mean() )

# Calculating a 95% confidence interval from boot_mean_diff
confidence_interval = pd.Series(boot_mean_diff).quantile([0.025, 0.975])
confidence_interval
```




    0.025   -0.101600
    0.975   -0.067949
    dtype: float64



## 9. The fate of Dr. Semmelweis
<p>So handwashing reduced the proportion of deaths by between 6.7 and 10 percentage points, according to a 95% confidence interval. All in all, it would seem that Semmelweis had solid evidence that handwashing was a simple but highly effective procedure that could save many lives.</p>
<p>The tragedy is that, despite the evidence, Semmelweis' theory — that childbed fever was caused by some "substance" (what we today know as <em>bacteria</em>) from autopsy room corpses — was ridiculed by contemporary scientists. The medical community largely rejected his discovery and in 1849 he was forced to leave the Vienna General Hospital for good.</p>
<p>One reason for this was that statistics and statistical arguments were uncommon in medical science in the 1800s. Semmelweis only published his data as long tables of raw data, but he didn't show any graphs nor confidence intervals. If he would have had access to the analysis we've just put together he might have been more successful in getting the Viennese doctors to wash their hands.</p>


```python
# The data Semmelweis collected points to that:
doctors_should_wash_their_hands = True
```
