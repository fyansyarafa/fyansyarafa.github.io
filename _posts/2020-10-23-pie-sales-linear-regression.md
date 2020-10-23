---
title: "Pie Sales Multiple Linear Regression"
data: 2020-10-23
tags: []
header:
excerpt: ""
mathjax: "true"
toc: true
toc_sticky: false
---

# Tujuan

Ditujukan untuk menghasilkan model prediktif (*multiple linear regression*) untuk melakukan prediksi terhadap penjualan pie (`pie_sales`) berdasarkan fitur *independet* (`week`, `price`, `advertising`).

# Import Library & Data

## Libraries


LIbrary yang akan digunakan
*   Pandas
*   Numpy
* matplotlib.pyplot
* seaborn
* plotly express
* plotly graph object
* statsmodels.api
* sklearn linear model





```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


```

## Data


```
df = pd.read_csv('https://raw.githubusercontent.com/rc-dbe/dti/main/data/pie-sales.csv', sep =";")
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
      <th>week</th>
      <th>pie_sales</th>
      <th>price</th>
      <th>advertising</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>350</td>
      <td>5.5</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>460</td>
      <td>7.5</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>350</td>
      <td>8.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>430</td>
      <td>8.0</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>350</td>
      <td>6.8</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



Melihat informasi yang ada pada dataset, seperti jumlah *entry*/baris, jumlah kolom, tipe data setiap kolom, serta *memory*:


```
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 15 entries, 0 to 14
    Data columns (total 4 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   week         15 non-null     int64  
     1   pie_sales    15 non-null     int64  
     2   price        15 non-null     float64
     3   advertising  15 non-null     float64
    dtypes: float64(2), int64(2)
    memory usage: 608.0 bytes


Semua kolom/fitur bertipe numerik, yang berarti dapat langsung digunakan dalam pembuatan model regresi linear. Juga tidak ada *missing values* pada dataset.

Melihat deskripsi statistik numerik pada data:


```
df.describe().transpose()
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
      <th>week</th>
      <td>15.0</td>
      <td>8.000000</td>
      <td>4.472136</td>
      <td>1.0</td>
      <td>4.5</td>
      <td>8.0</td>
      <td>11.50</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>pie_sales</th>
      <td>15.0</td>
      <td>399.333333</td>
      <td>63.523524</td>
      <td>300.0</td>
      <td>350.0</td>
      <td>430.0</td>
      <td>450.00</td>
      <td>490.0</td>
    </tr>
    <tr>
      <th>price</th>
      <td>15.0</td>
      <td>6.613333</td>
      <td>1.171609</td>
      <td>4.5</td>
      <td>5.7</td>
      <td>7.0</td>
      <td>7.50</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>advertising</th>
      <td>15.0</td>
      <td>3.480000</td>
      <td>0.488730</td>
      <td>2.7</td>
      <td>3.1</td>
      <td>3.5</td>
      <td>3.85</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>



# Exploratory Data Analysis

Pada *section* ini, akan dilakukan visualisasi terhadap data sehingga dapat di-*extract* beberapa *insight* secara deskriptif dari data.

## Distribusi Fitur


```
df.columns
```




    Index(['week', 'pie_sales', 'price', 'advertising'], dtype='object')




```
plt.figure(figsize=(15,5))
for i in range(len(df.columns)):
  plt.subplot(2, 2, i+1)
  sns.distplot(df[df.columns[i]])
  plt.title(df.columns[i])
  plt.axvline(df[df.columns[i]].mean(), label='Mean = {}'.format(round(df[df.columns[i]].mean(),3)), color='g')
  plt.axvline(df[df.columns[i]].median(), label='Median = {}'.format(round(df[df.columns[i]].median(),3)), color='y')
  plt.legend(bbox_to_anchor=(1.1, 1.05))

plt.tight_layout();
```

    /usr/local/lib/python3.6/dist-packages/seaborn/distributions.py:2551: FutureWarning:

    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).

    /usr/local/lib/python3.6/dist-packages/seaborn/distributions.py:2551: FutureWarning:

    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).

    /usr/local/lib/python3.6/dist-packages/seaborn/distributions.py:2551: FutureWarning:

    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).

    /usr/local/lib/python3.6/dist-packages/seaborn/distributions.py:2551: FutureWarning:

    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).




![png](images/Pie_Sales_Linear_Regression_files/Pie_Sales_Linear_Regression_15_1.png)


Penjelasan:

*   Week: Pelanggan banyak melakukan pembelian pie di minggu ke-8
*   Pie sales: Terdapat dua grup besar dalam distribusi pie_sales. Grup pertama, dengan rentang 300 sampai sekitar 380 pada jumlah penjualan dan rentang 420 sampai 490. Secara rata-rata, penjualan pie berada pada angka 430.
* Price: Untuk price, lebih variatif dengan rata-rata di harga 7 dollar
* Advertising: Pengeluaran untuk promosi banyak berada pada angka 3.48 dollar. Lebih rendah daripada rata-rata harga satu buah pie.



Melihat hubungan dari masing-masing fitur yang berisi variabel variabel independen terhadap variabel dependen. Apakah memiliki relasi yang linear atau tidak?


```
week_piesales = px.scatter(
    df,
    x = 'week', y = 'pie_sales',
    trendline='ols', trendline_color_override='darkblue',
    template = 'seaborn',
    title = 'Regression fit: Week vs Pie sales'
)

price_piesales = px.scatter(
    df,
    x = 'price', y = 'pie_sales',
    trendline='ols', trendline_color_override='darkblue',
    template = 'seaborn',
    title = 'Regression fit: Price vs Pie sales'
)

advertising_piesales = px.scatter(
    df,
    x = 'advertising', y = 'pie_sales',
    trendline='ols', trendline_color_override='darkblue',
    template= 'seaborn',
    title = 'Regression fit: Advertising vs Pie sales'
)
```

## Week - Pie Sales


```
week_piesales_line = go.Figure()
week_piesales_line.add_trace(go.Scatter(x=df['week'], y=df['pie_sales'],
                    mode='lines+markers'))
week_piesales_line.update_layout(title='Total pie terjual per minggu',
                   xaxis_title='Week',
                   yaxis_title='Pie Sales')

week_piesales_line.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="147c347e-10ec-4f13-9a9c-8d720980c981" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("147c347e-10ec-4f13-9a9c-8d720980c981")) {
                    Plotly.newPlot(
                        '147c347e-10ec-4f13-9a9c-8d720980c981',
                        [{"mode": "lines+markers", "type": "scatter", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "y": [350, 460, 350, 430, 350, 380, 430, 470, 450, 490, 340, 300, 440, 450, 300]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Total pie terjual per minggu"}, "xaxis": {"title": {"text": "Week"}}, "yaxis": {"title": {"text": "Pie Sales"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('147c347e-10ec-4f13-9a9c-8d720980c981');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


Grafik di atas menunjukkan bahwa penjualan pie fluktuatif pada awal minggu, hingga mencapai puncaknya di minggu ke-10 dengan total penjualan pie 490 buah. Namun mengalami penurunan tajam dan menjadi paling rendah selama pekan penjualan, tepatnya di minggu ke-12 dengan angka penjualan 300 pie. Sempat naik pada minggu 13 dan 14 namun turun tajam kembali di minggu berkutnya dengan total penjualan yang sama pada minggu ke-12.

### Regression Fit

Digunakan untuk melihat relasi linear fitur independen (`week`) terhadap targetnya (`pie_sales`). Apakah memiliki relasi linear yang kuat atau tidak?


```
week_piesales.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="758ba08f-c2aa-4c93-966f-b7979caea5c6" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("758ba08f-c2aa-4c93-966f-b7979caea5c6")) {
                    Plotly.newPlot(
                        '758ba08f-c2aa-4c93-966f-b7979caea5c6',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "week=%{x}<br>pie_sales=%{y}", "legendgroup": "", "marker": {"color": "rgb(76,114,176)", "symbol": "circle"}, "mode": "markers", "name": "", "showlegend": false, "type": "scatter", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "xaxis": "x", "y": [350, 460, 350, 430, 350, 380, 430, 470, 450, 490, 340, 300, 440, 450, 300], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>OLS trendline</b><br>pie_sales = -0.964286 * week + 407.047619<br>R<sup>2</sup>=0.004609<br><br>week=%{x}<br>pie_sales=%{y} <b>(trend)</b>", "legendgroup": "", "line": {"color": "darkblue"}, "marker": {"color": "rgb(76,114,176)", "symbol": "circle"}, "mode": "lines", "name": "", "showlegend": false, "type": "scatter", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "xaxis": "x", "y": [406.08333333333354, 405.1190476190478, 404.15476190476215, 403.19047619047643, 402.2261904761907, 401.26190476190504, 400.2976190476193, 399.3333333333336, 398.3690476190479, 397.40476190476215, 396.4404761904765, 395.47619047619077, 394.51190476190504, 393.5476190476194, 392.58333333333366], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "rgb(36,36,36)"}, "error_y": {"color": "rgb(36,36,36)"}, "marker": {"line": {"color": "rgb(234,234,242)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(234,234,242)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "rgb(36,36,36)", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "rgb(36,36,36)"}, "baxis": {"endlinecolor": "rgb(36,36,36)", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "rgb(36,36,36)"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "rgb(231,231,240)"}, "line": {"color": "white"}}, "header": {"fill": {"color": "rgb(183,183,191)"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "rgb(67,103,167)"}, "coloraxis": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "colorscale": {"sequential": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "sequentialminus": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]]}, "colorway": ["rgb(76,114,176)", "rgb(221,132,82)", "rgb(85,168,104)", "rgb(196,78,82)", "rgb(129,114,179)", "rgb(147,120,96)", "rgb(218,139,195)", "rgb(140,140,140)", "rgb(204,185,116)", "rgb(100,181,205)"], "font": {"color": "rgb(36,36,36)"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "rgb(234,234,242)", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "paper_bgcolor": "white", "plot_bgcolor": "rgb(234,234,242)", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}, "bgcolor": "rgb(234,234,242)", "radialaxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(234,234,242)", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "showgrid": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "rgb(234,234,242)", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "showgrid": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "rgb(234,234,242)", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "showgrid": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"fillcolor": "rgb(67,103,167)", "line": {"width": 0}, "opacity": 0.5}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}, "bgcolor": "rgb(234,234,242)", "caxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white"}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white"}}}, "title": {"text": "Regression fit: Week vs Pie sales"}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "week"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "pie_sales"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('758ba08f-c2aa-4c93-966f-b7979caea5c6');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


## Price - Pie Sales


```
price_piesales.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="948b6c4d-66ed-417b-92f3-6b2b5813703a" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("948b6c4d-66ed-417b-92f3-6b2b5813703a")) {
                    Plotly.newPlot(
                        '948b6c4d-66ed-417b-92f3-6b2b5813703a',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "price=%{x}<br>pie_sales=%{y}", "legendgroup": "", "marker": {"color": "rgb(76,114,176)", "symbol": "circle"}, "mode": "markers", "name": "", "showlegend": false, "type": "scatter", "x": [5.5, 7.5, 8.0, 8.0, 6.8, 7.5, 4.5, 6.4, 7.0, 5.0, 7.2, 7.9, 5.9, 5.0, 7.0], "xaxis": "x", "y": [350, 460, 350, 430, 350, 380, 430, 470, 450, 490, 340, 300, 440, 450, 300], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>OLS trendline</b><br>pie_sales = -24.033858 * price + 558.277250<br>R<sup>2</sup>=0.196491<br><br>price=%{x}<br>pie_sales=%{y} <b>(trend)</b>", "legendgroup": "", "line": {"color": "darkblue"}, "marker": {"color": "rgb(76,114,176)", "symbol": "circle"}, "mode": "lines", "name": "", "showlegend": false, "type": "scatter", "x": [4.5, 5.0, 5.0, 5.5, 5.9, 6.4, 6.8, 7.0, 7.0, 7.2, 7.5, 7.5, 7.9, 8.0, 8.0], "xaxis": "x", "y": [450.1248872545617, 438.10795809338777, 438.10795809338777, 426.09102893221376, 416.4774856032746, 404.4605564421006, 394.84701311316144, 390.04024144869186, 390.04024144869186, 385.2334697842223, 378.0233122875179, 378.0233122875179, 368.40976895857875, 366.00638312634396, 366.00638312634396], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "rgb(36,36,36)"}, "error_y": {"color": "rgb(36,36,36)"}, "marker": {"line": {"color": "rgb(234,234,242)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(234,234,242)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "rgb(36,36,36)", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "rgb(36,36,36)"}, "baxis": {"endlinecolor": "rgb(36,36,36)", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "rgb(36,36,36)"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "rgb(231,231,240)"}, "line": {"color": "white"}}, "header": {"fill": {"color": "rgb(183,183,191)"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "rgb(67,103,167)"}, "coloraxis": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "colorscale": {"sequential": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "sequentialminus": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]]}, "colorway": ["rgb(76,114,176)", "rgb(221,132,82)", "rgb(85,168,104)", "rgb(196,78,82)", "rgb(129,114,179)", "rgb(147,120,96)", "rgb(218,139,195)", "rgb(140,140,140)", "rgb(204,185,116)", "rgb(100,181,205)"], "font": {"color": "rgb(36,36,36)"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "rgb(234,234,242)", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "paper_bgcolor": "white", "plot_bgcolor": "rgb(234,234,242)", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}, "bgcolor": "rgb(234,234,242)", "radialaxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(234,234,242)", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "showgrid": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "rgb(234,234,242)", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "showgrid": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "rgb(234,234,242)", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "showgrid": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"fillcolor": "rgb(67,103,167)", "line": {"width": 0}, "opacity": 0.5}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}, "bgcolor": "rgb(234,234,242)", "caxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white"}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white"}}}, "title": {"text": "Regression fit: Price vs Pie sales"}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "price"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "pie_sales"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('948b6c4d-66ed-417b-92f3-6b2b5813703a');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


## Advertising - Pie Sales


```
advertising_piesales
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="27fa2b97-cb88-4879-a937-b65030e2caec" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("27fa2b97-cb88-4879-a937-b65030e2caec")) {
                    Plotly.newPlot(
                        '27fa2b97-cb88-4879-a937-b65030e2caec',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "advertising=%{x}<br>pie_sales=%{y}", "legendgroup": "", "marker": {"color": "rgb(76,114,176)", "symbol": "circle"}, "mode": "markers", "name": "", "showlegend": false, "type": "scatter", "x": [3.3, 3.3, 3.0, 4.5, 3.0, 4.0, 3.0, 3.7, 3.5, 4.0, 3.5, 3.2, 4.0, 3.5, 2.7], "xaxis": "x", "y": [350, 460, 350, 430, 350, 380, 430, 470, 450, 490, 340, 300, 440, 450, 300], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>OLS trendline</b><br>pie_sales = 72.308612 * advertising + 147.699362<br>R<sup>2</sup>=0.309492<br><br>advertising=%{x}<br>pie_sales=%{y} <b>(trend)</b>", "legendgroup": "", "line": {"color": "darkblue"}, "marker": {"color": "rgb(76,114,176)", "symbol": "circle"}, "mode": "lines", "name": "", "showlegend": false, "type": "scatter", "x": [2.7, 3.0, 3.0, 3.0, 3.2, 3.3, 3.3, 3.5, 3.5, 3.5, 3.7, 4.0, 4.0, 4.0, 4.5], "xaxis": "x", "y": [342.932615629984, 364.62519936204137, 364.62519936204137, 364.62519936204137, 379.08692185007965, 386.31778309409873, 386.31778309409873, 400.779505582137, 400.779505582137, 400.779505582137, 415.2412280701753, 436.93381180223264, 436.93381180223264, 436.93381180223264, 473.0881180223283], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "rgb(36,36,36)"}, "error_y": {"color": "rgb(36,36,36)"}, "marker": {"line": {"color": "rgb(234,234,242)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(234,234,242)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "rgb(36,36,36)", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "rgb(36,36,36)"}, "baxis": {"endlinecolor": "rgb(36,36,36)", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "rgb(36,36,36)"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "rgb(231,231,240)"}, "line": {"color": "white"}}, "header": {"fill": {"color": "rgb(183,183,191)"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "rgb(67,103,167)"}, "coloraxis": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "colorscale": {"sequential": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "sequentialminus": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]]}, "colorway": ["rgb(76,114,176)", "rgb(221,132,82)", "rgb(85,168,104)", "rgb(196,78,82)", "rgb(129,114,179)", "rgb(147,120,96)", "rgb(218,139,195)", "rgb(140,140,140)", "rgb(204,185,116)", "rgb(100,181,205)"], "font": {"color": "rgb(36,36,36)"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "rgb(234,234,242)", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "paper_bgcolor": "white", "plot_bgcolor": "rgb(234,234,242)", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}, "bgcolor": "rgb(234,234,242)", "radialaxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(234,234,242)", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "showgrid": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "rgb(234,234,242)", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "showgrid": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "rgb(234,234,242)", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "showgrid": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"fillcolor": "rgb(67,103,167)", "line": {"width": 0}, "opacity": 0.5}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}, "bgcolor": "rgb(234,234,242)", "caxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white"}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white"}}}, "title": {"text": "Regression fit: Advertising vs Pie sales"}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "advertising"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "pie_sales"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('27fa2b97-cb88-4879-a937-b65030e2caec');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


# Feature Selection

Di sini, akan dilakukan pemilihan fitur/kolom agar model regresi yang dihasilkan lebih merepresentasikan sebaran data yang sebenarnya. Atau dengan kata lain, untuk menghasilkan model regresi yang kuat.

Pada kolom `week`, sebenarnya terdapat relasi linear dengan arah negatif. Akan tetapi sangat lemah, dengan R square = 0,0046. Oleh karena itu, kolom ini tidak akan diikutkan dalam model.

Kolom `price`, dapat diikutkan ke dalam pembangunan model walaupun dengan R Square 0,196 dan berelasi linear dengan arah negatif.

`advertising` memiliki hubungan linear yang lebih tinggi terhadap pie_sales dengan angka R square 0,309 dan berarah positif. Yang artinya, semakin tinggi pengeluaran untuk promosi, maka akan naik pula total penjualannya. Walaupun dengan kenaikan yang tidak terlalu tinggi.

Jadi, variabel-variabel independen yang akan dilibatkan dalam pembangunan model adalah `price` dan `advertising`.

# Split Data (Independent(Xs) dan Dependent(y))

Dataset akan dibagi menjadi data training dan data testing. Proporsi pembagiannya adalah 25% akan digunakan sebagai data testing, sisanya untuk data training.

Data training digunakan dalam pembuatan model regresi, sedangkan data testing digunakan sebagai data uji keakuratan model yang dihasilkan.


```
y = df['pie_sales']
X = df[['price','advertising']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
```

# Model Training

## Menggunakan Scikit-learn LinearRegression


```
regressor = LinearRegression(fit_intercept = True)
regressor.fit(X_train,y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```
print('Coefficient (slope) = ', regressor.coef_)
print('Intercept = ', regressor.intercept_)
```

    Coefficient (slope) =  [-26.17250718  75.66278583]
    Intercept =  306.0709478818248


Koefisien-koefisien slope/kemiringan garis yang dihasilkan oleh model adalah -26.17250718 dan 75.66278583 serta intersep 306.0709478818248.

Formula Multiple linear regression:

![mlr.PNG](data:image/png;base64,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)



Jika koefisien-koefisien tersebut disubtitusi ke dalam formula, maka model yang dihasilkan memiliki formula:

y = 306.07 - 26.17*x1 + 75.66*x2

Jika diinterpretasikan lebih lanjut, harga 1 unit x1 berada pada angka -26.17. Sedangkan harga 1 unit x2 berada pada angka 75.66.

## Menggunakan Statsmodels
 Selain dengan menggunakan library scikit-learn, model juga dapat dihasilkan dengan statsmodel.api:


```
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

print_model = model.summary()
print(print_model)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:              pie_sales   R-squared:                       0.521
    Model:                            OLS   Adj. R-squared:                  0.442
    Method:                 Least Squares   F-statistic:                     6.539
    Date:                Fri, 23 Oct 2020   Prob (F-statistic):             0.0120
    Time:                        15:09:16   Log-Likelihood:                -77.510
    No. Observations:                  15   AIC:                             161.0
    Df Residuals:                      12   BIC:                             163.1
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const         306.5262    114.254      2.683      0.020      57.588     555.464
    price         -24.9751     10.832     -2.306      0.040     -48.576      -1.374
    advertising    74.1310     25.967      2.855      0.014      17.553     130.709
    ==============================================================================
    Omnibus:                        1.505   Durbin-Watson:                   1.683
    Prob(Omnibus):                  0.471   Jarque-Bera (JB):                0.937
    Skew:                           0.595   Prob(JB):                        0.626
    Kurtosis:                       2.709   Cond. No.                         72.2
    ==============================================================================

    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


    /usr/local/lib/python3.6/dist-packages/scipy/stats/stats.py:1535: UserWarning:

    kurtosistest only valid for n>=20 ... continuing anyway, n=15



Model regresi yang dihasilkan oleh statmodel.api adalah

y = 306.52 - 24.97*x1 + 74.13*x2

Dengan R square 0.52.



# Evaluasi Model (Scikit-learn)

Melakukan prediksi `pie_sales`(y_test) pada X_test:


```
y_predict = regressor.predict(X_test)
```

Bandingkan terhadap data aktual pada y_test:


```
y_test
```




    14    300
    1     460
    6     430
    10    340
    Name: pie_sales, dtype: int64



Dalam dataframe:


```
pd.DataFrame({
    'pred':y_predict,
    'actual':y_test
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
      <th>pred</th>
      <th>actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>327.152919</td>
      <td>300</td>
    </tr>
    <tr>
      <th>1</th>
      <td>359.464337</td>
      <td>460</td>
    </tr>
    <tr>
      <th>6</th>
      <td>415.283023</td>
      <td>430</td>
    </tr>
    <tr>
      <th>10</th>
      <td>382.448647</td>
      <td>340</td>
    </tr>
  </tbody>
</table>
</div>



Dapat dilihat, model dapat melakukan prediksi dari model yang dihasilkan pada metode yang ada pada Scikit-learn. Namun, sepertinya model yang dihasilkan masih lemah. Salah satu sebabnya adalah adalah jumlah data yang tidak besar.

Menurut *machine learning algorithm map* pada scikit-learn, dibutuhkan setidaknya 51 jumlah row data agar dapat menghasilkan model yang lebih optimal.

Jika divisualisasikan:


```
plt.scatter(y_test, y_predict, color = 'r')
plt.ylabel('Prediksi')
plt.xlabel('Aktual')
plt.title('Prediksi vs Aktual');
```


![png](images/Pie_Sales_Linear_Regression_files/Pie_Sales_Linear_Regression_48_0.png)


## Evaluation Metrics

Akan digunakan RMSE dan R square.


```
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)) , '.3f'))
MSE = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
```


```
print('RMSE = ', RMSE, '\nMSE =', MSE , '\nR2 =', r2)
```

    RMSE =  56.708
    MSE = 3215.7943794485604
    R2 = 0.237737628575156


RMSE menghitung seberapa jauh residual/error yang dihasilkan dari selisih hasil titik aktual dan prediksi yang dihasilkan oleh model. Karena yang diharapkan adalah model yang dihasilkan memiliki selisih jarak yang minim, maka semakin kecil nilai RMSE, maka akan semakin baik model tersebut.

Jika RMSE merupakan standar deviasi dari residual, maka MSE merupakan variansi dari residual. Serupa dengan RMSE, namun MSE berada dalam satuan unit yang berbeda dengan satuan unit pada nilai pada data yang sesungguhnya.

R square berada pada rentang 0-1 yang menunjukkan seberapa terwakilkan/terepresentasikan suatu model dengan sebaran data yang sesungguhnya. Semakin mendekati 1, maka semakin baik. Nilai R2 yang dihasilkan lebih buruk daripada R2 pada statsmodel. Hal ini dapat disebabkan karena pada metode scikit learn, dataset telah dibagi menjadi data training dan data testing sehingga jumlah data yang diproses sebagai model berkurang. Berbeda dengan model pada statsmodel yang menggunakan keseluruhan data.



# Visualisasi Hasil

Jika divisualisasikan dalam bentuk 3D:


```
from mpl_toolkits.mplot3d import Axes3D
x_surf, y_surf = np.meshgrid(np.linspace(df['advertising'].min(),
                                         df['price'].max(), 10),  
                                         np.linspace(df['advertising'].min(),
                                                     df['price'].max(), 10))
```


```
onlyX = pd.DataFrame({'advertising': x_surf.ravel(), 'price':y_surf.ravel()} )
fittedY = regressor.predict(onlyX)
fittedY = fittedY.reshape(x_surf.shape)
```


```
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(df['advertising'], df['price'] , df['pie_sales'] , c = 'blue', marker ='x')
ax.plot_surface(x_surf, y_surf, fittedY, color = 'red', alpha = 0.3)
ax.set_xlabel('Advertising')
ax.set_ylabel('Price')
ax.set_zlabel('Pie Sales')
```




    Text(0.5, 0, 'Pie Sales')




![png](images/Pie_Sales_Linear_Regression_files/Pie_Sales_Linear_Regression_56_1.png)


# Prediksi data di luar dataset


```
np.linspace(df['price'].min(),df['price'].max(),10)
```




    array([4.5       , 4.88888889, 5.27777778, 5.66666667, 6.05555556,
           6.44444444, 6.83333333, 7.22222222, 7.61111111, 8.        ])




```
np.linspace(df['advertising'].min(),df['advertising'].max(),10)
```




    array([2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5])




```
df_for_pred = pd.DataFrame({
    'price' : np.linspace(df['price'].min(),df['price'].max(),10),
    'advertising': np.linspace(df['advertising'].min(),df['advertising'].max(),10)
})
```


```
df_for_pred
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
      <th>price</th>
      <th>advertising</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.500000</td>
      <td>2.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.888889</td>
      <td>2.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.277778</td>
      <td>3.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.666667</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.055556</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.444444</td>
      <td>3.7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6.833333</td>
      <td>3.9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.222222</td>
      <td>4.1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7.611111</td>
      <td>4.3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8.000000</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>



Hasil prediksi `pie_sale` berdasarkan `df_for_pred`:


```
regressor.predict(df_for_pred)
```




    array([392.58418731, 397.53854724, 402.49290717, 407.4472671 ,
           412.40162703, 417.35598696, 422.31034689, 427.26470682,
           432.21906675, 437.17342667])



Dalam bentuk dataframe:


```
pd.DataFrame({
    'price' : np.linspace(df['price'].min(),df['price'].max(),10),
    'advertising': np.linspace(df['advertising'].min(),df['advertising'].max(),10),
    'pie_sale': regressor.predict(df_for_pred)
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
      <th>price</th>
      <th>advertising</th>
      <th>pie_sale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.500000</td>
      <td>2.7</td>
      <td>392.584187</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.888889</td>
      <td>2.9</td>
      <td>397.538547</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.277778</td>
      <td>3.1</td>
      <td>402.492907</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.666667</td>
      <td>3.3</td>
      <td>407.447267</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.055556</td>
      <td>3.5</td>
      <td>412.401627</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.444444</td>
      <td>3.7</td>
      <td>417.355987</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6.833333</td>
      <td>3.9</td>
      <td>422.310347</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.222222</td>
      <td>4.1</td>
      <td>427.264707</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7.611111</td>
      <td>4.3</td>
      <td>432.219067</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8.000000</td>
      <td>4.5</td>
      <td>437.173427</td>
    </tr>
  </tbody>
</table>
</div>



Maka pie dengan harga 4.5 dollar dan dengan biaya promosi 2.7 dollar diprediksi akan terjual dengan jumlah 392.58 buah pie. Dan seterusnya hingga index ke-9.
