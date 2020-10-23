---

title: "Pie Sales Multiple Liner Regression"
data: 2020-10-23
tags: [python, matplotlib, data visualization, linear regression]
header:

mathjax: "true"
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
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="4a3b3281-c558-4ca4-a8b4-218269048da7" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("4a3b3281-c558-4ca4-a8b4-218269048da7")) {
                    Plotly.newPlot(
                        '4a3b3281-c558-4ca4-a8b4-218269048da7',
                        [{"mode": "lines+markers", "type": "scatter", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "y": [350, 460, 350, 430, 350, 380, 430, 470, 450, 490, 340, 300, 440, 450, 300]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Total pie terjual per minggu"}, "xaxis": {"title": {"text": "Week"}}, "yaxis": {"title": {"text": "Pie Sales"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('4a3b3281-c558-4ca4-a8b4-218269048da7');
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



```
week_piesales
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="901e3113-6235-4884-91e2-00ae3663ca53" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("901e3113-6235-4884-91e2-00ae3663ca53")) {
                    Plotly.newPlot(
                        '901e3113-6235-4884-91e2-00ae3663ca53',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "week=%{x}<br>pie_sales=%{y}", "legendgroup": "", "marker": {"color": "rgb(76,114,176)", "symbol": "circle"}, "mode": "markers", "name": "", "showlegend": false, "type": "scatter", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "xaxis": "x", "y": [350, 460, 350, 430, 350, 380, 430, 470, 450, 490, 340, 300, 440, 450, 300], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>OLS trendline</b><br>pie_sales = -0.964286 * week + 407.047619<br>R<sup>2</sup>=0.004609<br><br>week=%{x}<br>pie_sales=%{y} <b>(trend)</b>", "legendgroup": "", "line": {"color": "darkblue"}, "marker": {"color": "rgb(76,114,176)", "symbol": "circle"}, "mode": "lines", "name": "", "showlegend": false, "type": "scatter", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "xaxis": "x", "y": [406.08333333333354, 405.1190476190478, 404.15476190476215, 403.19047619047643, 402.2261904761907, 401.26190476190504, 400.2976190476193, 399.3333333333336, 398.3690476190479, 397.40476190476215, 396.4404761904765, 395.47619047619077, 394.51190476190504, 393.5476190476194, 392.58333333333366], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "rgb(36,36,36)"}, "error_y": {"color": "rgb(36,36,36)"}, "marker": {"line": {"color": "rgb(234,234,242)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(234,234,242)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "rgb(36,36,36)", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "rgb(36,36,36)"}, "baxis": {"endlinecolor": "rgb(36,36,36)", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "rgb(36,36,36)"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "rgb(231,231,240)"}, "line": {"color": "white"}}, "header": {"fill": {"color": "rgb(183,183,191)"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "rgb(67,103,167)"}, "coloraxis": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "colorscale": {"sequential": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "sequentialminus": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]]}, "colorway": ["rgb(76,114,176)", "rgb(221,132,82)", "rgb(85,168,104)", "rgb(196,78,82)", "rgb(129,114,179)", "rgb(147,120,96)", "rgb(218,139,195)", "rgb(140,140,140)", "rgb(204,185,116)", "rgb(100,181,205)"], "font": {"color": "rgb(36,36,36)"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "rgb(234,234,242)", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "paper_bgcolor": "white", "plot_bgcolor": "rgb(234,234,242)", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}, "bgcolor": "rgb(234,234,242)", "radialaxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(234,234,242)", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "showgrid": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "rgb(234,234,242)", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "showgrid": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "rgb(234,234,242)", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "showgrid": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"fillcolor": "rgb(67,103,167)", "line": {"width": 0}, "opacity": 0.5}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}, "bgcolor": "rgb(234,234,242)", "caxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white"}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white"}}}, "title": {"text": "Regression fit: Week vs Pie sales"}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "week"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "pie_sales"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('901e3113-6235-4884-91e2-00ae3663ca53');
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
price_piesales
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="ce417d94-5093-4b4b-ba87-ab4a507073f6" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("ce417d94-5093-4b4b-ba87-ab4a507073f6")) {
                    Plotly.newPlot(
                        'ce417d94-5093-4b4b-ba87-ab4a507073f6',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "price=%{x}<br>pie_sales=%{y}", "legendgroup": "", "marker": {"color": "rgb(76,114,176)", "symbol": "circle"}, "mode": "markers", "name": "", "showlegend": false, "type": "scatter", "x": [5.5, 7.5, 8.0, 8.0, 6.8, 7.5, 4.5, 6.4, 7.0, 5.0, 7.2, 7.9, 5.9, 5.0, 7.0], "xaxis": "x", "y": [350, 460, 350, 430, 350, 380, 430, 470, 450, 490, 340, 300, 440, 450, 300], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>OLS trendline</b><br>pie_sales = -24.033858 * price + 558.277250<br>R<sup>2</sup>=0.196491<br><br>price=%{x}<br>pie_sales=%{y} <b>(trend)</b>", "legendgroup": "", "line": {"color": "darkblue"}, "marker": {"color": "rgb(76,114,176)", "symbol": "circle"}, "mode": "lines", "name": "", "showlegend": false, "type": "scatter", "x": [4.5, 5.0, 5.0, 5.5, 5.9, 6.4, 6.8, 7.0, 7.0, 7.2, 7.5, 7.5, 7.9, 8.0, 8.0], "xaxis": "x", "y": [450.1248872545617, 438.10795809338777, 438.10795809338777, 426.09102893221376, 416.4774856032746, 404.4605564421006, 394.84701311316144, 390.04024144869186, 390.04024144869186, 385.2334697842223, 378.0233122875179, 378.0233122875179, 368.40976895857875, 366.00638312634396, 366.00638312634396], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "rgb(36,36,36)"}, "error_y": {"color": "rgb(36,36,36)"}, "marker": {"line": {"color": "rgb(234,234,242)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(234,234,242)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "rgb(36,36,36)", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "rgb(36,36,36)"}, "baxis": {"endlinecolor": "rgb(36,36,36)", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "rgb(36,36,36)"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}, "colorscale": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "rgb(231,231,240)"}, "line": {"color": "white"}}, "header": {"fill": {"color": "rgb(183,183,191)"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "rgb(67,103,167)"}, "coloraxis": {"colorbar": {"outlinewidth": 0, "tickcolor": "rgb(36,36,36)", "ticklen": 8, "ticks": "outside", "tickwidth": 2}}, "colorscale": {"sequential": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]], "sequentialminus": [[0.0, "rgb(2,4,25)"], [0.06274509803921569, "rgb(24,15,41)"], [0.12549019607843137, "rgb(47,23,57)"], [0.18823529411764706, "rgb(71,28,72)"], [0.25098039215686274, "rgb(97,30,82)"], [0.3137254901960784, "rgb(123,30,89)"], [0.3764705882352941, "rgb(150,27,91)"], [0.4392156862745098, "rgb(177,22,88)"], [0.5019607843137255, "rgb(203,26,79)"], [0.5647058823529412, "rgb(223,47,67)"], [0.6274509803921569, "rgb(236,76,61)"], [0.6901960784313725, "rgb(242,107,73)"], [0.7529411764705882, "rgb(244,135,95)"], [0.8156862745098039, "rgb(245,162,122)"], [0.8784313725490196, "rgb(246,188,153)"], [0.9411764705882353, "rgb(247,212,187)"], [1.0, "rgb(250,234,220)"]]}, "colorway": ["rgb(76,114,176)", "rgb(221,132,82)", "rgb(85,168,104)", "rgb(196,78,82)", "rgb(129,114,179)", "rgb(147,120,96)", "rgb(218,139,195)", "rgb(140,140,140)", "rgb(204,185,116)", "rgb(100,181,205)"], "font": {"color": "rgb(36,36,36)"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "rgb(234,234,242)", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "paper_bgcolor": "white", "plot_bgcolor": "rgb(234,234,242)", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}, "bgcolor": "rgb(234,234,242)", "radialaxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(234,234,242)", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "showgrid": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "rgb(234,234,242)", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "showgrid": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "rgb(234,234,242)", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "showgrid": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"fillcolor": "rgb(67,103,167)", "line": {"width": 0}, "opacity": 0.5}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}, "bgcolor": "rgb(234,234,242)", "caxis": {"gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": ""}}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white"}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "showgrid": true, "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white"}}}, "title": {"text": "Regression fit: Price vs Pie sales"}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "price"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "pie_sales"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('ce417d94-5093-4b4b-ba87-ab4a507073f6');
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


# Split Data (Independent(Xs) dan Dependent(y))


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
print('Coefficient (m) = ', regressor.coef_)
print('Intercept (b) = ', regressor.intercept_)
```

    Coefficient (m) =  [-26.17250718  75.66278583]
    Intercept (b) =  306.0709478818248


## Menggunakan Statsmodels


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



# Evaluasi Model (Scikit-learn)


```
y_predict = regressor.predict(X_test)
y_predict
```




    array([327.15291937, 359.46433727, 415.28302306, 382.44864659])




```
y_test
```




    14    300
    1     460
    6     430
    10    340
    Name: pie_sales, dtype: int64




```
plt.scatter(y_test, y_predict, color = 'r')
plt.ylabel('Prediksi')
plt.xlabel('Aktual')
plt.title('Prediksi vs Aktual');
```


![png](images/Pie_Sales_Linear_Regression_files/Pie_Sales_Linear_Regression_36_0.png)



```
k = X_test.shape[1]
n = len(X_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)) , '.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.mean(np.abs( (y_test-y_predict) / y_test  )  )*100
```


```
print('RMSE = ', RMSE, '\nMSE =', MSE , '\nMAE =', MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2, '\nMAPE =', MAPE)
```

    RMSE =  56.708
    MSE = 3215.7943794485604
    MAE = 46.2135514067125
    R2 = 0.237737628575156
    Adjusted R2 = -1.286787114274532
    MAPE = 11.703500202358564


# Visualisasi Hasil


```
from mpl_toolkits.mplot3d import Axes3D
x_surf, y_surf = np.meshgrid(np.linspace(df['advertising'].min(),
                                         df['price'].max(), 100),  
                                         np.linspace(df['advertising'].min(),
                                                     df['price'].max(), 100))
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




![png](images/Pie_Sales_Linear_Regression_files/Pie_Sales_Linear_Regression_42_1.png)


Dari angel yang berbeda:


```
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['advertising'],df['price'],df['pie_sales'],c='blue', marker='x', alpha=1)
ax.plot_surface(x_surf, y_surf, fittedY, color='red', alpha=0.3)
ax.set_xlabel('Advertising')
ax.set_ylabel('Price')
ax.set_zlabel('Pie Sales')
ax.view_init(60, 30)
plt.show()
```


![png](images/Pie_Sales_Linear_Regression_files/Pie_Sales_Linear_Regression_44_0.png)
