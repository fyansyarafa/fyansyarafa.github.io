---

title: "Matplotlib Excercise"
data: 2020-02-01
tags: [python, matplotlib, data visualization]
header:
excerpt: "Matplotlib Portofolio"
mathjax: "true"
---


Welcome to the exercises for reviewing matplotlib! Take your time with these, Matplotlib can be tricky to understand at first. These are relatively simple plots, but they can be hard if this is your first time with matplotlib, feel free to reference the solutions as you go along.

Also don't worry if you find the matplotlib syntax frustrating, we actually won't be using it that often throughout the course, we will switch to using seaborn and pandas built-in visualization capabilities. But, those are built-off of matplotlib, which is why it is still important to get exposure to it!

** * NOTE: ALL THE COMMANDS FOR PLOTTING A FIGURE SHOULD ALL GO IN THE SAME CELL. SEPARATING THEM OUT INTO MULTIPLE CELLS MAY CAUSE NOTHING TO SHOW UP. * **

# Exercises

Follow the instructions to recreate the plots using this data:

## Data


```python
import numpy as np
x = np.arange(0,100)
y = x*2
z = x**2
```

** Import matplotlib.pyplot as plt and set %matplotlib inline if you are using the jupyter notebook. What command do you use if you aren't using the jupyter notebook?**


```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
plt.show()
```

## Exercise 1

** Follow along with these steps: **
* ** Create a figure object called fig using plt.figure() **
* ** Use add_axes to add an axis to the figure canvas at [0,0,1,1]. Call this new axis ax. **
* ** Plot (x,y) on that axes and set the labels and titles to match the plot below:**


```python
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])

axes.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x1dad6d83f60>]




![png](/images/matweb_files/matweb_6_1.png)


## Exercise 2
** Create a figure object and put two axes on it, ax1 and ax2. Located at [0,0,1,1] and [0.2,0.5,.2,.2] respectively.**


```python
fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5,.2,.2])
```


![png](/images/matweb_files/matweb_8_0.png)


** Now plot (x,y) on both axes. And call your figure object to show it.**


```python
ax1.plot
```




![png](/images/matweb_files/matweb_10_0.png)




```python
fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5,.2,.2])

ax1.plot(x,y)
ax1.set_title('Figure 1')
ax2.plot(x,y)
ax2.set_title('Figure 2')
ax1.set_xlabel('x')
```




    Text(0.5,0,'x')




![png](/images/matweb_files/matweb_11_1.png)


## Exercise 3

** Create the plot below by adding two axes to a figure object at [0,0,1,1] and [0.2,0.5,.4,.4]**


```python
fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5,.4,.4])
```


![png](/images/matweb_files/matweb_13_0.png)


** Now use x,y, and z arrays to recreate the plot below. Notice the xlimits and y limits on the inserted plot:**


```python

```




![png](/images/matweb_files/matweb_15_0.png)




![png](/images/matweb_files/matweb_15_1.png)



```python
fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5,.4,.4])
ax1.plot(x,z)
ax1.set_xlabel('X')
ax1.set_ylabel('Z')
#ax2.set_xlim([40,45])
ax2.plot(x,y)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('zoom')

ax2.set_xlim([20,22])
ax2.set_ylim([30,50])
fig
```




![png](/images/matweb_files/matweb_16_0.png)




![png](/images/matweb_files/matweb_16_1.png)



```python

```




    [<matplotlib.lines.Line2D at 0x1dad7410470>]



## Exercise 4

** Use plt.subplots(nrows=1, ncols=2) to create the plot below.**


```python
fig, axes = plt.subplots(nrows=1,ncols=2)
```


![png](/images/matweb_files/matweb_19_0.png)


** Now plot (x,y) and (x,z) on the axes. Play around with the linewidth and style**


```python

```




![png](/images/matweb_files/matweb_21_0.png)




```python
fig, axes = plt.subplots(nrows=1,ncols=2)
axes[0].plot(x,y, ls='--',color='blue',lw=3)
axes[1].plot(x,z, ls='-',color='red',lw=3)
```




    [<matplotlib.lines.Line2D at 0x1dad8e18550>]




![png](/images/matweb_files/matweb_22_1.png)


** See if you can resize the plot by adding the figsize() argument in plt.subplots() are copying and pasting your previous code.**


```python
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,3))
axes[0].plot(x,y, ls='--',color='blue',lw=3)
axes[1].plot(x,z, ls='-',color='red',lw=3)
```




    [<matplotlib.lines.Line2D at 0x1dada260400>]




![png](/images/matweb_files/matweb_24_1.png)


# Great Job!
