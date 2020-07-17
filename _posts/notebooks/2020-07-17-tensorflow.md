---
title: "tensorflow 1.x : Simple Linear Regression and Classification"
data: 2020-07-17
tags: [python,  tensorflow, linear regression, Classification]
header:
excerpt: "tensorflow notebook"
mathjax: "true"
---

```python
import tensorflow as tf
```


```python
import numpy as np
```


```python
np.random.seed(101)
tf.set_random_seed(101)
```


```python
# set a random uniform variable from 0 to 100 with size 5x5
rand_a = np.random.uniform(low = 0,
                           high = 100,
                           size = (5,5)
                          )
rand_a
```




    array([[51.63986277, 57.06675869,  2.84742265, 17.15216562, 68.52769817],
           [83.38968626, 30.69662197, 89.36130797, 72.15438618, 18.99389542],
           [55.42275911, 35.2131954 , 18.18924027, 78.56017619, 96.54832224],
           [23.23536618,  8.35614337, 60.35484223, 72.89927573, 27.62388285],
           [68.53063288, 51.78674742,  4.84845374, 13.78692376, 18.69674261]])




```python
# set another random uniform variable from 0 to 100 with size 5x1
rand_b = np.random.uniform(low = 0,
                           high = 100,
                           size = (5,1)
                          )
rand_b
```




    array([[99.43179012],
           [52.06653967],
           [57.87895355],
           [73.48190583],
           [54.19617722]])




```python
# set placeholder a and b
a = tf.placeholder(dtype = tf.float32)
b = tf.placeholder(dtype= tf.float32)
```


```python
# operation
add_op = a + b
mul_op = a * b
```


```python
# run operation
with tf.Session() as sess:

    # add operation
    add_result = sess.run(add_op,
                          feed_dict = {
                              a : rand_a,
                              b : rand_b
                          }
                         )
    print(add_result)

    print('\n')

    # mul operation
    mul_result = sess.run(mul_op,
                          feed_dict = {
                              a : rand_a,
                              b : rand_b
                          }
                         )
    print(mul_result)
```

    [[151.07166  156.49855  102.27921  116.58396  167.95949 ]
     [135.45622   82.76316  141.42784  124.22093   71.06043 ]
     [113.30171   93.09215   76.06819  136.43912  154.42728 ]
     [ 96.71727   81.83804  133.83675  146.38118  101.10579 ]
     [122.72681  105.982925  59.044632  67.9831    72.89292 ]]


    [[5134.644   5674.25     283.12433 1705.4707  6813.8315 ]
     [4341.8125  1598.267   4652.734   3756.8293   988.94635]
     [3207.8113  2038.1029  1052.7742  4546.9805  5588.1157 ]
     [1707.379    614.02527 4434.989   5356.7773  2029.8555 ]
     [3714.0984  2806.6438   262.76764  747.19855 1013.292  ]]



```python

```


```python

```


```python

```

example X*w + b
```
x : placeholder
w : variable
b : variable
```


```python
n_features = 10 # amount of features
n_dense_neurons = 3 # amount of layers
```


```python
x = tf.placeholder(tf.float32, shape=(None, n_features))
w = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
b = tf.Variable(tf.ones([n_dense_neurons]))
```


```python
# set matmul
xW = tf.matmul(x,w)

# set add
z = tf.add(xW,b)

# add activation function
a = tf.sigmoid(z)
```


```python
# init variables
init = tf.global_variables_initializer()
```


```python
# run operation
with tf.Session() as sess:
    sess.run(init)
    layer_out = sess.run(a, feed_dict = {
        x : np.random.random([1,n_features])
    })
    print(w)
    print(x)
    print(b)
    print(layer_out)

```

    <tf.Variable 'Variable_22:0' shape=(10, 3) dtype=float32_ref>
    Tensor("Placeholder_24:0", shape=(?, 10), dtype=float32)
    <tf.Variable 'Variable_23:0' shape=(3,) dtype=float32_ref>
    [[0.58175945 0.8834179  0.89205456]]



```python
# interactive session
Is = tf.InteractiveSession()
```

    C:\Users\abulu\anaconda3\lib\site-packages\tensorflow_core\python\client\session.py:1750: UserWarning: An interactive session is already active. This can cause out-of-memory errors in some cases. You must explicitly call `InteractiveSession.close()` to release resources held by the other session(s).
      warnings.warn('An interactive session is already active. This can '



```python
initt = tf.global_variables_initializer()
```


```python
Is.run(initt)
```


```python
Is.run(w)

```




    array([[ 0.26770195,  0.14685939,  0.9291129 ],
           [-1.8898729 , -0.80983925, -0.9824627 ],
           [-0.01287324, -1.877487  ,  0.9715939 ],
           [ 1.1782739 , -1.3546664 ,  0.8460633 ],
           [ 0.56966436,  1.4744045 , -0.3378433 ],
           [ 0.4205022 , -1.2170011 , -1.3850214 ],
           [-0.29671642,  1.6556927 ,  1.0600923 ],
           [ 0.5325541 ,  0.70352495, -0.3847609 ],
           [-0.7484736 ,  0.09566908,  0.02216081],
           [-0.29627335, -0.291333  ,  0.59812504]], dtype=float32)




```python
Is.run(b)
```




    array([1., 1., 1.], dtype=float32)




```python

```


```python

```

# Simple Regression Example


```python
# set data
x_data = np.linspace(0,10,10)
y_label = np.linspace(0,10,10)
```


```python
from matplotlib import pyplot as plt
plt.plot(x_data, y_label, 'r')
```




    [<matplotlib.lines.Line2D at 0x299168fbd88>]




![png](output_26_1.png)



```python
# add data noise
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
```


```python
plt.plot(x_data, y_label, '*')
```




    [<matplotlib.lines.Line2D at 0x2991325bf88>]




![png](output_28_1.png)


y = mx + b


```python
np.random.rand(2)
```




    array([0.44236813, 0.87758732])




```python
m = tf.Variable(0.51)
b = tf.Variable(0.22)
```


```python
error  = 0

for x, y in zip(x_data,y_label):
    y_hat = m*x + b
    error = (y-y_hat)**2
```

optimizer


```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
```


```python
init = tf.global_variables_initializer()
```


```python
with tf.Session() as sess:
    sess.run(init)
    training_steps = 10
    for i in range(training_steps):
        sess.run(train)

    final_slope, final_intercept = sess.run([m,b])
    print(final_intercept, final_slope)
```

    0.25326803 0.84144604


```
y = mx + b
model = 0.88x + 0.25
```


```python
# test data
x_test = np.linspace(-1,11,10)


# fit data to model
y_pred_plot = (final_slope*x_test + final_intercept)
```


```python
y_pred_plot
```




    array([-0.58817801,  0.53375005,  1.6556781 ,  2.77760616,  3.89953422,
            5.02146227,  6.14339033,  7.26531838,  8.38724644,  9.5091745 ])




```python
plt.plot(x_test, y_pred_plot , 'r')
plt.plot(x_data, y_label,'*');
```


![png](output_40_0.png)



```python

```

## Regression example 2


```python
%matplotlib inline
```


```python
# data and noise
x_data = np.linspace(0.0,10.0,1000000)

```


```python
noise = np.random.randn(len(x_data))
```


```python
noise
```




    array([-0.03157914,  0.64982583,  2.15484644, ..., -0.39165848,
            0.26840566, -1.24638586])




```python
noise.shape, x_data.shape
```




    ((1000000,), (1000000,))



y = mx + b

b = 5


```python
y_true = (0.5 * x_data) + 5 + noise
```


```python
import pandas as pd
```


```python
x_df = pd.DataFrame(data=x_data,
                    columns=['X Data']
                   )

y_df = pd.DataFrame(data=y_true,
                    columns=['Y']
                   )
```


```python
x_df.head()
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
      <th>X Data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00002</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00003</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00004</td>
    </tr>
  </tbody>
</table>
</div>




```python
 y_df.head()
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
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.968421</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.649831</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.154856</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.389756</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.244695</td>
    </tr>
  </tbody>
</table>
</div>




```python
# concat both of the dataframe
my_data = pd.concat([x_df, y_df], axis=1)
my_data.head()
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
      <th>X Data</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00000</td>
      <td>4.968421</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.00001</td>
      <td>5.649831</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00002</td>
      <td>7.154856</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00003</td>
      <td>4.389756</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00004</td>
      <td>4.244695</td>
    </tr>
  </tbody>
</table>
</div>



Plot the data, using sample of 250 entries of my_data



```python
my_data.sample(n=250).plot(kind='scatter', x= 'X Data', y= 'Y');
```


![png](output_56_0.png)



```python
batch_size = 8
np.random.randn(2)
```




    array([-0.0333475 ,  1.67028503])




```python
m = tf.Variable(-0.77)
```


```python
b = tf.Variable(0.77)
```


```python
xph = tf.placeholder(tf.float32,[batch_size])
```


```python
yph = tf.placeholder(tf.float32,[batch_size])
```


```python
y_model = m*xph + b #graph/operation

```


```python
error = tf.reduce_sum(tf.square(yph-y_model))
```


```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
```


```python
init = tf.global_variables_initializer()
```


```python
with tf.Session() as sess:
    sess.run(init)
    batches = 1000
    for i in range(batches):
        rand_ind = np.random.randint(len(x_data), size=batch_size)
        feed = {
            xph : x_data[rand_ind],
            yph : y_true[rand_ind]
        }
        sess.run(train,feed_dict = feed)

    model_m, model_b = sess.run([m,b])
```


```python
model_b, model_m
```




    (4.930831, 0.5364137)




```python

```


```python

```

## Using tf.estimator

### List of Feature Columns


```python
feat_cols = [tf.feature_column.numeric_column(key='x',
                                              shape=[1]
                                             )]
```


```python
feat_cols
```




    [NumericColumn(key='x', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)]




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

### Create estimator model


```python
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)
```

    INFO:tensorflow:Using default config.
    WARNING:tensorflow:Using temporary folder as model directory: C:\Users\abulu\AppData\Local\Temp\tmp24ahw8x1
    INFO:tensorflow:Using config: {'_model_dir': 'C:\\Users\\abulu\\AppData\\Local\\Temp\\tmp24ahw8x1', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x000002990B959208>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}


### Create data input function


```python
from sklearn.model_selection import train_test_split
```


```python
x_data
```




    array([0.000000e+00, 1.000001e-05, 2.000002e-05, ..., 9.999980e+00,
           9.999990e+00, 1.000000e+01])




```python
y_label
```




    array([0.5420333 , 1.17575569, 0.85241231, 2.50514314, 4.67005971,
           4.41685654, 6.66701681, 6.69180648, 7.54731409, 9.03483077])




```python
 x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size=0.3, random_state=101)
```


```python
x_train.shape, y_train.shape
```




    ((700000,), (700000,))




```python
#estimator input
input_func = tf.estimator.inputs.numpy_input_fn({
    'x' : x_train,
    'y' : y_train
}, y=y_train, batch_size=8,num_epochs=None,shuffle=True)
```


```python
#estimator train
train_input_func = tf.estimator.inputs.numpy_input_fn({
    'x' : x_train
}, y=y_train, batch_size=8,num_epochs=1000, shuffle=False)
```


```python
#estimator eval
eval_input_func = tf.estimator.inputs.numpy_input_fn({
    'x' : x_eval
}, y=y_eval, batch_size=8, num_epochs=1000,shuffle=False)
```

### Call train, evaluate, and predict methods


```python
# train
estimator.train(input_fn=input_func,steps=1000)
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 0 into C:\Users\abulu\AppData\Local\Temp\tmp24ahw8x1\model.ckpt.
    INFO:tensorflow:loss = 550.22766, step = 1
    INFO:tensorflow:global_step/sec: 441.377
    INFO:tensorflow:loss = 22.077072, step = 101 (0.229 sec)
    INFO:tensorflow:global_step/sec: 581.347
    INFO:tensorflow:loss = 13.727564, step = 201 (0.171 sec)
    INFO:tensorflow:global_step/sec: 534.768
    INFO:tensorflow:loss = 8.303132, step = 301 (0.187 sec)
    INFO:tensorflow:global_step/sec: 511.451
    INFO:tensorflow:loss = 25.267738, step = 401 (0.197 sec)
    INFO:tensorflow:global_step/sec: 558.466
    INFO:tensorflow:loss = 10.732901, step = 501 (0.178 sec)
    INFO:tensorflow:global_step/sec: 588.108
    INFO:tensorflow:loss = 9.602923, step = 601 (0.170 sec)
    INFO:tensorflow:global_step/sec: 606.017
    INFO:tensorflow:loss = 8.716262, step = 701 (0.165 sec)
    INFO:tensorflow:global_step/sec: 550.864
    INFO:tensorflow:loss = 7.5503354, step = 801 (0.183 sec)
    INFO:tensorflow:global_step/sec: 544.799
    INFO:tensorflow:loss = 25.699596, step = 901 (0.183 sec)
    INFO:tensorflow:Saving checkpoints for 1000 into C:\Users\abulu\AppData\Local\Temp\tmp24ahw8x1\model.ckpt.
    INFO:tensorflow:Loss for final step: 3.2968612.





    <tensorflow_estimator.python.estimator.canned.linear.LinearRegressor at 0x299084ed7c8>




```python
# evaluate train
train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Starting evaluation at 2020-07-17T02:54:10Z
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\abulu\AppData\Local\Temp\tmp24ahw8x1\model.ckpt-1000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Evaluation [100/1000]
    INFO:tensorflow:Evaluation [200/1000]
    INFO:tensorflow:Evaluation [300/1000]
    INFO:tensorflow:Evaluation [400/1000]
    INFO:tensorflow:Evaluation [500/1000]
    INFO:tensorflow:Evaluation [600/1000]
    INFO:tensorflow:Evaluation [700/1000]
    INFO:tensorflow:Evaluation [800/1000]
    INFO:tensorflow:Evaluation [900/1000]
    INFO:tensorflow:Evaluation [1000/1000]
    INFO:tensorflow:Finished evaluation at 2020-07-17-02:54:12
    INFO:tensorflow:Saving dict for global step 1000: average_loss = 1.0885203, global_step = 1000, label/mean = 7.5061226, loss = 8.708162, prediction/mean = 7.4360123
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 1000: C:\Users\abulu\AppData\Local\Temp\tmp24ahw8x1\model.ckpt-1000



```python
# evaluate eval
eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Starting evaluation at 2020-07-17T02:54:12Z
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\abulu\AppData\Local\Temp\tmp24ahw8x1\model.ckpt-1000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Evaluation [100/1000]
    INFO:tensorflow:Evaluation [200/1000]
    INFO:tensorflow:Evaluation [300/1000]
    INFO:tensorflow:Evaluation [400/1000]
    INFO:tensorflow:Evaluation [500/1000]
    INFO:tensorflow:Evaluation [600/1000]
    INFO:tensorflow:Evaluation [700/1000]
    INFO:tensorflow:Evaluation [800/1000]
    INFO:tensorflow:Evaluation [900/1000]
    INFO:tensorflow:Evaluation [1000/1000]
    INFO:tensorflow:Finished evaluation at 2020-07-17-02:54:14
    INFO:tensorflow:Saving dict for global step 1000: average_loss = 1.0605388, global_step = 1000, label/mean = 7.4655814, loss = 8.48431, prediction/mean = 7.41421
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 1000: C:\Users\abulu\AppData\Local\Temp\tmp24ahw8x1\model.ckpt-1000



```python
print("TRAIN DATA METRICS")
print(train_metrics)

print('\n')

print("EVAL DATA METRICS")
print(eval_metrics)
```

    TRAIN DATA METRICS
    {'average_loss': 1.0885203, 'label/mean': 7.5061226, 'loss': 8.708162, 'prediction/mean': 7.4360123, 'global_step': 1000}


    EVAL DATA METRICS
    {'average_loss': 1.0605388, 'label/mean': 7.4655814, 'loss': 8.48431, 'prediction/mean': 7.41421, 'global_step': 1000}



```python
# new data
new_data = np.linspace(0,10,10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({
    'x' : new_data
}, shuffle=False)
```


```python
estimator.predict(input_fn=input_fn_predict)
```




    <generator object Estimator.predict at 0x000002990B9D99C8>




```python
list(estimator.predict(input_fn=input_fn_predict))
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\abulu\AppData\Local\Temp\tmp24ahw8x1\model.ckpt-1000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.





    [{'predictions': array([4.416166], dtype=float32)},
     {'predictions': array([5.0869412], dtype=float32)},
     {'predictions': array([5.757717], dtype=float32)},
     {'predictions': array([6.4284925], dtype=float32)},
     {'predictions': array([7.099268], dtype=float32)},
     {'predictions': array([7.7700434], dtype=float32)},
     {'predictions': array([8.440819], dtype=float32)},
     {'predictions': array([9.111594], dtype=float32)},
     {'predictions': array([9.78237], dtype=float32)},
     {'predictions': array([10.453146], dtype=float32)}]




```python
predictions = []
for pred in list(estimator.predict(input_fn=input_fn_predict)):
    predictions.append(pred['predictions'])
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\abulu\AppData\Local\Temp\tmp24ahw8x1\model.ckpt-1000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.



```python
predictions
```




    [array([4.416166], dtype=float32),
     array([5.0869412], dtype=float32),
     array([5.757717], dtype=float32),
     array([6.4284925], dtype=float32),
     array([7.099268], dtype=float32),
     array([7.7700434], dtype=float32),
     array([8.440819], dtype=float32),
     array([9.111594], dtype=float32),
     array([9.78237], dtype=float32),
     array([10.453146], dtype=float32)]




```python
my_data.sample(n=250).plot(kind='scatter', x='X Data', y = 'Y')
plt.plot(new_data,predictions,'r*');
```


![png](output_100_0.png)


# Classification: Linear Classifier


```python
diabetes = pd.read_csv("../pima-indians-diabetes.csv")
```


```python
diabetes.head()
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
      <th>Number_pregnant</th>
      <th>Glucose_concentration</th>
      <th>Blood_pressure</th>
      <th>Triceps</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>Pedigree</th>
      <th>Age</th>
      <th>Class</th>
      <th>Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>0.743719</td>
      <td>0.590164</td>
      <td>0.353535</td>
      <td>0.000000</td>
      <td>0.500745</td>
      <td>0.234415</td>
      <td>50</td>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.427136</td>
      <td>0.540984</td>
      <td>0.292929</td>
      <td>0.000000</td>
      <td>0.396423</td>
      <td>0.116567</td>
      <td>31</td>
      <td>0</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>0.919598</td>
      <td>0.524590</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.347243</td>
      <td>0.253629</td>
      <td>32</td>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.447236</td>
      <td>0.540984</td>
      <td>0.232323</td>
      <td>0.111111</td>
      <td>0.418778</td>
      <td>0.038002</td>
      <td>21</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.688442</td>
      <td>0.327869</td>
      <td>0.353535</td>
      <td>0.198582</td>
      <td>0.642325</td>
      <td>0.943638</td>
      <td>33</td>
      <td>1</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



## Clean the Data


```python
diabetes.columns
```




    Index(['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
           'Insulin', 'BMI', 'Pedigree', 'Age', 'Class', 'Group'],
          dtype='object')




```python
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps', 'Insulin', 'BMI', 'Pedigree']
```


```python
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x : (x-x.min()) / (x.max()-x.min()))
```


```python
diabetes.head()
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
      <th>Number_pregnant</th>
      <th>Glucose_concentration</th>
      <th>Blood_pressure</th>
      <th>Triceps</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>Pedigree</th>
      <th>Age</th>
      <th>Class</th>
      <th>Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.352941</td>
      <td>0.743719</td>
      <td>0.590164</td>
      <td>0.353535</td>
      <td>0.000000</td>
      <td>0.500745</td>
      <td>0.234415</td>
      <td>50</td>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.058824</td>
      <td>0.427136</td>
      <td>0.540984</td>
      <td>0.292929</td>
      <td>0.000000</td>
      <td>0.396423</td>
      <td>0.116567</td>
      <td>31</td>
      <td>0</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.470588</td>
      <td>0.919598</td>
      <td>0.524590</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.347243</td>
      <td>0.253629</td>
      <td>32</td>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.058824</td>
      <td>0.447236</td>
      <td>0.540984</td>
      <td>0.232323</td>
      <td>0.111111</td>
      <td>0.418778</td>
      <td>0.038002</td>
      <td>21</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.688442</td>
      <td>0.327869</td>
      <td>0.353535</td>
      <td>0.198582</td>
      <td>0.642325</td>
      <td>0.943638</td>
      <td>33</td>
      <td>1</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



## Feature Columns


```python
diabetes.columns
```




    Index(['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
           'Insulin', 'BMI', 'Pedigree', 'Age', 'Class', 'Group'],
          dtype='object')



### Continuous Features


```python
diabetes[diabetes.columns[:8]]
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
      <th>Number_pregnant</th>
      <th>Glucose_concentration</th>
      <th>Blood_pressure</th>
      <th>Triceps</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>Pedigree</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.352941</td>
      <td>0.743719</td>
      <td>0.590164</td>
      <td>0.353535</td>
      <td>0.000000</td>
      <td>0.500745</td>
      <td>0.234415</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.058824</td>
      <td>0.427136</td>
      <td>0.540984</td>
      <td>0.292929</td>
      <td>0.000000</td>
      <td>0.396423</td>
      <td>0.116567</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.470588</td>
      <td>0.919598</td>
      <td>0.524590</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.347243</td>
      <td>0.253629</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.058824</td>
      <td>0.447236</td>
      <td>0.540984</td>
      <td>0.232323</td>
      <td>0.111111</td>
      <td>0.418778</td>
      <td>0.038002</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.688442</td>
      <td>0.327869</td>
      <td>0.353535</td>
      <td>0.198582</td>
      <td>0.642325</td>
      <td>0.943638</td>
      <td>33</td>
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
    </tr>
    <tr>
      <th>763</th>
      <td>0.588235</td>
      <td>0.507538</td>
      <td>0.622951</td>
      <td>0.484848</td>
      <td>0.212766</td>
      <td>0.490313</td>
      <td>0.039710</td>
      <td>63</td>
    </tr>
    <tr>
      <th>764</th>
      <td>0.117647</td>
      <td>0.613065</td>
      <td>0.573770</td>
      <td>0.272727</td>
      <td>0.000000</td>
      <td>0.548435</td>
      <td>0.111870</td>
      <td>27</td>
    </tr>
    <tr>
      <th>765</th>
      <td>0.294118</td>
      <td>0.608040</td>
      <td>0.590164</td>
      <td>0.232323</td>
      <td>0.132388</td>
      <td>0.390462</td>
      <td>0.071307</td>
      <td>30</td>
    </tr>
    <tr>
      <th>766</th>
      <td>0.058824</td>
      <td>0.633166</td>
      <td>0.491803</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.448584</td>
      <td>0.115713</td>
      <td>47</td>
    </tr>
    <tr>
      <th>767</th>
      <td>0.058824</td>
      <td>0.467337</td>
      <td>0.573770</td>
      <td>0.313131</td>
      <td>0.000000</td>
      <td>0.453055</td>
      <td>0.101196</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 8 columns</p>
</div>




```python
indexes = diabetes[diabetes.columns[:8]].columns
feat_con = indexes.to_list()
```


```python
for i in range(len(feat_con)):
    for k in diabetes[diabetes.columns[:8]]:
        feat_con[i] = tf.feature_column.numeric_column(diabetes[diabetes.columns[:8]].columns[i])


```


```python
num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, age = feat_con[:]
num_preg
```




    NumericColumn(key='Number_pregnant', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)



### Categorical Features


```python
assigned_group = tf.feature_column.categorical_column_with_hash_bucket(key='Group',
                                                                       hash_bucket_size=10

                                                                      )
```


```python
assigned_group
```




    HashedCategoricalColumn(key='Group', hash_bucket_size=10, dtype=tf.string)



### Convert Continuous to Categorical


```python
import matplotlib.pyplot as plt
%matplotlib inline

```


```python
diabetes['Age'].hist(bins=20);
```


![png](output_121_0.png)



```python
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])
```

Now we have continuous variables:

1. num_preg
2. plasma_gluc
3. dias_press
4. tricep
5. insulin
6. bmi
7. diabetes_pedigree
8. age


And categorical variable

1. class
2. group
3. age_buckets

### Putting them together



```python
feat_cols = [num_preg ,plasma_gluc,dias_press ,tricep ,insulin,bmi,diabetes_pedigree ,assigned_group, age_buckets]
```

## Train test split


```python
diabetes.head()
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
      <th>Number_pregnant</th>
      <th>Glucose_concentration</th>
      <th>Blood_pressure</th>
      <th>Triceps</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>Pedigree</th>
      <th>Age</th>
      <th>Class</th>
      <th>Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.352941</td>
      <td>0.743719</td>
      <td>0.590164</td>
      <td>0.353535</td>
      <td>0.000000</td>
      <td>0.500745</td>
      <td>0.234415</td>
      <td>50</td>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.058824</td>
      <td>0.427136</td>
      <td>0.540984</td>
      <td>0.292929</td>
      <td>0.000000</td>
      <td>0.396423</td>
      <td>0.116567</td>
      <td>31</td>
      <td>0</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.470588</td>
      <td>0.919598</td>
      <td>0.524590</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.347243</td>
      <td>0.253629</td>
      <td>32</td>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.058824</td>
      <td>0.447236</td>
      <td>0.540984</td>
      <td>0.232323</td>
      <td>0.111111</td>
      <td>0.418778</td>
      <td>0.038002</td>
      <td>21</td>
      <td>0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.688442</td>
      <td>0.327869</td>
      <td>0.353535</td>
      <td>0.198582</td>
      <td>0.642325</td>
      <td>0.943638</td>
      <td>33</td>
      <td>1</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_data = diabetes.drop('Class', axis=1)
labels = diabetes['Class']
```


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.33, random_state=101)
```

## Input function


```python
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,
                                                 y=y_train,
                                                 batch_size=10,
                                                 num_epochs=1000,
                                                 shuffle=True
                                                )
```

## Creating model


```python
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
```

    INFO:tensorflow:Using default config.
    WARNING:tensorflow:Using temporary folder as model directory: C:\Users\abulu\AppData\Local\Temp\tmpwnbx423x
    INFO:tensorflow:Using config: {'_model_dir': 'C:\\Users\\abulu\\AppData\\Local\\Temp\\tmpwnbx423x', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x000002990B6B18C8>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}



```python
model.train(input_fn=input_func, steps=1000)
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 0 into C:\Users\abulu\AppData\Local\Temp\tmpwnbx423x\model.ckpt.
    INFO:tensorflow:loss = 6.931472, step = 1
    INFO:tensorflow:global_step/sec: 133.78
    INFO:tensorflow:loss = 6.0543165, step = 101 (0.771 sec)
    INFO:tensorflow:global_step/sec: 245.008
    INFO:tensorflow:loss = 4.606823, step = 201 (0.393 sec)
    INFO:tensorflow:global_step/sec: 207.782
    INFO:tensorflow:loss = 5.849765, step = 301 (0.476 sec)
    INFO:tensorflow:global_step/sec: 229.071
    INFO:tensorflow:loss = 6.593772, step = 401 (0.440 sec)
    INFO:tensorflow:global_step/sec: 224.186
    INFO:tensorflow:loss = 7.709773, step = 501 (0.441 sec)
    INFO:tensorflow:global_step/sec: 244.163
    INFO:tensorflow:loss = 5.3710237, step = 601 (0.418 sec)
    INFO:tensorflow:global_step/sec: 242.7
    INFO:tensorflow:loss = 3.1055427, step = 701 (0.409 sec)
    INFO:tensorflow:global_step/sec: 219.758
    INFO:tensorflow:loss = 4.2785654, step = 801 (0.457 sec)
    INFO:tensorflow:global_step/sec: 245.956
    INFO:tensorflow:loss = 7.26054, step = 901 (0.405 sec)
    INFO:tensorflow:Saving checkpoints for 1000 into C:\Users\abulu\AppData\Local\Temp\tmpwnbx423x\model.ckpt.
    INFO:tensorflow:Loss for final step: 7.219046.





    <tensorflow_estimator.python.estimator.canned.linear.LinearClassifier at 0x2990b6a4c08>



## Evaluation


```python
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                                      y=y_test,
                                                      batch_size=10,
                                                      shuffle=False,
                                                      num_epochs=1
                                                     )
```


```python
result = model.evaluate(eval_input_func)
```

    INFO:tensorflow:Calling model_fn.
    WARNING:tensorflow:From C:\Users\abulu\anaconda3\lib\site-packages\tensorflow_core\python\ops\metrics_impl.py:2026: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Deprecated in favor of operator or tf.math.divide.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Starting evaluation at 2020-07-17T07:35:04Z
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\abulu\AppData\Local\Temp\tmpwnbx423x\model.ckpt-1000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Finished evaluation at 2020-07-17-07:35:06
    INFO:tensorflow:Saving dict for global step 1000: accuracy = 0.72440946, accuracy_baseline = 0.65748036, auc = 0.7812651, auc_precision_recall = 0.6176144, average_loss = 0.5387462, global_step = 1000, label/mean = 0.34251967, loss = 5.2631354, precision = 0.60240966, prediction/mean = 0.38184115, recall = 0.57471263
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 1000: C:\Users\abulu\AppData\Local\Temp\tmpwnbx423x\model.ckpt-1000



```python
result
```




    {'accuracy': 0.72440946,
     'accuracy_baseline': 0.65748036,
     'auc': 0.7812651,
     'auc_precision_recall': 0.6176144,
     'average_loss': 0.5387462,
     'label/mean': 0.34251967,
     'loss': 5.2631354,
     'precision': 0.60240966,
     'prediction/mean': 0.38184115,
     'recall': 0.57471263,
     'global_step': 1000}



## Prediction of new data


```python
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                                      batch_size=10,
                                                      num_epochs=1,
                                                      shuffle=False)
```


```python
predictions = model.predict(input_fn=pred_input_func)
```


```python
predictions
```




    <generator object Estimator.predict at 0x0000029909BC2C48>




```python
mypred=list(predictions)
mypred
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\abulu\AppData\Local\Temp\tmpwnbx423x\model.ckpt-1000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.





    [{'logits': array([0.15652645], dtype=float32),
      'logistic': array([0.5390519], dtype=float32),
      'probabilities': array([0.46094808, 0.5390519 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.49707556], dtype=float32),
      'logistic': array([0.6217718], dtype=float32),
      'probabilities': array([0.37822816, 0.6217718 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.3423611], dtype=float32),
      'logistic': array([0.41523606], dtype=float32),
      'probabilities': array([0.58476394, 0.41523603], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.6890718], dtype=float32),
      'logistic': array([0.3342396], dtype=float32),
      'probabilities': array([0.6657604, 0.3342396], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.6648747], dtype=float32),
      'logistic': array([0.1591087], dtype=float32),
      'probabilities': array([0.8408913 , 0.15910873], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.7803595], dtype=float32),
      'logistic': array([0.6857576], dtype=float32),
      'probabilities': array([0.31424242, 0.6857576 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.35092163], dtype=float32),
      'logistic': array([0.58684105], dtype=float32),
      'probabilities': array([0.41315898, 0.5868411 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.5202713], dtype=float32),
      'logistic': array([0.37278882], dtype=float32),
      'probabilities': array([0.6272112 , 0.37278882], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.0711479], dtype=float32),
      'logistic': array([0.25518483], dtype=float32),
      'probabilities': array([0.7448152 , 0.25518486], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.0656369], dtype=float32),
      'logistic': array([0.25623372], dtype=float32),
      'probabilities': array([0.74376625, 0.2562337 ], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.8398626], dtype=float32),
      'logistic': array([0.13706753], dtype=float32),
      'probabilities': array([0.8629325 , 0.13706756], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.6099801], dtype=float32),
      'logistic': array([0.16659138], dtype=float32),
      'probabilities': array([0.83340865, 0.16659139], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.71776295], dtype=float32),
      'logistic': array([0.3278858], dtype=float32),
      'probabilities': array([0.67211425, 0.3278858 ], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.39483047], dtype=float32),
      'logistic': array([0.597445], dtype=float32),
      'probabilities': array([0.40255502, 0.597445  ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.5282011], dtype=float32),
      'logistic': array([0.17825705], dtype=float32),
      'probabilities': array([0.821743  , 0.17825705], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.6267849], dtype=float32),
      'logistic': array([0.34823993], dtype=float32),
      'probabilities': array([0.6517601, 0.3482399], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.44008678], dtype=float32),
      'logistic': array([0.3917203], dtype=float32),
      'probabilities': array([0.6082797 , 0.39172027], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.8401628], dtype=float32),
      'logistic': array([0.13703203], dtype=float32),
      'probabilities': array([0.86296797, 0.13703205], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.44096327], dtype=float32),
      'logistic': array([0.39151147], dtype=float32),
      'probabilities': array([0.6084885 , 0.39151144], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.7228401], dtype=float32),
      'logistic': array([0.15150571], dtype=float32),
      'probabilities': array([0.8484943 , 0.15150571], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.2049282], dtype=float32),
      'logistic': array([0.23059967], dtype=float32),
      'probabilities': array([0.76940036, 0.23059969], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.77639294], dtype=float32),
      'logistic': array([0.3150978], dtype=float32),
      'probabilities': array([0.6849022, 0.3150978], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.70183], dtype=float32),
      'logistic': array([0.66859335], dtype=float32),
      'probabilities': array([0.33140662, 0.6685934 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.2171028], dtype=float32),
      'logistic': array([0.22844669], dtype=float32),
      'probabilities': array([0.7715533, 0.2284467], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.412579], dtype=float32),
      'logistic': array([0.19582757], dtype=float32),
      'probabilities': array([0.8041724 , 0.19582759], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([1.2446325], dtype=float32),
      'logistic': array([0.77636933], dtype=float32),
      'probabilities': array([0.22363067, 0.77636933], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.4116753], dtype=float32),
      'logistic': array([0.19596994], dtype=float32),
      'probabilities': array([0.80403006, 0.19596994], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.77069294], dtype=float32),
      'logistic': array([0.31632924], dtype=float32),
      'probabilities': array([0.68367076, 0.31632924], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-2.1080256], dtype=float32),
      'logistic': array([0.10831922], dtype=float32),
      'probabilities': array([0.8916808 , 0.10831922], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.43039107], dtype=float32),
      'logistic': array([0.60596704], dtype=float32),
      'probabilities': array([0.39403293, 0.60596704], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([2.7407637], dtype=float32),
      'logistic': array([0.9393896], dtype=float32),
      'probabilities': array([0.06061041, 0.9393896 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.29752433], dtype=float32),
      'logistic': array([0.4261628], dtype=float32),
      'probabilities': array([0.5738372, 0.4261628], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.7392054], dtype=float32),
      'logistic': array([0.32317793], dtype=float32),
      'probabilities': array([0.67682207, 0.3231779 ], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.365196], dtype=float32),
      'logistic': array([0.59029764], dtype=float32),
      'probabilities': array([0.40970236, 0.59029764], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.030357], dtype=float32),
      'logistic': array([0.26301488], dtype=float32),
      'probabilities': array([0.73698515, 0.2630149 ], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.3631996], dtype=float32),
      'logistic': array([0.41018528], dtype=float32),
      'probabilities': array([0.5898147 , 0.41018525], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.11858785], dtype=float32),
      'logistic': array([0.52961224], dtype=float32),
      'probabilities': array([0.47038773, 0.52961224], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.62737036], dtype=float32),
      'logistic': array([0.6518929], dtype=float32),
      'probabilities': array([0.34810707, 0.65189296], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([1.0936623], dtype=float32),
      'logistic': array([0.74907076], dtype=float32),
      'probabilities': array([0.2509293 , 0.74907076], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.62206125], dtype=float32),
      'logistic': array([0.34931278], dtype=float32),
      'probabilities': array([0.6506872, 0.3493128], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.393826], dtype=float32),
      'logistic': array([0.19879764], dtype=float32),
      'probabilities': array([0.80120236, 0.19879767], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.19331682], dtype=float32),
      'logistic': array([0.54817927], dtype=float32),
      'probabilities': array([0.45182073, 0.54817927], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.5891826], dtype=float32),
      'logistic': array([0.16949889], dtype=float32),
      'probabilities': array([0.8305011 , 0.16949894], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.16544294], dtype=float32),
      'logistic': array([0.5412666], dtype=float32),
      'probabilities': array([0.45873338, 0.5412667 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.680789], dtype=float32),
      'logistic': array([0.156991], dtype=float32),
      'probabilities': array([0.843009  , 0.15699103], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-2.0334938], dtype=float32),
      'logistic': array([0.11573091], dtype=float32),
      'probabilities': array([0.88426906, 0.11573089], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.4931754], dtype=float32),
      'logistic': array([0.62085414], dtype=float32),
      'probabilities': array([0.37914583, 0.6208542 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.3864852], dtype=float32),
      'logistic': array([0.19996944], dtype=float32),
      'probabilities': array([0.8000305 , 0.19996946], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.2637606], dtype=float32),
      'logistic': array([0.22032721], dtype=float32),
      'probabilities': array([0.7796728, 0.2203272], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.9777828], dtype=float32),
      'logistic': array([0.12155538], dtype=float32),
      'probabilities': array([0.87844455, 0.12155538], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.5123078], dtype=float32),
      'logistic': array([0.18059707], dtype=float32),
      'probabilities': array([0.81940293, 0.18059702], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.0574362], dtype=float32),
      'logistic': array([0.2577997], dtype=float32),
      'probabilities': array([0.7422003 , 0.25779969], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.0267159], dtype=float32),
      'logistic': array([0.26372135], dtype=float32),
      'probabilities': array([0.7362787 , 0.26372132], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.40214896], dtype=float32),
      'logistic': array([0.40079615], dtype=float32),
      'probabilities': array([0.5992038 , 0.40079612], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.7521031], dtype=float32),
      'logistic': array([0.14778212], dtype=float32),
      'probabilities': array([0.85221785, 0.14778213], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.2050265], dtype=float32),
      'logistic': array([0.23058224], dtype=float32),
      'probabilities': array([0.76941776, 0.23058222], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.224959], dtype=float32),
      'logistic': array([0.22706494], dtype=float32),
      'probabilities': array([0.7729351 , 0.22706495], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.2284904], dtype=float32),
      'logistic': array([0.2264458], dtype=float32),
      'probabilities': array([0.77355427, 0.22644576], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.07060373], dtype=float32),
      'logistic': array([0.5176436], dtype=float32),
      'probabilities': array([0.48235637, 0.5176436 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.5549314], dtype=float32),
      'logistic': array([0.17437516], dtype=float32),
      'probabilities': array([0.8256249 , 0.17437518], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.1003025], dtype=float32),
      'logistic': array([0.24968326], dtype=float32),
      'probabilities': array([0.7503168 , 0.24968323], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.69223976], dtype=float32),
      'logistic': array([0.666465], dtype=float32),
      'probabilities': array([0.333535, 0.666465], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.540225], dtype=float32),
      'logistic': array([0.6318648], dtype=float32),
      'probabilities': array([0.36813524, 0.6318647 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.06415558], dtype=float32),
      'logistic': array([0.5160334], dtype=float32),
      'probabilities': array([0.48396662, 0.5160334 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.5642401], dtype=float32),
      'logistic': array([0.36256695], dtype=float32),
      'probabilities': array([0.63743305, 0.36256698], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.03477478], dtype=float32),
      'logistic': array([0.5086928], dtype=float32),
      'probabilities': array([0.49130717, 0.5086928 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.4814208], dtype=float32),
      'logistic': array([0.18521294], dtype=float32),
      'probabilities': array([0.8147871 , 0.18521293], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.3018522], dtype=float32),
      'logistic': array([0.21385345], dtype=float32),
      'probabilities': array([0.7861465 , 0.21385345], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.5979596], dtype=float32),
      'logistic': array([0.16826697], dtype=float32),
      'probabilities': array([0.831733  , 0.16826697], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.7520052], dtype=float32),
      'logistic': array([0.14779447], dtype=float32),
      'probabilities': array([0.8522056 , 0.14779447], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.4185698], dtype=float32),
      'logistic': array([0.60314095], dtype=float32),
      'probabilities': array([0.39685905, 0.60314095], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.0935931], dtype=float32),
      'logistic': array([0.25094226], dtype=float32),
      'probabilities': array([0.74905777, 0.2509423 ], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.21585143], dtype=float32),
      'logistic': array([0.4462457], dtype=float32),
      'probabilities': array([0.5537543 , 0.44624573], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.40172982], dtype=float32),
      'logistic': array([0.5991032], dtype=float32),
      'probabilities': array([0.40089676, 0.59910315], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.8387807], dtype=float32),
      'logistic': array([0.30179167], dtype=float32),
      'probabilities': array([0.69820833, 0.30179164], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.63169694], dtype=float32),
      'logistic': array([0.6528741], dtype=float32),
      'probabilities': array([0.34712586, 0.6528741 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.825357], dtype=float32),
      'logistic': array([0.1387923], dtype=float32),
      'probabilities': array([0.86120766, 0.13879232], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.44804382], dtype=float32),
      'logistic': array([0.610174], dtype=float32),
      'probabilities': array([0.38982597, 0.61017406], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.8843212], dtype=float32),
      'logistic': array([0.13189332], dtype=float32),
      'probabilities': array([0.8681067 , 0.13189332], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.2403022], dtype=float32),
      'logistic': array([0.5597881], dtype=float32),
      'probabilities': array([0.44021186, 0.5597881 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.5972296], dtype=float32),
      'logistic': array([0.3549778], dtype=float32),
      'probabilities': array([0.6450222 , 0.35497776], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.511801], dtype=float32),
      'logistic': array([0.18067202], dtype=float32),
      'probabilities': array([0.81932795, 0.18067203], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.26319373], dtype=float32),
      'logistic': array([0.5654212], dtype=float32),
      'probabilities': array([0.43457878, 0.5654212 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.40432477], dtype=float32),
      'logistic': array([0.59972626], dtype=float32),
      'probabilities': array([0.4002737, 0.5997263], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.27770674], dtype=float32),
      'logistic': array([0.5689839], dtype=float32),
      'probabilities': array([0.4310161, 0.5689839], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.4868379], dtype=float32),
      'logistic': array([0.18439683], dtype=float32),
      'probabilities': array([0.8156032 , 0.18439682], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.5091523], dtype=float32),
      'logistic': array([0.37539226], dtype=float32),
      'probabilities': array([0.62460774, 0.3753923 ], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.6554847], dtype=float32),
      'logistic': array([0.16036907], dtype=float32),
      'probabilities': array([0.8396309 , 0.16036905], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.9426429], dtype=float32),
      'logistic': array([0.7196332], dtype=float32),
      'probabilities': array([0.28036678, 0.7196332 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([1.5261445], dtype=float32),
      'logistic': array([0.8214415], dtype=float32),
      'probabilities': array([0.17855848, 0.8214415 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.53744996], dtype=float32),
      'logistic': array([0.631219], dtype=float32),
      'probabilities': array([0.368781, 0.631219], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.2351558], dtype=float32),
      'logistic': array([0.22528031], dtype=float32),
      'probabilities': array([0.77471966, 0.2252803 ], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.21030796], dtype=float32),
      'logistic': array([0.44761595], dtype=float32),
      'probabilities': array([0.552384  , 0.44761592], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.42725807], dtype=float32),
      'logistic': array([0.3947813], dtype=float32),
      'probabilities': array([0.60521877, 0.3947813 ], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.07854843], dtype=float32),
      'logistic': array([0.480373], dtype=float32),
      'probabilities': array([0.51962703, 0.48037302], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.6105846], dtype=float32),
      'logistic': array([0.35192588], dtype=float32),
      'probabilities': array([0.64807415, 0.35192585], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.11339515], dtype=float32),
      'logistic': array([0.47168157], dtype=float32),
      'probabilities': array([0.52831846, 0.47168157], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.5606086], dtype=float32),
      'logistic': array([0.17355934], dtype=float32),
      'probabilities': array([0.82644063, 0.17355932], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.25853372], dtype=float32),
      'logistic': array([0.5642758], dtype=float32),
      'probabilities': array([0.43572417, 0.5642758 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.28293192], dtype=float32),
      'logistic': array([0.5702649], dtype=float32),
      'probabilities': array([0.42973512, 0.5702649 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.0933548], dtype=float32),
      'logistic': array([0.2509871], dtype=float32),
      'probabilities': array([0.74901295, 0.25098708], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.4264697], dtype=float32),
      'logistic': array([0.19364938], dtype=float32),
      'probabilities': array([0.80635065, 0.19364934], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.4421829], dtype=float32),
      'logistic': array([0.19120756], dtype=float32),
      'probabilities': array([0.8087925 , 0.19120754], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.8940967], dtype=float32),
      'logistic': array([0.29026514], dtype=float32),
      'probabilities': array([0.70973486, 0.29026514], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.3874625], dtype=float32),
      'logistic': array([0.40432832], dtype=float32),
      'probabilities': array([0.5956717 , 0.40432832], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.7156084], dtype=float32),
      'logistic': array([0.15243769], dtype=float32),
      'probabilities': array([0.8475623, 0.1524377], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.6310084], dtype=float32),
      'logistic': array([0.34728193], dtype=float32),
      'probabilities': array([0.65271807, 0.34728193], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.1580088], dtype=float32),
      'logistic': array([0.23902929], dtype=float32),
      'probabilities': array([0.7609707 , 0.23902929], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-2.141654], dtype=float32),
      'logistic': array([0.1051137], dtype=float32),
      'probabilities': array([0.8948863 , 0.10511371], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.8988707], dtype=float32),
      'logistic': array([0.13023634], dtype=float32),
      'probabilities': array([0.8697637 , 0.13023634], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([1.1183], dtype=float32),
      'logistic': array([0.7536732], dtype=float32),
      'probabilities': array([0.24632676, 0.75367326], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.4320714], dtype=float32),
      'logistic': array([0.19277614], dtype=float32),
      'probabilities': array([0.80722386, 0.19277613], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.7548261], dtype=float32),
      'logistic': array([0.14743951], dtype=float32),
      'probabilities': array([0.8525605 , 0.14743952], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.7412794], dtype=float32),
      'logistic': array([0.14915046], dtype=float32),
      'probabilities': array([0.8508495, 0.1491505], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.07148433], dtype=float32),
      'logistic': array([0.51786345], dtype=float32),
      'probabilities': array([0.48213655, 0.5178635 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.04997742], dtype=float32),
      'logistic': array([0.48750824], dtype=float32),
      'probabilities': array([0.51249176, 0.48750827], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.075425], dtype=float32),
      'logistic': array([0.2543728], dtype=float32),
      'probabilities': array([0.7456273 , 0.25437278], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.19048673], dtype=float32),
      'logistic': array([0.4525218], dtype=float32),
      'probabilities': array([0.5474782, 0.4525218], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.3832451], dtype=float32),
      'logistic': array([0.20048833], dtype=float32),
      'probabilities': array([0.79951173, 0.20048834], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.1457813], dtype=float32),
      'logistic': array([0.2412605], dtype=float32),
      'probabilities': array([0.75873953, 0.24126051], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.00764507], dtype=float32),
      'logistic': array([0.5019113], dtype=float32),
      'probabilities': array([0.49808878, 0.5019113 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.02720392], dtype=float32),
      'logistic': array([0.50680053], dtype=float32),
      'probabilities': array([0.4931994 , 0.50680053], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.6897668], dtype=float32),
      'logistic': array([0.15580651], dtype=float32),
      'probabilities': array([0.84419346, 0.15580651], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.3706205], dtype=float32),
      'logistic': array([0.40839112], dtype=float32),
      'probabilities': array([0.5916089, 0.4083911], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.84302115], dtype=float32),
      'logistic': array([0.6991011], dtype=float32),
      'probabilities': array([0.30089888, 0.6991011 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.2518865], dtype=float32),
      'logistic': array([0.5626408], dtype=float32),
      'probabilities': array([0.4373592, 0.5626408], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.31961983], dtype=float32),
      'logistic': array([0.4207684], dtype=float32),
      'probabilities': array([0.5792316 , 0.42076844], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.30967748], dtype=float32),
      'logistic': array([0.57680655], dtype=float32),
      'probabilities': array([0.42319345, 0.57680655], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.1961896], dtype=float32),
      'logistic': array([0.23215374], dtype=float32),
      'probabilities': array([0.7678462 , 0.23215374], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.3148222], dtype=float32),
      'logistic': array([0.21168104], dtype=float32),
      'probabilities': array([0.788319  , 0.21168104], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.7530949], dtype=float32),
      'logistic': array([0.14765725], dtype=float32),
      'probabilities': array([0.8523427 , 0.14765728], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.0165224], dtype=float32),
      'logistic': array([0.26570538], dtype=float32),
      'probabilities': array([0.73429465, 0.26570535], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.05639964], dtype=float32),
      'logistic': array([0.48590383], dtype=float32),
      'probabilities': array([0.5140962 , 0.48590386], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.7975259], dtype=float32),
      'logistic': array([0.689445], dtype=float32),
      'probabilities': array([0.310555  , 0.68944496], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.1806737], dtype=float32),
      'logistic': array([0.23493108], dtype=float32),
      'probabilities': array([0.7650689 , 0.23493108], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.6170547], dtype=float32),
      'logistic': array([0.16561145], dtype=float32),
      'probabilities': array([0.83438855, 0.16561146], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.24539268], dtype=float32),
      'logistic': array([0.43895784], dtype=float32),
      'probabilities': array([0.56104213, 0.43895784], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.1256213], dtype=float32),
      'logistic': array([0.24497005], dtype=float32),
      'probabilities': array([0.7550299 , 0.24497008], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.6694522], dtype=float32),
      'logistic': array([0.15849723], dtype=float32),
      'probabilities': array([0.8415028 , 0.15849723], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.4103087], dtype=float32),
      'logistic': array([0.19618538], dtype=float32),
      'probabilities': array([0.80381465, 0.19618537], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.8941054], dtype=float32),
      'logistic': array([0.13077709], dtype=float32),
      'probabilities': array([0.86922294, 0.13077709], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.4383155], dtype=float32),
      'logistic': array([0.39214242], dtype=float32),
      'probabilities': array([0.6078576 , 0.39214242], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([1.2780626], dtype=float32),
      'logistic': array([0.7821198], dtype=float32),
      'probabilities': array([0.21788019, 0.7821198 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([1.1880805], dtype=float32),
      'logistic': array([0.7663976], dtype=float32),
      'probabilities': array([0.2336024, 0.7663976], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.25707316], dtype=float32),
      'logistic': array([0.5639167], dtype=float32),
      'probabilities': array([0.43608332, 0.5639167 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.00436449], dtype=float32),
      'logistic': array([0.49890888], dtype=float32),
      'probabilities': array([0.5010911, 0.4989089], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.25761342], dtype=float32),
      'logistic': array([0.56404954], dtype=float32),
      'probabilities': array([0.4359505 , 0.56404954], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.9332423], dtype=float32),
      'logistic': array([0.12639216], dtype=float32),
      'probabilities': array([0.8736079 , 0.12639214], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.16000938], dtype=float32),
      'logistic': array([0.53991723], dtype=float32),
      'probabilities': array([0.4600828 , 0.53991723], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([1.201807], dtype=float32),
      'logistic': array([0.76884604], dtype=float32),
      'probabilities': array([0.2311539 , 0.76884604], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.2794661], dtype=float32),
      'logistic': array([0.4305847], dtype=float32),
      'probabilities': array([0.56941533, 0.4305847 ], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.898512], dtype=float32),
      'logistic': array([0.28935638], dtype=float32),
      'probabilities': array([0.71064365, 0.28935638], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.4349565], dtype=float32),
      'logistic': array([0.3929434], dtype=float32),
      'probabilities': array([0.6070566, 0.3929434], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.44896233], dtype=float32),
      'logistic': array([0.61039245], dtype=float32),
      'probabilities': array([0.38960755, 0.6103925 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.0735512], dtype=float32),
      'logistic': array([0.25472832], dtype=float32),
      'probabilities': array([0.7452717 , 0.25472835], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.63958025], dtype=float32),
      'logistic': array([0.65465856], dtype=float32),
      'probabilities': array([0.34534144, 0.6546586 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.3593638], dtype=float32),
      'logistic': array([0.4111136], dtype=float32),
      'probabilities': array([0.58888644, 0.41111362], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.30105555], dtype=float32),
      'logistic': array([0.42529947], dtype=float32),
      'probabilities': array([0.57470053, 0.42529947], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.5351486], dtype=float32),
      'logistic': array([0.17724162], dtype=float32),
      'probabilities': array([0.8227584 , 0.17724164], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.3619018], dtype=float32),
      'logistic': array([0.20393138], dtype=float32),
      'probabilities': array([0.7960686 , 0.20393139], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.4525678], dtype=float32),
      'logistic': array([0.18960667], dtype=float32),
      'probabilities': array([0.81039333, 0.1896067 ], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.1918807], dtype=float32),
      'logistic': array([0.45217648], dtype=float32),
      'probabilities': array([0.54782355, 0.45217648], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.6330588], dtype=float32),
      'logistic': array([0.16341177], dtype=float32),
      'probabilities': array([0.8365882 , 0.16341177], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.99882734], dtype=float32),
      'logistic': array([0.7308279], dtype=float32),
      'probabilities': array([0.26917204, 0.7308279 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.2879733], dtype=float32),
      'logistic': array([0.21619606], dtype=float32),
      'probabilities': array([0.7838039 , 0.21619605], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.09101772], dtype=float32),
      'logistic': array([0.52273875], dtype=float32),
      'probabilities': array([0.47726128, 0.52273875], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.6054876], dtype=float32),
      'logistic': array([0.16721606], dtype=float32),
      'probabilities': array([0.832784  , 0.16721605], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.7242882], dtype=float32),
      'logistic': array([0.6735506], dtype=float32),
      'probabilities': array([0.3264494, 0.6735506], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.67606175], dtype=float32),
      'logistic': array([0.33714086], dtype=float32),
      'probabilities': array([0.66285914, 0.33714083], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.337137], dtype=float32),
      'logistic': array([0.20798126], dtype=float32),
      'probabilities': array([0.7920188 , 0.20798127], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.5232364], dtype=float32),
      'logistic': array([0.37209576], dtype=float32),
      'probabilities': array([0.6279042 , 0.37209573], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.7069396], dtype=float32),
      'logistic': array([0.33027548], dtype=float32),
      'probabilities': array([0.6697245 , 0.33027542], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.5309724], dtype=float32),
      'logistic': array([0.17785147], dtype=float32),
      'probabilities': array([0.8221485 , 0.17785145], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.9711175], dtype=float32),
      'logistic': array([0.72534215], dtype=float32),
      'probabilities': array([0.27465782, 0.72534215], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.3316518], dtype=float32),
      'logistic': array([0.20888627], dtype=float32),
      'probabilities': array([0.79111373, 0.20888628], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.514461], dtype=float32),
      'logistic': array([0.1802786], dtype=float32),
      'probabilities': array([0.8197214 , 0.18027861], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.94388086], dtype=float32),
      'logistic': array([0.28011712], dtype=float32),
      'probabilities': array([0.7198829, 0.2801171], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.17495155], dtype=float32),
      'logistic': array([0.45637333], dtype=float32),
      'probabilities': array([0.54362667, 0.45637333], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.9097699], dtype=float32),
      'logistic': array([0.28704694], dtype=float32),
      'probabilities': array([0.7129531 , 0.28704694], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.94418], dtype=float32),
      'logistic': array([0.12518936], dtype=float32),
      'probabilities': array([0.87481064, 0.12518936], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([1.6155245], dtype=float32),
      'logistic': array([0.834177], dtype=float32),
      'probabilities': array([0.16582301, 0.83417696], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.7691226], dtype=float32),
      'logistic': array([0.683331], dtype=float32),
      'probabilities': array([0.31666893, 0.6833311 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.20079637], dtype=float32),
      'logistic': array([0.5500311], dtype=float32),
      'probabilities': array([0.4499689, 0.5500311], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.9826883], dtype=float32),
      'logistic': array([0.7276413], dtype=float32),
      'probabilities': array([0.2723587, 0.7276413], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-2.0075142], dtype=float32),
      'logistic': array([0.11841625], dtype=float32),
      'probabilities': array([0.88158375, 0.11841623], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.6508651], dtype=float32),
      'logistic': array([0.16099203], dtype=float32),
      'probabilities': array([0.839008  , 0.16099207], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.2933464], dtype=float32),
      'logistic': array([0.21528694], dtype=float32),
      'probabilities': array([0.7847131 , 0.21528694], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.62236273], dtype=float32),
      'logistic': array([0.3492443], dtype=float32),
      'probabilities': array([0.6507557 , 0.34924427], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.48445654], dtype=float32),
      'logistic': array([0.38120034], dtype=float32),
      'probabilities': array([0.6187997 , 0.38120034], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.3560735], dtype=float32),
      'logistic': array([0.58808964], dtype=float32),
      'probabilities': array([0.4119104 , 0.58808964], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.20214903], dtype=float32),
      'logistic': array([0.55036587], dtype=float32),
      'probabilities': array([0.44963413, 0.55036587], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.42306024], dtype=float32),
      'logistic': array([0.3957847], dtype=float32),
      'probabilities': array([0.6042153, 0.3957847], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.46006835], dtype=float32),
      'logistic': array([0.38696963], dtype=float32),
      'probabilities': array([0.6130304, 0.3869696], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.49475527], dtype=float32),
      'logistic': array([0.378774], dtype=float32),
      'probabilities': array([0.621226, 0.378774], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.3907281], dtype=float32),
      'logistic': array([0.19929153], dtype=float32),
      'probabilities': array([0.8007085 , 0.19929156], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.6189307], dtype=float32),
      'logistic': array([0.6499753], dtype=float32),
      'probabilities': array([0.35002467, 0.6499753 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.8592539], dtype=float32),
      'logistic': array([0.13479], dtype=float32),
      'probabilities': array([0.86521   , 0.13479005], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.97619784], dtype=float32),
      'logistic': array([0.2736469], dtype=float32),
      'probabilities': array([0.7263531 , 0.27364686], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.17821217], dtype=float32),
      'logistic': array([0.5444355], dtype=float32),
      'probabilities': array([0.4555645, 0.5444355], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.51610017], dtype=float32),
      'logistic': array([0.3737646], dtype=float32),
      'probabilities': array([0.6262354, 0.3737646], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.373675], dtype=float32),
      'logistic': array([0.20202672], dtype=float32),
      'probabilities': array([0.7979733 , 0.20202675], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.89841306], dtype=float32),
      'logistic': array([0.28937674], dtype=float32),
      'probabilities': array([0.71062326, 0.2893767 ], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.96706027], dtype=float32),
      'logistic': array([0.27546686], dtype=float32),
      'probabilities': array([0.72453314, 0.27546683], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.7902675], dtype=float32),
      'logistic': array([0.14303991], dtype=float32),
      'probabilities': array([0.85696006, 0.14303994], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.4414214], dtype=float32),
      'logistic': array([0.1913253], dtype=float32),
      'probabilities': array([0.8086747 , 0.19132535], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.3340757], dtype=float32),
      'logistic': array([0.58275074], dtype=float32),
      'probabilities': array([0.4172493 , 0.58275074], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.7905768], dtype=float32),
      'logistic': array([0.68795514], dtype=float32),
      'probabilities': array([0.31204483, 0.68795514], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.50846803], dtype=float32),
      'logistic': array([0.6244473], dtype=float32),
      'probabilities': array([0.37555274, 0.6244473 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.8651918], dtype=float32),
      'logistic': array([0.70374423], dtype=float32),
      'probabilities': array([0.29625577, 0.70374423], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.08803403], dtype=float32),
      'logistic': array([0.4780057], dtype=float32),
      'probabilities': array([0.5219943 , 0.47800568], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.82374364], dtype=float32),
      'logistic': array([0.30496958], dtype=float32),
      'probabilities': array([0.69503045, 0.30496958], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.0548985], dtype=float32),
      'logistic': array([0.51372117], dtype=float32),
      'probabilities': array([0.48627883, 0.51372117], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.4291216], dtype=float32),
      'logistic': array([0.19323558], dtype=float32),
      'probabilities': array([0.8067644 , 0.19323559], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.7127969], dtype=float32),
      'logistic': array([0.1528013], dtype=float32),
      'probabilities': array([0.84719867, 0.15280129], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.3925333], dtype=float32),
      'logistic': array([0.19900364], dtype=float32),
      'probabilities': array([0.80099636, 0.19900364], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.07810092], dtype=float32),
      'logistic': array([0.51951534], dtype=float32),
      'probabilities': array([0.48048472, 0.51951534], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-2.5814633], dtype=float32),
      'logistic': array([0.07034099], dtype=float32),
      'probabilities': array([0.929659  , 0.07034098], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.874089], dtype=float32),
      'logistic': array([0.2944042], dtype=float32),
      'probabilities': array([0.70559585, 0.2944042 ], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.5292218], dtype=float32),
      'logistic': array([0.17810757], dtype=float32),
      'probabilities': array([0.82189244, 0.17810759], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.9009876], dtype=float32),
      'logistic': array([0.28884757], dtype=float32),
      'probabilities': array([0.71115243, 0.2888476 ], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.0809021], dtype=float32),
      'logistic': array([0.25333536], dtype=float32),
      'probabilities': array([0.74666464, 0.25333533], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.96174246], dtype=float32),
      'logistic': array([0.27652943], dtype=float32),
      'probabilities': array([0.72347057, 0.27652946], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([1.1846085], dtype=float32),
      'logistic': array([0.76577544], dtype=float32),
      'probabilities': array([0.2342246 , 0.76577544], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.4440498], dtype=float32),
      'logistic': array([0.19091898], dtype=float32),
      'probabilities': array([0.809081, 0.190919], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.8508105], dtype=float32),
      'logistic': array([0.13577777], dtype=float32),
      'probabilities': array([0.8642223 , 0.13577776], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.40428722], dtype=float32),
      'logistic': array([0.59971726], dtype=float32),
      'probabilities': array([0.40028277, 0.5997173 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.01913691], dtype=float32),
      'logistic': array([0.49521592], dtype=float32),
      'probabilities': array([0.5047841 , 0.49521595], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([1.696264], dtype=float32),
      'logistic': array([0.84504616], dtype=float32),
      'probabilities': array([0.15495384, 0.8450462 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.5734564], dtype=float32),
      'logistic': array([0.63956034], dtype=float32),
      'probabilities': array([0.36043966, 0.63956034], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.8265579], dtype=float32),
      'logistic': array([0.13864884], dtype=float32),
      'probabilities': array([0.8613512 , 0.13864884], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.63882864], dtype=float32),
      'logistic': array([0.6544886], dtype=float32),
      'probabilities': array([0.34551135, 0.6544886 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.21626043], dtype=float32),
      'logistic': array([0.55385536], dtype=float32),
      'probabilities': array([0.4461446 , 0.55385536], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([1.2162428], dtype=float32),
      'logistic': array([0.77140164], dtype=float32),
      'probabilities': array([0.22859833, 0.77140164], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.161502], dtype=float32),
      'logistic': array([0.23839447], dtype=float32),
      'probabilities': array([0.7616055 , 0.23839445], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.5250192], dtype=float32),
      'logistic': array([0.1787236], dtype=float32),
      'probabilities': array([0.82127637, 0.1787236 ], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.30953908], dtype=float32),
      'logistic': array([0.57677275], dtype=float32),
      'probabilities': array([0.42322725, 0.57677275], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.621764], dtype=float32),
      'logistic': array([0.16496176], dtype=float32),
      'probabilities': array([0.83503824, 0.16496174], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([1.241158], dtype=float32),
      'logistic': array([0.77576554], dtype=float32),
      'probabilities': array([0.22423448, 0.77576554], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.11652744], dtype=float32),
      'logistic': array([0.529099], dtype=float32),
      'probabilities': array([0.4709011, 0.529099 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.4399824], dtype=float32),
      'logistic': array([0.19154808], dtype=float32),
      'probabilities': array([0.80845195, 0.19154808], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.29614162], dtype=float32),
      'logistic': array([0.573499], dtype=float32),
      'probabilities': array([0.42650095, 0.573499  ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.09454119], dtype=float32),
      'logistic': array([0.4763823], dtype=float32),
      'probabilities': array([0.5236177, 0.4763823], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.9825604], dtype=float32),
      'logistic': array([0.12104616], dtype=float32),
      'probabilities': array([0.8789538 , 0.12104616], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.15765989], dtype=float32),
      'logistic': array([0.5393335], dtype=float32),
      'probabilities': array([0.46066645, 0.5393335 ], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([0.4118098], dtype=float32),
      'logistic': array([0.60152173], dtype=float32),
      'probabilities': array([0.3984782 , 0.60152173], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([1.0424118], dtype=float32),
      'logistic': array([0.7393151], dtype=float32),
      'probabilities': array([0.26068494, 0.73931515], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([1.5019126], dtype=float32),
      'logistic': array([0.81785953], dtype=float32),
      'probabilities': array([0.18214044, 0.81785953], dtype=float32),
      'class_ids': array([1], dtype=int64),
      'classes': array([b'1'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.18321651], dtype=float32),
      'logistic': array([0.4543236], dtype=float32),
      'probabilities': array([0.5456764 , 0.45432353], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.5613787], dtype=float32),
      'logistic': array([0.17344889], dtype=float32),
      'probabilities': array([0.82655114, 0.1734489 ], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-2.2088246], dtype=float32),
      'logistic': array([0.09896083], dtype=float32),
      'probabilities': array([0.9010392 , 0.09896083], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-1.9908977], dtype=float32),
      'logistic': array([0.12016193], dtype=float32),
      'probabilities': array([0.87983805, 0.12016193], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.9935088], dtype=float32),
      'logistic': array([0.2702196], dtype=float32),
      'probabilities': array([0.7297804 , 0.27021956], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.7111473], dtype=float32),
      'logistic': array([0.32934538], dtype=float32),
      'probabilities': array([0.6706546 , 0.32934538], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)},
     {'logits': array([-0.94634223], dtype=float32),
      'logistic': array([0.27962103], dtype=float32),
      'probabilities': array([0.720379  , 0.27962103], dtype=float32),
      'class_ids': array([0], dtype=int64),
      'classes': array([b'0'], dtype=object),
      'all_class_ids': array([0, 1]),
      'all_classes': array([b'0', b'1'], dtype=object)}]




```python
mypred[0:5][2]['class_ids']
```




    array([0], dtype=int64)




```python
X_test
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
      <th>Number_pregnant</th>
      <th>Glucose_concentration</th>
      <th>Blood_pressure</th>
      <th>Triceps</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>Pedigree</th>
      <th>Age</th>
      <th>Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>766</th>
      <td>0.058824</td>
      <td>0.633166</td>
      <td>0.491803</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.448584</td>
      <td>0.115713</td>
      <td>47</td>
      <td>C</td>
    </tr>
    <tr>
      <th>748</th>
      <td>0.176471</td>
      <td>0.939698</td>
      <td>0.573770</td>
      <td>0.222222</td>
      <td>0.236407</td>
      <td>0.542474</td>
      <td>0.140905</td>
      <td>36</td>
      <td>B</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.411765</td>
      <td>0.532663</td>
      <td>0.754098</td>
      <td>0.181818</td>
      <td>0.000000</td>
      <td>0.338301</td>
      <td>0.067037</td>
      <td>48</td>
      <td>A</td>
    </tr>
    <tr>
      <th>485</th>
      <td>0.000000</td>
      <td>0.678392</td>
      <td>0.557377</td>
      <td>0.424242</td>
      <td>0.295508</td>
      <td>0.630402</td>
      <td>0.122545</td>
      <td>24</td>
      <td>C</td>
    </tr>
    <tr>
      <th>543</th>
      <td>0.235294</td>
      <td>0.422111</td>
      <td>0.737705</td>
      <td>0.232323</td>
      <td>0.066194</td>
      <td>0.588674</td>
      <td>0.034586</td>
      <td>25</td>
      <td>C</td>
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
      <th>106</th>
      <td>0.058824</td>
      <td>0.482412</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.333830</td>
      <td>0.055081</td>
      <td>27</td>
      <td>A</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0.058824</td>
      <td>0.366834</td>
      <td>0.409836</td>
      <td>0.101010</td>
      <td>0.000000</td>
      <td>0.342772</td>
      <td>0.072588</td>
      <td>21</td>
      <td>A</td>
    </tr>
    <tr>
      <th>291</th>
      <td>0.000000</td>
      <td>0.537688</td>
      <td>0.508197</td>
      <td>0.303030</td>
      <td>0.087470</td>
      <td>0.545455</td>
      <td>0.289923</td>
      <td>25</td>
      <td>C</td>
    </tr>
    <tr>
      <th>212</th>
      <td>0.411765</td>
      <td>0.899497</td>
      <td>0.778689</td>
      <td>0.313131</td>
      <td>0.000000</td>
      <td>0.509687</td>
      <td>0.036721</td>
      <td>60</td>
      <td>B</td>
    </tr>
    <tr>
      <th>269</th>
      <td>0.117647</td>
      <td>0.733668</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.409836</td>
      <td>0.069172</td>
      <td>28</td>
      <td>D</td>
    </tr>
  </tbody>
</table>
<p>254 rows × 9 columns</p>
</div>




```python
y_test
```




    766    1
    748    1
    42     0
    485    1
    543    0
          ..
    106    0
    55     0
    291    1
    212    0
    269    1
    Name: Class, Length: 254, dtype: int64




```python

```

# Classification: DNN Classifier


```python
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],
                                       feature_columns=feat_cols,
                                       n_classes=2)
```

    INFO:tensorflow:Using default config.
    WARNING:tensorflow:Using temporary folder as model directory: C:\Users\abulu\AppData\Local\Temp\tmp_ujo6e0w
    INFO:tensorflow:Using config: {'_model_dir': 'C:\\Users\\abulu\\AppData\\Local\\Temp\\tmp_ujo6e0w', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x0000029916BAD348>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}



```python
dnn_model.train(input_fn=input_func,steps=1000)
```

    INFO:tensorflow:Calling model_fn.



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-580-f8c9922af341> in <module>
    ----> 1 dnn_model.train(input_fn=input_func,steps=1000)


    ~\anaconda3\lib\site-packages\tensorflow_estimator\python\estimator\estimator.py in train(self, input_fn, hooks, steps, max_steps, saving_listeners)
        368
        369       saving_listeners = _check_listeners_type(saving_listeners)
    --> 370       loss = self._train_model(input_fn, hooks, saving_listeners)
        371       logging.info('Loss for final step: %s.', loss)
        372       return self


    ~\anaconda3\lib\site-packages\tensorflow_estimator\python\estimator\estimator.py in _train_model(self, input_fn, hooks, saving_listeners)
       1159       return self._train_model_distributed(input_fn, hooks, saving_listeners)
       1160     else:
    -> 1161       return self._train_model_default(input_fn, hooks, saving_listeners)
       1162
       1163   def _train_model_default(self, input_fn, hooks, saving_listeners):


    ~\anaconda3\lib\site-packages\tensorflow_estimator\python\estimator\estimator.py in _train_model_default(self, input_fn, hooks, saving_listeners)
       1189       worker_hooks.extend(input_hooks)
       1190       estimator_spec = self._call_model_fn(
    -> 1191           features, labels, ModeKeys.TRAIN, self.config)
       1192       global_step_tensor = training_util.get_global_step(g)
       1193       return self._train_with_estimator_spec(estimator_spec, worker_hooks,


    ~\anaconda3\lib\site-packages\tensorflow_estimator\python\estimator\estimator.py in _call_model_fn(self, features, labels, mode, config)
       1147
       1148     logging.info('Calling model_fn.')
    -> 1149     model_fn_results = self._model_fn(features=features, **kwargs)
       1150     logging.info('Done calling model_fn.')
       1151


    ~\anaconda3\lib\site-packages\tensorflow_estimator\python\estimator\canned\dnn.py in _model_fn(features, labels, mode, config)
        809           input_layer_partitioner=input_layer_partitioner,
        810           config=config,
    --> 811           batch_norm=batch_norm)
        812
        813     super(DNNClassifier, self).__init__(


    ~\anaconda3\lib\site-packages\tensorflow_estimator\python\estimator\canned\dnn.py in _dnn_model_fn(features, labels, mode, head, hidden_units, feature_columns, optimizer, activation_fn, dropout, input_layer_partitioner, config, use_tpu, batch_norm)
        461         input_layer_partitioner=input_layer_partitioner,
        462         batch_norm=batch_norm)
    --> 463     logits = logit_fn(features=features, mode=mode)
        464
        465     return _get_dnn_estimator_spec(use_tpu, head, features, labels, mode,


    ~\anaconda3\lib\site-packages\tensorflow_estimator\python\estimator\canned\dnn.py in dnn_logit_fn(features, mode)
        106         input_layer_partitioner,
        107         batch_norm,
    --> 108         name='dnn')
        109     return dnn_model(features, mode)
        110


    ~\anaconda3\lib\site-packages\tensorflow_estimator\python\estimator\canned\dnn.py in __init__(self, units, hidden_units, feature_columns, activation_fn, dropout, input_layer_partitioner, batch_norm, name, **kwargs)
        184     if feature_column_lib.is_feature_column_v2(feature_columns):
        185       self._input_layer = dense_features.DenseFeatures(
    --> 186           feature_columns=feature_columns, name='input_layer')
        187     else:
        188       self._input_layer = feature_column.InputLayer(


    ~\anaconda3\lib\site-packages\tensorflow_core\python\feature_column\dense_features.py in __init__(self, feature_columns, trainable, name, **kwargs)
         84         name=name,
         85         expected_column_type=fc.DenseColumn,
    ---> 86         **kwargs)
         87
         88   @property


    ~\anaconda3\lib\site-packages\tensorflow_core\python\feature_column\feature_column_v2.py in __init__(self, feature_columns, expected_column_type, trainable, name, **kwargs)
        378             'You can wrap a categorical column with an '
        379             'embedding_column or indicator_column. Given: {}'.format(
    --> 380                 expected_column_type, column))
        381
        382   def build(self, _):


    ValueError: Items of feature_columns must be a <class 'tensorflow.python.feature_column.feature_column_v2.DenseColumn'>. You can wrap a categorical column with an embedding_column or indicator_column. Given: HashedCategoricalColumn(key='Group', hash_bucket_size=10, dtype=tf.string)


Need embedded columns


```python
embedded_group_column =tf.feature_column.embedding_column(categorical_column=assigned_group, dimension=4)
```


```python
feat_cols = [num_preg ,plasma_gluc,dias_press ,tricep ,insulin,bmi,diabetes_pedigree ,embedded_group_column, age_buckets]
```


```python
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,
                                                 y=y_train,
                                                 batch_size=10,
                                                 num_epochs=1000,
                                                 shuffle=True)
```


```python
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],
                                       feature_columns=feat_cols,
                                       n_classes=2)
```

    INFO:tensorflow:Using default config.
    WARNING:tensorflow:Using temporary folder as model directory: C:\Users\abulu\AppData\Local\Temp\tmpzsb8oio0
    INFO:tensorflow:Using config: {'_model_dir': 'C:\\Users\\abulu\\AppData\\Local\\Temp\\tmpzsb8oio0', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x0000029918834808>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}



```python
dnn_model.train(input_fn=input_func, steps=1000)
```

    INFO:tensorflow:Calling model_fn.
    WARNING:tensorflow:From C:\Users\abulu\anaconda3\lib\site-packages\tensorflow_core\python\feature_column\feature_column_v2.py:3079: HashedCategoricalColumn._num_buckets (from tensorflow.python.feature_column.feature_column_v2) is deprecated and will be removed in a future version.
    Instructions for updating:
    The old _FeatureColumn APIs are being deprecated. Please use the new FeatureColumn APIs instead.
    WARNING:tensorflow:From C:\Users\abulu\anaconda3\lib\site-packages\tensorflow_core\python\training\adagrad.py:76: calling Constant.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 0 into C:\Users\abulu\AppData\Local\Temp\tmpzsb8oio0\model.ckpt.
    INFO:tensorflow:loss = 6.824401, step = 1
    INFO:tensorflow:global_step/sec: 145.966
    INFO:tensorflow:loss = 8.172454, step = 101 (0.694 sec)
    INFO:tensorflow:global_step/sec: 223.282
    INFO:tensorflow:loss = 6.0655146, step = 201 (0.436 sec)
    INFO:tensorflow:global_step/sec: 240.363
    INFO:tensorflow:loss = 3.9746408, step = 301 (0.426 sec)
    INFO:tensorflow:global_step/sec: 228.016
    INFO:tensorflow:loss = 7.733548, step = 401 (0.432 sec)
    INFO:tensorflow:global_step/sec: 240.059
    INFO:tensorflow:loss = 6.992091, step = 501 (0.423 sec)
    INFO:tensorflow:global_step/sec: 208.749
    INFO:tensorflow:loss = 2.8215663, step = 601 (0.470 sec)
    INFO:tensorflow:global_step/sec: 205.095
    INFO:tensorflow:loss = 4.136054, step = 701 (0.488 sec)
    INFO:tensorflow:global_step/sec: 196.86
    INFO:tensorflow:loss = 1.5443542, step = 801 (0.509 sec)
    INFO:tensorflow:global_step/sec: 216.109
    INFO:tensorflow:loss = 2.54969, step = 901 (0.461 sec)
    INFO:tensorflow:Saving checkpoints for 1000 into C:\Users\abulu\AppData\Local\Temp\tmpzsb8oio0\model.ckpt.
    INFO:tensorflow:Loss for final step: 4.408986.





    <tensorflow_estimator.python.estimator.canned.dnn.DNNClassifier at 0x29919ab07c8>




```python
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                                      y=y_test,
                                                      num_epochs=1,
                                                      batch_size=10,
                                                      shuffle=False)
```


```python
dnn_model.evaluate(input_fn=eval_input_func)
```

    INFO:tensorflow:Calling model_fn.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Starting evaluation at 2020-07-17T08:54:22Z
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\abulu\AppData\Local\Temp\tmpzsb8oio0\model.ckpt-1000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Finished evaluation at 2020-07-17-08:54:23
    INFO:tensorflow:Saving dict for global step 1000: accuracy = 0.7519685, accuracy_baseline = 0.65748036, auc = 0.80401266, auc_precision_recall = 0.6598675, average_loss = 0.5088996, global_step = 1000, label/mean = 0.34251967, loss = 4.971558, precision = 0.6395349, prediction/mean = 0.35957855, recall = 0.6321839
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 1000: C:\Users\abulu\AppData\Local\Temp\tmpzsb8oio0\model.ckpt-1000





    {'accuracy': 0.7519685,
     'accuracy_baseline': 0.65748036,
     'auc': 0.80401266,
     'auc_precision_recall': 0.6598675,
     'average_loss': 0.5088996,
     'label/mean': 0.34251967,
     'loss': 4.971558,
     'precision': 0.6395349,
     'prediction/mean': 0.35957855,
     'recall': 0.6321839,
     'global_step': 1000}




```python

```
