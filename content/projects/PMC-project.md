+++
showonlyimage = false
draft = false
image = "projects/PMC-project_files/shots_area.jpg"
date = "2017-02-01"
title = "Predict when Stephen Curry shoots the ball"
weight = 2
type = "post"
tags = [
"NBA","Python",
"EDA"]
author = "Vincent Zhang"
+++
IIDATA 2017 Predictive Modeling Competition is hosted by UC Davis. This year the challenge is to predict when Stephen Curry shoots the ball, given the movement tracking data from sportVU.
<!--more-->

-   [IIDATA 2017 Predictive Modeling Competition](#iidata-2017-predictive-modeling-competition)
-   [Exploratory Data Analysis](#exploratory-data-analysis)
-   [Fisrt Shot](#first-shot)
-   [Can we get the labels?](#can-we-get-the-labels?)
-   [Can we use the labels?](#can-we-use-the-labels?)
-   [Identify the offense movement patterns](#identify-the-offense-movement-patterns)
-   [Change Point Detection](#change-point-detection)
-   [Afterthoughts](#afterthoughts)



<h2>IIDATA 2017 Predictive Modeling Competition</h2>
<p>
<strong>IIDATA</strong> is a one-day data science convention hosted by UC Davis undergraduate students. This year will be on Feb.4, 2017. There will be hands-on workshops, tech talks and modeling competitions in this event. I think it would be a great learning opportunity for students interesed in data science to attend. For more information, please visit the <a href="http://www.iidata.net/">website</a>. Lately, the organization team released their dataset for the Predictive Modeling Competition(PMC). This year the challenge is quite interesting. </p>

> "Given Steph Curry’s distance to ball, hoops and the opposing team’s defenders, can you determine when he is releasing the ball?" 

They also have a leaderboard with more detailed information about the competition <a href="https://pmc-leaderboard.com/">here</a>, especially the submission format.
</p>

<h2> Data </h2>

<p>This dataset contains temporal snapshots taken every 0.04 seconds during the November 12th, 2015 Golden State Warriors vs. Minnesota Timberwolves game (upon which Curry is on the court). It was told that Curry took **24** shots. For each shot, you are expected to find when he is releasing the ball from his hands with as much accuracy as possible. The data source is from [SportVU](https://www.stats.com/sportvu-basketball/). These are the variables:
<ul type="disc">
<li>“Time” - time left in the game (In fractional minutes)</li>
<li>“currylhoop” - distance from Curry to the left hoop</li>
<li>“curryrhoop” - distance from Curry to the right hoop</li>
<li>“balllhoop” - distance from the ball to the left hoop</li>
<li>“ballrhoop” - distance from the ball to the right hoop</li>
<li>“def1dist” - distance from the 1st defender to Curry</li>
<li>“def2dist” - distance from the 2nd defender to Curry</li>
<li>“def3dist” - distance from the 3rd defender to Curry</li>
<li>“def4dist” - distance from the 4th defender to Curry</li>
<li>“def5dist” - distance from the 5th defender to Curry</li>
</ul>
</p>

<p>It is a little bit surprise to me at first because it's not a typically supervised learning problem. There are some challenges or ambiguities involved: 
<ol type="a">
<li>They don't provide the actual times that Curry shot the balls. This makes it difficult to apply a supervised learning algorithm. Maybe it leaves for competitors to get the information so that it's possible to use in a supervised training and testing framework, since it allows competitors to ultilize other data/resources. On the other hand, it's possible to be treated as a unsupervised problem, for example, anomaly detection based on the activity patterns. So I see there are two direction here.</li>
<li>Note that they will evaluate the final score of our algorithm/model with another new dataset. So we need to be careful about generalization ability of our model.</li>
<li>Interestingly, it asks to predict the interval of times and it would evaluate an overall score based on the following formula. 
\begin{equation}
score = \frac{\sum_{i=1}^{|s|}I_{T_i}\frac{1}{1+s_{i2}-s_{i1}}}{|T|+(|S|-|T|)^2}
\end{equation}
where $S_i: \{(s_{i1}, s_{i2})\mid s_{i1} \leq s_{i2}\}$, $\| T\|=24$, $\|S\|\geq \|T\|$ and $I_{T_i}:\{1 \mid S_i \in T_i\}$.
Thus, there is room and trade-off for us to determine the interval of the times to get a higher overall prediction score to win the competition.
</li> 
</ol>
</p>


<h2> Load Libraries</h2>


```python
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
```


```python
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
```


```python
pd.set_option('display.max_colwidth', -1)
plt.rcdefaults()
sns.set(style="whitegrid")
```


```python
curry_df = pd.read_csv("curry3.csv")
```


```python
curry_df.head()
```




<div style="height:100%;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>curryball</th>
      <th>currylhoop</th>
      <th>curryrhoop</th>
      <th>balllhoop</th>
      <th>ballrhoop</th>
      <th>def1dist</th>
      <th>def2dist</th>
      <th>def3dist</th>
      <th>def4dist</th>
      <th>def5dist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>47.92533</td>
      <td>0.965270</td>
      <td>37.686696</td>
      <td>51.420125</td>
      <td>37.697605</td>
      <td>50.813065</td>
      <td>8.603274</td>
      <td>16.950232</td>
      <td>27.471575</td>
      <td>30.409634</td>
      <td>35.474795</td>
    </tr>
    <tr>
      <th>1</th>
      <td>47.92450</td>
      <td>0.797786</td>
      <td>37.290305</td>
      <td>51.855878</td>
      <td>37.555269</td>
      <td>51.177335</td>
      <td>8.298388</td>
      <td>16.730251</td>
      <td>27.240581</td>
      <td>30.249721</td>
      <td>35.246274</td>
    </tr>
    <tr>
      <th>2</th>
      <td>47.92383</td>
      <td>1.035342</td>
      <td>36.870045</td>
      <td>52.285237</td>
      <td>36.950707</td>
      <td>51.578687</td>
      <td>7.992415</td>
      <td>16.471735</td>
      <td>26.975070</td>
      <td>30.059684</td>
      <td>34.973485</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47.92317</td>
      <td>2.126335</td>
      <td>36.451216</td>
      <td>52.728202</td>
      <td>36.566778</td>
      <td>51.347007</td>
      <td>7.693763</td>
      <td>16.237772</td>
      <td>26.712746</td>
      <td>29.885916</td>
      <td>34.705748</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47.92250</td>
      <td>2.704138</td>
      <td>36.022605</td>
      <td>53.181823</td>
      <td>35.945881</td>
      <td>51.625402</td>
      <td>7.393487</td>
      <td>15.996931</td>
      <td>26.437723</td>
      <td>29.697227</td>
      <td>34.429396</td>
    </tr>
  </tbody>
</table>
</div>




```python
curry_df.shape
```




    (55615, 11)



It's tricky to note that the dataset include all the movement data of Curry on the field and he may play offense or defense.  And he played nearly equal time in the first half and second half. So we could expect the probability that Curry was in the side closer to either left or right hoop should be somewhat equal with each other. Anyways We can do some sanity checks first to verify that. It is what you expected.


```python
sum(map(lambda i,j: i<j, curry_df['currylhoop'],cur ry_df['curryrhoop']))
```




    27334




```python
27334/55615.0 
```




    0.4914861098624472




```python
sum(map(lambda i,j: i<j, curry_df['balllhoop'],curry_df['ballrhoop']))
```




    28468




```python
28468/55615.0 
```




    0.5118762923671671



## Exploratory Data Analysis

As we know, Stephen Curry is a special shooter in the NBA history, especially in 2015-2016 Season. Our goal is to infer his shot times based on his movement patterns. 

First, since Stephen Curry is known for his 3-point and long-distance shots, we would be interested in how the distance between Curry and the hoop varies around the 3-point line. Through some research about the specific game online, we know for the first half, the left hoop is the goal of Golden Warriors, and vice versa. However, as we can see, there are a lot of fluatuations because Curry were playing defense or offense. So we somehow need to come up with some clever ways to identify whether Curry is in attacking mode or not.


```python
curry_df['curryhoop_min'] = map(lambda i,j: min(i,j), curry_df['currylhoop'] ,curry_df['curryrhoop'] )
```


```python
plt.figure(figsize=(100,40))
plt.plot(curry_df.index, curry_df['curryhoop_min'], '-')
plt.title("Closest Distance between Hoop and Curry over time", fontsize = 50)
plt.xlabel('Index', fontsize=30)
plt.ylabel('Distance', fontsize=30)
```






<img src="../PMC-project_files/output_17_1.png" class="img-responsive" style="display: block; margin: auto;" />


We have to transform the **`time`** variable to make it more understandable and easy to inteprete.


```python
def getQuarter(x):
    if x >=36: 
        return 1
    elif 36 > x >= 24:
        return 2
    elif 24 >= x >= 12:
        return 3
    else:
        return 4
```


```python
curry_df['quarter'] = map(lambda x: getQuarter(x), curry_df['time'])
```


```python
def getMinSecInQuarter(time, quarter):
    mins = int(time - 12 * (4-quarter))
    secs = int(((time - 12 * (4-quarter))-mins)*60)
    millisecs = round((((time - 12 * (4-quarter))-mins)*60)- secs,2)
    return pd.Series({'minute': mins,'second':secs, 'milisecs': millisecs})
```


```python
curry_df = curry_df.merge(curry_df.apply(lambda row: getMinSecInQuarter(row['time'],row['quarter']), axis = 1), left_index=True, right_index=True)
```

<h2>First Shot</h2>
<p> Let's try to use our reasoning first. As we know when a ball is shot, the distance between the player and the ball will increase while the distance between the ball and the hoop will decrease. So we define _ballhoopmin_ as the actual distance between the ball and the hoop. Thus, we can infer that if the ratio between _curryball_ and _ballhoopmin_ is large, it's likely that the ball is in the fly.</p>


```python
curry_df['ballhoop_min'] = map(lambda i, j: min(i,j), curry_df['balllhoop'], curry_df['ballrhoop'])
```


```python
curry_df.head()
```


<div style="height:100%;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>time</th>
      <th>curryball</th>
      <th>currylhoop</th>
      <th>curryrhoop</th>
      <th>balllhoop</th>
      <th>ballrhoop</th>
      <th>def1dist</th>
      <th>def2dist</th>
      <th>def3dist</th>
      <th>def4dist</th>
      <th>def5dist</th>
      <th>curryhoop_min</th>
      <th>quarter</th>
      <th>milisecs</th>
      <th>minute</th>
      <th>second</th>
      <th>ballhoop_min</th>
      <th>dist_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>47.92533</td>
      <td>0.965270</td>
      <td>37.686696</td>
      <td>51.420125</td>
      <td>37.697605</td>
      <td>50.813065</td>
      <td>8.603274</td>
      <td>16.950232</td>
      <td>27.471575</td>
      <td>30.409634</td>
      <td>35.474795</td>
      <td>37.686696</td>
      <td>1</td>
      <td>0.52</td>
      <td>11.0</td>
      <td>55.0</td>
      <td>37.697605</td>
      <td>0.025606</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>47.92450</td>
      <td>0.797786</td>
      <td>37.290305</td>
      <td>51.855878</td>
      <td>37.555269</td>
      <td>51.177335</td>
      <td>8.298388</td>
      <td>16.730251</td>
      <td>27.240581</td>
      <td>30.249721</td>
      <td>35.246274</td>
      <td>37.290305</td>
      <td>1</td>
      <td>0.47</td>
      <td>11.0</td>
      <td>55.0</td>
      <td>37.555269</td>
      <td>0.021243</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>47.92383</td>
      <td>1.035342</td>
      <td>36.870045</td>
      <td>52.285237</td>
      <td>36.950707</td>
      <td>51.578687</td>
      <td>7.992415</td>
      <td>16.471735</td>
      <td>26.975070</td>
      <td>30.059684</td>
      <td>34.973485</td>
      <td>36.870045</td>
      <td>1</td>
      <td>0.43</td>
      <td>11.0</td>
      <td>55.0</td>
      <td>36.950707</td>
      <td>0.028020</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>47.92317</td>
      <td>2.126335</td>
      <td>36.451216</td>
      <td>52.728202</td>
      <td>36.566778</td>
      <td>51.347007</td>
      <td>7.693763</td>
      <td>16.237772</td>
      <td>26.712746</td>
      <td>29.885916</td>
      <td>34.705748</td>
      <td>36.451216</td>
      <td>1</td>
      <td>0.39</td>
      <td>11.0</td>
      <td>55.0</td>
      <td>36.566778</td>
      <td>0.058149</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>47.92250</td>
      <td>2.704138</td>
      <td>36.022605</td>
      <td>53.181823</td>
      <td>35.945881</td>
      <td>51.625402</td>
      <td>7.393487</td>
      <td>15.996931</td>
      <td>26.437723</td>
      <td>29.697227</td>
      <td>34.429396</td>
      <td>36.022605</td>
      <td>1</td>
      <td>0.35</td>
      <td>11.0</td>
      <td>55.0</td>
      <td>35.945881</td>
      <td>0.075228</td>
    </tr>
  </tbody>
</table>
</div>




```python
curry_df['dist_ratio'] = curry_df['curryball'] / curry_df['ballhoop_min']
```

Let's see whether we can observe any pattern of this distance ratio. We can see that there are some obvious spikes, which are probably what we are looking for. Next, we can extract those corresponding times.


```python
curry_df.reset_index(inplace=True)
```


```python
%matplotlib inline
plt.figure(figsize=(100,40))
plt.plot(curry_df['index'], curry_df['dist_ratio'], '-')
plt.title("Distance Ratio over time", fontsize = 50)
plt.xlabel('Index', fontsize=30)
plt.ylabel('Distance Ratio', fontsize=30)
plt.show()
```
<img src="../PMC-project_files/output_29_0.png" class="img-responsive" style="display: block; margin: auto;" />





```python
# Find the times with the top 24 highest spikes
curry_df.nlargest(24, 'dist_ratio')[["time", "dist_ratio", "curryhoop_min","quarter"]]
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>dist_ratio</th>
      <th>curryhoop_min</th>
      <th>quarter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7600</th>
      <td>42.86000</td>
      <td>941.658649</td>
      <td>26.536377</td>
      <td>1</td>
    </tr>
    <tr>
      <th>984</th>
      <td>47.26917</td>
      <td>673.596547</td>
      <td>24.232998</td>
      <td>1</td>
    </tr>
    <tr>
      <th>52328</th>
      <td>2.19100</td>
      <td>580.250936</td>
      <td>18.255410</td>
      <td>4</td>
    </tr>
    <tr>
      <th>43072</th>
      <td>14.02417</td>
      <td>531.475767</td>
      <td>28.710639</td>
      <td>3</td>
    </tr>
    <tr>
      <th>47706</th>
      <td>5.41300</td>
      <td>434.798704</td>
      <td>23.713664</td>
      <td>4</td>
    </tr>
    <tr>
      <th>41432</th>
      <td>15.11750</td>
      <td>433.106794</td>
      <td>30.614339</td>
      <td>3</td>
    </tr>
    <tr>
      <th>52329</th>
      <td>2.19033</td>
      <td>369.158118</td>
      <td>18.407326</td>
      <td>4</td>
    </tr>
    <tr>
      <th>19805</th>
      <td>29.68417</td>
      <td>281.887963</td>
      <td>22.637169</td>
      <td>2</td>
    </tr>
    <tr>
      <th>982</th>
      <td>47.27050</td>
      <td>274.460240</td>
      <td>24.283017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47707</th>
      <td>5.41233</td>
      <td>266.074215</td>
      <td>24.107225</td>
      <td>4</td>
    </tr>
    <tr>
      <th>981</th>
      <td>47.27117</td>
      <td>264.607287</td>
      <td>24.304749</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19276</th>
      <td>30.09017</td>
      <td>235.350832</td>
      <td>36.780734</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9096</th>
      <td>41.81467</td>
      <td>234.948026</td>
      <td>7.722907</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14598</th>
      <td>38.08533</td>
      <td>223.859015</td>
      <td>9.388604</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6861</th>
      <td>43.35267</td>
      <td>215.994452</td>
      <td>30.577963</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42100</th>
      <td>14.67217</td>
      <td>211.754106</td>
      <td>31.718987</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6860</th>
      <td>43.35333</td>
      <td>209.289207</td>
      <td>30.424288</td>
      <td>1</td>
    </tr>
    <tr>
      <th>43073</th>
      <td>14.02350</td>
      <td>207.605952</td>
      <td>28.960756</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16937</th>
      <td>36.52900</td>
      <td>204.341708</td>
      <td>33.667907</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18823</th>
      <td>30.39250</td>
      <td>201.161102</td>
      <td>25.614859</td>
      <td>2</td>
    </tr>
    <tr>
      <th>52327</th>
      <td>2.19167</td>
      <td>199.797752</td>
      <td>18.103289</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10283</th>
      <td>41.02433</td>
      <td>198.874455</td>
      <td>34.357564</td>
      <td>1</td>
    </tr>
    <tr>
      <th>52330</th>
      <td>2.18967</td>
      <td>190.416307</td>
      <td>18.566167</td>
      <td>4</td>
    </tr>
    <tr>
      <th>33665</th>
      <td>20.34900</td>
      <td>185.174150</td>
      <td>14.383419</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



It seems reasonable. We then prepare our submission to test our rough model. We simply contructed the intervals by adding 0.05 in minutes. It turned out we only got the prediction accuracy about 0.198. So we had to look for other methods.


```python
curry_df.columns
```




    Index([u'time', u'curryball', u'currylhoop', u'curryrhoop', u'balllhoop',
           u'ballrhoop', u'def1dist', u'def2dist', u'def3dist', u'def4dist',
           u'def5dist', u'curryhoop_min', u'quarter', u'milisecs', u'minute',
           u'second', u'ballhoop_min', u'dist_ratio', u'curryball_pctChg'],
          dtype='object')




```python
curry_times = curry_df.nlargest(24, 'dist_ratio').time.values
# prepare submission file
curry_submit = pd.DataFrame({'upper': curry_times + 0.05,'lower': curry_times}, )[['upper', 'lower']]
curry_submit.to_csv('curry_pred1.csv', index = False)
```


```python
# Another heauristic approach
curry_df['curryball_pctChg'] = curry_df['curryball'].pct_change(25*3) # 3 seconds
curry_df['ballhoop_min_pcgChg'] = curry_df['ballhoop_min'].pct_change(25*3) # 3 seconds
curry_df['dist_ratio_pctChg'] = curry_df['dist_ratio'].pct_change(25*3) # 3 seconds

curry_df.nlargest(24, ['dist_ratio_pctChg', 'ballhoop_min_pcgChg','curryball_pctChg'])[["time", "quarter", "curryball_pctChg", "ballhoop_min_pcgChg","dist_ratio_pctChg"]]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>quarter</th>
      <th>curryball_pctChg</th>
      <th>ballhoop_min_pcgChg</th>
      <th>dist_ratio_pctChg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7600</th>
      <td>42.86000</td>
      <td>1</td>
      <td>19.900084</td>
      <td>-0.998690</td>
      <td>15957.179434</td>
    </tr>
    <tr>
      <th>41432</th>
      <td>15.11750</td>
      <td>3</td>
      <td>23.893631</td>
      <td>-0.998094</td>
      <td>13060.238297</td>
    </tr>
    <tr>
      <th>43072</th>
      <td>14.02417</td>
      <td>3</td>
      <td>17.547261</td>
      <td>-0.998397</td>
      <td>11570.539712</td>
    </tr>
    <tr>
      <th>2062</th>
      <td>46.55050</td>
      <td>1</td>
      <td>26.098719</td>
      <td>-0.995940</td>
      <td>6672.902973</td>
    </tr>
    <tr>
      <th>43073</th>
      <td>14.02350</td>
      <td>3</td>
      <td>18.545367</td>
      <td>-0.995844</td>
      <td>4701.474897</td>
    </tr>
    <tr>
      <th>15516</th>
      <td>37.47500</td>
      <td>1</td>
      <td>18.436189</td>
      <td>-0.995833</td>
      <td>4662.975981</td>
    </tr>
    <tr>
      <th>48990</th>
      <td>4.55700</td>
      <td>4</td>
      <td>39.575986</td>
      <td>-0.990790</td>
      <td>4404.777254</td>
    </tr>
    <tr>
      <th>21572</th>
      <td>28.50617</td>
      <td>2</td>
      <td>54.559725</td>
      <td>-0.985603</td>
      <td>3858.244006</td>
    </tr>
    <tr>
      <th>21573</th>
      <td>28.50550</td>
      <td>2</td>
      <td>54.482756</td>
      <td>-0.985187</td>
      <td>3744.661281</td>
    </tr>
    <tr>
      <th>47706</th>
      <td>5.41300</td>
      <td>4</td>
      <td>9.309178</td>
      <td>-0.997226</td>
      <td>3715.907761</td>
    </tr>
    <tr>
      <th>10283</th>
      <td>41.02433</td>
      <td>1</td>
      <td>26.174887</td>
      <td>-0.992601</td>
      <td>3671.606449</td>
    </tr>
    <tr>
      <th>21574</th>
      <td>28.50483</td>
      <td>2</td>
      <td>60.650835</td>
      <td>-0.982348</td>
      <td>3491.645703</td>
    </tr>
    <tr>
      <th>41433</th>
      <td>15.11683</td>
      <td>3</td>
      <td>24.428133</td>
      <td>-0.992558</td>
      <td>3415.769885</td>
    </tr>
    <tr>
      <th>41431</th>
      <td>15.11833</td>
      <td>3</td>
      <td>27.986632</td>
      <td>-0.990870</td>
      <td>3174.013392</td>
    </tr>
    <tr>
      <th>48989</th>
      <td>4.55767</td>
      <td>4</td>
      <td>39.985800</td>
      <td>-0.986866</td>
      <td>3119.690776</td>
    </tr>
    <tr>
      <th>1510</th>
      <td>46.91850</td>
      <td>1</td>
      <td>16.986352</td>
      <td>-0.993484</td>
      <td>2759.132521</td>
    </tr>
    <tr>
      <th>27372</th>
      <td>24.56750</td>
      <td>2</td>
      <td>42.719127</td>
      <td>-0.984090</td>
      <td>2746.966359</td>
    </tr>
    <tr>
      <th>1513</th>
      <td>46.91650</td>
      <td>1</td>
      <td>19.801880</td>
      <td>-0.992428</td>
      <td>2746.179540</td>
    </tr>
    <tr>
      <th>27373</th>
      <td>24.56683</td>
      <td>2</td>
      <td>50.255761</td>
      <td>-0.981097</td>
      <td>2710.527264</td>
    </tr>
    <tr>
      <th>1512</th>
      <td>46.91717</td>
      <td>1</td>
      <td>20.080091</td>
      <td>-0.992072</td>
      <td>2657.842502</td>
    </tr>
    <tr>
      <th>27371</th>
      <td>24.56817</td>
      <td>2</td>
      <td>30.435812</td>
      <td>-0.987464</td>
      <td>2506.588780</td>
    </tr>
    <tr>
      <th>15517</th>
      <td>37.47433</td>
      <td>1</td>
      <td>19.043743</td>
      <td>-0.992002</td>
      <td>2505.107109</td>
    </tr>
    <tr>
      <th>47707</th>
      <td>5.41233</td>
      <td>4</td>
      <td>10.563076</td>
      <td>-0.995336</td>
      <td>2478.333671</td>
    </tr>
    <tr>
      <th>48991</th>
      <td>4.55633</td>
      <td>4</td>
      <td>43.048869</td>
      <td>-0.981782</td>
      <td>2416.834482</td>
    </tr>
  </tbody>
</table>
</div>




```python
curry_times = curry_df.nlargest(24, ['dist_ratio_pctChg', 'ballhoop_min_pcgChg','curryball_pctChg']).time.values
# prepare submission file
curry_submit = pd.DataFrame({'upper': curry_times + 0.06,'lower': curry_times+ 0.01}, )[['upper', 'lower']]
curry_submit.to_csv('curry_pred1.csv', index = False)
```

<h2>Can we get the "labels"?</h2>

It's obvious that the raw data doesn't directly tell us when Curry made the shots. However, given the background information about the game, it's possible to find out those actual shooting times, for example, by watching this specific past [game](https://www.youtube.com/watch?v=5PozYV_qHD8). Although we are not supposed to use it as the submission, ideally, we can use the information to denote the corresponding Curry's shooting times as a way to tune our pattern regconition algorithm.

Acutally, there is a more clever way to get the labels. We found we can scrape the actual shot times by Curry on the official NBA statistics website [here](http://stats.nba.com/game/#!/0021500125/playbyplay/). It's a little tricky that the website is dynamically generated. To do the web scraping, first we can try to find the API access point by locating Develop Tools -> Network -> XHR in Chrome. Luckily, we found it exists. Then we can go ahead to scrape the information we want.


```python
import requests
#may need to open on the browser first
shots_url = 'http://stats.nba.com/stats/playbyplayv2?EndPeriod=10&EndRange=55800&GameID=0021500125&RangeType=2&Season=2015-16&SeasonType=Regular+Season&StartPeriod=1&StartRange=0'

# request the URL and parse the JSON
response = requests.get(shots_url)
#response.raise_for_status() # raise exception if invalid response
shot_logs = response.json()['resultSets'][0]['rowSet']
headers = response.json()['resultSets'][0]['headers']
```


```python
shots_df = pd.DataFrame(shot_logs, columns = headers)
shots_df.head()
```

<div style="height:100%;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GAME_ID</th>
      <th>EVENTNUM</th>
      <th>EVENTMSGTYPE</th>
      <th>EVENTMSGACTIONTYPE</th>
      <th>PERIOD</th>
      <th>WCTIMESTRING</th>
      <th>PCTIMESTRING</th>
      <th>HOMEDESCRIPTION</th>
      <th>NEUTRALDESCRIPTION</th>
      <th>VISITORDESCRIPTION</th>
      <th>...</th>
      <th>PLAYER2_TEAM_CITY</th>
      <th>PLAYER2_TEAM_NICKNAME</th>
      <th>PLAYER2_TEAM_ABBREVIATION</th>
      <th>PERSON3TYPE</th>
      <th>PLAYER3_ID</th>
      <th>PLAYER3_NAME</th>
      <th>PLAYER3_TEAM_ID</th>
      <th>PLAYER3_TEAM_CITY</th>
      <th>PLAYER3_TEAM_NICKNAME</th>
      <th>PLAYER3_TEAM_ABBREVIATION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0021500125</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>8:16 PM</td>
      <td>12:00</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0021500125</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>8:16 PM</td>
      <td>12:00</td>
      <td>Jump Ball Towns vs. Ezeli: Tip to Green</td>
      <td>None</td>
      <td>None</td>
      <td>...</td>
      <td>Golden State</td>
      <td>Warriors</td>
      <td>GSW</td>
      <td>5</td>
      <td>203110</td>
      <td>Draymond Green</td>
      <td>1.610613e+09</td>
      <td>Golden State</td>
      <td>Warriors</td>
      <td>GSW</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0021500125</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>8:16 PM</td>
      <td>11:46</td>
      <td>None</td>
      <td>None</td>
      <td>Green 15' Jump Shot (2 PTS) (Curry 1 AST)</td>
      <td>...</td>
      <td>Golden State</td>
      <td>Warriors</td>
      <td>GSW</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0021500125</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>8:17 PM</td>
      <td>11:31</td>
      <td>MISS Wiggins 20' Jump Shot</td>
      <td>None</td>
      <td>None</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0021500125</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>8:17 PM</td>
      <td>11:30</td>
      <td>None</td>
      <td>None</td>
      <td>Curry REBOUND (Off:0 Def:1)</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>



Thus, we can use regular expression to identify the times of Curry's shots based on the descriptions in the column **`VISITORDESCRIPTION`**.


```python
pattern = r'Curry.*(Layup|Shot)'
shots_df[shots_df['VISITORDESCRIPTION'].str.contains(pattern, na=False)][['PERIOD', 'PCTIMESTRING','VISITORDESCRIPTION']]
```

    C:\Users\jpzha\Anaconda2\lib\site-packages\ipykernel\__main__.py:2: UserWarning: This pattern has match groups. To actually get the groups, use str.extract.
      from ipykernel import kernelapp as app
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PERIOD</th>
      <th>PCTIMESTRING</th>
      <th>VISITORDESCRIPTION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>10:54</td>
      <td>Curry 28' 3PT Jump Shot (3 PTS)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>9:48</td>
      <td>Curry 25' 3PT Jump Shot (9 PTS) (Barnes 1 AST)</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>8:07</td>
      <td>MISS Curry 3' Layup</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1</td>
      <td>6:51</td>
      <td>Curry 24' 3PT Jump Shot (12 PTS) (Ezeli 2 AST)</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1</td>
      <td>3:38</td>
      <td>Curry 18' Pullup Jump Shot (16 PTS)</td>
    </tr>
    <tr>
      <th>86</th>
      <td>1</td>
      <td>2:13</td>
      <td>MISS Curry  3PT Jump Shot</td>
    </tr>
    <tr>
      <th>91</th>
      <td>1</td>
      <td>1:50</td>
      <td>Curry  Layup (18 PTS) (Bogut 1 AST)</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1</td>
      <td>0:53</td>
      <td>MISS Curry 7' Floating Jump Shot</td>
    </tr>
    <tr>
      <th>101</th>
      <td>1</td>
      <td>0:12</td>
      <td>Curry 24' 3PT Jump Shot (21 PTS) (Iguodala 1 AST)</td>
    </tr>
    <tr>
      <th>185</th>
      <td>2</td>
      <td>4:13</td>
      <td>MISS Curry 27' 3PT Jump Shot</td>
    </tr>
    <tr>
      <th>191</th>
      <td>2</td>
      <td>3:25</td>
      <td>Curry 11' Floating Jump Shot (23 PTS) (Green 6 AST)</td>
    </tr>
    <tr>
      <th>220</th>
      <td>2</td>
      <td>0:00</td>
      <td>Curry 2' Driving Layup (25 PTS)</td>
    </tr>
    <tr>
      <th>254</th>
      <td>3</td>
      <td>8:33</td>
      <td>MISS Curry 21' Jump Shot</td>
    </tr>
    <tr>
      <th>275</th>
      <td>3</td>
      <td>6:00</td>
      <td>MISS Curry 1' Layup</td>
    </tr>
    <tr>
      <th>284</th>
      <td>3</td>
      <td>5:17</td>
      <td>Curry 16' Jump Shot (27 PTS)</td>
    </tr>
    <tr>
      <th>311</th>
      <td>3</td>
      <td>3:06</td>
      <td>Curry 26' 3PT Jump Shot (32 PTS)</td>
    </tr>
    <tr>
      <th>313</th>
      <td>3</td>
      <td>2:39</td>
      <td>Curry 25' 3PT Jump Shot (35 PTS) (Green 9 AST)</td>
    </tr>
    <tr>
      <th>315</th>
      <td>3</td>
      <td>2:01</td>
      <td>MISS Curry 25' 3PT Jump Shot</td>
    </tr>
    <tr>
      <th>329</th>
      <td>3</td>
      <td>0:25</td>
      <td>MISS Curry 10' Floating Jump Shot</td>
    </tr>
    <tr>
      <th>391</th>
      <td>4</td>
      <td>5:51</td>
      <td>Curry 26' 3PT Jump Shot (38 PTS)</td>
    </tr>
    <tr>
      <th>393</th>
      <td>4</td>
      <td>5:23</td>
      <td>Curry 17' Jump Shot (40 PTS)</td>
    </tr>
    <tr>
      <th>395</th>
      <td>4</td>
      <td>4:58</td>
      <td>MISS Curry 26' 3PT Jump Shot</td>
    </tr>
    <tr>
      <th>419</th>
      <td>4</td>
      <td>2:54</td>
      <td>MISS Curry 28' 3PT Jump Shot</td>
    </tr>
    <tr>
      <th>432</th>
      <td>4</td>
      <td>1:55</td>
      <td>Curry 19' Jump Bank Shot (42 PTS) (Iguodala 5 AST)</td>
    </tr>
    <tr>
      <th>452</th>
      <td>4</td>
      <td>0:29</td>
      <td>Curry 26' 3PT Jump Shot (46 PTS) (Green 12 AST)</td>
    </tr>
  </tbody>
</table>
</div>



### Can we use the "labels"?


```python
tmp_time_df = shots_df[shots_df['VISITORDESCRIPTION'].str.contains(pattern, na=False)][['PERIOD', 'PCTIMESTRING']]
```

    C:\Users\jpzha\Anaconda2\lib\site-packages\ipykernel\__main__.py:1: UserWarning: This pattern has match groups. To actually get the groups, use str.extract.
      if __name__ == '__main__':
    


```python
def calGameTime(period, pctimestring):
    mins = int(pctimestring.split(':')[0])
    secs = float(pctimestring.split(':')[1])
    return (4-period)*12 + mins + secs/60.0
```


```python
tmp_time_df['shot_time'] = map(lambda x,y: calGameTime(x,y), tmp_time_df.PERIOD, tmp_time_df.PCTIMESTRING)
```


```python
tmp_time_df['upper'] = tmp_time_df['shot_time'] + 0.08
tmp_time_df['lower'] = tmp_time_df['shot_time'] + 0.02 
```


```python
tmp_time_df
```




<div style="height:100%;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PERIOD</th>
      <th>PCTIMESTRING</th>
      <th>shot_time</th>
      <th>upper</th>
      <th>lower</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>10:54</td>
      <td>46.900000</td>
      <td>46.980000</td>
      <td>46.920000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>9:48</td>
      <td>45.800000</td>
      <td>45.880000</td>
      <td>45.820000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>8:07</td>
      <td>44.116667</td>
      <td>44.196667</td>
      <td>44.136667</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1</td>
      <td>6:51</td>
      <td>42.850000</td>
      <td>42.930000</td>
      <td>42.870000</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1</td>
      <td>3:38</td>
      <td>39.633333</td>
      <td>39.713333</td>
      <td>39.653333</td>
    </tr>
    <tr>
      <th>86</th>
      <td>1</td>
      <td>2:13</td>
      <td>38.216667</td>
      <td>38.296667</td>
      <td>38.236667</td>
    </tr>
    <tr>
      <th>91</th>
      <td>1</td>
      <td>1:50</td>
      <td>37.833333</td>
      <td>37.913333</td>
      <td>37.853333</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1</td>
      <td>0:53</td>
      <td>36.883333</td>
      <td>36.963333</td>
      <td>36.903333</td>
    </tr>
    <tr>
      <th>101</th>
      <td>1</td>
      <td>0:12</td>
      <td>36.200000</td>
      <td>36.280000</td>
      <td>36.220000</td>
    </tr>
    <tr>
      <th>185</th>
      <td>2</td>
      <td>4:13</td>
      <td>28.216667</td>
      <td>28.296667</td>
      <td>28.236667</td>
    </tr>
    <tr>
      <th>191</th>
      <td>2</td>
      <td>3:25</td>
      <td>27.416667</td>
      <td>27.496667</td>
      <td>27.436667</td>
    </tr>
    <tr>
      <th>220</th>
      <td>2</td>
      <td>0:00</td>
      <td>24.000000</td>
      <td>24.080000</td>
      <td>24.020000</td>
    </tr>
    <tr>
      <th>254</th>
      <td>3</td>
      <td>8:33</td>
      <td>20.550000</td>
      <td>20.630000</td>
      <td>20.570000</td>
    </tr>
    <tr>
      <th>275</th>
      <td>3</td>
      <td>6:00</td>
      <td>18.000000</td>
      <td>18.080000</td>
      <td>18.020000</td>
    </tr>
    <tr>
      <th>284</th>
      <td>3</td>
      <td>5:17</td>
      <td>17.283333</td>
      <td>17.363333</td>
      <td>17.303333</td>
    </tr>
    <tr>
      <th>311</th>
      <td>3</td>
      <td>3:06</td>
      <td>15.100000</td>
      <td>15.180000</td>
      <td>15.120000</td>
    </tr>
    <tr>
      <th>313</th>
      <td>3</td>
      <td>2:39</td>
      <td>14.650000</td>
      <td>14.730000</td>
      <td>14.670000</td>
    </tr>
    <tr>
      <th>315</th>
      <td>3</td>
      <td>2:01</td>
      <td>14.016667</td>
      <td>14.096667</td>
      <td>14.036667</td>
    </tr>
    <tr>
      <th>329</th>
      <td>3</td>
      <td>0:25</td>
      <td>12.416667</td>
      <td>12.496667</td>
      <td>12.436667</td>
    </tr>
    <tr>
      <th>391</th>
      <td>4</td>
      <td>5:51</td>
      <td>5.850000</td>
      <td>5.930000</td>
      <td>5.870000</td>
    </tr>
    <tr>
      <th>393</th>
      <td>4</td>
      <td>5:23</td>
      <td>5.383333</td>
      <td>5.463333</td>
      <td>5.403333</td>
    </tr>
    <tr>
      <th>395</th>
      <td>4</td>
      <td>4:58</td>
      <td>4.966667</td>
      <td>5.046667</td>
      <td>4.986667</td>
    </tr>
    <tr>
      <th>419</th>
      <td>4</td>
      <td>2:54</td>
      <td>2.900000</td>
      <td>2.980000</td>
      <td>2.920000</td>
    </tr>
    <tr>
      <th>432</th>
      <td>4</td>
      <td>1:55</td>
      <td>1.916667</td>
      <td>1.996667</td>
      <td>1.936667</td>
    </tr>
    <tr>
      <th>452</th>
      <td>4</td>
      <td>0:29</td>
      <td>0.483333</td>
      <td>0.563333</td>
      <td>0.503333</td>
    </tr>
  </tbody>
</table>
</div>



Let's validate our labels by submitting this "cheated" scraping results. It gives us an accuracy rate about 87% on the leader board(top 1) after some twisting on the bounds. Although it's not perfect, we still prefer to ultilize this information since it's at least "NBA official" labels.:) Notice that we will focus on modeling after this step.


```python
tmp_time_df[['upper', 'lower']].to_csv('curry_pred2.csv', index=False)
```

### More EDA


```python
%matplotlib inline
#plt.figure(figsize=(100,40))
plt.plot(curry_df[(curry_df.time>46.8) & (curry_df.time<47)]['time'], curry_df[(curry_df.time>46.8) & (curry_df.time<47)]['curryball_pctChg'], '-')
plt.legend(['curryball_pctChg'], bbox_to_anchor=(1.3, 0.5))
plt.title("Distances over time")
plt.xlabel('Time')
plt.ylabel('Distance')
plt.show()
```

<img src="../PMC-project_files/output_51_0.png" class="img-responsive" style="display: block; margin: auto;" />


For example, at 10:54 in the first quarter of the time, he made a 3 PT shot. We plot the measurement trends in a certain time range.


```python
%matplotlib inline
#plt.figure(figsize=(100,40))
plt.plot(curry_df[(curry_df.time>46.8) & (curry_df.time<47)]['time'], curry_df[(curry_df.time>46.8) & (curry_df.time<47)]['curryball'], '-')
plt.plot(curry_df[(curry_df.time>46.8) & (curry_df.time<47)]['time'], curry_df[(curry_df.time>46.8) & (curry_df.time<47)]['currylhoop'], '-')
plt.plot(curry_df[(curry_df.time>46.8) & (curry_df.time<47)]['time'], curry_df[(curry_df.time>46.8) & (curry_df.time<47)]['balllhoop'], '-')
plt.plot(curry_df[(curry_df.time>46.8) & (curry_df.time<47)]['time'], curry_df[(curry_df.time>46.8) & (curry_df.time<47)]['def1dist'], '-')
plt.legend(['curryball', 'currylhoop', 'balllhoop','def1dist'], bbox_to_anchor=(1.3, 0.5))
plt.title("Distances over time")
plt.xlabel('Time')
plt.ylabel('Distance')
plt.show()
```

<img src="../PMC-project_files/output_53_0.png" class="img-responsive" style="display: block; margin: auto;" />


Note that we have to observe the trend by scanning from the right to the left because the time stamp here is to indicate the remaining time. We can see that Curry was holding the ball initially since currylhoop and balllhoop are close. Then a ball was released around 46.94 as the balllhoop distance decreases(towards to 0) and the curryball distance increases significantly. This is helpful for us to understand the pattern characteristics.

For example, around 6:00 in the third quarter of the time, he intended to layup. We plot the measurement trends in a certain time range.


```python
%matplotlib inline
#plt.figure(figsize=(100,40))
plt.plot(curry_df[(curry_df.time>17.9) & (curry_df.time<18.1)]['time'], curry_df[(curry_df.time>17.9) & (curry_df.time<18.1)]['curryball'], '-')
plt.plot(curry_df[(curry_df.time>17.9) & (curry_df.time<18.1)]['time'], curry_df[(curry_df.time>17.9) & (curry_df.time<18.1)]['curryrhoop'], '-')
plt.plot(curry_df[(curry_df.time>17.9) & (curry_df.time<18.1)]['time'], curry_df[(curry_df.time>17.9) & (curry_df.time<18.1)]['ballrhoop'], '-')
plt.plot(curry_df[(curry_df.time>17.9) & (curry_df.time<18.1)]['time'], curry_df[(curry_df.time>17.9) & (curry_df.time<18.1)]['def1dist'], '-')
plt.legend(['curryball', 'curryrhoop', 'ballrhoop', 'def1dist'], bbox_to_anchor=(1.3, 0.5))
plt.title("Distances over time")
plt.xlabel('Time')
plt.ylabel('Distance')
plt.show()
```

<img src="../PMC-project_files/output_56_0.png" class="img-responsive" style="display: block; margin: auto;" />


Note that in the second half, the right-hand side is of our interest when curry was scoring. In this case, we can see that curryball, curryrhoop and ballrhoop are close to each other towards to 0. The take-aways from these visualizations are that we can understand the movement patterns better and we could focus on the short time frames that invovked significant fluntuations.

## Modeling Framework

It's known that the shooting time points should be relatively close to the time points when the distance between the ball and hoop is very small. Due to the noise and granuity of this kind of sensor measurement data, it is necessary to perform some data cleaning to extract the relevant movement sequences of our interest. We designed our algorithm as a general modeling framework for this problem:

__Steps__:

1. Identify the round turning points by "min(balllhoop, ballrhoop) ~= 0" to extract the relevant movement sequences.
   
2. Build a supervised learning algorithm to identify those sequences for curry playing offense so that we can apply the model to predict on a new dataset
   
3. With our model trained to predict an "offense" sequence, we can further investigate the exact shooting points from those "offense" sequence.

### Step 1 Data Preparation
Actually, we had to choose the ball-hoop distance threshold smaller than a reasonable value to advoid over-specification errors. Then, we filter out reduncdant sequences if turning points are too close and select the correponding sequences with 6 seconds ahead. 


```python
from itertools import groupby
from operator import itemgetter
```


```python
def getRoundTurningPoints(curry_df, threshold):
    tmp = curry_df[curry_df.ballhoop_min<threshold][['index', 'time', 'ballhoop_min']]
    tmp_ls = tmp['index'].values
    consecutive_ls = [map(itemgetter(1), g) for k, g in groupby(enumerate(tmp_ls), lambda (i, x): i-x)]
    stop_points = [curry_df[curry_df.index.isin(index_ls)]['ballhoop_min'].argmin() for index_ls in consecutive_ls]
    return stop_points
```

A remark here is that we tuned the threshold here so that we can ensure all the 'offense' sequences are included to build supervised learning model for our next steps.


```python
turniningpoint_ls = getRoundTurningPoints(curry_df, 5)
```


```python
def getSequenceData(curry_df, turningpoint, seconds_ahead):
    record_ahead = int(seconds_ahead / 0.04)
    start_index = turningpoint - record_ahead
    if start_index<0:
        start_index = 0
    return curry_df[curry_df['index'].between(start_index, turningpoint, inclusive=True)]
```


```python
getSequenceData(curry_df, 1510, 6).head()
```




<div style="height:100%;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>time</th>
      <th>curryball</th>
      <th>currylhoop</th>
      <th>curryrhoop</th>
      <th>balllhoop</th>
      <th>ballrhoop</th>
      <th>def1dist</th>
      <th>def2dist</th>
      <th>def3dist</th>
      <th>def4dist</th>
      <th>def5dist</th>
      <th>ballhoop_min</th>
      <th>dist_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1360</th>
      <td>1360</td>
      <td>47.01850</td>
      <td>1.126781</td>
      <td>81.490019</td>
      <td>21.637208</td>
      <td>80.638449</td>
      <td>20.907522</td>
      <td>8.515027</td>
      <td>15.470442</td>
      <td>33.152840</td>
      <td>38.125602</td>
      <td>45.525469</td>
      <td>20.907522</td>
      <td>0.053894</td>
    </tr>
    <tr>
      <th>1361</th>
      <td>1361</td>
      <td>47.01783</td>
      <td>1.189048</td>
      <td>81.166932</td>
      <td>21.570949</td>
      <td>80.457890</td>
      <td>20.633572</td>
      <td>8.522492</td>
      <td>15.740184</td>
      <td>33.025416</td>
      <td>38.164352</td>
      <td>45.470323</td>
      <td>20.633572</td>
      <td>0.057627</td>
    </tr>
    <tr>
      <th>1362</th>
      <td>1362</td>
      <td>47.01717</td>
      <td>1.302032</td>
      <td>80.806313</td>
      <td>21.527865</td>
      <td>80.238336</td>
      <td>20.379774</td>
      <td>8.510399</td>
      <td>15.951757</td>
      <td>32.897002</td>
      <td>38.179300</td>
      <td>45.415983</td>
      <td>20.379774</td>
      <td>0.063888</td>
    </tr>
    <tr>
      <th>1363</th>
      <td>1363</td>
      <td>47.01650</td>
      <td>1.284768</td>
      <td>80.425426</td>
      <td>21.498966</td>
      <td>80.140926</td>
      <td>20.262702</td>
      <td>8.500776</td>
      <td>16.129816</td>
      <td>32.782485</td>
      <td>38.185713</td>
      <td>45.343810</td>
      <td>20.262702</td>
      <td>0.063406</td>
    </tr>
    <tr>
      <th>1364</th>
      <td>1364</td>
      <td>47.01583</td>
      <td>1.410186</td>
      <td>80.030255</td>
      <td>21.500508</td>
      <td>79.830341</td>
      <td>20.122170</td>
      <td>8.492476</td>
      <td>16.317941</td>
      <td>32.686315</td>
      <td>38.186847</td>
      <td>45.273431</td>
      <td>20.122170</td>
      <td>0.070081</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(turniningpoint_ls) # our candidate sequence set
```




    232




```python
count = 0 
point_ls = []
for point in turniningpoint_ls:
    tmp = getSequenceData(curry_df, point, 4)
    test_ls = map(lambda x:  tmp['time'].iloc[0]> x >tmp['time'].iloc[-1], list(tmp_time_df.lower.values))
    if sum(test_ls) >= 1:
        count+=1
        point_ls.append(list(tmp_time_df.lower.values)[test_ls.index(1)])
print count 
print point_ls
print len(set(point_ls))
```

    23
    [46.920000000000002, 44.13666666666667, 42.870000000000005, 39.653333333333336, 38.236666666666672, 37.853333333333339, 36.903333333333336, 36.220000000000006, 28.236666666666665, 27.436666666666667, 24.02, 24.02, 20.57, 15.119999999999999, 14.036666666666667, 12.436666666666666, 12.436666666666666, 5.8699999999999992, 5.8699999999999992, 5.4033333333333333, 4.9866666666666664, 2.9199999999999999, 0.5033333333333333]
    20
    

It captures 20 unique acutal shot times in the first step of filtering sequences. 

### Step 2  Classify offense plays of Curry
The sequences extracted from last step serve as our candicate set to further identify the sequence that Curry are in attack mode. Based on the time frame of corresponding sequences , we will extract the features to construct our training and test sets.


```python
curry_df['defdist_mean'] = map(lambda a,b,c,d,e: np.nanmean([a,b,c,d,e]), curry_df['def1dist'],curry_df['def2dist'],curry_df['def3dist'],curry_df['def4dist'],curry_df['def5dist']) 
curry_df['defdist_var'] = map(lambda a,b,c,d,e: np.nanvar([a,b,c,d,e]), curry_df['def1dist'],curry_df['def2dist'],curry_df['def3dist'],curry_df['def4dist'],curry_df['def5dist']) 
```


```python
def extractFeatures(df, shot_time_points_ls = None):
    curryball_max = np.nanmax(df['curryball'].values)
    curryball_min = np.nanmin(df['curryball'].values)
    curryball_mean = np.nanmean(df['curryball'].values)
    curryball_var = np.nanvar(df['curryball'].values)
    
    curryhoop_max = np.nanmax(df['curryhoop_min'].values)
    curryhoop_min = np.nanmin(df['curryhoop_min'].values)
    curryhoop_mean = np.nanmean(df['curryhoop_min'].values)
    curryhoop_var = np.nanvar(df['curryhoop_min'].values)
    
    ballhoop_max = np.nanmax(df['ballhoop_min'].values)
    ballhoop_min =  np.nanmin(df['ballhoop_min'].values)
    ballhoop_mean = np.nanmean(df['ballhoop_min'].values)
    ballhoop_var = np.nanvar(df['ballhoop_min'].values)
    
    dist_ratio_mean = np.nanmean(df['dist_ratio'].values)
    dist_ratio_var = np.nanvar(df['dist_ratio'].values)
    
    def1dist_mean = np.nanmean(df['def1dist'].values)
    def1dist_var = np.nanvar(df['def1dist'].values)
    def2dist_mean = np.nanmean(df['def2dist'].values)
    def2dist_var =np.nanvar(df['def2dist'].values)
    def3dist_mean =np.nanmean(df['def3dist'].values)
    def3dist_var = np.nanvar(df['def3dist'].values)
    def4dist_mean = np.nanmean(df['def4dist'].values)
    def4dist_var = np.nanvar(df['def4dist'].values)
    def5dist_mean = np.nanmean(df['def5dist'].values)
    def5dist_var = np.nanvar(df['def5dist'].values)
    
    defdist_mean_m = np.nanmean(df['defdist_mean'].values)
    defdist_mean_v = np.nanvar(df['defdist_mean'].values)
    defdist_var_m = np.nanmean(df['defdist_var'].values)
    defdist_var_v = np.nanvar(df['defdist_var'].values)
    
    if type(shot_time_points_ls) is list:
        is_shot_ls = filter(lambda x:  df['time'].iloc[0]> x >df['time'].iloc[-1], shot_time_points_ls)
        if len(is_shot_ls) >= 1:
            offense = 1
            shot_time = is_shot_ls
        else:
            offense = 0
            shot_time = np.nan
    
        return pd.DataFrame({'curryball_max':[curryball_max], 'curryball_min':[curryball_min], 'curryball_mean':[curryball_mean],
                        'curryball_var':[curryball_var], 'curryhoop_max':[curryhoop_max], 'curryhoop_min':[curryhoop_min],
                        'curryhoop_mean':[curryhoop_mean], 'curryhoop_var':[curryhoop_var], 'ballhoop_max':[ballhoop_max],
                        'ballhoop_min':[ballhoop_min], 'ballhoop_mean':[ballhoop_mean], 'ballhoop_var':[ballhoop_var],
                        'dist_ratio_mean':[dist_ratio_mean], 'dist_ratio_var':[dist_ratio_var], 'def1dist_mean':[def1dist_mean],
                        'def1dist_var':[def1dist_var], 'def2dist_mean':[def2dist_mean], 'def2dist_var':[def2dist_var],
                        'def3dist_mean':[def3dist_mean],'def3dist_var':[def3dist_var], 'def4dist_mean':[def4dist_mean], 
                         'def4dist_var':[def4dist_mean],'def5dist_mean':[def5dist_mean], 'def5dist_var':[def5dist_var],
                        'defdist_mean_m':[defdist_mean_m], 'defdist_mean_v':[defdist_mean_v], 'defdist_var_m':[defdist_var_m],
                        'defdist_var_v':[defdist_var_v], 'offense': [offense], 'trace_start':[df['time'].iloc[0]],
                        'trace_end': [df['time'].iloc[-1]], 'shot_time':shot_time })
    else:
        return pd.DataFrame({'curryball_max':[curryball_max], 'curryball_min':[curryball_min], 'curryball_mean':[curryball_mean],
                        'curryball_var':[curryball_var], 'curryhoop_max':[curryhoop_max], 'curryhoop_min':[curryhoop_min],
                        'curryhoop_mean':[curryhoop_mean], 'curryhoop_var':[curryhoop_var], 'ballhoop_max':[ballhoop_max],
                        'ballhoop_min':[ballhoop_min], 'ballhoop_mean':[ballhoop_mean], 'ballhoop_var':[ballhoop_var],
                        'dist_ratio_mean':[dist_ratio_mean], 'dist_ratio_var':[dist_ratio_var], 'def1dist_mean':[def1dist_mean],
                        'def1dist_var':[def1dist_var], 'def2dist_mean':[def2dist_mean], 'def2dist_var':[def2dist_var],
                        'def3dist_mean':[def3dist_mean],'def3dist_var':[def3dist_var], 'def4dist_mean':[def4dist_mean], 
                         'def4dist_var':[def4dist_mean],'def5dist_mean':[def5dist_mean], 'def5dist_var':[def5dist_var],
                        'defdist_mean_m':[defdist_mean_m], 'defdist_mean_v':[defdist_mean_v], 'defdist_var_m':[defdist_var_m],
                        'defdist_var_v':[defdist_var_v], 'trace_start':[df['time'].iloc[0]],
                        'trace_end': [df['time'].iloc[-1]]})
```


```python
tmp = getSequenceData(curry_df, 1510, 6)
extractFeatures(tmp, list(tmp_time_df.lower.values))
```




<div style="height:100%;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ballhoop_max</th>
      <th>ballhoop_mean</th>
      <th>ballhoop_min</th>
      <th>ballhoop_var</th>
      <th>curryball_max</th>
      <th>curryball_mean</th>
      <th>curryball_min</th>
      <th>curryball_var</th>
      <th>curryhoop_max</th>
      <th>curryhoop_mean</th>
      <th>...</th>
      <th>defdist_mean_m</th>
      <th>defdist_mean_v</th>
      <th>defdist_var_m</th>
      <th>defdist_var_v</th>
      <th>dist_ratio_mean</th>
      <th>dist_ratio_var</th>
      <th>offense</th>
      <th>shot_time</th>
      <th>trace_end</th>
      <th>trace_start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42.249655</td>
      <td>26.443021</td>
      <td>0.231528</td>
      <td>96.82848</td>
      <td>35.99551</td>
      <td>5.338232</td>
      <td>0.095481</td>
      <td>73.990365</td>
      <td>42.354334</td>
      <td>30.966018</td>
      <td>...</td>
      <td>19.915984</td>
      <td>31.028272</td>
      <td>91.749817</td>
      <td>2709.719469</td>
      <td>1.983769</td>
      <td>172.652191</td>
      <td>1</td>
      <td>46.92</td>
      <td>46.9185</td>
      <td>47.0185</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 32 columns</p>
</div>




```python
def prepareMLdata(df, turniningpoint_ls, shot_time_points_ls):
    appended_data = []
    for point in turniningpoint_ls:
        tmp = getSequenceData(curry_df, point, 6)
        tmp = extractFeatures(tmp, shot_time_points_ls)
        appended_data.append(tmp)
    appended_data = pd.concat(appended_data, ignore_index= True)
    return appended_data 
```


```python
shot_time_points_ls = list(tmp_time_df.lower.values)
ML_data = prepareMLdata(curry_df, turniningpoint_ls, shot_time_points_ls)
```


```python
ML_data.head()
```




<div style="height:100%;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ballhoop_max</th>
      <th>ballhoop_mean</th>
      <th>ballhoop_min</th>
      <th>ballhoop_var</th>
      <th>curryball_max</th>
      <th>curryball_mean</th>
      <th>curryball_min</th>
      <th>curryball_var</th>
      <th>curryhoop_max</th>
      <th>curryhoop_mean</th>
      <th>...</th>
      <th>defdist_mean_m</th>
      <th>defdist_mean_v</th>
      <th>defdist_var_m</th>
      <th>defdist_var_v</th>
      <th>dist_ratio_mean</th>
      <th>dist_ratio_var</th>
      <th>offense</th>
      <th>shot_time</th>
      <th>trace_end</th>
      <th>trace_start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>28.885491</td>
      <td>13.326784</td>
      <td>0.231505</td>
      <td>93.175493</td>
      <td>20.219201</td>
      <td>8.185967</td>
      <td>0.444152</td>
      <td>43.916751</td>
      <td>27.706306</td>
      <td>15.357575</td>
      <td>...</td>
      <td>13.228087</td>
      <td>6.290439</td>
      <td>52.646155</td>
      <td>361.518488</td>
      <td>5.723790</td>
      <td>101.393295</td>
      <td>0</td>
      <td>NaN</td>
      <td>47.78783</td>
      <td>47.88783</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32.916573</td>
      <td>25.156415</td>
      <td>0.136773</td>
      <td>79.154517</td>
      <td>12.005317</td>
      <td>8.053691</td>
      <td>4.092887</td>
      <td>3.174108</td>
      <td>23.009739</td>
      <td>19.967946</td>
      <td>...</td>
      <td>18.643226</td>
      <td>3.331261</td>
      <td>43.505947</td>
      <td>216.642121</td>
      <td>1.186465</td>
      <td>52.984364</td>
      <td>0</td>
      <td>NaN</td>
      <td>47.54317</td>
      <td>47.64317</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29.044444</td>
      <td>14.148250</td>
      <td>0.035938</td>
      <td>73.572487</td>
      <td>39.326996</td>
      <td>30.617151</td>
      <td>20.185569</td>
      <td>27.224889</td>
      <td>31.454115</td>
      <td>24.031073</td>
      <td>...</td>
      <td>21.486210</td>
      <td>0.246793</td>
      <td>91.032794</td>
      <td>494.299733</td>
      <td>14.798147</td>
      <td>4093.662355</td>
      <td>0</td>
      <td>NaN</td>
      <td>47.26917</td>
      <td>47.36917</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.347788</td>
      <td>6.813383</td>
      <td>0.035938</td>
      <td>34.803312</td>
      <td>33.381643</td>
      <td>27.531473</td>
      <td>20.185569</td>
      <td>14.003907</td>
      <td>31.026790</td>
      <td>24.390090</td>
      <td>...</td>
      <td>21.380101</td>
      <td>0.376388</td>
      <td>65.100121</td>
      <td>904.094115</td>
      <td>21.210795</td>
      <td>4217.887524</td>
      <td>0</td>
      <td>NaN</td>
      <td>47.22650</td>
      <td>47.32650</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42.249655</td>
      <td>26.443021</td>
      <td>0.231528</td>
      <td>96.828480</td>
      <td>35.995510</td>
      <td>5.338232</td>
      <td>0.095481</td>
      <td>73.990365</td>
      <td>42.354334</td>
      <td>30.966018</td>
      <td>...</td>
      <td>19.915984</td>
      <td>31.028272</td>
      <td>91.749817</td>
      <td>2709.719469</td>
      <td>1.983769</td>
      <td>172.652191</td>
      <td>1</td>
      <td>46.92</td>
      <td>46.91850</td>
      <td>47.01850</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
ML_data.to_csv('ML_data.csv', index = False)
```

Then, with the labels we scraped online, we can build a classifier to classify whether a movement pattern indicates Curry plays offense or not, which helps us to narrow down the scope for searching shooting patterns. We use 1st and 3rd quarters as our training set and train with logistic regression and random forest algorithms. logistic regression model turned out to perform better in this case although there still exists overfitting problem.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
```


```python
#ML_data = pd.read_csv("ML_data.csv")
# split training and test set
train_set = ML_data[(ML_data.trace_end >= 36.0) | ((24 > ML_data.trace_end) &(ML_data.trace_end >= 12.0)) ] 
test_set = ML_data[((36 > ML_data.trace_end) & (ML_data.trace_end >= 24)) | (ML_data.trace_end < 12.0)]
```


```python
features = [name for name in train_set.columns.tolist() if name not in ['offense', 'shot_time', 'trace_end', 'trace_start']]
X_train = train_set[features]
y_train = train_set['offense']
X_test = test_set[features]
y_test = test_set['offense']
```


```python
X_train.shape
```




    (150, 28)




```python
lr_model = LogisticRegression() 
rf_model = RandomForestClassifier(n_estimators= 290, max_features= 15) 

lr_model.fit(X_train, y_train)

rf_model.fit(X_train, y_train)

fit_predicted_lr = lr_model.predict(X_train) # prediction performance in training set
report_lr = classification_report(y_train, fit_predicted_lr) 
print(report_lr)

fit_predicted_rf = rf_model.predict(X_train) # prediction performance in training set
report_rf = classification_report(y_train, fit_predicted_rf) 
print(report_rf)
```

                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99       135
              1       0.93      0.93      0.93        15
    
    avg / total       0.99      0.99      0.99       150
    
                 precision    recall  f1-score   support
    
              0       1.00      1.00      1.00       135
              1       1.00      1.00      1.00        15
    
    avg / total       1.00      1.00      1.00       150
    
    

We have to deal with missing value cases due to the measurement errors.


```python
X_test[X_test.isnull().any(axis=1)]
```




<div style="height:100%;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ballhoop_max</th>
      <th>ballhoop_mean</th>
      <th>ballhoop_min</th>
      <th>ballhoop_var</th>
      <th>curryball_max</th>
      <th>curryball_mean</th>
      <th>curryball_min</th>
      <th>curryball_var</th>
      <th>curryhoop_max</th>
      <th>curryhoop_mean</th>
      <th>...</th>
      <th>def4dist_mean</th>
      <th>def4dist_var</th>
      <th>def5dist_mean</th>
      <th>def5dist_var</th>
      <th>defdist_mean_m</th>
      <th>defdist_mean_v</th>
      <th>defdist_var_m</th>
      <th>defdist_var_v</th>
      <th>dist_ratio_mean</th>
      <th>dist_ratio_var</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>229</th>
      <td>30.100838</td>
      <td>16.739614</td>
      <td>0.389141</td>
      <td>68.798010</td>
      <td>38.949504</td>
      <td>26.843761</td>
      <td>0.448405</td>
      <td>126.753295</td>
      <td>25.918275</td>
      <td>15.388054</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.740568</td>
      <td>8.362249</td>
      <td>70.660123</td>
      <td>1202.564293</td>
      <td>1.644810</td>
      <td>0.282469</td>
    </tr>
    <tr>
      <th>230</th>
      <td>36.568824</td>
      <td>27.263279</td>
      <td>0.221226</td>
      <td>91.600931</td>
      <td>33.334161</td>
      <td>6.969266</td>
      <td>0.333314</td>
      <td>63.844736</td>
      <td>35.092890</td>
      <td>31.432368</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.895483</td>
      <td>3.236023</td>
      <td>110.514178</td>
      <td>3801.879614</td>
      <td>2.660365</td>
      <td>230.627245</td>
    </tr>
    <tr>
      <th>231</th>
      <td>27.563397</td>
      <td>16.694428</td>
      <td>0.130125</td>
      <td>58.094332</td>
      <td>22.415984</td>
      <td>10.877561</td>
      <td>1.328954</td>
      <td>20.481343</td>
      <td>23.652165</td>
      <td>18.631163</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.368485</td>
      <td>5.217303</td>
      <td>15.273074</td>
      <td>59.153009</td>
      <td>2.296235</td>
      <td>199.197441</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 28 columns</p>
</div>




```python
# Treatment with missing values by replacing with median values
X_test = X_test.fillna(X_test.median())
```


```python
lr_predicted = lr_model.predict(X_test) # prediction performance in training set
report_lr = classification_report(y_test, lr_predicted) 
print(report_lr)

rf_predicted = rf_model.predict(X_test) # prediction performance in training set
report_rf = classification_report(y_test, rf_predicted) 
print(report_rf)
```

                 precision    recall  f1-score   support
    
              0       0.91      0.89      0.90        71
              1       0.38      0.45      0.42        11
    
    avg / total       0.84      0.83      0.84        82
    
                 precision    recall  f1-score   support
    
              0       0.88      0.96      0.92        71
              1       0.40      0.18      0.25        11
    
    avg / total       0.82      0.85      0.83        82
    
    


```python
# save the model
with open('lr_model.pickle', 'w') as pf:
    pickle.dump(lr_model, pf)
```

### Step 3 Change point detection
Using our trained model from the last step, we can apply it on a new dataset to identify the "offense" sequences. In this case, to submit our prediction results, we would generate the predictions for the whole provided dataset.


```python
def getOffenseSeq(model, ndata, features, top = 24):
    ndata['preds'] = model.predict_proba(ndata[features])[:,1]
    #offense_data = ndata[ndata.preds == 1].reset_index()
    offense_data = ndata.nlargest(24, 'preds')
    return offense_data
```


```python
nML_data = ML_data.fillna(ML_data[features].median())
getOffenseSeq(lr_model, nML_data, features)[['preds', 'offense', 'shot_time', 'trace_end', 'trace_start']]
```




<div style="height:100%;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>preds</th>
      <th>offense</th>
      <th>shot_time</th>
      <th>trace_end</th>
      <th>trace_start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>204</th>
      <td>1.000000</td>
      <td>1</td>
      <td>5.403333</td>
      <td>5.36033</td>
      <td>5.46033</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1.000000</td>
      <td>1</td>
      <td>42.870000</td>
      <td>42.86000</td>
      <td>42.96000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.999993</td>
      <td>1</td>
      <td>46.920000</td>
      <td>46.84050</td>
      <td>46.94050</td>
    </tr>
    <tr>
      <th>193</th>
      <td>0.999788</td>
      <td>1</td>
      <td>12.436667</td>
      <td>12.38417</td>
      <td>12.48433</td>
    </tr>
    <tr>
      <th>102</th>
      <td>0.997139</td>
      <td>0</td>
      <td>NaN</td>
      <td>27.43833</td>
      <td>27.53833</td>
    </tr>
    <tr>
      <th>224</th>
      <td>0.995493</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.19100</td>
      <td>2.29100</td>
    </tr>
    <tr>
      <th>103</th>
      <td>0.984838</td>
      <td>1</td>
      <td>27.436667</td>
      <td>27.38033</td>
      <td>27.48033</td>
    </tr>
    <tr>
      <th>222</th>
      <td>0.979128</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.50533</td>
      <td>2.66983</td>
    </tr>
    <tr>
      <th>175</th>
      <td>0.966567</td>
      <td>1</td>
      <td>15.120000</td>
      <td>15.11750</td>
      <td>15.21767</td>
    </tr>
    <tr>
      <th>223</th>
      <td>0.965653</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.50400</td>
      <td>2.66850</td>
    </tr>
    <tr>
      <th>106</th>
      <td>0.965209</td>
      <td>0</td>
      <td>NaN</td>
      <td>27.07433</td>
      <td>27.17433</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.937037</td>
      <td>1</td>
      <td>46.920000</td>
      <td>46.91850</td>
      <td>47.01850</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.922539</td>
      <td>1</td>
      <td>36.903333</td>
      <td>36.87167</td>
      <td>36.97133</td>
    </tr>
    <tr>
      <th>180</th>
      <td>0.910724</td>
      <td>1</td>
      <td>14.036667</td>
      <td>14.02417</td>
      <td>14.12417</td>
    </tr>
    <tr>
      <th>176</th>
      <td>0.776233</td>
      <td>1</td>
      <td>15.120000</td>
      <td>15.03083</td>
      <td>15.13083</td>
    </tr>
    <tr>
      <th>109</th>
      <td>0.726532</td>
      <td>0</td>
      <td>NaN</td>
      <td>26.41567</td>
      <td>26.58767</td>
    </tr>
    <tr>
      <th>192</th>
      <td>0.722562</td>
      <td>1</td>
      <td>12.436667</td>
      <td>12.43350</td>
      <td>12.53367</td>
    </tr>
    <tr>
      <th>230</th>
      <td>0.712588</td>
      <td>1</td>
      <td>0.503333</td>
      <td>0.50083</td>
      <td>0.60100</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0.699003</td>
      <td>1</td>
      <td>39.653333</td>
      <td>39.61433</td>
      <td>39.71433</td>
    </tr>
    <tr>
      <th>64</th>
      <td>0.685481</td>
      <td>1</td>
      <td>38.236667</td>
      <td>38.22283</td>
      <td>38.32300</td>
    </tr>
    <tr>
      <th>148</th>
      <td>0.671682</td>
      <td>1</td>
      <td>20.570000</td>
      <td>20.56550</td>
      <td>20.66567</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.654600</td>
      <td>1</td>
      <td>44.136667</td>
      <td>44.11267</td>
      <td>44.21267</td>
    </tr>
    <tr>
      <th>203</th>
      <td>0.630730</td>
      <td>0</td>
      <td>NaN</td>
      <td>5.41300</td>
      <td>5.51300</td>
    </tr>
    <tr>
      <th>122</th>
      <td>0.602449</td>
      <td>1</td>
      <td>24.020000</td>
      <td>24.01100</td>
      <td>24.11200</td>
    </tr>
  </tbody>
</table>
</div>



Then, based on these selected time frames, we can apply other algorithms to detect the "change points" in the time series. We use the PELT algorithm[1] and the package `changepy`. The PERL algorithm requires a cost function. In our case, we chose the `exponential` for exponential distributed data with changing mean.

_**Reference**_:
[1] Killick R, Fearnhead P, Eckley IA (2012) Optimal detection of changepoints with a linear computational cost, JASA 107(500), 1590-1598


```python
def mapSeq(data, trace_start, trace_end):
    seq_data = data[data['time'].between(trace_end,trace_start, inclusive = True)]
    return seq_data['curryball'].values
```


```python
%matplotlib inline
curry_df.curryball.plot(kind="hist", bins =15) # exponential distribution
```




<img src="../PMC-project_files/output_94_1.png" class="img-responsive" style="display: block; margin: auto;" />



```python
from changepy import pelt
from changepy.costs import exponential,normal_meanvar
```


```python
curryball_seq = mapSeq(curry_df, 47.01850, 46.91850)
```


```python
cost = exponential(curryball_seq)
pelt(cost, len(curryball_seq))
```




    [0, 118]




```python
curryball_seq[118]
```




    4.49890067991059




```python
def getChangePoint(data, trace_start, trace_end):
    seq_data = data[data['time'].between(trace_end,trace_start, inclusive = True)].reset_index()
    curryball_seq = seq_data['curryball'].values
    cost = exponential(curryball_seq)
    try:
        chg_point = pelt(cost, len(curryball_seq))[1]
        pred_shot_time = round(seq_data.ix[chg_point, 'time'], 8)      
    except:
        pred_shot_time = np.nan
    return pred_shot_time
```


```python
getChangePoint(curry_df, 47.01850, 46.91850)
```




    46.93983




```python
shot_info = getOffenseSeq(lr_model, nML_data, features)[['preds', 'offense', 'shot_time', 'trace_end', 'trace_start']]
shot_info['pred_shot_time'] = map(lambda t1, t2: getChangePoint(curry_df, t1, t2), shot_info['trace_start'], shot_info['trace_end'])
```


```python
# apply some heauristic rules: 1. if we fail to detect change point
shot_info['pred_shot_time'] = map(lambda t1, t2: t2+0.05  if pd.isnull(t1) else t1,shot_info['pred_shot_time'], shot_info['trace_end'])
```


```python
shot_info
```




<div style="height:100%;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>preds</th>
      <th>offense</th>
      <th>shot_time</th>
      <th>trace_end</th>
      <th>trace_start</th>
      <th>pred_shot_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>204</th>
      <td>1.000000</td>
      <td>1</td>
      <td>5.403333</td>
      <td>5.36033</td>
      <td>5.46033</td>
      <td>5.43033</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1.000000</td>
      <td>1</td>
      <td>42.870000</td>
      <td>42.86000</td>
      <td>42.96000</td>
      <td>42.92400</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.999993</td>
      <td>1</td>
      <td>46.920000</td>
      <td>46.84050</td>
      <td>46.94050</td>
      <td>46.92717</td>
    </tr>
    <tr>
      <th>193</th>
      <td>0.999788</td>
      <td>1</td>
      <td>12.436667</td>
      <td>12.38417</td>
      <td>12.48433</td>
      <td>12.44083</td>
    </tr>
    <tr>
      <th>102</th>
      <td>0.997139</td>
      <td>0</td>
      <td>NaN</td>
      <td>27.43833</td>
      <td>27.53833</td>
      <td>27.47367</td>
    </tr>
    <tr>
      <th>224</th>
      <td>0.995493</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.19100</td>
      <td>2.29100</td>
      <td>2.27300</td>
    </tr>
    <tr>
      <th>103</th>
      <td>0.984838</td>
      <td>1</td>
      <td>27.436667</td>
      <td>27.38033</td>
      <td>27.48033</td>
      <td>27.47367</td>
    </tr>
    <tr>
      <th>222</th>
      <td>0.979128</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.50533</td>
      <td>2.66983</td>
      <td>2.65050</td>
    </tr>
    <tr>
      <th>175</th>
      <td>0.966567</td>
      <td>1</td>
      <td>15.120000</td>
      <td>15.11750</td>
      <td>15.21767</td>
      <td>15.18017</td>
    </tr>
    <tr>
      <th>223</th>
      <td>0.965653</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.50400</td>
      <td>2.66850</td>
      <td>2.65050</td>
    </tr>
    <tr>
      <th>106</th>
      <td>0.965209</td>
      <td>0</td>
      <td>NaN</td>
      <td>27.07433</td>
      <td>27.17433</td>
      <td>27.09367</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.937037</td>
      <td>1</td>
      <td>46.920000</td>
      <td>46.91850</td>
      <td>47.01850</td>
      <td>46.93983</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.922539</td>
      <td>1</td>
      <td>36.903333</td>
      <td>36.87167</td>
      <td>36.97133</td>
      <td>36.91267</td>
    </tr>
    <tr>
      <th>180</th>
      <td>0.910724</td>
      <td>1</td>
      <td>14.036667</td>
      <td>14.02417</td>
      <td>14.12417</td>
      <td>14.04683</td>
    </tr>
    <tr>
      <th>176</th>
      <td>0.776233</td>
      <td>1</td>
      <td>15.120000</td>
      <td>15.03083</td>
      <td>15.13083</td>
      <td>15.08083</td>
    </tr>
    <tr>
      <th>109</th>
      <td>0.726532</td>
      <td>0</td>
      <td>NaN</td>
      <td>26.41567</td>
      <td>26.58767</td>
      <td>26.55700</td>
    </tr>
    <tr>
      <th>192</th>
      <td>0.722562</td>
      <td>1</td>
      <td>12.436667</td>
      <td>12.43350</td>
      <td>12.53367</td>
      <td>12.44017</td>
    </tr>
    <tr>
      <th>230</th>
      <td>0.712588</td>
      <td>1</td>
      <td>0.503333</td>
      <td>0.50083</td>
      <td>0.60100</td>
      <td>0.56167</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0.699003</td>
      <td>1</td>
      <td>39.653333</td>
      <td>39.61433</td>
      <td>39.71433</td>
      <td>39.69167</td>
    </tr>
    <tr>
      <th>64</th>
      <td>0.685481</td>
      <td>1</td>
      <td>38.236667</td>
      <td>38.22283</td>
      <td>38.32300</td>
      <td>38.28367</td>
    </tr>
    <tr>
      <th>148</th>
      <td>0.671682</td>
      <td>1</td>
      <td>20.570000</td>
      <td>20.56550</td>
      <td>20.66567</td>
      <td>20.65967</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.654600</td>
      <td>1</td>
      <td>44.136667</td>
      <td>44.11267</td>
      <td>44.21267</td>
      <td>44.20950</td>
    </tr>
    <tr>
      <th>203</th>
      <td>0.630730</td>
      <td>0</td>
      <td>NaN</td>
      <td>5.41300</td>
      <td>5.51300</td>
      <td>5.42967</td>
    </tr>
    <tr>
      <th>122</th>
      <td>0.602449</td>
      <td>1</td>
      <td>24.020000</td>
      <td>24.01100</td>
      <td>24.11200</td>
      <td>24.06100</td>
    </tr>
  </tbody>
</table>
</div>




```python
shot_info['upper'] = shot_info['pred_shot_time'] + 0.02
shot_info['lower'] = shot_info['pred_shot_time'] - 0.06
modeling_result = shot_info[['upper', 'lower']]
modeling_result.to_csv('curry_model_pred.csv', index=False)
```


```python
modeling_result
```




<div style="height:100%;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>upper</th>
      <th>lower</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>204</th>
      <td>5.45033</td>
      <td>5.37033</td>
    </tr>
    <tr>
      <th>35</th>
      <td>42.94400</td>
      <td>42.86400</td>
    </tr>
    <tr>
      <th>5</th>
      <td>46.94717</td>
      <td>46.86717</td>
    </tr>
    <tr>
      <th>193</th>
      <td>12.46083</td>
      <td>12.38083</td>
    </tr>
    <tr>
      <th>102</th>
      <td>27.49367</td>
      <td>27.41367</td>
    </tr>
    <tr>
      <th>224</th>
      <td>2.29300</td>
      <td>2.21300</td>
    </tr>
    <tr>
      <th>103</th>
      <td>27.49367</td>
      <td>27.41367</td>
    </tr>
    <tr>
      <th>222</th>
      <td>2.67050</td>
      <td>2.59050</td>
    </tr>
    <tr>
      <th>175</th>
      <td>15.20017</td>
      <td>15.12017</td>
    </tr>
    <tr>
      <th>223</th>
      <td>2.67050</td>
      <td>2.59050</td>
    </tr>
    <tr>
      <th>106</th>
      <td>27.11367</td>
      <td>27.03367</td>
    </tr>
    <tr>
      <th>4</th>
      <td>46.95983</td>
      <td>46.87983</td>
    </tr>
    <tr>
      <th>74</th>
      <td>36.93267</td>
      <td>36.85267</td>
    </tr>
    <tr>
      <th>180</th>
      <td>14.06683</td>
      <td>13.98683</td>
    </tr>
    <tr>
      <th>176</th>
      <td>15.10083</td>
      <td>15.02083</td>
    </tr>
    <tr>
      <th>109</th>
      <td>26.57700</td>
      <td>26.49700</td>
    </tr>
    <tr>
      <th>192</th>
      <td>12.46017</td>
      <td>12.38017</td>
    </tr>
    <tr>
      <th>230</th>
      <td>0.58167</td>
      <td>0.50167</td>
    </tr>
    <tr>
      <th>54</th>
      <td>39.71167</td>
      <td>39.63167</td>
    </tr>
    <tr>
      <th>64</th>
      <td>38.30367</td>
      <td>38.22367</td>
    </tr>
    <tr>
      <th>148</th>
      <td>20.67967</td>
      <td>20.59967</td>
    </tr>
    <tr>
      <th>28</th>
      <td>44.22950</td>
      <td>44.14950</td>
    </tr>
    <tr>
      <th>203</th>
      <td>5.44967</td>
      <td>5.36967</td>
    </tr>
    <tr>
      <th>122</th>
      <td>24.08100</td>
      <td>24.00100</td>
    </tr>
  </tbody>
</table>
</div>



After submmiting this result, this ML approach gave us about 50% accuracy. One concern is that the pretrained model may not be good enough to detect the offense sequences correctly in the first step. Also, training on only one dataset is not adequate for buiding a robust machine learning model. 

As I mentioned, the organization committee would test our algorithm on a new test dataset. It turned out our ML approach didn't lead us to the final round for on-site presentation. After I attended the convention on Feb 4, the winning solutions discussed mostly are just related to use some "domain knowledge" with deeper analysis of shooting pattern matching. For this competition, the ML is probably not the good way to go. It somehow shows Machine Learning is not an elixir.

## Afterthoughts

Through my research, there are some good reasons to try Recurrent Neural Network, a Deep learning model known for processing language and sequence data.

- All the covariates are actually time series data. Like speech regonition or signal processing, the specific hidden shooting pattern is what we can look for in the time series. 
- Data is noisy. However, deep learning model is good at extracting latent representation/information. Thus, we can be less bothered by feature engineering.


