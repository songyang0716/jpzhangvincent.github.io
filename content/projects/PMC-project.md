+++
showonlyimage = false
draft = false
image = "projects/PMC-project_files/shots_area.jpg"
date = "2017-01-13"
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
-   [To be continued](#to-be-continued)


<h2>IIDATA 2017 Predictive Modeling Competition</h2>
<p>
<strong>IIDATA</strong> is a one-day data science convention hosted by UC Davis undergraduate students. This year will be on Feb.4, 2017. There will be hands-on workshops, tech talks and modeling competitions in this event. I think it would be a good learning opportunity for people interesed in data science to attend. For more information, please visit the <a href="http://www.iidata.net/">website</a>. Lately, the organization team released their dataset for the Predictive Modeling Competition(PMC). This year the challenge is quite interesting. </p>

<p><i>Given Stephen Curry’s distance to ball, hoops and the opposing team’s defenders, can 
you determine when he is releasing the ball?</i></p>

<p>
They also have a leaderboard with more detailed information about the competition <a href="https://pmc-leaderboard.com/">here</a>, especially the submission format.
</p>

<h2> Data </h2>

<p>This dataset contains temporal snapshots taken every 0.04 seconds during the November 12th, 2015 Golden State Warriors vs. Minnesota Timberwolves game (upon which Curry is on the court). It was told that Curry took **24** shots. For each shot, you are expected to find when he is releasing the ball from his hands with as much accuracy as possible. The data source is <a href="https://www.stats.com/sportvu-basketball/">SportVU</a>. These are the variables:
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
<div>
$$score = \frac{\sum_{i=1}^{|s|}I_{T_i}\frac{1}{1+s_{i2}-s_{i1}}}{|T|+(|S|-|T|)^2}$$</div>
where <div> $$S_i: \{(s_{i1}, s_{i2})\mid s_{i1} \leq s_{i2}\}, \| T\|=24, \|S\|\geq \|T\|, I_{T_i}:\{1 \mid S_i \in T_i\}$$. </div>

Thus, there is room and trade-off for us to determine the interval of the times to get a higher overall prediction score to win the competition.
</li> 
</ol>
</p>

<h2> Load Libraries</h2>


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
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
```


```python
sns.set(style="whitegrid")
```


```python
curry_df = pd.read_csv("curry.csv")
```


```python
curry_df.head()
```




<div>
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
      <td>47.925333</td>
      <td>0.965270</td>
      <td>15.652612</td>
      <td>92.829463</td>
      <td>16.323108</td>
      <td>91.876879</td>
      <td>8.603274</td>
      <td>16.950232</td>
      <td>27.471575</td>
      <td>30.409634</td>
      <td>35.474795</td>
    </tr>
    <tr>
      <th>1</th>
      <td>47.924500</td>
      <td>0.797786</td>
      <td>15.223746</td>
      <td>93.094109</td>
      <td>15.935113</td>
      <td>92.307938</td>
      <td>8.298388</td>
      <td>16.730251</td>
      <td>27.240581</td>
      <td>30.249721</td>
      <td>35.246274</td>
    </tr>
    <tr>
      <th>2</th>
      <td>47.923833</td>
      <td>1.035342</td>
      <td>14.806208</td>
      <td>93.324333</td>
      <td>15.593023</td>
      <td>92.293618</td>
      <td>7.992415</td>
      <td>16.471735</td>
      <td>26.975070</td>
      <td>30.059684</td>
      <td>34.973485</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47.923167</td>
      <td>2.126335</td>
      <td>14.375951</td>
      <td>93.577226</td>
      <td>15.994375</td>
      <td>91.469738</td>
      <td>7.693763</td>
      <td>16.237772</td>
      <td>26.712746</td>
      <td>29.885916</td>
      <td>34.705748</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47.922500</td>
      <td>2.704138</td>
      <td>13.937750</td>
      <td>93.836626</td>
      <td>15.895369</td>
      <td>91.200628</td>
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




    (55615, 18)


It's tricky to note that the dataset include all the movement data of Curry on the field and he may play offense or defense.  And he played nearly equal time in the first half and second half. So we could expect the probability that Curry was in the side closer to either left or right hoop should be somewhat equal with possibly some skewness since we know he's a PG and more likely in attack mode. We can do a sanity check first to verify that. It does turn out to be true.

```python
sum(map(lambda i,j: i<j, curry_df['currylhoop'],curry_df['curryrhoop']))
```




    28784




```python
28784/55615.0 
```




    0.5175582127123978



## Exploratory Data Analysis

As we know, Stephen Curry is a special shooter in the NBA history, especially in 2015-2016 Season. Using the movement data is a way to understand his shooting pattern, which would be helpful for us to infer his shot times. 

First, since Stephen Curry is known for his 3-point and long-distance shots, we would be interested in how the distance between Curry and the hoop varies around the 3-point line. Through some research about the specific game online, we know for the first half, the left hoop is the goal of Golden Warriors, and vice versa. However, as we can see, there are a lot of fluatuations because Curry were playing defense or offense. So we somehow need to come up with some clever ways to identify whether Curry is in attacking mode or not.


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
curry_df['curryhoop_min'] = map(lambda i,j: min(i,j), curry_df['currylhoop'] ,curry_df['curryrhoop'] )
```


```python
curry_df = curry_df.reset_index()
plt.figure(figsize=(100,40))
plt.plot(curry_df['index'], curry_df['curryhoop_min'], '-')
plt.title("Closest Distance between Hoop and Curry over time", fontsize = 50)
plt.xlabel('Index', fontsize=30)
plt.ylabel('Distance', fontsize=30)
```

<img src="../PMC-project_files/output_17_1.png" class="img-responsive" style="display: block; margin: auto;" />


We have to transform the **`time`** variable to make it more understandable and easy to inteprete.


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
      <th>quarter</th>
      <th>milisecs</th>
      <th>minute</th>
      <th>second</th>
      <th>curryhoop_min</th>
      <th>ballhoop_min</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>47.925333</td>
      <td>0.965270</td>
      <td>15.652612</td>
      <td>92.829463</td>
      <td>16.323108</td>
      <td>91.876879</td>
      <td>8.603274</td>
      <td>16.950232</td>
      <td>27.471575</td>
      <td>30.409634</td>
      <td>35.474795</td>
      <td>1</td>
      <td>0.52</td>
      <td>11.0</td>
      <td>55.0</td>
      <td>15.652612</td>
      <td>16.323108</td>
    </tr>
    <tr>
      <th>1</th>
      <td>47.924500</td>
      <td>0.797786</td>
      <td>15.223746</td>
      <td>93.094109</td>
      <td>15.935113</td>
      <td>92.307938</td>
      <td>8.298388</td>
      <td>16.730251</td>
      <td>27.240581</td>
      <td>30.249721</td>
      <td>35.246274</td>
      <td>1</td>
      <td>0.47</td>
      <td>11.0</td>
      <td>55.0</td>
      <td>15.223746</td>
      <td>15.935113</td>
    </tr>
    <tr>
      <th>2</th>
      <td>47.923833</td>
      <td>1.035342</td>
      <td>14.806208</td>
      <td>93.324333</td>
      <td>15.593023</td>
      <td>92.293618</td>
      <td>7.992415</td>
      <td>16.471735</td>
      <td>26.975070</td>
      <td>30.059684</td>
      <td>34.973485</td>
      <td>1</td>
      <td>0.43</td>
      <td>11.0</td>
      <td>55.0</td>
      <td>14.806208</td>
      <td>15.593023</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47.923167</td>
      <td>2.126335</td>
      <td>14.375951</td>
      <td>93.577226</td>
      <td>15.994375</td>
      <td>91.469738</td>
      <td>7.693763</td>
      <td>16.237772</td>
      <td>26.712746</td>
      <td>29.885916</td>
      <td>34.705748</td>
      <td>1</td>
      <td>0.39</td>
      <td>11.0</td>
      <td>55.0</td>
      <td>14.375951</td>
      <td>15.994375</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47.922500</td>
      <td>2.704138</td>
      <td>13.937750</td>
      <td>93.836626</td>
      <td>15.895369</td>
      <td>91.200628</td>
      <td>7.393487</td>
      <td>15.996931</td>
      <td>26.437723</td>
      <td>29.697227</td>
      <td>34.429396</td>
      <td>1</td>
      <td>0.35</td>
      <td>11.0</td>
      <td>55.0</td>
      <td>13.937750</td>
      <td>15.895369</td>
    </tr>
  </tbody>
</table>
</div>




```python
curry_df['dist_ratio'] = curry_df['curryball'] / curry_df['ballhoop_min']
```

Let's see whether we can observe any pattern of this distance ratio. We can see that there are some obvious spikes, which are probably what we are looking for. Next, we can extract those corresponding times.


```python
plt.figure(figsize=(100,40))
plt.plot(curry_df['index'], curry_df['dist_ratio'], '-')
plt.title("Distance Ratio over time", fontsize = 50)
plt.xlabel('Index', fontsize=30)
plt.ylabel('Distance Ratio', fontsize=30)
```

<img src="../PMC-project_files/output_26_1.png" class="img-responsive" style="display: block; margin: auto;" />



```python
# Find the times with the top 24 highest spikes
curry_df.nlargest(24, 'dist_ratio')[["time", "dist_ratio", "curryhoop_min","quarter"]]
```




<div style="height:100%;overflow:auto;">
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
      <th>42425</th>
      <td>14.455500</td>
      <td>363.651664</td>
      <td>18.843438</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42433</th>
      <td>14.450167</td>
      <td>363.327192</td>
      <td>19.129749</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42434</th>
      <td>14.449500</td>
      <td>241.701580</td>
      <td>19.158059</td>
      <td>3</td>
    </tr>
    <tr>
      <th>17819</th>
      <td>31.062000</td>
      <td>240.649549</td>
      <td>34.829993</td>
      <td>2</td>
    </tr>
    <tr>
      <th>42435</th>
      <td>14.448833</td>
      <td>176.939222</td>
      <td>19.185484</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42431</th>
      <td>14.451500</td>
      <td>115.415805</td>
      <td>19.068632</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42424</th>
      <td>14.456167</td>
      <td>113.908181</td>
      <td>18.783849</td>
      <td>3</td>
    </tr>
    <tr>
      <th>824</th>
      <td>47.375833</td>
      <td>111.561951</td>
      <td>36.922632</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42437</th>
      <td>14.447500</td>
      <td>106.837579</td>
      <td>19.243607</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42426</th>
      <td>14.454833</td>
      <td>92.652413</td>
      <td>18.893214</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42436</th>
      <td>14.448167</td>
      <td>87.928430</td>
      <td>19.214337</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42430</th>
      <td>14.452167</td>
      <td>82.046262</td>
      <td>19.041845</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42432</th>
      <td>14.450833</td>
      <td>77.040831</td>
      <td>19.100626</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42439</th>
      <td>14.446167</td>
      <td>74.102692</td>
      <td>19.302154</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42450</th>
      <td>14.438833</td>
      <td>67.965220</td>
      <td>19.557624</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42438</th>
      <td>14.446833</td>
      <td>65.596627</td>
      <td>19.276922</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42451</th>
      <td>14.438167</td>
      <td>63.899451</td>
      <td>19.572023</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42428</th>
      <td>14.453500</td>
      <td>60.944564</td>
      <td>18.977573</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42427</th>
      <td>14.454167</td>
      <td>60.595256</td>
      <td>18.935209</td>
      <td>3</td>
    </tr>
    <tr>
      <th>17820</th>
      <td>31.061333</td>
      <td>57.896951</td>
      <td>34.785746</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17818</th>
      <td>31.062667</td>
      <td>56.580318</td>
      <td>34.883618</td>
      <td>2</td>
    </tr>
    <tr>
      <th>42440</th>
      <td>14.445500</td>
      <td>54.899058</td>
      <td>19.330957</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42423</th>
      <td>14.456833</td>
      <td>53.134837</td>
      <td>18.717561</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42429</th>
      <td>14.452833</td>
      <td>50.605485</td>
      <td>19.012620</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



However, it turns out that the corresponding top 24 "time points" include the noise that many duplicated pointed are counted in a short time period. As we researched online, it just doesn't make sense to have most shots happened in the third quarter.

<h2>Can we get the labels?</h2>

It's obvious that the raw data doesn't directly tell us when Curry made the shots. However, given the background information about the game, it's possible to find out those actual shooting times, for example, by watching this specific past (game)[https://www.youtube.com/watch?v=5PozYV_qHD8]. Although we are not supposed to use it as the submission, ideally, we can use the information to denote the corresponding Curry's shooting times as a way to tune our pattern regconition algorithm.

Acutally, there is a more clever way to get the labels. We found we can scrape the actual shot times by Curry on the official NBA statistics website [here](http://stats.nba.com/game/#!/0021500125/playbyplay/). It's a little tricky that the website is dynamically generated. To do the web scraping, first we can try to find the API access point by locating Develop Tools -> Network -> XHR in Chrome. Luckily, we found it exists. Then we can go ahead to scrape the information we want.


```python
import requests

shots_url = 'http://stats.nba.com/stats/playbyplayv2?EndPeriod=10&EndRange=55800&GameID=0021500125&RangeType=2&Season=2015-16&SeasonType=Regular+Season&StartPeriod=1&StartRange=0'

# request the URL and parse the JSON
response = requests.get(shots_url)
response.raise_for_status() # raise exception if invalid response
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


## To be continued

Plans for the next step

- Doing more exploratory data analysis for the snippet time frames that Curry made the shots. 
- Learn and experiment with the Recurrent Neural Network model

Through my research, there are some good reasons why we should try Recurrent Neural Network, a Deep learning model known for processing language and sequence data.

- All the covariates are actually time series data. Like speech regonition or signal processing, the specific hidden shooting pattern is what we are looking for in the time series. 
- Data is noisy. However, deep learning model is good at extracting latent representation/information. Thus, we can be less bothered by feature engineering.

To be continued. Stay tuned. :)

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

