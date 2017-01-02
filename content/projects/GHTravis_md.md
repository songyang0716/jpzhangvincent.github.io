+++
showonlyimage = false
draft = false
image = "projects/GHTravis_md_files/travis.png"
date = "2017-01-02"
title = "TravisTorrent Data Challenge"
weight = 0
type = "post"
tags = [
"Github","SQL",
"data viz"]
author = "Vincent Zhang"
+++

Summary: TravisTorrent is a freely available data set synthesized from Travis CI and GitHub, which also serves as a current data challenge from The International Conference on Mining Software Repositories(MSR).

<!--more-->
You can embed an R code chunk like this:

``` r
summary(cars)
```

    ##      speed           dist       
    ##  Min.   : 4.0   Min.   :  2.00  
    ##  1st Qu.:12.0   1st Qu.: 26.00  
    ##  Median :15.0   Median : 36.00  
    ##  Mean   :15.4   Mean   : 42.98  
    ##  3rd Qu.:19.0   3rd Qu.: 56.00  
    ##  Max.   :25.0   Max.   :120.00

``` r
fit <- lm(dist ~ speed, data = cars)
fit
```

    ## 
    ## Call:
    ## lm(formula = dist ~ speed, data = cars)
    ## 
    ## Coefficients:
    ## (Intercept)        speed  
    ##     -17.579        3.932

Including Plots
===============

You can also embed plots. See Figure @ref(fig:pie) for example:

``` r
par(mar = c(0, 1, 0, 1))
pie(
  c(280, 60, 20),
  c('Sky', 'Sunny side of pyramid', 'Shady side of pyramid'),
  col = c('#0292D8', '#F7EA39', '#C4B632'),
  init.angle = -50, border = NA
)
```

![A fancy pie chart.](GHTravis_files/figure-markdown_github/pie-1.png)
