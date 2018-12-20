---
title: kmeans
date: 2018-12-11 15:00:00 +02:00
layout: default
published: true
---
# K-Means 
Those operators perform the unsupervised learning algorithm [K-Means](https://en.wikipedia.org/wiki/K-means_clustering). It is used to cluster the data in K homogeneous and provide a 2-dimensional synthetic visualization of the result. For each of the K groups, an "artificial mean point" called the centroid is built and displayed in the visualisation.

There are three different K-means algorithms in IKATS, depending on the nature of the inputs:

## [K-Means on SAX](kmeans_on_sax.md)
Perform K-Means algorithm on SAX words. It is dedicated to be applied to the output of the [SAX](https://ikats.org/doc/operators/sax.html) operator

## [K-Means on Patterns](kmeans_on_patterns.md)
Perform K-Means algorithm on patterns. It is dedicated to be applied to the output of the [Random Projections](https://ikats.org/doc/operators/randomProjections.html) operator.

## [K-Means on Time Series](kmeans_on_ts.md)
Perform K-Means algorithm directly on time series data. 

Return to the [list of all operators](https://ikats.org/operators.html)
