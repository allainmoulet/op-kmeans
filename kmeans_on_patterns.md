---
title: kmeansOnPatterns
date: 2018-01-15 15:00:00 +02:00
layout: default
published: true
---
# K-means on Patterns

This IKATS operator implements K-Means algorithm, like the [kmeans on SAX](https://ikats.org/doc/operators/kmeans.html) operator, but with a different input: `patterns` instead of `SAX` words. The principle is the same, i.e., Clusters data by trying to separate samples in *n* groups with the nearest mean.


## Input and parameters

In the current implementation, this operator only takes one input of the functional type `patterns`, and this way is dedicated to be applied to the output of the [Random Projections](https://ikats.org/doc/operators/randomProjections.html) operator.

It also takes 1 input from the user :

- **clusters** : The number of clusters to form as well as the number of centroids to generate


## Outputs

The operator has two outputs :

 - **Model** : a binary dump of the best model found by the procedure
 - **result** : Clusters visualisation of the inpout patterns, including centroïds positions


Return to the [list of all operators](https://ikats.org/operators.html)

