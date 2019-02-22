---
title: kmeansOnTS
date: 2018-12-11 15:00:00 +02:00
layout: default
published: true
---
# K-Means on time series operator
This IKATS operator implements a K-means algorithm. It is designed for time series.

## Input and parameters
This operator takes only one input of the functional type `ts_list`.
It also takes an optional parameter from the user:
- **nb_clusters**: the number of clusters K (and centroids) to form with the K-means algorithm. This is also the number of centroids.

## Output
The operator has one output:
- **result**: a Python dictionary containing the clustering performed.

## Warnings
- The `qual_ref_period` metadata must be in every time serie's metadata. That means that the user has to run the
`Quality Indicators` operator before using this one.
- The time series have to be aligned, that is to say with the same number of points and the exact same timestamps.


## Implementation remarks
- The choice of the value of K has a huge impact on the clustering.
- The K-means algorithm is sensible to initialisation, that means it is theoretically possible to get different results. 
- It can theoretically converge to a local minimum, that is to say a non optimal solution.
- As the classification is non supervised, label switching cases may occur. That means, if we run the algorithm several times, we may get the same clustering but with different names for the clusters:

## Example
Return to the [list of all operators](https://ikats.org/operators.html)
