---
title: kmeansOnTSPredict
date: 2019-01-24 18:00:00 +02:00
layout: default
published: true
---
# K-Means on time series - predict operator
This IKATS operator implements a K-means predict operator. It is designed for time series and have to be ran after the `K-Means on TS` operator.

## Input and parameters
This operator takes three inputs of the functional type : `kmeans_mds`, `sk_model` and `ts_list`.
This operator doesn't have any parameter.

## Output
The operator has one output:
- **Table**: in IKATS, it is seen as a table of 2 columns : functional ID and cluster ID. It is possible to click on each row. The `Curve` visualisation displays the TS of the cluster where the TS has been assigned. 

## Warnings
- The `qual_ref_period` and the `qual_nb_points` metadatas must be in every time serie's metadata. That means that the user has to run the
`Quality Indicators` operator before using this one.
- The time series must have the same period and the same number of points. They can have different start and end dates.

## Implementation remarks
- The Euclidian distance is used to affect the new time series. The prediction is made by finding the shortest centroids from the considered TS.

## Example
Return to the [list of all operators](https://ikats.org/operators.html)
