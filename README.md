# GraphGLRM

Graph Regularized Low Rank Models (http://github.com/madeleineudell/LowRankModels.jl)
Allows for the specification of graphs representing relation between different examples or features.

# Overview

Most data analysis techniques (linear models, nearest-neighbor methods, and other regressors and classifiers)
make assumptions about the nature of the data at hand. First, it is often assumed that the features (predictor variables, or columns of the dataset)
are real numbers. Second, these techniques assume that all of the values in a dataset are observed, so missing values are
usually discarded or crudely imputed (for example, using the mean of a column). Furthermore, they almost universally work their best
when the number of examples (rows of the dataset) is substantially greater than the number of features.

However, real-world datasets frequently have the problems of being heterogeneous, sparse, and high-dimensional. Columns may be
binary ("yes" or "no"), nominal (an element of a set of possible values), or ordinal (varying levels where the distances may be unknown, such as "very high", "high",...),
not just real numbers. Some or most of the values may be missing, corrupted, or unknown. There may be more features than examples, posing large problems for
most ordinary data mining techniques.

The traditional approach to these problems is often crude and ignores the structure of the data. Nominal or ordinal variables may be encoded as integers, despite
there being no implicit ordering to these variables. The column mean or mode may be imputed to missing values, completely ignoring the information from other features.
After these ad-hoc transformations, a method like Principal Component Analysis may be applied, despite its poor performance on data that is not normally distributed.

The Generalized Low-Rank Model (GLRM) aims to rectify all of these issues, by allowing a flexible approach to imputation and dimensionality reduction. It achieves
this through awareness of missing and heterogeneous data, by optimizing different distance metrics (loss functions) over only the observed entries, where the distance
metrics may be different over each column. By using a combination of these loss functions to model the different data types, the GLRM can find a low-rank representation of
a dataset that is composed of two real-valued matrices, easing the usage of other supervised methods. Furthermore, by reconstructing the approximation of the dataset using the
low-rank factors, missing values can be imputed in a manner dependent on their type, often much more accurately than imputing the mean. Furthermore, by regularizing or constraining
the model, overfitting can be reduced, preventing the model from specializing too much on a small subset of observed data.

GraphGLRMs take this concept even further and allow specification of relationships between rows and/or columns, allowing a data scientist to provide extra information about
the dataset. By specifying these relationships, entire columns or rows of missing values can "borrow strength" from other features or examples to provide a more accurate reconstruction.
This package provides the necessary extensions on the original LowRankModels to combine graph information with other regularizers and losses.

# IndexGraphs
This maps the indices of a matrix to their nodes in a graph (LightGraphs.jl).
This graph is then used to compute the Laplacian matrix for use in the graph regularizer.
Usage:
```julia
IndexGraph(nodes, edges)
```
Example:
```julia
using GraphGLRM
ig = IndexGraph([1,2,3,4,5], [(1,2), (1,3), (1,4), (1,5)])
```

# GraphQuadReg
A combined graph-Laplacian and quadratic regularizer for use in a Low Rank Model.
Usage:
```julia
GraphQuadReg(ig, graphscale, quadamt)
```
Example:
```julia
gq = GraphQuadReg(ig, 2., 0.05)
```
It is highly recommended to use at least a small amount of additional quadratic
regularization to ensure convergence because the laplacian matrix may not be
positive definite.

# NonNegGraphReg
A combined graph-Laplacian and non-negative constraint.
Usage:
```julia
nng = NonNegGraphReg(ig, 2.)
```

# GGLRM
In the style of most dimensionality reduction algorithms, the constructor
is an acronym (Graph Generalized Low Rank Model).
For a matrix with no missing data, usage is as follows:
```julia
gm = GGLRM(A, losses, rx, ry, k)
```
For a matrix with missing data, add an obs parameter to only optimize over
the observed set of data
```julia
gm = GGLRM(Amissing, losses, rx, ry, k, obs=observations(Amissing))
```
To fit this model, call:
```julia
fit!(gm, ProxGradParams(...))
```
