# GraphGLRM

Graph Regularized Low Rank Models (http://github.com/madeleineudellLowRankModels.jl)
Allows for the specification of graphs representing relation between different examples or features.
This uses the graph-Laplacian matrix to regularize the thin factor, the wide factor, or both, with
any loss function. Additionally, the other supported regularizers are
QuadReg() (Quadratic Regularization), OneReg() (Absolute-value regularization), NonNegConstraint(), and NonNegOneReg()

This package also provides closed-form solutions to particular low-rank models, where the loss
is quadratic, the regularizer on X is either quadratic, graph-laplacian, or a combination of both, and Y is either
graph-regularized or graph and quadratic regularized.

# IndexGraphs
This maps the indices of a matrix to their nodes in a graph (LightGraphs.jl).
This graph is then used to compute the Laplacian matrix for use in the graph regularizer.
The index-to-node mapping is useful for future implementation of these algorithms on DataFrames.
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

# GGLRM
In the style of most dimensionality reduction algorithms, the constructor
is an acronym (Graph Generalized Low Rank Model).
For a matrix with no missing data, usage is as follows:
```julia
gm = GGLRM(A, loss, rx, ry, k)
```
For a matrix with missing data, add an obs parameter to only optimize over
the observed set of data
```julia
gm = GGLRM(Amissing, loss, rx, ry, k, obs=observations(Amissing))
```
To fit this model, call:
```julia
fit!(gm, ProxGradParams(...))
```
