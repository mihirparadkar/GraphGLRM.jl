module GraphGLRM

using LowRankModels
using LightGraphs
using DataArrays
import LowRankModels: prox, prox!,
      add_offset!, equilibrate_variance!, prob_scale!,
      evaluate, ObsArray, sort_observations, observations,
      fit!, row_objective
import Base.BLAS: axpy!
import Base.Threads

export impute_means, impute_zeros, standardize, standardize!, #Simple imputation for pre-preprocessing
      matrixRegFact, quadgraphRegFact, init_qqreg!,#Closed-form factorizations with regularization
      IndexGraph, #The data structure for easily initializing a GraphQuadReg

      #The regularizer
      MatrixRegularizer, GraphQuadReg, matrix, prox, prox!, evaluate,

      #The constructors for the GGLRM itself
      GGLRM,

      #The offsets and scaling


      #The alternating minimization and objective calculation
      fit!, whole_objective

# package code goes here
include("indexgraph.jl")
include("graphquadreg.jl")
include("gglrm.jl")
include("initialize.jl")
include("offsetscale.jl")
include("objective.jl")
#include("fit.jl")
include("proxgrad.jl")

end # module
