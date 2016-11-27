#using LowRankModels
#import LowRankModels: ObsArray, sort_observations, observations
#export GraphGLRM

type GGLRM <: AbstractGLRM
  A::AbstractArray{Float64,2}  # The data table
  losses::Array{Loss,1}        # array of loss functions
  rx::Regularizer              # Regularizer to apply to each row of X
  ry::Dict{Regularizer, Union{Int, Array{Int,1}}} # Mapping of regularizers to column of Y
  k::Int                       # Desired rank
  observed_features::ObsArray  # for each example, an array telling which features were observed
  observed_examples::ObsArray  # for each feature, an array telling in which examples the feature was observed
  X::AbstractArray{Float64,2}  # Representation of data in low-rank space. A ≈ X'Y
  Y::AbstractArray{Float64,2}  # Representation of features in low-rank space. A ≈ X'Y
end

function checkGraphDims(ry::Dict, losses::Array)
  for r in keys(ry)
    #check that only a multi-column regularizer is mapped to more than one index
    if !isa(r, MatrixRegularizer)
      if !isa(ry[r], Number)
        error("Only graph regularizers can be mapped to multiple columns")
      end
    end
    if isa(r, GraphQuadReg)
      p = length(ry[r])
      if (p,p) != size(matrix(r))
        error("Graph regularizer on columns $(ry[r]) must have same number of nodes")
      end
      #check that the graph regularized columns are not embedded in more than one dimension
      for c in ry[r]
        if embedding_dim(losses[c]) > 1
          error("Cannot have a graph regularizer on a multidimensional loss (column $c with loss $(losses[c]))")
        end
      end
    end
  end
end

function checkLossDims(losses::Array, n)
  if length(losses)!=n error("There must be as many losses as there are columns in the data matrix") end
end

# fill an array of length n with copies of the object foo
fillcopies(foo, n::Int; arraytype=typeof(foo)) = arraytype[copy(foo) for i=1:n]

function GGLRM(A::AbstractMatrix, losses::Array, rx::Regularizer, ry::Dict, k::Int;
          X = randn(k, size(A,1)), Y = randn(k, embedding_dim(losses)),
          obs = nothing,                                    # [(i₁,j₁), (i₂,j₂), ... (iₒ,jₒ)]
          observed_features = fill(1:size(A,2), size(A,1)), # [1:n, 1:n, ... 1:n] m times
          observed_examples = fill(1:size(A,1), size(A,2)), # [1:m, 1:m, ... 1:m] n times)# [(i₁,j₁), (i₂,j₂), ... (iₒ,jₒ)]
          offset=false, scale=false)

  n, d = size(A)
  if isa(rx, GraphQuadReg)
    #A graph regularizer on X should be able to multiply with X
    if (n,n) != size(matrix(rx))
      error("Graph regularizer on X must have as many nodes as there are rows in X")
    end
  end

  if size(X)!=(k,n) error("X must be of size (k,m) where m is the number of rows in the data matrix. This is the transpose of the standard notation used in the paper, but it makes for better memory management. \nsize(X) = $(size(X)), size(A) = $(size(A)), k = $k") end
  if size(Y)!=(k,sum(map(embedding_dim, losses))) error("Y must be of size (k,d) where d is the sum of the embedding dimensions of all the losses. \n(1 for real-valued losses, and the number of categories for categorical losses).") end

  #Make sure there are as many losses as columns
  checkLossDims(losses, d)

  #make sure that every column has one and only one regularizer on it
  if (length(union(values(ry)...)) != sum(map(length, values(ry)))) || (sum(map(length, values(ry))) != d)
    error("Each column of Y must have one and only one regularizer targeting it")
  end

  #Check that all multi-column regularizers are of the same size
  checkGraphDims(ry, losses)

  if obs==nothing # if no specified array of tuples, use what was explicitly passed in or the defaults (all)
  # println("no obs given, using observed_features and observed_examples")
    glrm = GGLRM(A,losses,rx,ry,k, observed_features, observed_examples, X,Y)
  else
    glrm = GGLRM(A,losses,rx,ry,k, sort_observations(obs,size(A)...)..., X,Y)
  end
  if scale # scale losses (and regularizers) so they all have equal variance
      equilibrate_variance!(glrm)
  end
  if offset # don't penalize the offset of the columns
      add_offset!(glrm)
  end
  glrm
end

function GGLRM(A::AbstractMatrix, loss::Loss, rx::Regularizer, ry::Regularizer, k::Int; kwargs...)
  losses = [copy(loss) for i in 1:size(A,2)]
  rys = Dict(copy(ry) => i for i in 1:size(A,2))
  GGLRM(A, losses, rx, rys, k, kwargs...)
end

function GGLRM(A::AbstractMatrix, losses::Array, rx::Regularizer, ry::Regularizer, k::Int; kwargs...)
  rys = Dict(copy(ry) => i for i in 1:size(A,2))
  GGLRM(A, losses, rx, rys, k, kwargs...)
end

function GGLRM(A::AbstractMatrix, loss::Loss, rx::Regularizer, ry::Dict, k::Int; kwargs...)
  losses = [copy(loss) for i in 1:size(A,2)]
  GGLRM(A, losses, rx, ry, k, kwargs...)
end
