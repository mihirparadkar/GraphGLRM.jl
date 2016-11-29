#using LowRankModels
#import LowRankModels: ObsArray, sort_observations, observations
#export GraphGLRM

type GGLRM <: AbstractGLRM
  A                            # The data table
  losses::Array{Loss,1}        # array of loss functions
  rx::Regularizer              # Regularizer to apply to each row of X
  ry::Regularizer              # Array of regularizers to be applied to each column of Y
  k::Int                       # Desired rank
  observed_features::ObsArray  # for each example, an array telling which features were observed
  observed_examples::ObsArray  # for each feature, an array telling in which examples the feature was observed
  X::AbstractArray{Float64,2}  # Representation of data in low-rank space. A ≈ X'Y
  Y::AbstractArray{Float64,2}  # Representation of features in low-rank space. A ≈ X'Y
end

function no_multidim_edges(ry::AbstractGraphReg, losses::Array)
  for (i,j) in edges(ry.idxgraph.graph)
    if (embedding_dim(losses[i]) > 1) || (embedding_dim(losses[j]) > 1)
      error("Graph regularizer cannot have any edges into or out of a multidimensional loss")
    end
  end
end

function GGLRM(A::AbstractMatrix, losses::Array, rx::Regularizer, ry::Regularizer, k::Int;
          X = randn(k, size(A,1)), Y = randn(k,embedding_dim(losses)),
          obs = nothing,                                    # [(i₁,j₁), (i₂,j₂), ... (iₒ,jₒ)]
          observed_features = fill(1:size(A,2), size(A,1)), # [1:n, 1:n, ... 1:n] m times
          observed_examples = fill(1:size(A,1), size(A,2))) # [1:m, 1:m, ... 1:m] n times)# [(i₁,j₁), (i₂,j₂), ... (iₒ,jₒ)]

  n, d = size(A)
  if isa(rx, AbstractGraphReg)
    #A graph regularizer on X should be able to multiply with X
    if (n,n) != size(matrix(rx))
      error("Graph regularizer on X must have as many nodes as there are rows in X")
    end
  end

  if isa(ry, AbstractGraphReg)
    #Same goes for Y
    if (d,d) != size(matrix(ry))
      error("Graph regularizer on Y must have as many nodes as there are columns in A")
    end
    no_multidim_edges(ry, losses)
    #ry = GraphQuadReg(embed_graph(ry.idxgraph, get_yidxs(losses)), ry.scale, ry.quadamt)
    ry = embed(ry, get_yidxs(losses))
  end

  if size(X) != (k, size(A,1))
    error("X must have the same number of columns as there are rows in A. This is the transpose of the monograph's description but makes for efficient memory access")
  end

  if size(Y)!=(k,sum(map(embedding_dim, losses)))
    error("Y must be of size (k,d) where d is the sum of the embedding dimensions of all the losses. \n(1 for real-valued losses, and the number of categories for categorical losses).")
  end

  if obs==nothing # if no specified array of tuples, use what was explicitly passed in or the defaults (all)
  # println("no obs given, using observed_features and observed_examples")
    glrm = GGLRM(A,losses,rx,ry,k, observed_features, observed_examples, X,Y)
  else
    glrm = GGLRM(A,losses,rx,ry,k, sort_observations(obs,size(A)...)..., X,Y)
  end
  glrm
end

function GGLRM(A::AbstractMatrix, loss::Loss, rx::Regularizer, ry::Regularizer, k::Int; kwargs...)
  losses = [copy(loss) for i in 1:size(A,2)]
  GGLRM(A, losses, rx, ry, k, kwargs...)
end
