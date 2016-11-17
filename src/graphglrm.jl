using LowRankModels
import LowRankModels: ObsArray, sort_observations, observations
export GraphGLRM

type GraphGLRM <: AbstractGLRM
  A::AbstractArray{Float64,2}  # The data table
  loss::Loss                   # array of loss functions
  rx::Regularizer              # Regularizer to apply to each row of X
  ry::Regularizer              # Array of regularizers to be applied to each column of Y
  k::Int                       # Desired rank
  observed_features::ObsArray  # for each example, an array telling which features were observed
  observed_examples::ObsArray  # for each feature, an array telling in which examples the feature was observed
  X::AbstractArray{Float64,2}  # Representation of data in low-rank space. A ≈ X'Y
  Y::AbstractArray{Float64,2}  # Representation of features in low-rank space. A ≈ X'Y
end

function GraphGLRM(A::AbstractMatrix, loss::Loss, rx::Regularizer, ry::Regularizer, k::Int;
          X = randn(size(A,1), k), Y = randn(k, size(A,2)),
          obs = nothing,                                    # [(i₁,j₁), (i₂,j₂), ... (iₒ,jₒ)]
          observed_features = fill(1:size(A,2), size(A,1)), # [1:n, 1:n, ... 1:n] m times
          observed_examples = fill(1:size(A,1), size(A,2))) # [1:m, 1:m, ... 1:m] n times)# [(i₁,j₁), (i₂,j₂), ... (iₒ,jₒ)]

  n, d = size(A)
  if isa(rx, GraphQuadReg)
    #A graph regularizer on X should be able to multiply with X
    if (n,n) != size(rx.QL)
      error("Graph regularizer on X must have as many nodes as there are rows in X")
    end
  end

  if isa(ry, GraphQuadReg)
    #Same goes for Y
    if (d,d) != size(ry.QL)
      error("Graph regularizer on Y must have as many nodes as there are columns in Y")
    end
  end

  if obs==nothing # if no specified array of tuples, use what was explicitly passed in or the defaults (all)
  # println("no obs given, using observed_features and observed_examples")
    glrm = GraphGLRM(A,loss,rx,ry,k, observed_features, observed_examples, X,Y)
  else
    glrm = GraphGLRM(A,loss,rx,ry,k, sort_observations(obs,size(A)...)..., X,Y)
  end
  glrm
end
