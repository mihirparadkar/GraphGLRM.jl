#Helper function for a regularizer that's actually a constraint
function inf_or_zero(r::Regularizer, a::AbstractMatrix)
  for i in 1:size(a,2)
    if evaluate(r, a[:,i]) === Inf
      return Inf
    else
      continue
    end
  end
  return 0
end

## Takes a prox step for each column, mutable method
function colwise_prox!(r::Regularizer, u::AbstractMatrix, alpha::Number)
  for i in 1:size(u,2)
    prox!(r, view(u, :, i), alpha)
  end
  u
end

## Immutable wrapper around colwise_prox!
function colwise_prox(r::Regularizer, u::AbstractMatrix, alpha::Number)
  v = copy(u)
  colwise_prox!(r, v, alpha)
  v
end

## Unit one-sparse constraint
##
function evaluate(r::UnitOneSparseConstraint, a::AbstractMatrix)
  inf_or_zero(r, a)
end

function prox!(r::UnitOneSparseConstraint, u::Matrix, alpha::Number)
  colwise_prox!(r, u, alpha)
end

function prox(r::UnitOneSparseConstraint, u::AbstractMatrix, alpha::Number)
  colwise_prox(r, u, alpha)
end


## K-sparse constraint
##
function evaluate(r::KSparseConstraint, a::AbstractMatrix)
  inf_or_zero(r, a)
end

function prox!(r::KSparseConstraint, u::Matrix, alpha::Number)
  colwise_prox!(r, u, alpha)
end

function prox(r::KSparseConstraint, u::AbstractMatrix, alpha::Number)
  colwise_prox(r, u, alpha)
end


## One-sparse constraint
##
function evaluate(r::OneSparseConstraint, a::AbstractMatrix)
  inf_or_zero(r, a)
end

function prox!(r::OneSparseConstraint, u::Matrix, alpha::Number)
  colwise_prox!(r, u, alpha)
end

function prox(r::OneSparseConstraint, u::AbstractMatrix, alpha::Number)
  colwise_prox(r, u, alpha)
end
