#export impute_means, impute_zeros,
#      #Closed-form factorizations with regularization
#      matrixRegFact, quadgraphRegFact

# Subtracts each column's mean (if center=true),
# divides by each column's standard deviation (if scale=true).
# Returns (scaledData, mean, std), where mean or std may be
#  empty matrices if center or scale are false
function standardize!{T}(X::Matrix{T} ; center=true, scale=true)
    n = size(X,1)

    local m
    if center
        m = mean(X,1)
        X = X .- m
    else
        m = similar(X, 0, 0)
    end

    local s
    if scale
        s = std(X,1)
        X = X ./ s
    else
        s = similar(X, 0, 0)
    end

    X, m, s
end

##Subtracts each column mean (if center=true)
##Divides each column standard deviation (if scale=true)
##Skips NA entries in DataArrays
##Returns (scaledData, mean, std)
function standardize!{T}(X::DataMatrix{T}; center=true, scale=true)
  n = size(X,1)

  local m
  if center
    m = mean(X,1,skipna=true)
    X = X .- m
  else
    m = similar(X,0,0)
  end

  local s
  if scale
    s = sqrt(var(X,1,skipna=true))
    X = X ./ s
  else
    s = similar(X,0,0)
  end

  X, m, s
end

standardize(X; center=true, scale=true) = standardize!(copy(X), center=center, scale=scale)

##Imputes the column means to a DataMatrix and returns an ordinary Matrix
##of the same size.
function impute_means{T}(X::DataMatrix{T})
  m, n = size(X)
  colmeans = mean(X,1,skipna=true)
  Xmat = zeros(m,n)
  for j in 1:n
    #For entirely NA columns, just use zero
    if all(isna(X[:,j]))
      Xmat[:,j] = 0
    else
      #Impute either the mean or the existing value
      for i in 1:m
        if X[i,j] === NA
          Xmat[i,j] = colmeans[j]
        else
          Xmat[i,j] = X[i,j]
        end
      end
    end
  end
  Xmat
end

##Impute zeros on every NA value of a DataMatrix
function impute_zeros{T}(X::DataMatrix{T})
  m, n = size(X)
  Xmat = zeros(m,n)
  for j in 1:n
    for i in 1:m
      if !(X[i,j] === NA)
        Xmat[i,j] = X[i,j]
      end
    end
  end
  Xmat
end

##Closed-form solution to a quadratic loss function with graph+quadratic
##regularization on both X and Y. Performance may be poor for large number of rows
function matrixRegFact(A::AbstractMatrix, k::Int,
  α::Number, K::AbstractMatrix,
  β::Number, L::AbstractMatrix)

  n, d = size(A)
  @assert (n,n) == size(K)
  @assert (d,d) == size(L)
  iKA = (I + α*K) \ A # (I + αK)⁻¹ * A
  iKAt = iKA' # A'*(I + αK)⁻¹
  fst = -2*A'*iKA
  snd = iKAt*iKA
  thd = α*iKAt*K*iKA
  res = β*L
  for i in 1:length(res)
    res[i] += fst[i] + snd[i] + thd[i] #Add the results in a loop to save space
  end
  #=
  invIαK = inv(I + α*K)
  #Three terms to find the eigenvectors of, save time by adding in a loop
  fst = -2*A'*invIαK*A
  snd = A'*invIαK^2*A
  thd = α*A'*invIαK*K*invIαK*A
  res = β*L
  for i in 1:length(res)
    res[i] += fst[i] + snd[i] + thd[i]
  end
  =#
  Yt = eigvecs(res)[:,1:k]
  X = invIαK*A*Yt
  X,Yt'
end

##Closed-form solution to quadratic loss function with quadratic regularization
##on X and a graph+quadratic regularizer on Y.
function quadgraphRegFact(A::Matrix{Float64}, k::Int, α::Number,
  β::Number, L::AbstractMatrix)

  n, d = size(A)
  @assert (d,d) == size(L)

  ALmat = -1/(1+α)*A'*A + β*L
  Yt = eigvecs(ALmat)[:,1:k]
  X = 1/(1 + α)*A*Yt
  X,Yt'
end
