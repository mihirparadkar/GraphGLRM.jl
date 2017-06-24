mutable struct NonNegGraphReg <: AbstractGraphReg
  L::AbstractMatrix{Float64}
  scale::Float64
  idxgraph::IndexGraph
end

#Retrieve the matrix component of the regularizer for use in initialization
matrix(g::NonNegGraphReg) = g.L

#
function NonNegGraphReg(IG::IndexGraph, scale::Float64=1.)
  L = laplacian_matrix(IG.graph)
  NonNegGraphReg(L, scale, IG)
end

function prox(g::NonNegGraphReg, Y::AbstractMatrix{Float64}, α::Number)
  #Y*(2α*g.scale*g.QL + eye(g.QL))⁻¹
  #g.QL is guaranteed to be sparse and symmetric positive definite
  #Factorize (2α*g.scale*g.QL + I)
  L = Symmetric((2α*g.scale)*g.L)
  max(A_ldiv_Bt(cholfact(L, shift=1.), Y)', 0)
end

function prox!(g::NonNegGraphReg, Y::AbstractMatrix{Float64}, α::Number)
  #Y*(2α*g.scale*g.QL + eye(g.QL))⁻¹
  #g.QL is guaranteed to be sparse and symmetric positive definite
  #Factorize (2α*g.scale*g.QL + I)
  L = Symmetric((2α*g.scale)*g.L)
  #invQLpI = cholfact(QL, shift=1.) \ eye(QL)
  #Y*invQLpI
  transpose!(Y, A_ldiv_Bt(cholfact(L, shift=1.), Y))
  Ymax = maximum(Y)
  clamp!(Y, 0, Ymax)
end

function evaluate(g::NonNegGraphReg, Y::AbstractMatrix{Float64})
  for yi in Y
    if yi < 0
      return Inf
    end
  end
  g.scale*sum((Y'*Y) .* g.L)
end

function embed(g::NonNegGraphReg, yidxs::Array)
  NonNegGraphReg(embed_graph(g.idxgraph, yidxs), g.scale)
end
