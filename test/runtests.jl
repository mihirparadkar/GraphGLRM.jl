using GraphGLRM, LowRankModels
using Base.Test

# write your own tests here
include("realvalueglrm.jl")
include("classificationglrm.jl")
include("multidimglrm.jl")
include("sparseglrm.jl")

#Graph regularizer
A1 = randn(6,2) * diagm([128, 16])
A2 = 2*A1 + randn(6,2)
A = [A1 A2]
A /= vecnorm(A)/√(8)
ig = IndexGraph(1:4, [(1,3),(2,4)])
GRG = GGLRM(A, QuadLoss(), QuadReg(0.02), GraphQuadReg(ig, 2., 0.01), 2)
GRGM = GGLRM(A, QuadLoss(), QuadReg(0.02), GraphQuadReg(ig, 2., 0.01), 2, X=GRG.X, Y=GRG.Y)
NGRG = GGLRM(A, QuadLoss(), NonNegConstraint(), NonNegGraphReg(ig, 2.), 2)
fit!(GRG)
fit_multithread!(GRGM)
fit!(NGRG)
GRGobj = whole_objective(GRG, GRG.X'GRG.Y)
GRGMobj = whole_objective(GRGM, GRGM.X'GRGM.Y)
@test ≈(GRGobj, GRGMobj, atol=0.1*length(A))
