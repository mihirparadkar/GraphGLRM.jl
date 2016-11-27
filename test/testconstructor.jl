using GraphGLRM, LowRankModels

A = randn(10,5)
rx = QuadReg()
ry = Dict(QuadReg() => i for i in 1:5)
losses = [QuadLoss() for i in 1:5]

g = GGLRM(A, losses, rx, ry, 5)
