A = sprand(100, 100, 0.2)
Am = Matrix(A)
l = QuadLoss()
rx = QuadReg()
ry = QuadReg()
obs = [(f[1], f[2]) for f in zip(findnz(A)[1], findnz(A)[2])]

gg = GGLRM(A, l, rx, ry, 10, obs=obs)
ggs = GGLRM(A, l, rx, ry, 10, obs=obs)

fit!(gg); fit_sparse!(ggs);
ggobj = whole_objective(gg, gg.X'gg.Y)
ggsobj = whole_objective(ggs, ggs.X'ggs.Y)
@test_approx_eq_eps(ggobj, ggsobj, 0.1*length(A))
