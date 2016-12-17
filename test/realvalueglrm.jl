#Create a true low-rank model
Xtrue = randn(100,10)
Ytrue = randn(10, 60)
A = Xtrue*Ytrue + 0.01*randn(100,60)
obs = [(i,j) for i in 1:100 for j in 1:60 if !(j in [3,59])]

#Test that the results from LowRankModels are approximately equal to GraphGLRM
glrm_quad = GLRM(A, QuadLoss(), QuadReg(1.), QuadReg(1.), 5)
gg = GGLRM(A, QuadLoss(), QuadReg(1.), QuadReg(1.), 5)
fit!(gg); fit!(glrm_quad)
ggobj = whole_objective(gg, gg.X'*gg.Y)
glrmobj = whole_objective(gg, glrm_quad.X'*glrm_quad.Y)
@test_approx_eq_eps(ggobj, glrmobj, 0.1*length(A))

#Try for a different loss function and regularizers
glrm_hquad = GLRM(A, HuberLoss(), OneReg(1.), OneReg(1.), 5)
ggh = GGLRM(A, HuberLoss(), OneReg(1.), OneReg(1.), 5)
fit!(ggh); fit!(glrm_hquad)
gghobj = whole_objective(ggh, ggh.X'*ggh.Y)
glrmhobj = whole_objective(ggh, glrm_hquad.X'*glrm_hquad.Y)
@test_approx_eq_eps(gghobj, glrmhobj, 0.1*length(A))

#Make sure that it works with missing data
glrm_missing = GLRM(A, L1Loss(), NonNegConstraint(), NonNegConstraint(), 5, obs=obs)
gg_missing = GGLRM(A, L1Loss(), NonNegConstraint(), NonNegConstraint(), 5, obs=obs)
fit!(gg_missing); fit!(glrm_missing);
ggmobj = whole_objective(gg_missing, gg_missing.X'gg_missing.Y)
glrmmobj = whole_objective(gg_missing, glrm_missing.X'glrm_missing.Y)
@test_approx_eq_eps(ggmobj, glrmmobj, 0.1*length(A))
