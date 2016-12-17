import StatsBase: sample, WeightVec

# test logistic loss

## generate data
srand(1);
m,n,k = 1000,1000,3;
kfit = k+1
# variance of measurement
sigmasq = .1

# coordinates of covariates
X_real = randn(m,k)
# directions of observations
Y_real = randn(k,n)

XY = X_real*Y_real;
A = zeros(Int, (m, n))
logistic(x) = 1/(1+exp(-x))
for i=1:m
	for j=1:n
		A[i,j] = (logistic(XY[i,j]) >= rand()) ? true : false
	end
end

# and the model
losses = LogisticLoss()
rx, ry = QuadReg(.1), QuadReg(.1);
glrm = GLRM(A,losses,rx,ry,kfit)
gg = GGLRM(A, losses, rx, ry, kfit)
fit!(glrm);
fit!(gg);
@test_approx_eq_eps(whole_objective(gg, gg.X'gg.Y), whole_objective(gg, glrm.X'glrm.Y), 0.1*length(A))
