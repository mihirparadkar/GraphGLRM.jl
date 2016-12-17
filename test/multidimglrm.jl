# tests MNL loss

srand(1);
m,n,k = 200,50,2;
kfit = k+1
K = 4; # number of categories
d = n*K;
# matrix to encode
X_real, Y_real = randn(m,k), randn(k,d);
XY = X_real*Y_real;
# subtract the mean so we can compare the truth with the fit;
# the loss function is invariant under shifts
losses = fill(MultinomialLoss(K),n)
yidxs = get_yidxs(losses)
for i=1:m
	for j=1:n
		mef = mean(XY[i,yidxs[j]])
		XY[i,yidxs[j]] = XY[i,yidxs[j]] - mef
	end
end

A = zeros(Int, (m, n))
for i=1:m
	for j=1:n
		wv = WeightVec(Float64[exp(-XY[i, K*(j-1) + l]) for l in 1:K])
		l = sample(wv)
		A[i,j] = l
	end
end

# and the model
losses = Loss[fill(MultinomialLoss(K),n)...]
rx, ry = QuadReg(), QuadReg();
glrm = GLRM(A,losses,rx,ry,kfit, scale=false, offset=false, X=randn(kfit,m), Y=randn(kfit,d));
gg = GGLRM(A, losses, rx, ry, kfit, X=randn(kfit,m), Y=randn(kfit,d))
fit!(glrm);
fit!(gg);
@test_approx_eq_eps(whole_objective(gg, gg.X'gg.Y), whole_objective(gg, glrm.X'glrm.Y), 0.1*length(A))
