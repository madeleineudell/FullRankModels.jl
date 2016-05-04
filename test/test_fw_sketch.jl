import LowRankModels: sort_observations
using LowRankModels, FullRankModels, Compat, Convex

srand(1)
# parameters
m, n, k = 30, 30, 3

# generate data
X0 = randn(m,k)
Y0 = randn(k,n)
A = X0*Y0 #+ .1*randn(m,n)
u,s,v = svd(A)
tau = sum(s)
P = sparse(float(randn(m,n) .>= .5)) # observed entries

I,J = findn(P) # observed indices (vectors)
@compat obs = Tuple{Int,Int}[(I[a],J[a]) for a = 1:length(I)]
observed_features, observed_examples = sort_observations(obs,size(P)...)

losses = fill(QuadLoss(1), n)
reg = TraceNormConstraint(tau)
gfrm = GFRM(A, losses, reg, k, observed_features, observed_examples, 
	zeros(m,n), zeros(m+n,m+n))

X_sketched, ch = fit!(gfrm, FrankWolfeParams())
Xhat = Array(X_sketched)

# compare with SDP solver
U = Variable(m,n)
W = Variable(m+n, m+n)

obj = sumsquares(P.*(U-A))
p = minimize(obj, 
	nuclearnorm(U) <= tau)
solve!(p)

println("mean square error is ", vecnorm(U.value - Xhat) / vecnorm(U.value))
U.value = Xhat
println("objective of Convex problem evaluated at frank wolfe solution is $(Convex.evaluate(obj))
")