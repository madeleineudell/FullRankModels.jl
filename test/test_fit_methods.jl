import LowRankModels: sort_observations
import FirstOrderOptimization: HopefulStepSize, DecreasingStepSize
using LowRankModels, FullRankModels, Compat
using Convex

srand(1)
# parameters
m, n, k = 5, 4, 2

# generate data
X0 = randn(m,k)
Y0 = randn(k,n)
A = X0*Y0 #+ .1*randn(m,n)
u,s,v = svd(A)
tau = sum(s)
P = sparse(float(randn(m,n) .>= .1)) # observed entries

I,J = findn(P) # observed indices (vectors)
@compat obs = Tuple{Int,Int}[(I[a],J[a]) for a = 1:length(I)]
observed_features, observed_examples = sort_observations(obs,size(P)...)

losses = fill(QuadLoss(1), n)
reg = TraceNormConstraint(tau)
gfrm1 = GFRM(A, losses, reg, k, observed_features, observed_examples,
	zeros(m,n), zeros(m+n,m+n))
gfrm2 = gfrm1

params = FrankWolfeParams(maxiters = 10, reltol = 1e-10, stepsize = DecreasingStepSize());

# println("frank wolfe")
#
# @time X_fw, ch = fit_fw!(gfrm2, copy(params))
# @time X_fw, ch = fit_fw!(gfrm2, copy(params))

println("thin frank wolfe")

@time X_thin, ch = fit_thin!(gfrm2, copy(params))
# @time X_thin, ch = fit_thin!(gfrm2, copy(params))

println("sketched frank wolfe")

@time X_sketched, ch = fit_sketch!(gfrm1, copy(params))
# @time X_sketched, ch = fit_sketch!(gfrm1, copy(params))

# @show norm(Array(X_fw) - Array(X_sketched))
@show norm(Array(X_thin) - Array(X_sketched))
