### Fit a full rank model with thin Frank Wolfe

# todo: make it work for mpca

import FirstOrderOptimization: frank_wolfe_thin, FrankWolfeParams, DecreasingStepSize,
                               IndexingOperator, LowRankOperator, mysvds
import LowRankModels: fit!, ConvergenceHistory, get_yidxs, grad, evaluate
import Base: axpy!, scale!

export fit_thin!, FrankWolfeParams

ZeroOp(m::Int,n::Int) = LowRankOperator(
    tuple(randn(m,1), sparse([1],[1],[0.],1,1), randn(n,1)),
    (:N,:N,:T))

### FITTING
function fit_thin!(gfrm::GFRM, params::FrankWolfeParams = FrankWolfeParams();
			  ch::ConvergenceHistory=ConvergenceHistory("FrankWolfeGFRM"),
        X::LowRankOperator=ZeroOp(size(gfrm.A)...), # starting point
        fenchel_tol::Float64 = 1e-5,
			  verbose=true,
			  kwargs...)

    if !isa(gfrm.r, TraceNormConstraint)
        error("Frank Wolfe fitting is only implemented for trace norm constrained problems")
    end

    # the functions below close over all this problem data
    yidxs = get_yidxs(gfrm.losses)
    d = maximum(yidxs[end])
    m,n = size(gfrm.A)
    alpha = gfrm.r.scale

    nobsj = map(length, gfrm.observed_examples)
    startobsj = cumsum(vcat(1, nobsj))
    nobs = sum(nobsj)
    obs = Array(Int, nobs)

    Threads.@threads for j=1:n
        obs[startobsj[j]:(startobsj[j+1]-1)] = m*(j-1) + gfrm.observed_examples[j]
    end
    indexing_operator = IndexingOperator(m, n, obs)
    @assert size(indexing_operator, 1) == nobs

    objs = zeros(Threads.nthreads())
    function f(X::LowRankOperator)
        scale!(objs, 0)
        # println("obj")
        # @time
        Threads.@threads for j=1:n
            objs[Threads.threadid()] += evaluate(gfrm.losses[j],
                         Float64[X[i,j] for i in gfrm.observed_examples[j]],
                         gfrm.A[gfrm.observed_examples[j],j])
        end
        return sum(objs)
    end

    ## Grad of f
    g = Array(Float64,nobs) # working variable for computing gradient; grad_f mutates this
    function grad_f(X::LowRankOperator)
        Threads.@threads for j=1:n
            ii = startobsj[j]:(startobsj[j+1]-1)
            g[ii] = grad(gfrm.losses[j],
                         Float64[X[i,j] for i in gfrm.observed_examples[j]],
                         gfrm.A[gfrm.observed_examples[j],j])
        end
        # return G = A'*g as a sparse matrix
        # @show g
        G = Array(IndexedLowRankOperator(indexing_operator, g))
        return G
    end

    const_nucnorm(X) = alpha # we'll always saturate the constraint, don't bother computing it
    # returns solution, optval of min <G, Delta> st ||Delta||_* \leq alpha
    function min_lin_st_nucnorm_sketched(G::SparseMatrixCSC, alpha, tol::Float64 = fenchel_tol)
        u,s,v = mysvds(G, nsv=1, tol=tol)
        return LowRankOperator(-alpha*u, v, transpose = (:N, :T)), -alpha*s[1]
    end

    # recover
    t = time()
    X = frank_wolfe_thin(
        X,
        f, grad_f,
        const_nucnorm,
        alpha,
        min_lin_st_nucnorm_sketched,
        params,
        ch,
        verbose=verbose
        )

    t = time() - t
    update_ch!(ch, t, f(X))

    gfrm.W = X

    return X, ch
end
