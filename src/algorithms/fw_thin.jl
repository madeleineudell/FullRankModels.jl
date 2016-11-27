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
    startobsj = cumsum(vcat(0, nobsj))
    nobs = sum(nobsj)
    # start_obsj = append!([0], cumsum(nobsj)[1:end-1])
    # end_obsj = start_obsj+nobsj
    obs = Array(Int, nobs)
    iobs = 1
    for j=1:n
        for i=gfrm.observed_examples[j]
            obs[iobs] = m*(j-1) + i
            iobs += 1
        end
    end
    indexing_operator = IndexingOperator(m, n, obs)
    @assert size(indexing_operator, 1) == nobs

    function f(X)
        #X = Array(X)
        obj = 0
        for j=1:n
            for (iobs,i) in enumerate(gfrm.observed_examples[j])
                obj += evaluate(gfrm.losses[j], X[i,j], gfrm.A[i,j])
            end
        end
        return obj
    end

    ## Grad of f
    function grad_f(X)
        #X = Array(X)
        g = Array(Float64,nobs)
        iobs = 1
        for j=1:n
            lj = gfrm.losses[j]
            for i=gfrm.observed_examples[j]
                g[iobs] = grad(lj, X[i,j], gfrm.A[i,j])
                iobs += 1
            end
        end
        # return G = A'*g as a sparse array
        return Array(IndexedLowRankOperator(indexing_operator, g))
    end

    const_nucnorm(X) = alpha # we'll always saturate the constraint, don't bother computing it
    # returns solution, optval of min <G, Delta> st ||Delta||_* \leq alpha
    function min_lin_st_nucnorm_sketched(G, alpha, tol::Float64 = fenchel_tol)
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
