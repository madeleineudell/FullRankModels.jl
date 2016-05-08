### Fit a full rank model with sketched Frank Wolfe

# todo: make it work for mpca

import FirstOrderOptimization: frank_wolfe_sketched, FrankWolfeParams, DecreasingStepSize,
                               AsymmetricSketch, IndexingOperator, LowRankOperator
import LowRankModels: fit!, ConvergenceHistory, get_yidxs, grad, evaluate
import Base: axpy!, scale!

export fit!, FrankWolfeParams

### FITTING
function fit!(gfrm::GFRM, params::FrankWolfeParams = FrankWolfeParams();
			  ch::ConvergenceHistory=ConvergenceHistory("FrankWolfeGFRM"), 
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

    nobs = sum(map(length, gfrm.observed_examples))
    obs = Array(Int, nobs)
    iobs = 1
    for i in 1:m
        for j in gfrm.observed_features[i]
            obs[iobs] = m*(j-1) + i
            iobs += 1
        end
    end
    indexing_operator = IndexingOperator(m, n, obs)

    function f(z)
        obj = 0
        iobs = 1
        for i=1:m
            for j=gfrm.observed_features[i]
                obj += evaluate(gfrm.losses[j], z[iobs], gfrm.A[i,j])
                iobs += 1
            end
        end
        return obj
    end

    ## Grad of f
    function grad_f(z)
        g = zeros(size(z))
        iobs = 1
        for i=1:m
            for j=gfrm.observed_features[i]
                g[iobs] = grad(gfrm.losses[j], z[iobs], gfrm.A[i,j])
                iobs += 1
            end
        end
        # return G = A'*g
        return LowRankOperator(indexing_operator, g, transpose = Symbol[:T, :N])
    end

    const_nucnorm(z) = alpha # we'll always saturate the constraint, don't bother computing it
    # returns solution, optval of min <G, Delta> st ||Delta||_* \leq alpha
    function min_lin_st_nucnorm_sketched(G, alpha)
        u,s,v = svds(Array(G), nsv=1) # for this case, g is a sparse matrix so representing it is O(m)
        return LowRankOperator(-alpha*u, v')
    end

    # initialize
    z = zeros(nobs)

    # recover
    t = time()
    X_sketched = frank_wolfe_sketched(
        z,
        f, grad_f,
        const_nucnorm,
        alpha,
        min_lin_st_nucnorm_sketched,
        AsymmetricSketch(m,n,gfrm.k),
        params,
        ch,
        LB = 0,
        verbose=true
        )

    t = time() - t
    update_ch!(ch, t, f(z))

    gfrm.W = X_sketched

    return X_sketched, ch
end
