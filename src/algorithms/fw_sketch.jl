### Fit a full rank model with sketched Frank Wolfe

# todo: make it work for mpca

import FirstOrderOptimization: frank_wolfe_sketched, FrankWolfeParams, DecreasingStepSize,
                               AbstractSketch, AsymmetricSketch, 
                               IndexingOperator, LowRankOperator
import LowRankModels: fit!, ConvergenceHistory, get_yidxs, grad, evaluate
import Base: axpy!, scale!

export fit!, FrankWolfeParams

### FITTING
function fit!(gfrm::GFRM, params::FrankWolfeParams = FrankWolfeParams();
			  ch::ConvergenceHistory=ConvergenceHistory("FrankWolfeGFRM"), 
              z::AbstractVector = zeros(sum(map(length, gfrm.observed_examples))),
              sketch::AbstractSketch = AsymmetricSketch(size(gfrm.A)..., gfrm.k),
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
    @assert length(z) == nobs

    function f(z)
        obj = 0
        iobs = 1
        for j=1:n
            lj = gfrm.losses[j]
            for i=gfrm.observed_examples[j]
                obj += evaluate(lj, z[iobs], gfrm.A[i,j])
                iobs += 1
            end
        end
        return obj
    end

    ## Grad of f
    function grad_f(z; 
                    g = Array(Float64, size(z)))
        iobs = 1
        for j=1:n
            lj = gfrm.losses[j]
            for i=gfrm.observed_examples[j]
                g[iobs] = grad(lj, z[iobs], gfrm.A[i,j])
                iobs += 1
            end
        end
        # return G = A'*g
        return LowRankOperator(indexing_operator, g, transpose = Symbol[:T, :N])
    end

    const_nucnorm(z) = alpha # we'll always saturate the constraint, don't bother computing it
    # returns solution, optval of min <G, Delta> st ||Delta||_* \leq alpha
    function min_lin_st_nucnorm_sketched(G, alpha)
        ga = Array(G)
        u,s,v = svds(ga, nsv=1)
        return LowRankOperator(-alpha*u, v')
    end

    # recover
    t = time()
    X_sketched = frank_wolfe_sketched(
        z,
        f, grad_f,
        const_nucnorm,
        alpha,
        min_lin_st_nucnorm_sketched,
        sketch,
        params,
        ch,
        verbose=verbose
        )

    t = time() - t
    update_ch!(ch, t, f(z))

    gfrm.W = X_sketched

    return X_sketched, ch
end
