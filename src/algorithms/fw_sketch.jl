### Fit a full rank model with sketched Frank Wolfe

# todo: make it work for mpca

import FirstOrderOptimization: frank_wolfe_sketched, FrankWolfeParams, DecreasingStepSize,
                               AsymmetricSketch, IndexingOperator, LowRankOperator
import LowRankModels: fit!, ConvergenceHistory, get_yidxs, grad, evaluate
import Base: axpy!, scale!

export fit!, FrankWolfeParams

defaultFrankWolfeParams = FrankWolfeParams(100, 1e-2, DecreasingStepSize(2,1))

### FITTING
function fit!(gfrm::GFRM, params::FrankWolfeParams = defaultFrankWolfeParams;
			  ch::ConvergenceHistory=ConvergenceHistory("FrankWolfeGFRM"), 
			  verbose=true,
			  kwargs...)

    # the functions below close over all this problem data
    yidxs = get_yidxs(gfrm.losses)
    d = maximum(yidxs[end])
    m,n = size(gfrm.A)
    tau = gfrm.r.scale

    nobs = sum(map(length, gfrm.observed_examples))
    obs = Array(Int, nobs)
    iobs = 1
    for i=1:m
        for j=gfrm.observed_features[i]
            obs[iobs] = n*(j-1) + i
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
        return LowRankOperator(indexing_operator, g, transpose = Symbol[:T, :N])
    end

    const_nucnorm(z) = tau # we'll always saturate the constraint, don't bother computing it
    function min_lin_st_nucnorm_sketched(g, tau)
        u,s,v = svds(Array(g), nsv=1) # for this case, g is a sparse matrix so representing it is O(m)
        return LowRankOperator(-tau*u, v')
    end
    # I can't think of a pretty way to compute
    # <g, tilde_x - x> = <A'(z-b), u tau v' - x> = <z-b, A(u tau v') - A(x)> = <z-b, A(u tau v') - z>
    dot_g_w_tx_minus_x(g, Delta, z) = dot(vec(g.factors[2]), g.factors[1]*Delta - z)
    # we end up computing A*Delta twice with this scheme
    update_var!(z, Delta, a) = (scale!(z, 1-a); axpy!(a, indexing_operator*Delta, z))

    # initialize
    z = zeros(nobs)

    # recover
    t = time()
    X_sketched = frank_wolfe_sketched(
        z,
        f, grad_f,
        const_nucnorm,
        tau,
        min_lin_st_nucnorm_sketched,
        AsymmetricSketch(m,n,gfrm.k),
        FrankWolfeParams(100, 1e-2, DecreasingStepSize(2,1)),
        ch,
        dot_g_w_tx_minus_x = dot_g_w_tx_minus_x,
        update_var! = update_var!,
        LB = 0,
        verbose=true
        )

    t = time() - t
    update_ch!(ch, t, f(z))

    # gfrm.U = X_sketched

    return X_sketched, ch
end
