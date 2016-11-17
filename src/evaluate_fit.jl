import LowRankModels: objective
import FirstOrderOptimization: LowRankOperator

export objective

## Objective evaluation
# we're not going to bother checking whether W is psd or not
# when evaluating the objective; in the course of the prisma
# algo this makes no difference
function objective(gfrm::GFRM, W::Array{Float64,2};
                   yidxs=get_yidxs(gfrm.losses),
                   include_regularization=true)
    # W is the symmetric parameter; U is the upper right block
    m,n = size(gfrm.A)
    UW = W[1:m, m+1:end]
    err = 0.0
    for j=1:n
        for i in gfrm.observed_examples[j]
            err += evaluate(gfrm.losses[j], UW[i,yidxs[j]], gfrm.A[i,j])
        end
    end
    if include_regularization
        err += evaluate(gfrm.r, W)
    end
    return err
end
function objective(gfrm::GFRM; kwargs...)
    objective(gfrm::GFRM, gfrm.W; kwargs...)
end

function objective(gfrm::GFRM, W::LowRankOperator;
                   yidxs=get_yidxs(gfrm.losses),
                   include_regularization=true)
    m,n = size(gfrm.A)
    err = 0.0
    for j=1:n
        for i in gfrm.observed_examples[j]
            err += evaluate(gfrm.losses[j], W[i,yidxs[j]], gfrm.A[i,j])
        end
    end
    if include_regularization
        err += evaluate(gfrm.r, W)
    end
    return err
end

# objective(gfrm::GFRM, args...; kwargs...) = objective(gfrm, gfrm.W, args...; kwargs...)
