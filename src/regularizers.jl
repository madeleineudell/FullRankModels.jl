export ProductRegularizer, MaxNormReg, TraceNormReg,

abstract ProductRegularizer<:Regularizer
prox(r::ProductRegularizer,W::AbstractArray,alpha::Number) = (Wc = copy(W); prox!(r,Wc,alpha))

# Max norm
type MaxNormReg<:ProductRegularizer
    scale::Float64
end
scale(r::MaxNormReg) = r.scale
scale!(r::MaxNormReg, newscale::Number) = (r.scale = newscale)

function evaluate(r::MaxNormReg, W::AbstractArray)
    r.scale*maximum(diag(W))
end

function prox!(r::MaxNormReg, W::AbstractArray, alpha::Number)
    oldmax = maximum(diag(W))
    newmax = oldmax - r.scale*alpha/2
    for i=1:size(W,1)
        if W[i,i] > newmax 
            W[i,i] = newmax
        end
    end
    W
end

# Trace norm
type TraceNormReg<:ProductRegularizer
    scale::Float64
end
TraceNormReg() = TraceNormReg(1)
scale(r::TraceNormReg) = r.scale
scale!(r::TraceNormReg, newscale::Number) = (r.scale = newscale)

function evaluate(r::TraceNormReg, W::AbstractArray)
    r.scale*sum(diag(W))/2
end

# note: this prox does *not* project onto the PSD cone
# that's ok in prisma, b/c the other regularizer does it
function prox!(r::TraceNormReg, W::AbstractArray, alpha::Number)
    for i=1:size(W,1)
        W[i,i] -= r.scale*alpha/2
    end
    W
end

# Trace norm
type TraceNormConstraint<:ProductRegularizer
    scale::Float64
end
TraceNormReg() = TraceNormReg(1)
scale(r::TraceNormReg) = r.scale
scale!(r::TraceNormReg, newscale::Number) = (r.scale = newscale)

function evaluate(r::TraceNormReg, W::AbstractArray)
    sum(diag(W))/2 <= r.scale ? 0 : Inf
end

# note: this prox does *not* project onto the PSD cone
# that's ok in prisma, b/c the other regularizer does it
function prox!(r::TraceNormReg, W::AbstractArray, alpha::Number)
    for i=1:size(W,1)
        W[i,i] -= r.scale*alpha/2
    end
    W
end