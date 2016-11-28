__precompile__()

module FullRankModels

import LowRankModels: ConvergenceHistory, update_ch!, fit!

include("regularizers.jl")
include("gfrm.jl")
include("evaluate_fit.jl")

include("algorithms/prisma.jl")
# include("algorithms/fw_sketch.jl")
include("algorithms/fw.jl")
include("algorithms/fw_sketch_multithread.jl")
include("algorithms/fw_thin.jl")

include("utilities.jl")
include("cross_validate.jl")

end # module
