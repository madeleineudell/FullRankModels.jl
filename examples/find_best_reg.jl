reg_param = parse(Float64, ARGS[1])
big = true

using LowRankModels, FullRankModels, PyPlot, MAT
import PyPlot.plot
import FirstOrderOptimization: HopefulStepSize, BacktrackingStepSize

if big
	t = readdlm("/media/datdisk3/udell/ml-20m/ratings.csv", ',', header=true);
	A = sparse(round(Int, t[1][:,1]), round(Int, t[1][:,2]), t[1][:,3]);
else
	t = readdlm("/media/datdisk3/udell/ml-100k/u.data", '\t', Int);
	A = sparse(t[:,1], t[:,2], t[:,3]);
end

maxrank = 100
cv_params = FrankWolfeParams(maxiters = 10, reltol = 1e-1, stepsize = BacktrackingStepSize(1, .5, .1));
params = FrankWolfeParams(maxiters = 150, reltol = 1e-2, stepsize = BacktrackingStepSize(1, .5, .1));

function plot(ch::ConvergenceHistory, title_string::AbstractString="", filename::AbstractString="")
	figure()
	# yscale("log")
	plot(ch.objective, linewidth=4.0)
	plot(ch.dual_objective, linewidth=4.0, linestyle = "--")
	xlabel("iterations")
	ylabel("objective")
	title(title_string)
	savefig(filename*".pdf")
end

function logplot(ch::ConvergenceHistory, title_string::AbstractString="", filename::AbstractString="")
	l = min(length(ch.objective), length(ch.dual_objective))
	figure()
	yscale("log")
	plot(ch.objective[1:l] - ch.dual_objective[1:l], linewidth=4.0)
	xlabel("iterations")
	ylabel("duality gap")
	title(title_string)
	savefig(filename*".pdf")
end

# quadratic loss
glrm_quad = GLRM(A, QuadLoss(), QuadReg(), QuadReg(), maxrank);
gfrm_quad = GFRM(glrm_quad);
gfrm_quad.r = TraceNormConstraint();

gfrm_quad.r = TraceNormConstraint(reg_param);
X_quad, ch_quad = fit!(gfrm_quad, params);

file = matopen("glrm_quad_reg=$(gfrm_quad.r.scale).mat", "w")
write(file, "ch", ch_quad)
write(file, "Xhat", X_quad)
close(file)

plot(ch_quad, "Quadratic loss", "glrm_quad_reg=$(gfrm_quad.r.scale)")
logplot(ch_quad, "Quadratic loss", "glrm_quad_log")


