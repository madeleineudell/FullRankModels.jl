opt_param = parse(Float64, ARGS[1])
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

glrm_poisson = GLRM(A, fill(PoissonLoss(5), size(A,2)), QuadReg(), QuadReg(), maxrank);
gfrm_poisson = GFRM(glrm_poisson);
gfrm_poisson.r = TraceNormConstraint(opt_param);
X_poisson, ch_poisson = fit!(gfrm_poisson, params);


file = matopen("glrm_poisson_reg=$(gfrm_poisson.r.scale).mat", "w")
write(file, "ch", ch_poisson)
write(file, "Xhat", X_poisson)
close(file)

plot(ch_poisson, "Poisson loss", "glrm_poisson_reg=$(gfrm_poisson.r.scale)")
logplot(ch_poisson, "Poisson loss", "glrm_poisson_log_reg=$(gfrm_poisson.r.scale)")


