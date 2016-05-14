using LowRankModels, FullRankModels, PyPlot
import PyPlot.plot
import FirstOrderOptimization: HopefulStepSize, BacktrackingStepSize

loss = ARGS[1]
maybebig = ARGS[2]

if maybebig == "big"
	println("using big data set")
	t = readdlm("/media/datdisk3/udell/ml-20m/ratings.csv", ',', header=true);
	A = sparse(round(Int, t[1][:,1]), round(Int, t[1][:,2]), t[1][:,3]);
else
	println("using small data set")
	t = readdlm("/media/datdisk3/udell/ml-100k/u.data", '\t', Int);
	A = sparse(t[:,1], t[:,2], t[:,3]);
end
maxrank = 100

params = FrankWolfeParams(maxiters = 1000, reltol = 1e-2, 
	                      #stepsize = BacktrackingStepSize(1, .5, .01));
						  stepsize = HopefulStepSize(1, 1, .5, 1.2, .01));

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

opt_param = 540000

# julia> (mp,np) = size(A)
# (138493,131262)

# julia> (m,n) = size(Asmall)
# (943,1682)

# julia> (mp/m + np/n)/2*4800 # 4800 is best reg param for 10k data set, chosen via cv
# 539768.4047799481

# quadratic loss
if loss == "quad"
glrm_quad = GLRM(A, QuadLoss(), QuadReg(), QuadReg(), maxrank);
gfrm_quad = GFRM(glrm_quad);
gfrm_quad.r = TraceNormConstraint(opt_param);
X_quad, ch_quad = fit!(gfrm_quad, params);
plot(ch_quad, "Quadratic loss", "glrm_quad_reg=$(gfrm_quad.r.scale)")
logplot(ch_quad, "Quadratic loss", "glrm_quad_log_reg=$(gfrm_quad.r.scale)")

file = matopen("glrm_quad_reg=$(gfrm_quad.r.scale).mat", "w")
write(file, "ch", ch_quad)
write(file, "Xhat", X_quad)
close(file)

# huber loss
elseif loss == "huber"
println("\n\nHuber\n\n")
glrm_huber = GLRM(A, HuberLoss(), QuadReg(), QuadReg(), maxrank);
gfrm_huber = GFRM(glrm_huber);
gfrm_huber.r = TraceNormConstraint();

# train_error, test_error, train_time, reg_params = 
# 	regularization_path(gfrm_huber, params=cv_params, reg_params=reg_params);
# @show opt_param = reg_params[indmin(test_error)]
gfrm_huber.r = TraceNormConstraint(opt_param);
X_huber, ch_huber = fit!(gfrm_huber, params);
plot(ch_huber, "Huber loss", "glrm_huber_reg=$(gfrm_huber.r.scale)")
logplot(ch_huber, "Huber loss", "glrm_huber_log_reg=$(gfrm_huber.r.scale)")

file = matopen("glrm_huber_reg=$(gfrm_huber.r.scale).mat", "w")
write(file, "ch", ch_huber)
write(file, "Xhat", X_huber)
close(file)


# logistic loss
elseif loss == "logistic"
println("\n\nLogistic\n\n")
A_bool = copy(A)
A_bool.nzval = A_bool.nzval .>= 2
glrm_logistic = GLRM(A_bool, LogisticLoss(), QuadReg(), QuadReg(), maxrank);
gfrm_logistic = GFRM(glrm_logistic);
gfrm_logistic.r = TraceNormConstraint();

# train_error, test_error, train_time, reg_params = 
# 	regularization_path(gfrm_logistic, params=cv_params, reg_params=reg_params);
# @show opt_param = reg_params[indmin(test_error)]
gfrm_logistic.r = TraceNormConstraint(opt_param);
X_logistic, ch_logistic = fit!(gfrm_logistic, params);
plot(ch_logistic, "Logistic loss", "glrm_logistic_reg=$(gfrm_logistic.r.scale)")
logplot(ch_logistic, "Logistic loss", "glrm_logistic_log_reg=$(gfrm_logistic.r.scale)")

file = matopen("glrm_logistic_reg=$(gfrm_logistic.r.scale).mat", "w")
write(file, "ch", ch_logistic)
write(file, "Xhat", X_logistic)
close(file)


# poisson loss
elseif loss == "poisson"
println("\n\nPoisson\n\n")
reg_params = linspace(5000,7000,11)/5

glrm_poisson = GLRM(A, fill(PoissonLoss(5), size(A,2)), QuadReg(), QuadReg(), maxrank);
gfrm_poisson = GFRM(glrm_poisson);
gfrm_poisson.r = TraceNormConstraint(opt_param/5);
X_poisson, ch_poisson = fit!(gfrm_poisson, params);
plot(ch_poisson, "Poisson loss", "glrm_poisson_reg=$(gfrm_poisson.r.scale)")
logplot(ch_poisson, "Poisson loss", "glrm_poisson_log_reg=$(gfrm_poisson.r.scale)")

file = matopen("glrm_poisson_reg=$(gfrm_poisson.r.scale).mat", "w")
write(file, "ch", ch_poisson)
write(file, "Xhat", X_poisson)
close(file)
# MNL loss

end


