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
maxrank = 50

params = FrankWolfeParams(maxiters = 500, reltol = 1e-2, 
	                      #stepsize = BacktrackingStepSize(1, .5, .01));
						  stepsize = HopefulStepSize(1, 1, .5, 1.2, .01));

function plot(ch::ConvergenceHistory, filename::AbstractString="")
	figure()
	# yscale("log")
	plot(ch.objective, linewidth=5.0)
	plot(ch.dual_objective, linewidth=5.0, linestyle = "--")
	xlabel("iterations")
	ylabel("objective")
	savefig(filename*".pdf")
end

function logplot(ch::ConvergenceHistory, filename::AbstractString="")
	l = min(length(ch.objective), length(ch.dual_objective))
	figure()
	yscale("log")
	plot(ch.objective[1:l] - ch.dual_objective[1:l], linewidth=5.0)
	xlabel("iterations")
	ylabel("duality gap")
	savefig(filename*".pdf")
end

function plot(train_error, test_error, filename::AbstractString="")
	figure()
	# yscale("log")
	plot(train_error, linewidth=5.0, label="train")
	plot(test_error, linewidth=5.0, label="test")
	xlabel("iterations")
	ylabel("error")
	legend()
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
	glrm = GLRM(A, QuadLoss(), QuadReg(), QuadReg(), maxrank);
elseif loss == "huber"
	glrm = GLRM(A, HuberLoss(), QuadReg(), QuadReg(), maxrank);
elseif loss == "logistic"
	glrm = GLRM(A, LogisticLoss(), QuadReg(), QuadReg(), maxrank);
elseif loss == "poisson"
	glrm = GLRM(A, fill(PoissonLoss(5), size(A,2)), QuadReg(), QuadReg(), maxrank);
	opt_param = opt_param/5
end

println("\n\n$loss\n\n")
gfrm = GFRM(glrm);
gfrm.r = TraceNormConstraint(opt_param);
ch = ConvergenceHistory("cv")
train_error, test_error = cv_by_iter(gfrm, .1, params = params, ch = ch)

file = matopen("glrm_$loss_reg=$(gfrm.r.scale).mat", "w")
write(file, "ch", ch)
write(file, "train_error", train_error)
write(file, "test_error", test_error)
plot(ch, "cv_convergence_$loss_reg=$(gfrm.r.scale)")
logplot(ch, "cv_logconvergence_$loss_reg=$(gfrm.r.scale)")
plot(train_error, test_error, "cv_$loss_reg=$(gfrm.r.scale)")
close(file)
