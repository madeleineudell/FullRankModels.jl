using LowRankModels, FullRankModels, DataFrames

# importing relevant data (matrix with missing values and index of categoricals) 
df = readtable("/Users/madeleine/Downloads/GSS2014cleanestCV10.csv");
df1 = df[:, 2:size(df)[2] ];

# normalize reals
for i=1:23
	notmissing = !isna(df1[i])
	df1[notmissing,i] -= mean(df1[notmissing,i])
	df1[notmissing,i] /= std(df1[notmissing,i])
end

# vector of column types
expand_categoricals!(df1, 55:size(df1)[2])
datatype = Array(Symbol, size(df1)[2])
datatype[1:23] = :real
datatype[24:54] = :ord
datatype[55:size(df1)[2]] = :bool

# # range of low ranks 
# K = [5:5:50]

# # setting up round 1
# lowrank_prl = @time pmap(k -> lowrank(df1, k, datatype), K)
# tstE = [x[1] for x in lowrank_prl]
# trnE = [x[2] for x in lowrank_prl]

# kstar = K[findmin(tstE)[2]] 

# # searching around round 1 best
# K2 = [kstar-4: kstar+4]


# lowrank_prl2 = @time pmap(k -> lowrank(df1, k, datatype), K2)
# tstE2 = [x[1] for x in lowrank_prl2]
# trnE2 = [x[2] for x in lowrank_prl2]
# kstar2 = K2[findmin(tstE2)[2]] 

# final fit
prob_losses = Dict{Symbol, Any}(
	  :real        => QuadLoss,
	  :bool        => LogisticLoss,
	  :ord         => QuadLoss,
	  :cat         => MultinomialLoss)

glrm = GLRM(df1, 100, datatype, loss_map = prob_losses, scale=false, prob_scale=false, offset=false ) 
gfrm = GFRM(glrm)

# min(size(df)...)*1000 is too small ...?
# 
gfrm.r = TraceNormConstraint(min(size(df)...)*100)
params = FrankWolfeParams(maxiters = 200)
X_sketched, ch = fit!(gfrm, params)

gfrm.r = TraceNormConstraint(min(size(df)...)*1000)
params = FrankWolfeParams(maxiters = 200)
X_sketched, ch = fit!(gfrm, params)

train_error, test_error, train_time, model_onenorm, reg_params = regularization_path(gfrm, params, linspace())
