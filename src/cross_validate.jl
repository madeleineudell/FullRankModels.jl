import LowRankModels: get_train_and_test, cv_by_iter, flatten_observations
import FirstOrderOptimization: HopefulStepSize, FrankWolfeParams

export cv_by_iter

default_params = FrankWolfeParams() # maxiters = 100, reltol = 1e-10, stepsize = HopefulStepSize(1, 1, .5, 1.2, .01));

function cv_by_iter(glrm::GFRM, holdout_proportion::Number = .1; 
                    params = default_params,
                    ch::ConvergenceHistory = ConvergenceHistory("cv_by_iter"),
                    verbose = true)
    # obs = flattenarray(map(ijs->map(j->(ijs[1],j),ijs[2]),zip(1:length(glrm.observed_features),glrm.observed_features)))
    gfrm = glrm
    z = zeros(sum(map(length, gfrm.observed_examples)))
    sketch = AsymmetricSketch(size(gfrm.A)..., gfrm.k)
    obs = flatten_observations(glrm.observed_features)

    train_observed_features, train_observed_examples, test_observed_features, test_observed_examples = 
        get_train_and_test(obs, size(glrm.A)..., holdout_proportion)
    
    # form glrm on training dataset 
    train_glrm = copy_estimate(glrm)
    train_glrm.observed_examples = train_observed_examples
    train_glrm.observed_features = train_observed_features

    # form glrm on testing dataset
    test_glrm = copy_estimate(glrm)
    test_glrm.observed_examples = test_observed_examples
    test_glrm.observed_features = test_observed_features

    ntrain = sum(map(length, train_glrm.observed_features))
    ntest = sum(map(length, test_glrm.observed_features))
        
    niters = params.maxiters
    params.maxiters = 1
    train_error = Array(Float64, niters)
    test_error = Array(Float64, niters)
    if verbose
        @printf("%12s%12s%12s\n", "train error", "test error", "time")  
        t0 = time()
    end
    z = zeros(sum(map(length, train_glrm.observed_examples)))
    sketch = AsymmetricSketch(size(train_glrm.A)..., gfrm.k)    
    for iter=1:niters
        # evaluate train and test error
        fit!(train_glrm, params, z=z, sketch=sketch, ch=ch, verbose=false)
        train_error[iter] = ch.objective[end] # objective(train_glrm, parameter_estimate(train_glrm)..., include_regularization=false)/ntrain
        test_error[iter] = objective(test_glrm, parameter_estimate(train_glrm)..., include_regularization=false)/ntest
        if verbose
            @printf("%12.4e%12.4e%12.4e\n", train_error[iter], test_error[iter], time() - t0)
        end
    end
    params.maxiters = niters
    return train_error, test_error
end


