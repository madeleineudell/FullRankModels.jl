

function regularization_path(train_glrm::AbstractGLRM, test_glrm::AbstractGLRM; 
                                         params=Params(), 
                                         reg_params=logspace(2,-2,5), 
                                         verbose=true,
                                         ch::ConvergenceHistory=ConvergenceHistory("reg_path"))
    train_error = Array(Float64, length(reg_params))
    test_error = Array(Float64, length(reg_params))
    ntrain = sum(map(length, train_glrm.observed_features))
    ntest = sum(map(length, test_glrm.observed_features))
    if verbose println("training model on $ntrain samples and testing on $ntest") end
    @show params
    train_time = Array(Float64, length(reg_params))
    for iparam=1:length(reg_params)
        reg_param = reg_params[iparam]
        # evaluate train and test error
        if verbose println("fitting train GLRM for reg_param $reg_param") end
        scale_regularizer!(train_glrm, reg_param)
        # no need to restart glrm X and Y even if they went to zero at the higher regularization
        # b/c fit! does that automatically
        fit!(train_glrm, params, ch=ch, verbose=verbose)
        train_time[iparam] = ch.times[end]
        if verbose println("computing mean train and test error for reg_param $reg_param:") end
        train_error[iparam] = objective(train_glrm, parameter_estimate(train_glrm)..., include_regularization=false) / ntrain
        if verbose println("\ttrain error: $(train_error[iparam])") end
        test_error[iparam] = objective(test_glrm, parameter_estimate(train_glrm)..., include_regularization=false) / ntest
        if verbose println("\ttest error:  $(test_error[iparam])") end
    end
    return train_error, test_error, train_time, reg_params
end