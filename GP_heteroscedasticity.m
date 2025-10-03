% GP_heteroscedasticity.m
% -------------------------------------------------------------------------
% Heteroscedastic Gaussian Process Regression for data with non-constant variance.
% Automatically optimizes both a "main" GP (for f(x)) and an "auxiliary" GP (for log sigma^2(x)).
% This version fixes the row-count bug, groups replicates properly when x has multiple columns,
% ensures theta_main_hat/theta_aux_hat are always defined,
% excludes singleton x-values from auxiliary GP training,
% enforces a longer aux-GP length-scale, and uses a soft-clamp + adaptive shrink on noise.
%
% USAGE:
%   [y_hat, CI, theta_main_hat, theta_aux_hat, var_total] = ...
%       GP_heteroscedasticity(x, y, kernel, x_test, alpha_error)
%
% INPUTS:
%   x                - N-by-D matrix of training inputs (D may be >=1)
%   y                - N-by-1 vector of training outputs (replicates allowed)
%   kernel           - Function handle: K = kernel(X1, X2, theta),
%                      where theta = [l; sigma_f]
%   x_test           - M-by-D matrix of test inputs
%   alpha_error      - significance level for confidence interval (e.g. 0.05 for 95%)
%
% OUTPUTS:
%   y_hat            - M-by-1 predictive mean (de-standardized)
%   CI               - M-by-1 confidence-interval half-width (de-standardized)
%   theta_main_hat   - [l; sigma_f] for the main GP
%   theta_aux_hat    - [l_aux; sigma_f_aux; sigma_n_aux] for the auxiliary GP on log sigma^2
%   var_total        - M-by-1 total predictive variance (main GP + heteroscedastic noise, de-standardized)
%
% AUTHOR:
%   Carl Emil Aae Eskildsen, Imperial College London
% REVISED BY:
%   Carl Emil Aae Eskildsen, May 2025  (added l_min and stable CI logic; fixed undefined theta_main_hat/theta_aux_hat; added tau clamp + shrink; excluded singletons from aux GP)
%   Carl Emil Aae Eskildsen, July 2025 (added repCount; defined auxIdx; trained auxiliary GP on x_aux; enforced aux-GP length-scale floor; soft-clamp & adaptive shrink)
%   Carl Emil Aae Eskildsen, Sept 2025 (simplified outputs: replaced sigma_y2 and sigma_y2_test_hat with var_total)
%
% LICENSE:
%   MIT Licence
% -------------------------------------------------------------------------
function [y_hat, CI, theta_main_hat, theta_aux_hat, var_total] = GP_heteroscedasticity(x, y, kernel, x_test, alpha_error)

    %% 1) Standardize inputs and outputs
    mu_x = mean(x, 1);
    sx   = std(x,  0, 1);
    sx(sx == 0) = 1;
    x_s = (x - mu_x) ./ sx;

    mu_y = mean(y);
    sy   = std(y);
    if sy == 0, sy = 1; end
    y_s = (y - mu_y) / sy;

    N = size(x_s, 1);

    %% 2) Estimate training noise variance from replicates
    sigma_y  = nan(N,1);
    repCount = nan(N,1);
    ux = unique(x_s, 'rows');
    for i = 1:size(ux, 1)
        idx = ismember(x_s, ux(i,:), 'rows');
        replicate_s = std(y_s(idx));
        if replicate_s < 1e-6
            replicate_s = 1e-6;
        end
        sigma_y(idx)  = replicate_s;
        repCount(idx) = sum(idx);  % count replicates per x
    end
    sigma_y2 = sigma_y.^2;  % N-by-1 noise variances

    % Which points have true replicates?
    auxIdx = (repCount > 1);

    %% 3) Initial hyperparameter guess from data
    d     = pdist(x_s);
    med_d = median(d);
    l0    = med_d;
    sf0   = std(y_s);
    theta0_main = [l0; sf0];

    %% 4) Optimize main GP hyperparameters
    obj_main = @(th) negativeLogMarginalLikelihood_main(th, x_s, y_s, sigma_y, kernel);

    multiplier = 2.5;
    l_min      = multiplier * l0;
    if l_min <= 0, l_min = 1e-2; end

    lower_main = [l_min;  1e-5];
    upper_main = [2*l0;   1e5];

    opt = optimoptions('fmincon','Algorithm','sqp','Display','off','MaxFunctionEvaluations',1000);

    theta_main_hat = theta0_main;
    bestNL = inf;
    nStarts = 5;
    for k = 1:nStarts
        th0 = theta0_main .* (1 + 0.5*(2*rand(size(theta0_main)) - 1));
        try
            th_k = fmincon(obj_main, th0, [], [], [], [], lower_main, upper_main, [], opt);
            nl_k = obj_main(th_k);
            if nl_k < bestNL
                bestNL         = nl_k;
                theta_main_hat = th_k;
            end
        catch
            % skip if fmincon fails
        end
    end

    %% 5) Main GP prediction
    K       = kernel(x_s, x_s, theta_main_hat) + diag(sigma_y2);
    L       = chol(K + 1e-5*eye(N), 'lower');
    alpha   = L' \ (L \ y_s);

    x_test_s  = (x_test - mu_x) ./ sx;
    K_star    = kernel(x_s, x_test_s, theta_main_hat);
    y_hat_s   = K_star' * alpha;

    v       = L \ K_star;
    K_test  = kernel(x_test_s, x_test_s, theta_main_hat);
    var_main = diag(K_test) - sum(v.^2, 1)';
    var_main(var_main < 0) = 0;

    %% 6) Train auxiliary GP on log-variance of training noise
    span = max(x_s,[],1) - min(x_s,[],1);    % 1-by-D
    L_floor_aux = 0.5 * min(span);

    eps_log    = 1e-9;
    x_aux      = x_s(auxIdx, :);
    y_aux      = log(sigma_y2(auxIdx) + eps_log);
    theta0_aux = [l0; sf0; 0.1*sf0];

    lower_aux = [L_floor_aux; 1e-5;    1e-9];
    upper_aux = [2*l0;        1e5;     1e2];
    obj_aux   = @(th) negativeLogMarginalLikelihood_aux_log(th, x_aux, y_aux, kernel);

    theta_aux_hat = theta0_aux;
    bestNL_a = inf;
    for k = 1:nStarts
        th0 = theta0_aux .* (1 + 0.5*(2*rand(size(theta0_aux)) - 1));
        try
            th_k = fmincon(obj_aux, th0, [], [], [], [], lower_aux, upper_aux, [], opt);
            nl_k = obj_aux(th_k);
            if nl_k < bestNL_a
                bestNL_a      = nl_k;
                theta_aux_hat = th_k;
            end
        catch
            % skip if fmincon fails
        end
    end

    M = sum(auxIdx);
    K_aux       = kernel(x_aux, x_aux, theta_aux_hat(1:2)) + diag(theta_aux_hat(3)^2 * ones(M,1));
    L_aux       = chol(K_aux + 1e-5*eye(M), 'lower');
    alpha_aux   = L_aux' \ (L_aux \ y_aux);
    K_aux_star  = kernel(x_aux, x_test_s, theta_aux_hat(1:2));

    logVar_test_mean = K_aux_star' * alpha_aux;
    v_aux            = L_aux \ K_aux_star;
    K_aux_test       = kernel(x_test_s, x_test_s, theta_aux_hat(1:2));
    var_logVar       = diag(K_aux_test) - sum(v_aux.^2, 1)';
    var_logVar(var_logVar < 0) = 0;

    mu_L          = logVar_test_mean;
    v_L           = var_logVar;
    E_sigma2_test = exp(mu_L + 0.5 * v_L);

    sigma_y2_test_hat = E_sigma2_test;

    %% 7) Combine variances: soft-clamp + adaptive shrink
    tau = prctile(sigma_y2, 95);
    E_sigma2_clamped = tau .* (E_sigma2_test ./ (E_sigma2_test + tau));
    w = 1 ./ (1 + E_sigma2_test / tau);
    var_total_s = var_main + w .* E_sigma2_clamped;

    z = norminv(1-alpha_error/2);
    CI_s = z .* sqrt(var_total_s);

    %% 8) De-standardize outputs
    y_hat             = y_hat_s * sy + mu_y;
    CI             = CI_s * sy;
    sigma_y2          = sigma_y2 * (sy^2);
    sigma_y2_test_hat = sigma_y2_test_hat * (sy^2);
    var_total = var_total_s * (sy^2);
end

% ------------------------------------------------------------------------
function nlml = negativeLogMarginalLikelihood_main(theta, x, y, sigma_y, kernel)
    N = size(x,1);
    K = kernel(x, x, theta) + diag(sigma_y.^2) + 1e-5*eye(N);
    L = chol(K, 'lower');
    alpha = L' \ (L \ y);
    nlml = 0.5 * (y' * alpha) + sum(log(diag(L))) + 0.5 * N * log(2*pi);
end

% ------------------------------------------------------------------------
function nlml = negativeLogMarginalLikelihood_aux_log(theta, x_aux, y_aux, kernel)
    N      = size(x_aux,1);
    K_aux  = kernel(x_aux, x_aux, theta(1:2)) + diag(theta(3)^2 * ones(N,1)) + 1e-5*eye(N);
    L_aux  = chol(K_aux, 'lower');
    alpha_aux = L_aux' \ (L_aux \ y_aux);
    nlml   = 0.5 * (y_aux' * alpha_aux) + sum(log(diag(L_aux))) + 0.5 * N * log(2*pi);
end
