% GP_heteroscadiscity.m
% -------------------------------------------------------------------------
% Heteroscedastic Gaussian Process Regression for data with non-constant variance.
% Automatically optimizes both a “main” GP (for f(x)) and an “auxiliary” GP (for log σ²(x)).
% This version fixes the row‐count bug, groups replicates properly when x has multiple columns,
% and ensures theta_main_hat/theta_aux_hat are always defined.
%
% USAGE:
%   [y_hat, CI_95, theta_main_hat, theta_aux_hat, sigma_y2, sigma_y2_test_hat] = ...
%       GP_heteroscadiscity(x, y, kernel, x_test)
%
% INPUTS:
%   x                - N×D matrix of training inputs (D may be ≥1).
%   y                - N×1 vector of training outputs (replicates allowed).
%   kernel           - Function handle: kernel(X1, X2, theta),
%                      returns M×P Gram matrix if X1 is M×D and X2 is P×D.
%                      theta = [ℓ; σ_f].
%   x_test           - M×D matrix of test inputs.
%
% OUTPUTS:
%   y_hat             - M×1 predictive mean (de‐standardized).
%   CI_95             - M×1 95% confidence‐interval half‐width (de‐standardized).
%   theta_main_hat    - [ℓ; σ_f] for the main GP.
%   theta_aux_hat     - [ℓ_aux; σ_f_aux; σ_n_aux] for the auxiliary GP on log σ².
%   sigma_y2          - N×1 estimated training‐noise variances (de‐standardized).
%   sigma_y2_test_hat - M×1 predicted mean noise variance at x_test (de‐standardized).
%
% AUTHOR:
%   Carl Emil Aae Eskildsen, Imperial College London
% REVISED BY:
%   Carl Emil Aae Eskildsen, May 2025  (added ℓ_min and stable CI logic; fixed undefined theta_main_hat/theta_aux_hat, n=size(x,1); added data‐driven tau clamp + shrink weight w)
%
% LICENSE:
%   MIT Licence
% -------------------------------------------------------------------------
function [y_hat, CI_95, theta_main_hat, theta_aux_hat, sigma_y2, sigma_y2_test_hat] = GP_heteroscadiscity(x, y, kernel, x_test)

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
    % Group by identical x_s rows
    sigma_y = nan(N,1);
    ux = unique(x_s, 'rows');
    for i = 1:size(ux, 1)
        idx = ismember(x_s, ux(i,:), 'rows');
        replicate_s = std(y_s(idx));
        if replicate_s < 1e-6
            replicate_s = 1e-6;
        end
        sigma_y(idx) = replicate_s;
    end
    sigma_y2 = sigma_y.^2;   % N×1

    %% 3) Initial hyperparameter guess from data
    d     = pdist(x_s);
    med_d = median(d);
    l0    = med_d;
    sf0   = std(y_s);
    theta0_main = [l0; sf0];

    %% 4) Optimize main GP hyperparameters
    obj_main = @(th) negativeLogMarginalLikelihood_main(th, x_s, y_s, sigma_y, kernel);

    % Enforce a minimum length-scale ℓ_min = multiplier * l0
    multiplier = 2.5;          % increase to force smoother f(x)
    l_min      = multiplier * l0;
    if l_min <= 0, l_min = 1e-2; end

    lower_main = [l_min;  1e-5];    % [ℓ; σ_f]
    upper_main = [2*l0;   1e5];     % [ℓ; σ_f]

    opt = optimoptions('fmincon', ...
                       'Algorithm','sqp', ...
                       'Display','off', ...
                       'MaxFunctionEvaluations', 1000);

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

    %% 6) Train auxiliary GP on log‐variance of training noise
    eps_log    = 1e-9;
    y_aux      = log(sigma_y2 + eps_log);
    theta0_aux = [l0; sf0; 0.1*sf0];

    lower_aux = [l_min;  1e-5;    1e-9];
    upper_aux = [2*l0;   1e5;     1e2];
    obj_aux   = @(th) negativeLogMarginalLikelihood_aux_log(th, x_s, y_aux, kernel);

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

    K_aux       = kernel(x_s, x_s, theta_aux_hat(1:2)) + diag(theta_aux_hat(3)^2 * ones(N,1));
    L_aux       = chol(K_aux + 1e-5*eye(N), 'lower');
    alpha_aux   = L_aux' \ (L_aux \ y_aux);
    K_aux_star  = kernel(x_s, x_test_s, theta_aux_hat(1:2));

    logVar_test_mean = K_aux_star' * alpha_aux;
    v_aux            = L_aux \ K_aux_star;
    K_aux_test       = kernel(x_test_s, x_test_s, theta_aux_hat(1:2));
    var_logVar       = diag(K_aux_test) - sum(v_aux.^2, 1)';
    var_logVar(var_logVar < 0) = 0;

    mu_L          = logVar_test_mean;
    v_L           = var_logVar;
    E_sigma2_test = exp(mu_L + 0.5 * v_L);

    sigma_y2_test_hat = E_sigma2_test;

    %% 7) Combine variances for a stable CI (data‐driven clamp + shrinkage)
    % ---------------------------------------------------------------------
    % (a) Choose tau from the training noise variances (e.g., 90th percentile):
    tau = prctile(sigma_y2, 90);

    % (b) Clamp the predicted test noise variances at tau:
    E_sigma2_test_clamped = E_sigma2_test;
    E_sigma2_test_clamped(E_sigma2_test_clamped > tau) = tau;

    % (c) Apply a shrinkage weight w to dampen the noise effect further:
    w = 0.5;   % 0 ≤ w ≤ 1 (for example, 0.5 means "take half of the clamped noise variance")

    % (d) Form the total variance as main‐GP variance + weighted & clamped noise:
    var_total = var_main + w * E_sigma2_test_clamped;

    % (e) Build a 95% confidence interval using z = norminv(0.975):
    z       = norminv(0.975);
    CI_95_s = z .* sqrt(var_total);

    %% 8) De‐standardize outputs
    y_hat             = y_hat_s * sy + mu_y;          
    CI_95             = CI_95_s * sy;                
    sigma_y2          = sigma_y2 * (sy^2);    
    sigma_y2_test_hat = sigma_y2_test_hat * (sy^2);
end

%%-----------------------------------------------------------------------
%% Helper: Negative log marginal likelihood for main GP
function nlml = negativeLogMarginalLikelihood_main(theta, x, y, sigma_y, kernel)
    N = size(x,1);
    K = kernel(x, x, theta) + diag(sigma_y.^2) + 1e-5*eye(N);
    L = chol(K, 'lower');
    alpha = L' \ (L \ y);
    nlml = 0.5 * (y' * alpha) + sum(log(diag(L))) + 0.5 * N * log(2*pi);
end

%%-----------------------------------------------------------------------
%% Helper: Negative log marginal likelihood for auxiliary GP in log space
function nlml = negativeLogMarginalLikelihood_aux_log(theta, x, y_aux, kernel)
    N      = size(x,1);
    K_aux  = kernel(x, x, theta(1:2)) + diag(theta(3)^2 * ones(N,1)) + 1e-5*eye(N);
    L_aux  = chol(K_aux, 'lower');
    alpha_aux = L_aux' \ (L_aux \ y_aux);
    nlml   = 0.5 * (y_aux' * alpha_aux) + sum(log(diag(L_aux))) + 0.5 * N * log(2*pi);
end
