% GP_heteroscadiscity.m
% -------------------------------------------------------------------------
% Heteroscedastic Gaussian Process Regression for data with non-constant variance.
% Automatically optimizes both main GP hyperparameters and an auxiliary GP for noise variance.
%
% USAGE:
%   [y_hat, CI_95, theta_main_hat, theta_aux_hat, sigma_y2, sigma_y2_test_hat] = ...
%       GP_heteroscadiscity(x, y, kernel, x_test)
%
% INPUTS:
%   x                - n×1 vector of training inputs.
%   y                - n×1 vector of training outputs (replicates allowed).
%   kernel           - Function handle: kernel(x1, x2, theta).
%                      For main GP: theta = [length-scale; signal-variance].
%   x_test           - m×1 vector of test inputs.
%
% OUTPUTS:
%   y_hat             - m×1 predictive mean.
%   CI_95             - m×1 95% confidence interval half-width.
%   theta_main_hat    - Optimized main GP hyperparameters [l; sigma_f].
%   theta_aux_hat     - Optimized auxiliary GP hyperparameters [l_aux; sigma_f_aux; sigma_n_aux].
%   sigma_y2          - n×1 estimated training noise variance.
%   sigma_y2_test_hat - m×1 predicted noise variance at test inputs.
%
% AUTHOR:
%   Carl Emil Aae Eskildsen, Imperial College London (c.eskildsen@imperial.ac.uk)
%
% LICENSE:
%   Creative Commons Attribution-NonCommercial 4.0 International
% -------------------------------------------------------------------------
function [y_hat, CI_95, theta_main_hat, theta_aux_hat, sigma_y2, sigma_y2_test_hat] = GP_heteroscadiscity(x, y, kernel, x_test)

%% 1) Standardize inputs and outputs
mu_x      = mean(x);       sx = std(x);       if sx==0, sx=1;   end
x_s       = (x - mu_x) / sx;
x_test_s  = (x_test - mu_x) / sx;
mu_y      = mean(y);       sy = std(y);       if sy==0, sy=1;   end
y_s       = (y - mu_y) / sy;
n         = numel(x_s);

%% 2) Estimate training noise variance from replicates
sigma_y   = nan(n,1);
ux        = unique(x_s);
for i = 1:numel(ux)
    idx         = (x_s == ux(i));
    replicate_s = std(y_s(idx));
    if replicate_s < 1e-6, replicate_s = 1e-6; end
    sigma_y(idx) = replicate_s;
end
sigma_y2 = sigma_y.^2;

%% 3) Initial hyperparameter guess from data
d    = pdist(x_s);
med_d = median(d);
l0    = med_d;
sf0   = std(y_s);
theta0_main = [l0; sf0];

%% 4) Optimize main GP hyperparameters
obj_main   = @(th) negativeLogMarginalLikelihood_main(th, x_s, y_s, sigma_y, kernel);
lower_main = [0.5*l0; 1e-5];
upper_main = [2*l0;   1e5];
opt        = optimoptions('fmincon','Algorithm','sqp','Display','off','MaxFunctionEvaluations',1000);

nStarts = 5;
bestNL   = inf;
for k = 1:nStarts
    th0 = theta0_main .* (1 + 0.5*(2*rand(size(theta0_main))-1));
    try
        th_k = fmincon(obj_main, th0, [], [], [], [], lower_main, upper_main, [], opt);
        nl   = obj_main(th_k);
        if nl < bestNL
            bestNL          = nl;
            theta_main_hat  = th_k;
        end
    catch
    end
end

%% 5) Main GP prediction
K       = kernel(x_s, x_s, theta_main_hat) + diag(sigma_y2);
L       = chol(K + 1e-5*eye(n), 'lower');
alpha   = L' \ (L \ y_s);
K_star  = kernel(x_s, x_test_s, theta_main_hat);
y_hat_s = K_star' * alpha;
v       = L \ K_star;
K_test  = kernel(x_test_s, x_test_s, theta_main_hat);
var_main= diag(K_test) - sum(v.^2,1)';

%% 6) Train auxiliary GP on log-variance of training noise
eps_log    = 1e-9;
y_aux      = log(sigma_y2 + eps_log);
theta0_aux = [l0; sf0; 0.1*sf0];
lower_aux  = [0.5*l0; 1e-5;    1e-9];
upper_aux  = [2*l0;   1e5;     1e2];
obj_aux    = @(th) negativeLogMarginalLikelihood_aux_log(th, x_s, y_aux, kernel);

bestNL_a   = inf;
for k = 1:nStarts
    th0 = theta0_aux .* (1 + 0.5*(2*rand(size(theta0_aux))-1));
    try
        th_k = fmincon(obj_aux, th0, [], [], [], [], lower_aux, upper_aux, [], opt);
        nl   = obj_aux(th_k);
        if nl < bestNL_a
            bestNL_a        = nl;
            theta_aux_hat   = th_k;
        end
    catch
    end
end

%% 7) Auxiliary GP prediction at test inputs
K_aux           = kernel(x_s, x_s, theta_aux_hat(1:2)) + diag(theta_aux_hat(3)^2 * ones(n,1));
L_aux           = chol(K_aux + 1e-5*eye(n), 'lower');
alpha_aux       = L_aux' \ (L_aux \ y_aux);
K_aux_star      = kernel(x_s, x_test_s, theta_aux_hat(1:2));
logVar_test     = K_aux_star' * alpha_aux;
sigma_y2_test_hat = exp(logVar_test);

%% 8) Combine variances and compute 95% confidence interval
var_total = var_main + sigma_y2_test_hat;
z         = norminv(0.975);
CI_95_s   = z .* sqrt(var_total);

%% 9) De-standardize outputs
y_hat             = y_hat_s * sy + mu_y;
CI_95             = CI_95_s * sy;
sigma_y2_test_hat = sigma_y2_test_hat * sy^2;
sigma_y2          = sigma_y2 * sy^2;
end

%% Helper: Negative log marginal likelihood for main GP
function nlml = negativeLogMarginalLikelihood_main(theta, x, y, sigma_y, kernel)
    n = numel(x);
    K = kernel(x, x, theta) + diag(sigma_y.^2) + 1e-5*eye(n);
    L = chol(K, 'lower');
    alpha = L' \ (L \ y);
    nlml = 0.5 * (y' * alpha) + sum(log(diag(L))) + 0.5 * n * log(2*pi);
end

%% Helper: Negative log marginal likelihood for auxiliary GP in log space
function nlml = negativeLogMarginalLikelihood_aux_log(theta, x, y_aux, kernel)
    n     = numel(x);
    K_aux = kernel(x, x, theta(1:2)) + diag(theta(3)^2 * ones(n,1)) + 1e-5*eye(n);
    L_aux = chol(K_aux, 'lower');
    alpha_aux = L_aux' \ (L_aux \ y_aux);
    nlml = 0.5 * (y_aux' * alpha_aux) + sum(log(diag(L_aux))) + 0.5 * n * log(2*pi);
end
