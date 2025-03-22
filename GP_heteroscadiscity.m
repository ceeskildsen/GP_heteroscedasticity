%% GP_heteroscadiscity.m
%
% Description:
%   Performs heteroscedastic Gaussian Process (GP) regression with hyperparameter 
%   optimization for both the main GP (2 hyperparameters) and an auxiliary GP 
%   (3 hyperparameters) that models the log of the noise variance.
%
%   The auxiliary GP predicts the log-variance, which is then exponentiated 
%   to guarantee strictly positive noise variance estimates.
%
% Usage:
%   [y_hat, CI_95, theta_main_hat, theta_aux_hat, sigma_y2, sigma_y2_test_hat] = ...
%       GP_heteroscadiscity(x, y, kernel, x_test, theta0_main, theta0_aux)
%
% Inputs:
%   x          - n x 1 vector of training inputs.
%   y          - n x 1 vector of training outputs.
%                (Repeated x values indicate replicate measurements.)
%   kernel     - Function handle for the kernel with call: kernel(x1, x2, theta)
%                For the main GP, theta is a 2-element vector: [l, sigma_f].
%   x_test     - m x 1 vector of test inputs.
%   theta0_main- Initial 2-element hyperparameter vector for the main GP 
%                (e.g., [l, sigma_f]).
%   theta0_aux - Initial 3-element hyperparameter vector for the auxiliary GP 
%                (e.g., [l_aux, sigma_f_aux, sigma_n_aux]), where sigma_n_aux 
%                is the noise term in log-variance space.
%
% Outputs:
%   y_hat             - m x 1 predictive mean from the main GP.
%   CI_95             - m x 1 vector: half-width of the 95% confidence interval.
%   theta_main_hat    - Optimized main GP hyperparameters [l, sigma_f].
%   theta_aux_hat     - Optimized auxiliary GP hyperparameters [l_aux, sigma_f_aux, sigma_n_aux].
%   sigma_y2          - n x 1 vector of estimated noise variances for the training data.
%   sigma_y2_test_hat - m x 1 vector of predicted noise variances at the test inputs.
%
% License: Creative Commons Attribution-NonCommercial 4.0 International
%
%   You are free to:
%       Share — copy and redistribute the material in any medium or format
%       Adapt — remix, transform, and build upon the material
%   under the following terms:
%
%       Attribution — You must give appropriate credit, provide a link to the license,
%                     and indicate if changes were made.
%       NonCommercial — You may not use the material for commercial purposes.
%       No additional restrictions — You may not apply legal terms or technological
%                                     measures that legally restrict others from doing anything the license permits.
%
%   Full license text is available at:
%       http://creativecommons.org/licenses/by-nc/4.0/
%
% Author: Carl Emil Aae Eskildsen, Imperial College London (c.eskildsen@imperial.ac.uk)
% Date: 2025-03-19

function [y_hat, CI_95, theta_main_hat, theta_aux_hat, sigma_y2, sigma_y2_test_hat] = GP_heteroscadiscity(x, y, kernel, x_test, theta0_main, theta0_aux)
    %% 1) Estimate training noise from replicates
    n = length(x);
    sigma_y = nan(n,1);
    u_x = unique(x);  % using u_x for unique training inputs
    for i = 1:length(u_x)
        idx = (x == u_x(i));
        replicate_std = std(y(idx));
        % Enforce a minimum noise level if there's only one replicate or near-zero std
        if replicate_std < 1e-6
            replicate_std = 1e-6;
        end
        sigma_y(idx) = replicate_std;
    end
    sigma_y2 = sigma_y.^2;  % Save noise variances for output
    
    %% 2) Optimize Main GP Hyperparameters
    % The main GP has 2 hyperparameters: [length scale, signal variance].
    % We incorporate heteroscedastic noise diag(sigma_y2) in the covariance.
    
    options = optimoptions('fmincon','Algorithm','sqp','Display','iter',...
        'MaxFunctionEvaluations',1000);
    lower_lim_main = [1e-5; 1e-5];
    upper_lim_main = [1e5; 1e5];
    
    % Define the NLML objective for the main GP using sigma_y2 directly
    obj_main = @(theta) negativeLogMarginalLikelihood_main(theta, x, y, sigma_y2, kernel);
    
    % Optimize hyperparameters
    theta_main_hat = fmincon(obj_main, theta0_main, [], [], [], [], lower_lim_main, upper_lim_main, [], options);
    
    %% 3) Fit the Main GP with the optimized hyperparameters
    K = kernel(x, x, theta_main_hat) + diag(sigma_y2);
    K_star = kernel(x, x_test, theta_main_hat);
    L = chol(K + 1e-5*eye(n), 'lower');
    alpha = L' \ (L \ y);
    
    % Predictive mean and variance (excluding test noise)
    y_hat = K_star' * alpha;
    v = L \ K_star;
    K_test = kernel(x_test, x_test, theta_main_hat);
    var_main = diag(K_test) - sum(v.^2, 1)';
    
    %% 4) Train Auxiliary GP on Log-Variance
    % Instead of modeling sigma_y^2 directly, we model:
    %   y_aux(i) = log( sigma_y(i)^2 + eps_log )
    eps_log = 1e-9;  % small constant to avoid log(0)
    y_aux = log(sigma_y2 + eps_log);
    
    % The auxiliary GP has 3 hyperparameters: [l_aux, sigma_f_aux, sigma_n_aux]
    lower_lim_aux = [1e-5; 1e-5; 1e-9];
    upper_lim_aux = [1e5; 1e5; 1e2];
    
    obj_aux = @(theta) negativeLogMarginalLikelihood_aux_log(theta, x, y_aux, kernel);
    theta_aux_hat = fmincon(obj_aux, theta0_aux, [], [], [], [], lower_lim_aux, upper_lim_aux, [], options);
    
    %% 5) Predict Log-Variance at Test Points and Exponentiate
    % Build covariance for the auxiliary GP:
    %   kernel(x,x, theta_aux_hat(1:2)) + diag( theta_aux_hat(3)^2 )
    K_aux = kernel(x, x, theta_aux_hat(1:2)) + diag(theta_aux_hat(3)^2 * ones(n,1));
    K_aux_star = kernel(x, x_test, theta_aux_hat(1:2));
    
    L_aux = chol(K_aux + 1e-5*eye(n), 'lower');
    alpha_aux = L_aux' \ (L_aux \ y_aux);
    
    % Predicted log-variance at test points and exponentiation
    logVar_test = K_aux_star' * alpha_aux;
    sigma_y2_test_hat = exp(logVar_test);
    
    %% 6) Combine Predictive Variances and Compute 95% Confidence Intervals
    var_total = var_main + sigma_y2_test_hat;
    z = norminv(0.975);  % 97.5th percentile for a two-sided 95% CI
    CI_95 = z * sqrt(var_total);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Subfunction: Negative Log Marginal Likelihood for Main GP
function nlml = negativeLogMarginalLikelihood_main(theta, x, y, sigma_y2, kernel)
    n = length(x);
    % Use sigma_y2 directly in the diagonal term
    K = kernel(x, x, theta) + diag(sigma_y2);
    K = K + 1e-5*eye(n);  % Add jitter for numerical stability
    L = chol(K, 'lower');
    alpha = L' \ (L \ y);
    
    nlml = 0.5 * (y' * alpha) + sum(log(diag(L))) + 0.5*n*log(2*pi);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Subfunction: Negative Log Marginal Likelihood for Auxiliary GP in LOG space
function nlml = negativeLogMarginalLikelihood_aux_log(theta, x, y_aux, kernel)
    n = length(x);
    K_aux = kernel(x, x, theta(1:2)) + diag(theta(3)^2 * ones(n,1));
    K_aux = K_aux + 1e-5*eye(n);  % Add jitter for numerical stability
    L_aux = chol(K_aux, 'lower');
    alpha_aux = L_aux' \ (L_aux \ y_aux);
    
    nlml = 0.5*(y_aux'*alpha_aux) + sum(log(diag(L_aux))) + 0.5*n*log(2*pi);
end
