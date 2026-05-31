function [y_hat, CI, theta_main, theta_aux, var_total, diagnostics] = ...
        GP_heteroscedasticity(x, y, kernel_main, x_test, alpha_error)
% GP_HETEROSCEDASTICITY  Two-step heteroscedastic GP regression using a
% Laplace-approximated variance GP.
%
% Replaces the previous log(r^2) auxiliary GP with a principled Laplace
% approximation.  The variance GP places a GP prior on
%
%       g(x) = log sigma^2(x)
%
% with non-Gaussian likelihood  r_i | x_i  ~  N(0, exp(g(x_i))).
% The MAP of g is found by damped Newton iterations and kernel hyper-
% parameters are optimised by minimising the Laplace approximation to
% the negative log marginal likelihood.  This eliminates the
% E[log r^2] != log E[r^2] bias and removes the need for ad-hoc bias
% correction.
%
% Pipeline:
%   Step 1 – Fit a homoscedastic GP for the mean function.
%   Step 2 – Compute debiased residuals.
%   Step 3 – Fit variance GP via Laplace approximation on residuals.
%   Step 4 – Predict mean (+ epistemic) and aleatoric variance at test
%            locations and combine for confidence intervals.
%
% INPUTS
%   x           : n-by-1 inputs  (e.g. log10(conc+1))
%   y           : n-by-1 responses (e.g. normalised intensity)
%   kernel_main : function handle  k(X1, X2, [l; sf]) -> covariance matrix
%   x_test      : m-by-1 test inputs
%   alpha_error : significance level (default 0.05 for 95% CI)
%
% OUTPUTS
%   y_hat       : m-by-1  posterior mean predictions (original y scale)
%   CI          : m-by-1  half-width of (1-alpha)*100% CI
%   theta_main  : [l; sf; sn]   optimised mean-GP hyperparameters
%   theta_aux   : [l_g; sf_g]   optimised variance-GP hyperparameters
%   var_total   : m-by-1  total predictive variance (epistemic + aleatoric)
%   diagnostics : struct with diagnostic information
%
% Carl Emil Eskildsen, Imperial College London/University of Oxford

    if nargin < 5 || isempty(alpha_error), alpha_error = 0.05; end

    % =================================================================
    % 1. DATA PREPARATION
    % =================================================================
    x = x(:);  y = y(:);  x_test = x_test(:);

    mu_x = mean(x);  sx = std(x);  if sx == 0, sx = 1; end
    x_s      = (x      - mu_x) ./ sx;
    x_test_s = (x_test - mu_x) ./ sx;

    mu_y = mean(y);  sy = std(y);  if sy == 0, sy = 1; end
    y_s = (y - mu_y) / sy;

    N = numel(x_s);

    % =================================================================
    % 2. FIT MEAN GP  (homoscedastic, multi-start)
    % =================================================================

    % --- Initial guess from data ---
    d_dist = pdist(x_s);
    d_pos  = d_dist(d_dist > 1e-12);           % drop zero / near-zero
    l0     = median(d_dist);
    if isnan(l0) || l0 <= 0, l0 = 1.0; end

    sf0 = std(y_s);
    if sf0 <= 0, sf0 = 1.0; end

    % --- Bounds (log-space optimisation) ---
    if isempty(d_pos), ell_floor = 0.05;
    else,              ell_floor = max(0.05, 0.5 * min(d_pos));
    end
    l_max_mean  = max(5 * l0, range(x_s));
    sf_max_mean = max(10 * sf0, 10);

    lb_m = log([ell_floor; 1e-3; 1e-6]);
    ub_m = log([l_max_mean; sf_max_mean; 10]);

    theta0_m = log([l0; sf0; 0.1 * sf0]);
    theta0_m = max(theta0_m, lb_m);
    theta0_m = min(theta0_m, ub_m);

    obj_mean = @(th) neg_log_likelihood_mean(th, x_s, y_s, kernel_main);
    fmin_opts = optimoptions('fmincon', 'Display', 'off', ...
                             'Algorithm', 'sqp', ...
                             'MaxFunctionEvaluations', 2000);

    nStarts_m  = 5;
    bestVal_m  = inf;
    bestTh_m   = theta0_m;

    for k = 1:nStarts_m
        if k == 1
            th_start = theta0_m;
        else
            th_start = theta0_m + 0.7 * randn(3, 1);
            th_start = max(th_start, lb_m);
            th_start = min(th_start, ub_m);
        end
        try
            th_k  = fmincon(obj_mean, th_start, [],[],[],[], ...
                            lb_m, ub_m, [], fmin_opts);
            val_k = obj_mean(th_k);
            if val_k < bestVal_m
                bestVal_m = val_k;
                bestTh_m  = th_k;
            end
        catch
        end
    end

    l_f  = exp(bestTh_m(1));
    sf_f = exp(bestTh_m(2));
    sn_f = exp(bestTh_m(3));

    % --- Cache Cholesky + alpha for prediction ---
    K_f_sig = kernel_main(x_s, x_s, [l_f; sf_f]);
    K_f     = K_f_sig + sn_f^2 * eye(N) + 1e-6 * eye(N);
    L_f     = chol(K_f, 'lower');
    alpha_f = L_f' \ (L_f \ y_s);

    % =================================================================
    % 3. DEBIASED RESIDUALS
    % =================================================================
    mu_train = K_f_sig * alpha_f;         % posterior mean at training x
    r        = y_s - mu_train;            % residuals in scaled-y space

    % =================================================================
    % 4. FIT VARIANCE GP  (Laplace approximation, multi-start)
    % =================================================================

    % Prior mean  m0 = log(median(r^2))
    r2 = r.^2 + 1e-12;
    m0 = log(median(r2));

    % Laplace Newton settings
    lap.maxIter = 50;
    lap.tol     = 1e-6;
    lap.jitterK = 1e-8;
    lap.gMin    = log(1e-12);
    lap.gMax    = log(1e+6);

    % Bounds for [l_g; sf_g]. Keep the variance GP from varying at scales
    % finer than the experimental spacing of unique x levels.
    ux_s = unique(x_s);
    dx_s = diff(sort(ux_s));
    dx_s = dx_s(dx_s > 1e-12);
    if isempty(dx_s)
        ell_floor_v = 0.05;
    else
        ell_floor_v = max(0.05, median(dx_s));
    end
    lb_v = [ell_floor_v; 1e-3];
    ub_v = [50;   1e+3];

    theta0_v = [1.0; 1.0];               % sensible for standardised x

    nStarts_v = 10;
    starts_v  = zeros(2, nStarts_v);
    starts_v(:,1) = theta0_v;
    for k = 2:nStarts_v
        starts_v(:,k) = theta0_v .* exp(0.7 * randn(2, 1));
    end
    starts_v = max(starts_v, lb_v);
    starts_v = min(starts_v, ub_v);

    fmin_opts_v = optimoptions('fmincon', 'Display', 'off', ...
                               'Algorithm', 'sqp', ...
                               'MaxFunctionEvaluations', 200);

    obj_var = @(theta) nlZ_laplace(theta, x_s, r, m0, lap);

    bestVal_v   = inf;
    bestTh_v    = theta0_v;
    bestCache_v = [];

    for k = 1:nStarts_v
        try
            th_k = fmincon(obj_var, starts_v(:,k), [],[],[],[], ...
                           lb_v, ub_v, [], fmin_opts_v);
            [val_k, cache_k] = nlZ_laplace(th_k, x_s, r, m0, lap);
            if val_k < bestVal_v
                bestVal_v   = val_k;
                bestTh_v    = th_k;
                bestCache_v = cache_k;
            end
        catch
        end
    end

    % Fallback: if every start failed, run one last attempt at theta0
    if isempty(bestCache_v)
        [bestVal_v, bestCache_v] = nlZ_laplace(theta0_v, x_s, r, m0, lap);
        bestTh_v = theta0_v;
    end

    l_g  = bestTh_v(1);
    sf_g = bestTh_v(2);

    % =================================================================
    % 5. PREDICT AT TEST POINTS
    % =================================================================
    z_score = norminv(1 - alpha_error / 2);

    % --- A) Mean prediction from mean GP ---
    K_star_f = kernel_main(x_s, x_test_s, [l_f; sf_f]);
    y_hat_s  = K_star_f' * alpha_f;

    % --- B) Epistemic variance from mean GP ---
    %   Var[f*] = sf^2 - v' v   (stationary RBF: k(*,*) = sf^2)
    v_f = L_f \ K_star_f;
    var_epistemic_s = max(0, sf_f^2 - sum(v_f.^2, 1)');

    % --- C) Aleatoric variance from Laplace variance GP ---
    %   g_mu  = m0 + k_*' alpha_g       (posterior mean of g)
    %   g_var = sf_g^2 - s' B^{-1} s    (posterior variance of g)
    %   E[sigma^2] = exp(g_mu + 0.5 g_var)
    K_star_g = rbf_kernel_var(x_s, x_test_s, bestTh_v);

    g_mu_test = m0 + K_star_g' * bestCache_v.alpha_g;

    s_g  = bsxfun(@times, bestCache_v.sqrtW, K_star_g);
    v_g  = bestCache_v.Lb \ s_g;
    g_var_test = max(0, sf_g^2 - sum(v_g.^2, 1)');

    var_aleatoric_s = exp(g_mu_test + 0.5 * g_var_test);

    % --- D) Total variance & rescale to original y scale ---
    var_total_s     = var_epistemic_s + var_aleatoric_s;

    y_hat         = y_hat_s * sy + mu_y;
    var_epistemic = var_epistemic_s * sy^2;
    var_aleatoric = var_aleatoric_s * sy^2;
    var_total     = var_total_s     * sy^2;

    CI = z_score * sqrt(var_total);

    theta_main = [l_f; sf_f; sn_f];
    theta_aux  = [l_g; sf_g];

    % =================================================================
    % 6. DIAGNOSTICS
    % =================================================================

    % --- Aleatoric variance at training points ---
    K_self_g    = rbf_kernel_var(x_s, x_s, bestTh_v);
    g_mu_train  = m0 + K_self_g' * bestCache_v.alpha_g;
    s_g_tr      = bsxfun(@times, bestCache_v.sqrtW, K_self_g);
    v_g_tr      = bestCache_v.Lb \ s_g_tr;
    g_var_train = max(0, sf_g^2 - sum(v_g_tr.^2, 1)');
    var_aleatoric_train_s = exp(g_mu_train + 0.5 * g_var_train);

    % --- Epistemic variance at training points ---
    v_f_tr = L_f \ K_f_sig;
    var_epistemic_train_s = max(0, sf_f^2 - sum(v_f_tr.^2, 1)');

    % --- Training coverage ---
    var_total_train_s = var_epistemic_train_s + var_aleatoric_train_s;
    CI_train_s = z_score * sqrt(var_total_train_s);
    train_coverage = mean(abs(r) <= CI_train_s);

    diagnostics.var_main            = var_epistemic;
    diagnostics.aux_eff             = var_aleatoric;
    diagnostics.mu_g                = g_mu_test;
    diagnostics.g_var               = g_var_test;
    diagnostics.g_mu_train          = g_mu_train;
    diagnostics.g_var_train         = g_var_train;
    diagnostics.residuals           = r * sy;
    diagnostics.log_r2              = log(r2);
    diagnostics.sn_f                = sn_f;
    diagnostics.m0                  = m0;
    diagnostics.nll_mean            = bestVal_m;
    diagnostics.nll_var             = bestVal_v;
    diagnostics.train_coverage      = train_coverage;
    diagnostics.var_aleatoric_train = var_aleatoric_train_s * sy^2;

end


% =========================================================================
% NEGATIVE LOG MARGINAL LIKELIHOOD – MEAN GP
% =========================================================================
function nll = neg_log_likelihood_mean(theta, x, y, kern)
    l  = exp(theta(1));
    sf = exp(theta(2));
    sn = exp(theta(3));
    N  = length(y);
    K  = kern(x, x, [l; sf]) + sn^2 * eye(N) + 1e-6 * eye(N);
    try
        L     = chol(K, 'lower');
        alpha = L' \ (L \ y);
        nll   = 0.5*(y'*alpha) + sum(log(diag(L))) + 0.5*N*log(2*pi);
    catch
        nll = inf;
    end
end


% =========================================================================
% ISOTROPIC RBF KERNEL – VARIANCE GP
% =========================================================================
function K = rbf_kernel_var(xa, xb, theta)
    ell = theta(1);
    sf2 = theta(2)^2;
    D2  = pdist2(xa, xb, 'euclidean').^2;
    K   = sf2 * exp(-0.5 * D2 / ell^2);
end


% =========================================================================
% LAPLACE NEGATIVE LOG EVIDENCE  +  CACHED MAP QUANTITIES
% =========================================================================
function [nlZ, cache] = nlZ_laplace(theta, xs, r, m0, opts)

    n = size(xs, 1);
    K = rbf_kernel_var(xs, xs, theta);
    K = K + opts.jitterK * eye(n);

    % Cholesky of prior covariance
    Lk = chol(K, 'lower');

    % ----- MAP for g via damped Newton -----
    g  = m0 * ones(n, 1);
    lp = logpost_var(g, r, m0, Lk);

    for it = 1:opts.maxIter

        % Prior contribution
        alpha_g = Lk' \ (Lk \ (g - m0));

        % Likelihood derivatives
        %   log p(r_i|g_i) = -0.5 [ g_i + r_i^2 exp(-g_i) ]
        exp_neg_g = exp(-g);
        dl    = -0.5 * (1 - (r.^2) .* exp_neg_g);        % gradient
        W     =  0.5 * (r.^2) .* exp_neg_g;               % neg-Hessian
        sqrtW = sqrt(max(W, 0));

        % B = I + sqrt(W) K sqrt(W)
        B  = eye(n) + (sqrtW * sqrtW') .* K;
        try
            Lb = chol(B, 'lower');
        catch
            B  = B + 1e-6 * eye(n);
            Lb = chol(B, 'lower');
        end

        % Posterior gradient
        grad = -alpha_g + dl;

        % Newton direction via Woodbury
        Kg    = K * grad;
        s     = sqrtW .* Kg;
        v     = Lb \ s;
        w     = Lb' \ v;
        delta = Kg - K * (sqrtW .* w);

        % Check convergence
        if max(abs(delta)) < opts.tol
            break;
        end

        % Backtracking line-search
        step   = 1.0;
        g_new  = clamp_g(g + step * delta, opts);
        lp_new = logpost_var(g_new, r, m0, Lk);

        while ~isfinite(lp_new) || lp_new < lp
            step = step / 2;
            if step < 1e-6, break; end
            g_new  = clamp_g(g + step * delta, opts);
            lp_new = logpost_var(g_new, r, m0, Lk);
        end

        g  = g_new;
        lp = lp_new;
    end

    % ----- Final MAP quantities -----
    alpha_g   = Lk' \ (Lk \ (g - m0));
    exp_neg_g = exp(-g);
    W         = 0.5 * (r.^2) .* exp_neg_g;
    sqrtW     = sqrt(max(W, 0));

    B  = eye(n) + (sqrtW * sqrtW') .* K;
    try
        Lb = chol(B, 'lower');
    catch
        B  = B + 1e-6 * eye(n);
        Lb = chol(B, 'lower');
    end

    % Laplace negative log evidence
    loglik  = sum(-0.5 * (g + (r.^2) .* exp_neg_g));
    quad    = 0.5 * (g - m0)' * alpha_g;
    logdetB = 2 * sum(log(diag(Lb)));

    nlZ = quad - loglik + 0.5 * logdetB;

    if ~isfinite(nlZ), nlZ = 1e12; end

    cache.g_hat   = g;
    cache.sqrtW   = sqrtW;
    cache.Lb      = Lb;
    cache.alpha_g = alpha_g;
    cache.sf2     = theta(end)^2;
end


% =========================================================================
% LOG POSTERIOR (for line-search in Newton iterations)
% =========================================================================
function lp = logpost_var(g, r, m0, Lk)
    alpha_g = Lk' \ (Lk \ (g - m0));
    prior   = -0.5 * (g - m0)' * alpha_g;
    lik     = sum(-0.5 * (g + (r.^2) .* exp(-g)));
    lp      = prior + lik;
end


% =========================================================================
% CLAMP g TO [gMin, gMax]
% =========================================================================
function g = clamp_g(g, opts)
    g = min(max(g, opts.gMin), opts.gMax);
end
