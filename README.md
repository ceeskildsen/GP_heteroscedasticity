# GP_heteroscedasticity

MATLAB implementation of heteroscedastic Gaussian-process regression with input-dependent noise variance.

The function fits a two-stage Gaussian-process model. First, a homoscedastic GP is fitted to estimate the conditional mean function. Second, the residual structure from the mean GP is used to fit a variance GP, in which the latent log-variance function is inferred using a Laplace approximation. The resulting prediction combines the posterior mean, epistemic variance from the mean GP, and aleatoric variance from the variance GP.

## Description

`GP_heteroscedasticity.m` provides a standalone implementation of heteroscedastic GP regression for data where the residual variance may change across the input space.

The workflow is:

1. Fit a homoscedastic GP for the mean function.
2. Compute debiased residuals.
3. Fit a variance GP to the residuals using a Laplace approximation for the latent log-variance function.
4. Predict the mean, epistemic variance, aleatoric variance, total predictive variance, and confidence intervals at test locations.

This version replaces the earlier auxiliary-GP approach based on direct log-residual variance modelling with a Laplace-approximated latent log-variance formulation.

## Usage

Call the function in MATLAB as follows:

```matlab
[y_hat, CI, theta_main, theta_aux, var_total, diagnostics] = ...
    GP_heteroscedasticity(x, y, kernel_main, x_test, alpha_error);
```

Example kernel:

```matlab
kernel_main = @(X1, X2, theta) theta(2)^2 .* ...
    exp(-0.5 .* pdist2(X1 ./ theta(1), X2 ./ theta(1)).^2);
```

Example call:

```matlab
x_test = linspace(min(x), max(x), 200)';

[y_hat, CI, theta_main, theta_aux, var_total, diagnostics] = ...
    GP_heteroscedasticity(x, y, kernel_main, x_test, 0.05);
```

The confidence interval can be plotted as:

```matlab
fill([x_test; flipud(x_test)], ...
     [y_hat - CI; flipud(y_hat + CI)], ...
     [0.8 0.8 0.8], 'EdgeColor', 'none');
hold on
plot(x_test, y_hat, 'k-', 'LineWidth', 2)
scatter(x, y, 25, 'filled')
```

## Inputs

- `x`: `n`-by-1 vector of training inputs.
- `y`: `n`-by-1 vector of training responses.
- `kernel_main`: function handle for the mean GP covariance function, with syntax `K = kernel_main(X1, X2, theta)`, where `theta = [l; sf]`.
- `x_test`: `m`-by-1 vector of test inputs.
- `alpha_error`: significance level for the confidence interval. The default is `0.05`, corresponding to a 95% confidence interval.

## Outputs

- `y_hat`: `m`-by-1 posterior mean predictions on the original response scale.
- `CI`: `m`-by-1 confidence-interval half-widths on the original response scale.
- `theta_main`: optimized mean-GP hyperparameters, `[l; sf; sn]`.
- `theta_aux`: optimized variance-GP hyperparameters, `[l_g; sf_g]`.
- `var_total`: `m`-by-1 total predictive variance, combining epistemic and aleatoric variance.
- `diagnostics`: structure containing intermediate quantities and diagnostic information from the fitted model.

## Requirements

- MATLAB R2022a or later recommended.
- Optimization Toolbox, for `fmincon`.
- Statistics and Machine Learning Toolbox, for functions such as `pdist`.

The function may also run in earlier MATLAB versions if the required functions are available.

## Notes

The model assumes a Gaussian-process prior for the mean function and a separate Gaussian-process prior for the latent log-variance function,

```text
g(x) = log sigma^2(x)
```

The variance GP is fitted using a Laplace approximation to handle the non-Gaussian likelihood induced by the residual-based variance model. The final predictive variance combines uncertainty in the mean prediction with input-dependent aleatoric variance.

This implementation is intended as a compact, reusable implementation of the heteroscedastic GP pipeline. For exact reproduction of manuscript figures and tables, use the full manuscript code repository associated with the publication.

## Citation

If you use this software, please cite the archived Zenodo version associated with the release.

A suggested citation format is:

```text
Eskildsen, C. E. GP_heteroscedasticity: MATLAB implementation of heteroscedastic Gaussian-process regression with input-dependent noise variance. Zenodo. https://doi.org/[insert DOI] ([year]).
```

Please replace `[insert DOI]` and `[year]` with the DOI and year of the Zenodo release you use.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

Carl Emil Eskildsen  
Imperial College London / University of Oxford
