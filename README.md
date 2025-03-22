GP Heteroscadiscity
This repository contains a MATLAB function for heteroscedastic Gaussian Process (GP) regression with log-variance modeling. The code demonstrates how to:

Estimate noise variances from replicate measurements,

Optimize hyperparameters for a main GP (modeling the mean function),

Train an auxiliary GP on the log of the noise variance to ensure strictly positive predictions,

Combine the main GP’s predictive variance with the predicted noise variance for robust confidence intervals.

Contents
GP_heteroscadiscity.m
The main MATLAB function that implements heteroscedastic GP regression with log-variance modeling.

Usage
Call the function as follows:

matlab
Copy
[y_hat, CI_95, theta_main_hat, theta_aux_hat, sigma_y2, sigma_y2_test_hat] = ...
    GP_heteroscadiscity(x, y, kernel, x_test, theta0_main, theta0_aux);
x: Vector of training inputs (n×1).

y: Vector of training outputs (n×1). Repeated x values indicate replicates.

kernel: Function handle for your chosen kernel (e.g., RBF).

x_test: Vector of test inputs (m×1).

theta0_main: Initial hyperparameters for the main GP (length scale, signal variance).

theta0_aux: Initial hyperparameters for the auxiliary GP (length scale, signal variance, noise in log space).

Example (Pseudo-Code)
matlab
Copy
% Define or load your data
x = [...];     % training inputs
y = [...];     % training outputs
x_test = [...];% test inputs

% Define your kernel function (example: RBF)
kernel = @(X1, X2, theta) rbfKernel(X1, X2, theta);

% Initial guesses for hyperparameters
theta0_main = [1; 1];      % e.g., [lengthScale; signalVariance]
theta0_aux  = [0.5; 1; 0.1];% e.g., [auxLengthScale; auxSignalVariance; auxNoise]

% Run heteroscedastic GP
[y_hat, CI_95, theta_main_hat, theta_aux_hat, sigma_y2, sigma_y2_test_hat] = ...
    GP_heteroscadiscity(x, y, kernel, x_test, theta0_main, theta0_aux);

% Now y_hat is your predictive mean, CI_95 is half-width of 95% confidence intervals, etc.
(Make sure you have an appropriate kernel function, such as an RBF kernel, defined in your MATLAB path.)

Requirements
MATLAB (tested on version R2021a or later, but should work on older versions).

Optimization Toolbox (for fmincon) or an equivalent method to minimize the negative log marginal likelihood.

License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). You are free to share and adapt the material under the following terms:

Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made.

NonCommercial: You may not use the material for commercial purposes.

For the full license text, see: https://creativecommons.org/licenses/by-nc/4.0/

Author
Carl Emil Aae Eskildsen
Imperial College London
c.eskildsen@imperial.ac.uk
