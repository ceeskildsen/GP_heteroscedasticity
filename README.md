# GP_heteroscedasticity

This repository provides a MATLAB implementation of heteroscedastic Gaussian Process regression with automatic noise variance modeling. The function estimates both predictive mean and uncertainty intervals for data exhibiting non-constant variance.

## Description: 
Estimates noise variances from replicate measurements. Optimizes hyperparameters for a main GP modeling the mean function. Trains an auxiliary GP on the log-variance to guarantee positive noise predictions. Combines the main GP’s predictive variance with the predicted noise variance for robust 95 percent confidence intervals.

## Usage: 
Call the function in MATLAB as follows: 
```matlab
[y_hat, CI, theta_main_hat, theta_aux_hat, var_total] = GP_heteroscedasticity(x, y, kernel, x_test, alpha_error)
```

## Inputs: 
-`x` is an n-by-1 vector of training inputs.  
-`y` is an n-by-1 vector of training outputs (replicates allowed).  
-`kernel` is a function handle with syntax kernel(X1, X2, theta).  
-`x_test` is an m-by-1 vector of test inputs.
-`alpha_error` is the significance level for confidence interval (e.g. 0.05 for 95%)

## Outputs: 
-`y_hat` is an m-by-1 predictive mean at test points.  
-`CI` is an m-by-1 confidence interval half-width.  
-`theta_main_hat` is a 2-by-1 vector of hyperparameters for the main GP (length-scale; signal variance).  
-`theta_aux_hat` is a 3-by-1 vector of hyperparameters for the auxiliary GP (length-scale; signal variance; noise).  
-`var_total` is an m-by-1 vector of total predictive variance (main GP + heteroscedastic noise, de-standardized).   

## Requirements: 
MATLAB R2018a or later. Optimization Toolbox (for fmincon) or equivalent nonlinear optimizer and the Statistics and Machine Learning Toolbox.

## License: 
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell   
copies of the Software, and to permit persons to whom the Software is        
furnished to do so, subject to the following conditions:                     

The above copyright notice and this permission notice shall be included in   
all copies or substantial portions of the Software.                            

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR   
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,     
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER       
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.

## Author:
Carl Emil Aae Eskildsen  
Imperial College London  
c.eskildsen@imperial.ac.uk
