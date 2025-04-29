orthogonality_constrained_pls
This repository contains a MATLAB implementation of Orthogonality-Constrained Partial Least Squares (OC-PLS) regression. The method enhances selectivity by forcing the latent-variable space to be orthogonal to known interfering signals, making it particularly useful in chemometric modeling of vibrational spectroscopy data.

📘 Citation
If you use this code in your research, please cite:

P. B. Skou, E. Hosseini, J. B. Ghasemi, A. K. Smilde, and C. E. Eskildsen
Orthogonality constrained inverse regression to improve model selectivity and analyte predictions from vibrational spectroscopic measurements
Analytica Chimica Acta, 2021, 1185:339073
https://doi.org/10.1016/j.aca.2021.339073

🔧 Usage
Function call:
[b, W, P, q, T] = pls_cons(X, y, LV, Sk);

Inputs:

X: n×p matrix of mean-centered predictors

y: n×1 mean-centered response vector

LV: number of latent variables to extract

Sk: p×k matrix of known interfering signals

Outputs:

b: regression vector

W, P, q, T: standard PLS model matrices

Example (pseudo-code):
% Define or load your data
x = [...]; % training inputs
y = [...]; % training outputs
x_test = [...]; % test inputs

% Define your kernel, e.g. RBF
kernel = @(X1,X2,θ) rbfKernel(X1,X2,θ);

% Run OC-PLS
[b, W, P, q, T] = pls_cons(x, y, LV, Sk);

🛠 Requirements
MATLAB R2021a or later (should work on older versions too)

Optimization Toolbox (for fmincon) or equivalent optimizer

📄 License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.
You are free to share and adapt the material under the following terms:
• Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
• NonCommercial — You may not use the material for commercial purposes.

Full license text: https://creativecommons.org/licenses/by-nc/4.0

✉️ Author
Carl Emil Aae Eskildsen
Imperial College London
c.eskildsen@imperial.ac.uk
