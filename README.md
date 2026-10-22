# Geometrically Local Quantum Kernel (GLQK)

Source code for the numerical experiments in

> K. Chinzei, Q. H. Tran, N. Matsumoto, Y. Endo, and H. Oshima, "Provably scalable learning from quantum many-body data via locality," [arXiv:2509.13705](https://arxiv.org/abs/2509.13705).

In file names, `sk` denotes the shadow kernel and `tesk`/`TESK` the polynomial GLQK with the truncated shadow kernel.

|Folder|Figure|Task|
|---|---|---|
|`Regression_Tsym/`|Fig. 3(a)|Regression, translationally symmetric data|
|`Regression_nonTsym/`|Fig. 3(b)|Regression, general data|
|`Quantum_phase_recognition/`|Fig. 4|Quantum phase recognition|

## Requirements

**Python 3.10** (tested with 3.10.12; 3.11 should also work):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyter
```

**Julia 1.8.x** (`Manifest.toml` was generated with 1.8.0). In the same shell, at the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia -e 'using Pkg; Pkg.add("IJulia")'    # for the Julia notebooks
export JULIA_PROJECT=@.                     # the scripts call plain `julia`
```

If `pyimport("numpy")` fails, rebuild PyCall: `julia -e 'ENV["PYTHON"]=""; using Pkg; Pkg.build("PyCall")'`.

## Usage

### Regression

Run inside `Regression_Tsym/` or `Regression_nonTsym/`:

```bash
julia data.jl $n                                  # MPS data, classical shadows, labels
julia gram_sk.jl $n                               # Gram matrix (shadow kernel)
julia gram_tesk.jl $n $w $e                       # Gram matrix (GLQK); w = ζ, e = κ
python krr_ndep_sk.py      -n $n -nt $nt -t $t    # R^2 vs n (10 random splits)
python krr_ndep_tesk.py    -n $n -nt $nt -t $t
python krr_scatter_sk.py   -n $n -nt $nt -t $t    # scatter plots
python krr_scatter_tesk.py -n $n -nt $nt -t $t
jupyter notebook results.ipynb                    # figures
```

|Parameter|`Regression_Tsym`|`Regression_nonTsym`|
|---|---|---|
|`n`|10, 20, ..., 80|same|
|(`w`, `e`)|{1, 2, 4, 6} × {1, 2}|same|
|`nt` (R^2 vs n)|50, 100, 250, 500|4000, 8000|
|`nt` (scatter, `n` = 80)|100, 500|8000|
|`t`|`xy_corr_center`, `xy_corr`, `xxyy_nonlinear`, `EE`, `expXY`|same|

The GLQK scripts select (`w`, `e`) by cross-validation, so compute all eight GLQK Gram matrices first.

The shell scripts in `scripts/` launch all jobs in the background. Run them from the regression folder and wait for each step to finish:

```bash
bash scripts/script_data.sh;  while pgrep -f data.jl > /dev/null; do sleep 60; done
bash scripts/script_gram.sh;  while pgrep -f gram_   > /dev/null; do sleep 60; done
bash scripts/script_ndep.sh
bash scripts/script_scatter.sh
```

### Quantum phase recognition

Run inside `Quantum_phase_recognition/`. Parameters are set in the first cell of each notebook; run each notebook from top to bottom in a fresh kernel:

1. `Data_dmrg.ipynb` and `Model_SK.ipynb` (Julia): for `N` = 10, 20, ..., 80.
2. `Model_TESK.ipynb` (Julia): for each `N` and (`width`, `exponent`) ∈ {2, 4, 6} × {1, 2}.
3. `SVM_SK.ipynb` and `SVM_TESK.ipynb` (Python): run the "n-dependence" cell with `num_train` = 500 and 250. The PCA cells give Figs. 4(d) and 4(e).
4. `results.ipynb`: Fig. 4(c).

## Correspondence with the paper

|`-t`|Paper|Implemented as|
|---|---|---|
|`xy_corr_center`|g1 = ⟨X₁Y₂⟩|⟨X_{n/2} Y_{n/2+1}⟩|
|`xxyy_nonlinear`|g2 = ⟨X₁X₂⟩⟨Y₁Y₂⟩|⟨X_{n/2} X_{n/2+1}⟩⟨Y_{n/2} Y_{n/2+1}⟩|
|`xy_corr`|g3 = ⟨X₁Y_{n/2+1}⟩|⟨X_{⌊n/4⌋} Y_{⌊3n/4⌋}⟩ (separation n/2)|
|`EE`|g4 = −tr(ρ₁₂ log ρ₁₂)|entropy of the RDM on sites n/2, n/2+1|
|`expXY`|g5 = exp(⟨X₁Y₂⟩)|exp(⟨X_{n/2} Y_{n/2+1}⟩)|

Both Hamiltonians are defined on a ring (periodic boundary condition). By translation symmetry (D1) or the translation invariance of the data distribution (D2), these sites are equivalent to those in the paper.