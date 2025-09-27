# Geometrically-Local-Quantum-Kernel (GLQK)
This repository contains the source codes for the Geometrically local quantum kernel (GLQK), a novel method for efficiently learning quantum many-body data.
It has been introduced in the paper titled "Learning quantum many-body data locally: A provably scalable framework."

Links: https://arxiv.org/abs/2509.13705

# Installation

1. **Clone the repository**
```bash
git clone https://github.com/FujitsuResearch/Geometrically-Local-Quantum-Kernel.git
cd Geometrically-Local-Quantum-Kernel
```


2. **Install the required packages for Python (tested with v3.10.12)**
```bash
# Create an environment
python -m venv .venv

# Activate the environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

3. **Install the required packages for Julia (tested with v1.8.0)**
```Julia
# Press the `]` key to enter Pkg mode
julia> ]

# Activate the local project environment
(@v1.8) pkg> activate .

# Install all dependencies from the Manifest.toml file
(Geometrically-Local-Quantum-Kernel) pkg> instantiate
```


# Usage
This repository is organized into three main folders, each corresponding to specific figures in our paper. 
Below is a guide to help you navigate the codebase and reproduce our results:
- `/Regression_Tsym` for Fig.3(a) in the paper
- `/Regression_nonTsym` for Fig.3(b) in the paper
- `/Quantum_phase_recognition` for Fig.4 in the paper

In the file names, sk and tesk represent shadow kernel and GLQK, respectively. 

## `/Regression_Tsym`
- `/Regression_Tsym/data.jl` generates MPS data and corresponding classical shadows.

- `/Regression_Tsym/gram_sk.jl` and `/Regression_Tsym/gram_tesk.jl` calculates gram matrices with shadow kernel and GLQK, respectively.

- `/Regression_Tsym/krr_ndep_sk.py` and `/Regression_Tsym/krr_ndep_tesk.py` calculates the learning accuracy with shadow kernel and GLQK, respectively.

- `/Regression_Tsym/krr_scatter_sk.py` and `/Regression_Tsym/krr_scatter_tesk.py` calculates the scatter plot of predictions with shadow kernel and GLQK, respectively.

- `/Regression_Tsym/results.ipynb` plots the results obtained from the above calculations.

```bash
# n is the number of qubits
julia data.jl $n

# n is the number of qubits, w is the subsystem size, and e is the exponent of polynomial GLQK
julia gram_sk.jl $n
julia gram_tesk.jl $n $w $e

# n is the number of qubits, nt is the number of training data, and t is the type of observables (xy_corr_center, xy_corr, xxyy_nonlinear)
python krr_ndep_sk.py -n $n -nt $nt -t $t
python krr_ndep_tesk.py -n $n -nt $nt -t $t

# n is the number of qubits, nt is the number of training data, and t is the type of observables (xy_corr_center, xy_corr, xxyy_nonlinear)
python krr_scatter_sk.py -n $n -nt $nt -t $t
python krr_scatter_tesk.py -n $n -nt $nt -t $t
```

## `/Regression_nonTsym`
- `/Regression_nonTsym/data.jl` generates MPS data and corresponding classical shadows.

- `/Regression_nonTsym/gram_sk.jl` and `/Regression_nonTsym/gram_tesk.jl` calculates gram matrices with shadow kernel and GLQK, respectively.

- `/Regression_nonTsym/krr_ndep_sk.py` and `/Regression_nonTsym/krr_ndep_tesk.py` calculates the learning accuracy with shadow kernel and GLQK, respectively.

- `/Regression_nonTsym/krr_scatter_sk.py` and `/Regression_nonTsym/krr_scatter_tesk.py` calculates the scatter plot of predictions with shadow kernel and GLQK, respectively.

- `/Regression_nonTsym/results.ipynb` plots the results obtained from the above calculations.

```bash
# n is the number of qubits
julia data.jl $n

# n is the number of qubits, w is the subsystem size, and e is the exponent of polynomial GLQK
julia gram_sk.jl $n
julia gram_tesk.jl $n $w $e

# n is the number of qubits, nt is the number of training data, and t is the type of observables (xy_corr_center, xy_corr, xxyy_nonlinear)
python krr_ndep_sk.py -n $n -nt $nt -t $t
python krr_ndep_tesk.py -n $n -nt $nt -t $t

# n is the number of qubits, nt is the number of training data, and t is the type of observables (xy_corr_center, xy_corr, xxyy_nonlinear)
python krr_scatter_sk.py -n $n -nt $nt -t $t
python krr_scatter_tesk.py -n $n -nt $nt -t $t
```


## `/Quantum_phase_recognition`
- `/Quantum_phase_recognition/Data_dmrg.ipynb` generates MPS data and corresponding classical shadows.

- `/Quantum_phase_recognition/Model_SK.ipynb` and `/Quantum_phase_recognition/Model_TESK.ipynb` calculates gram matrices with shadow kernel and GLQK, respectively.

- `/Quantum_phase_recognition/SVM_SK.ipynb` and `/Quantum_phase_recognition/SVM_TESK.ipynb` calculates the learning accuracy and PCA with shadow kernel and GLQK, respectively.

- `/Quantum_phase_recognition/results.ipynb` plots the results obtained from the above calculations.

Please execute the jupyter notebooks.

