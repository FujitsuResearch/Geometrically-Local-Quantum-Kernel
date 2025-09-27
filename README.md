# Geometrically-Local-Quantum-Kernel
This repository contains the source codes for the Geometrically local quantum kernel (GLQK), a novel method for efficiently learning quantum many-body data.
It has been introduced in the paper titled "Learning quantum many-body data locally: A provably scalable framework."

Links: https://arxiv.org/abs/2509.13705

# Usage
This repository is organized into three main folders, each corresponding to specific figures in our paper. 
Below is a guide to help you navigate the codebase and reproduce our results:
- `/Regression_Tsym` for Fig.3(a) in the paper
- `/Regression_nonTsym` for Fig.3(b) in the paper
- `/Quantum_phase_recognition` for Fig.4 in the paper

In the file names, sk and tesk represent shadow kernel and GLQK, respectively. 

## `/Regression_Tsym`
`/Regression_Tsym/data.jl` generates MPS data and corresponding classical shadows.

`/Regression_Tsym/gram_sk.jl` and `/Regression_Tsym/gram_tesk.jl` calculates gram matrices with shadow kernel and GLQK, respectively.

`/Regression_Tsym/krr_ndep_sk.py` and `/Regression_Tsym/krr_ndep_tesk.py` calculates the learning accuracy with shadow kernel and GLQK, respectively.

`/Regression_Tsym/krr_scatter_sk.py` and `/Regression_Tsym/krr_scatter_tesk.py` calculates the scatter plot of predictions with shadow kernel and GLQK, respectively.

`/Regression_Tsym/results.ipynb` plots the results obtained from the above calculations.

## `/Regression_nonTsym`
`/Regression_nonTsym/data.jl` generates MPS data and corresponding classical shadows.

`/Regression_nonTsym/gram_sk.jl` and `/Regression_nonTsym/gram_tesk.jl` calculates gram matrices with shadow kernel and GLQK, respectively.

`/Regression_nonTsym/krr_ndep_sk.py` and `/Regression_nonTsym/krr_ndep_tesk.py` calculates the learning accuracy with shadow kernel and GLQK, respectively.

`/Regression_nonTsym/krr_scatter_sk.py` and `/Regression_nonTsym/krr_scatter_tesk.py` calculates the scatter plot of predictions with shadow kernel and GLQK, respectively.

`/Regression_nonTsym/results.ipynb` plots the results obtained from the above calculations.


## `/Quantum_phase_recognition`
`/Quantum_phase_recognition/Data_dmrg.ipynb` generates MPS data and corresponding classical shadows.

`/Quantum_phase_recognition/Model_SK.ipynb` and `/Quantum_phase_recognition/Model_TESK.ipynb` calculates gram matrices with shadow kernel and GLQK, respectively.

`/Quantum_phase_recognition/SVM_SK.ipynb` and `/Quantum_phase_recognition/SVM_TESK.ipynb` calculates the learning accuracy and PCA with shadow kernel and GLQK, respectively.

`/Quantum_phase_recognition/results.ipynb` plots the results obtained from the above calculations.

