using SparseArrays
using LinearAlgebra
using KrylovKit
using Arpack
using Plots
using CSV
using DataFrames
using Printf
using PyCall
using ITensors, ITensorMPS
using ProgressBars
using Random
using JLD2
const pynp = pyimport("numpy")

BLAS.set_num_threads(1) 
ITensors.blas_get_num_threads() 

include("utils/my_cs_tools_v250625.jl")
include("utils/data_utils.jl")


# parameters
N = parse(Int, ARGS[1]) #number of qubits
J = 1.0 # parameter of Hamiltonian
h = 0.0 # parameter of Hamiltonian
ndata = 1500 # number of data
nsamp = 500 #shadow size



# time evolution & classical shadow & observable
seed_list = [i for i=1:ndata]

for seed in seed_list
    psi = TEBD(J, h, N, seed) 
    cs = random_measurement(N,nsamp,psi)
    save( "data_N=$(N)/qstate_seed=$(seed).jld2", Dict("psi"=>psi) )
    save( "data_N=$(N)/cs_nsamp=$(nsamp)_seed=$(seed).jld2", Dict("cs"=>cs) )
end



# exact observable
a=1
b = div(N,2)
c = div(N,4)
d = div(3*N,4)
obs1 = [] # xy_corr_center
obs2 = [] # xy_corr
obs3 = [] # xxyy_nonlinear

for seed in seed_list
    psi = load("data_N=$(N)/qstate_seed=$(seed).jld2","psi")
    push!(obs1, XY(psi, b, b+1))
    push!(obs2, XY(psi, c, d))
    push!(obs3, XXYY_nonlinear(psi, b, b+1, b, b+1))
end

filename = @sprintf("gram_matrix_N=%d/xy_corr_center.csv", N)
pynp.savetxt(filename, obs1, delimiter=",")

filename = @sprintf("gram_matrix_N=%d/xy_corr.csv", N)
pynp.savetxt(filename, obs2, delimiter=",")

filename = @sprintf("gram_matrix_N=%d/xxyy_nonlinear.csv", N)
pynp.savetxt(filename, obs3, delimiter=",")


# estimate Pauli observables from classical shadow
cs_obs1 = [] # xy_corr_center
cs_obs2 = [] # xy_corr
cs_obs3 = [] # xxyy_nonlinear

str_corr_center = [0 for i=1:N]
str_corr_center[b] = 1
str_corr_center[b+1] = 2

str_corr = [0 for i=1:N]
str_corr[c] = 1
str_corr[d] = 2

str_nl1 = [0 for i=1:N]
str_nl1[b] = 1
str_nl1[b+1] = 1

str_nl2 = [0 for i=1:N]
str_nl2[b] = 2
str_nl2[b+1] = 2

for seed in seed_list
    cs = load("data_N=$(N)/cs_nsamp=$(nsamp)_seed=$(seed).jld2","cs")
    push!(cs_obs1, Pauli_epval(cs, str_corr_center))
    push!(cs_obs2, Pauli_epval(cs, str_corr))
    push!(cs_obs3, Pauli_epval(cs, str_nl1) * Pauli_epval(cs, str_nl2))
end

filename = @sprintf("gram_matrix_N=%d/cs_estimate_xy_corr_center.csv", N)
pynp.savetxt(filename, cs_obs1, delimiter=",")

filename = @sprintf("gram_matrix_N=%d/cs_estimate_xy_corr.csv", N)
pynp.savetxt(filename, cs_obs2, delimiter=",")

filename = @sprintf("gram_matrix_N=%d/cs_estimate_xxyy_nonlinear.csv", N)
pynp.savetxt(filename, cs_obs3, delimiter=",")