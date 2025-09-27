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



# parameters
N=parse(Int, ARGS[1])
ndata = 1500
nsamp = 500
tau = 1.0
gamma = 1.0
width = parse(Int, ARGS[2])
exponent = parse(Int, ARGS[3])



# Calculate gram matrix
seed_list = [i for i=1:ndata]
AGL = [[(j-1)%N+1 for j=i:i+width-1] for i=1:N]

cs_list = []
for seed in seed_list
    x = load("data_N=$(N)/cs_nsamp=$(nsamp)_seed=$(seed).jld2","cs")
    y = to_ClassicalShadow_TESK(x, gamma, tau, exponent, AGL)
    push!(cs_list, y)
end

G = zeros(Float64, ndata, ndata)
for i=1:ndata
    for j=i:ndata
        cs1 = cs_list[i]
        cs2 = cs_list[j]
        G[i,j] = G[j,i] = TESK(cs1, cs2)
    end
end

filename = @sprintf("gram_matrix_N=%d/TESK_ndata=%d_nsamp=%d_w=%d_e=%d.csv", N, ndata, nsamp, width, exponent)
pynp.savetxt(filename, G, delimiter=",")
