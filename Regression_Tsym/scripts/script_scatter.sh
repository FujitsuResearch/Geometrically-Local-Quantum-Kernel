#!/bin/bash

for n in 80
do
for nt in 100 500
do
for t in xy_corr_center xy_corr xxyy_nonlinear EE expXY
do
nohup python krr_scatter_sk.py -n $n -nt $nt -t $t &
nohup python krr_scatter_tesk.py -n $n -nt $nt -t $t &
done
done
done
