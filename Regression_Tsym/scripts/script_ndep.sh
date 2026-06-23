#!/bin/bash

for n in 10 20 30 40 50 60 70 80
do
for nt in 50 100 250 500
do
for t in xy_corr_center xy_corr xxyy_nonlinear EE expXY
do
nohup python krr_ndep_sk.py -n $n -nt $nt -t $t &
nohup python krr_ndep_tesk.py -n $n -nt $nt -t $t &
done
done
done
