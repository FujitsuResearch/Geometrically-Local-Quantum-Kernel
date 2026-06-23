#!/bin/bash

for n in 10 20 30 40 50 60 70 80
do
nohup julia data.jl $n &
done
