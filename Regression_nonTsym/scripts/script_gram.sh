#!/bin/bash

for n in 10 20 30 40 50 60 70 80
do
nohup julia gram_sk.jl $n &
done

for n in 10 20 30 40 50 60 70 80
do
for w in 1 2 4 6
do
for e in 1 2
do
nohup julia gram_tesk.jl $n $w $e &
done
done
done
