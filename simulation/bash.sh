#!/bin/bash
q_LIST=("1" "2" "5")
for sig in {1..10}; do 
	for nt_id in {1..4}; do 
		for set_id in {1..8}; do
			for q in "${q_LIST[@]}"; do 
				for seed in {1..100}; do
Python3 ../utils/simu.py $sig $nt_id $set_id $q $seed 
done
done
done
done
done
