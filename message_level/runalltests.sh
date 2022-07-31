#!/bin/bash
#export PYTHONPATH="/home/ak2149/rds/hpc-work/multi-agent-disentanglement/"
export PYTHONPATH="/home/iris/Desktop/multi-agent-disentanglement/"
source ../../venv/bin/activate


python -m disentanglement.message_level.main g
python -m disentanglement.message_level.main f

#python -m disentanglement.message_level.main x
#python -m disentanglement.message_level.main a
#python -m disentanglement.message_level.main b
#python -m disentanglement.message_level.main d
#python -m disentanglement.message_level.main e
#python -m disentanglement.message_level.main c


#cd ../direct_model
#python -m disentanglement.direct_model.train_direct

#cd ../direct_timestep_model
#python -m disentanglement.direct_timestep_model.train_direct

#cd ../../adversarial_object_tracking/
#./run_many.sh 100 100


#alphas=(0.001 0.01 0.1 1 10 100)
#betas=("$1")
#betas=(0.001 0.01 0.1 1 10 100)
#gammas=(1 10 0.1)
#cd ../../
#export PYTHONPATH=.

#cd disentanglement/message_leve
#for b in ${betas[@]}; do
#    for g in ${gammas[@]}; do
#	for a in ${alphas[@]}; do 
#	    #echo "running" $a $b $g
#	    #echo "starting $a $b $c"
#	    #sleep 1m &
#	    python -m disentanglement.message_level.multi_experiment_training $a $b $g &
#	done
#
#	for job in `jobs -p`
#	do
#	    echo $job
#	    wait $job
#	done
#    done
#done
