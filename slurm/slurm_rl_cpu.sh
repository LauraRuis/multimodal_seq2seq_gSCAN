#!/bin/bash
#
# All commands that start with SBATCH contain commands that are just
# used by SLURM for scheduling
#################
# set a job name
#SBATCH --job-name=test_rl
#################
# a file for job output, you can check job progress
#SBATCH --output=test_rl.out
#################
# a file for errors from the job
#SBATCH --error=test_rl.err
#################
# time you think you need; default is one hour
# in minutes
# In this case, hh:mm:ss, select whatever time you want, the less you
# ask for the faster your job will run.
# Default is one hour, this example will run in less than 5 minutes.
#SBATCH --time=24:00:00
#################
# We are submitting to the gpu partition, if you can submit to the hns partition, change this to -p hns_gpu.
#SBATCH --qos=batch
#################
# number of nodes you are requesting
#SBATCH --nodes=1
#################
# memory per node; default is 4000 MB per CPU
#SBATCH --mem=32000
# job ends or fails, careful, the email could end up in your clutter folder
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=lr2715@courant.nyu.edu

module load python-3.7
echo "Starting Job.."
source rlenv/bin/activate
srun python3.7 -m seq2seq --mode=rl --data_directory=data/rl_simplified_dev --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=rl_method --training_batch_size=50 --timesteps_per_epoch=4000 --training_batch_size=200 --target_kl=0.05 --vf_coef=1. --evaluate_every_epoch=1 &> rl_method/test_rl.txt
