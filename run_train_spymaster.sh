#!/bin/bash

#SBATCH --time=1:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=64000M   # 64G memory per CPU core
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -J "spymaster_test"   # job name
#SBATCH --qos=cs
#SBATCH --partition=cs

cd /home/cayjobla/CodenamesAI
source codenames_env/bin/activate

python train_spymaster.py \
	--spymaster gpt2 \
	--num_games 10 \
	--save_dir models \
	--device cuda \
	--do_train 
