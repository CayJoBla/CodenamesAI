#!/bin/bash

#SBATCH --time=1:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=4
#SBATCH --mem-per-cpu=64000M   # 64G memory per CPU core
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -J "codenames_train_test"   # job name
#SBATCH --qos=gpu

cd /home/cayjobla/CodenamesAI
source codenames_env/bin/activate

python train_shared.py \
	--model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
	--num_games 5000 \
	--save_dir models \
	--device cuda \
	--do_train 