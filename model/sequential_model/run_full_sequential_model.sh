#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=128g                    # allocate 128 GB of memory
#SBATCH -J "industry_group_sequential_model"         # name of the job
#SBATCH -o industry_group_sequential_model%j.out     # name of the output file
#SBATCH -e industry_group_sequential_model%j.err     # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 24:00:00                   # time limit of 12 hours
#SBATCH --gres=gpu:H200:1             # request 1 H200 GPU

cd $SLURM_SUBMIT_DIR/../..

module load python/3.11.10
module load cuda/12.4.0/3mdaov5

python -m venv env
source env/bin/activate

pip install --upgrade pip

# Install dependencies
pip install numpy
pip install bitsandbytes
pip install peft
pip install torch
pip install transformers
pip install datasets
pip install tqdm
pip install scikit-learn
pip install joblib

python -m model.sequential_model.run_full_sequential_model