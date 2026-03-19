#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64g
#SBATCH -J "val_webscraping"
#SBATCH -o val_webscraping%j.out
#SBATCH -e val_webscraping%j.err
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:H200:1

cd $SLURM_SUBMIT_DIR/..

module load python/3.11.10
module load cuda/12.4.0/3mdaov5

python -m venv env
source env/bin/activate

pip install --upgrade pip
pip install unsloth
pip install numpy
pip install pandas
pip install playwright

playwright install 

python -m webscraping.update_website_dataset_val

