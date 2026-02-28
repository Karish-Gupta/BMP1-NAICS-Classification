#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8g
#SBATCH -J "gqp_webscraping"
#SBATCH -o gqp_webscraping%j.out
#SBATCH -e gqp_webscraping%j.err
#SBATCH -p academic
#SBATCH -t 05:00:00
#SBATCH --gres=gpu:A100:1

#cd $SLURM_SUBMIT_DIR/../..

#module load python/3.10.2/mqmlxcf

python -m venv env
source env/bin/active

pip install --upgrade pip
#pip install --upgrade -q accelerate bitsandbytes
pip install numpy
pip install pandas
pip install playwright

python update_website_dataset.py

