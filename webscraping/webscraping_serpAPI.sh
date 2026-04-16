#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64g
#SBATCH -J "test_webscraping"
#SBATCH -o test_webscraping%j.out
#SBATCH -e test_webscraping%j.err
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:H200:1

cd $SLURM_SUBMIT_DIR/..

module load python/3.11.10
module load cuda/12.4.0/3mdaov5

python -m venv serp_env
source serp_env/bin/activate

pip install --upgrade pip
pip install numpy
pip install pandas
pip install playwright
pip install vllm 
pip install bitsandbytes

playwright install 

python -m webscraping.webscraping_serpAPI

