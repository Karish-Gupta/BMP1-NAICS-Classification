#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64g
#SBATCH -J "gqp_webscraping"
#SBATCH -o gqp_webscraping%j.out
#SBATCH -e gqp_webscraping%j.err
#SBATCH -p academic
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:H200:1

module load python/3.10.2/mqmlxcf

python -m venv env
source env/bin/active

pip install --upgrade pip
pip install numpy
pip install pandas
pip install playwright

python update_website_dataset.py

