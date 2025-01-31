#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=02:00:00
#SBATCH --partition=short	
#SBATCH --job-name=forecast

module load Python/3.11.5-GCCcore-13.2.0
source .venv/bin/activate
echo "$We got this"
python src/bin/forecast.py
	