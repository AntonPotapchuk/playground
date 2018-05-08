#!/bin/bash
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --time 2-00:00:00
#SBATCH --gres=gpu:tesla
#SBATCH --mem 11000
#SBATCH --job-name Pommerman
#SBATCH --output ./logs/jupyter-log-%J.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=antpotapchuk@gmail.com
#SBATCH --cpus-per-task=16

# Some modules
module load python-3.6.0
module load cudnn-6.0
source activate pommerman

## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)

## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "\n\n   Copy/Paste this in your local terminal to ssh tunnel with remote "
echo        "   ------------------------------------------------------------------"
echo        "   ssh -N -L $ipnport:$ipnip:$ipnport $(whoami)@$(dig +short -x $ipnip | rev | cut -c 2- | rev)"
echo        "   ------------------------------------------------------------------"
echo -e "\n\n   Then open a browser on your local machine to the following address"
echo        "   ------------------------------------------------------------------"
echo        "   localhost:$ipnport                                                "
echo -e     "   ------------------------------------------------------------------\n\n"
sleep 1

## start an ipcluster instance and launch jupyter server
jupyter-notebook --no-browser --port=$ipnport --ip=$ipnip

