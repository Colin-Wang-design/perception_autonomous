#!/bin/sh
### ------------- specify queue name ---------------- 
#BSUB -q c02516

### ------------- specify gpu request---------------- 
#BSUB -gpu "num=1:mode=exclusive_process"

### ------------- specify job name ---------------- 
#BSUB -J testjob_piton

### ------------- specify number of cores ---------------- 
#BSUB -n 4 
#BSUB -R "span[hosts=1]"

### ------------- specify CPU memory requirements ---------------- 
#BSUB -R "rusage[mem=20GB]"

#BSUB -W 12:00 
#BSUB -o output/OUTPUT_FILE%J.out 
#BSUB -e output/OUTPUT_FILE%J.err

source "/zhome/15/3/203515/perception_autonomous/venv/bin/activate"
python "main.py"
