#!/bin/sh

#SBATCH --partition=WORKQ
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Memory per node specification is in MB. It is optional.
#SBATCH --job-name="prep"
#SBATCH --output=cpu-test-srun.out
#SBATCH --mail-user=johnew@stud.ntnu.no
#SBATCH --mail-type=ALL
 
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"


module purge

module load gcccuda/2018a
module load OpenMPI/2.1.2
module load cuDNN/7.0.5.15 
module load TensorFlow/1.8.0-Python-3.6.4 
module load Python/3.6.4 
module load GTK+/2.24.32 

#module purge
#module load fosscuda/2018b
#module load CUDA/9.1.85
#module load TensorFlow


# starter virtual environment.
source /lustre1/work/johnew/EiT/cuda/bin/activate

# kjører kode.
mpirun python dataset_generator.py
 
uname -a
