#!/bin/sh

#SBATCH --partition=WORKQ
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
# Memory per node specification is in MB. It is optional.
#SBATCH --job-name="hello_test"
#SBATCH --output=test-srun.out
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
module load intel/2018b
module load Python/3.6.6

# starter virtual environment.
source /lustre1/work/johnew/EiT/tf/bin/activate

# kjører kode.
mpirun python dataset_iterator.py 
 
uname -a
