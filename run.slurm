#!/bin/bash
#SBATCH --job-name=run-opa
#SBATCH --output=run-opa.out
#SBATCH --error=run-opa.err
#SBATCH --ntasks=22
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
##SBATCH --ntasks-per-socket=7
#SBATCH --cpus-per-task=28
##SBATCH --mem=150000
#SBATCH --time=02:00:00
#SBATCH --cluster=mpi
#SBATCH --partition=opa-high-mem


echo "SLURM_JOB_ID="$SLURM_JOB_ID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_CORES_NODES"=$SLURM_CPUS_PER_TASK
echo "SLURM_TASKS"=$SLURM_NTASKS
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
echo "************************************************"

module purge
module load gcc/8.2.0
module load intel

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

M=${SLURM_NNODES}
NP=$SLURM_NTASKS
MPI="mpirun -np ${NP}"
APP_PERFIX="bin"
APP=("./radixnet")

DATA_PERFIX="/zfs1/cs3580_2017F/moh18/sdnn/bin"
NEURONS=("1024" "4096" "16384" "65536")
LAYERS=("120" "480" "1920")


if [ "${M}" = "2" ]
then
    I="0"
elif [ "$M" = "4" ]
then
    I="1"
elif [ "$M" = "6" ]
then
    I="2"
elif [ "${M}" = "8" ]
then
    I="3"
else
    echo "Invalid input"
    exit 1;     
fi

N=${NEURONS[${I}]}
for L in "${LAYERS[@]}"
do
    CMD="${MPI} ${APP_PERFIX}/${APP} -m 60000 -n ${N} -l ${L} ${DATA_PERFIX}/MNIST/ ${DATA_PERFIX}/DNN/"
    echo "Command=${CMD}"
    ${CMD}
    #exit;
done
exit 0;
