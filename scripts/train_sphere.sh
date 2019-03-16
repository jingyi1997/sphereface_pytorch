export PYTHONPATH=$ROOT:$PYTHONPATH
now=$(date +"%Y%m%d_%H%M%S")
GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0 \
    srun --mpi=pmi2 --job-name $1 --partition=HA_vechicle -n1 --gres=gpu:8 --ntasks-per-node=8 \
        python -u train_sphere_epoch.py --loss_type a-softmax --ckpt experiment/a-softmax_epoch30_step18_bs512 --epochs 30  --lr_steps 18 --bs 512 --classnum 10575 

