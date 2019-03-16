export PYTHONPATH=$ROOT:$PYTHONPATH
now=$(date +"%Y%m%d_%H%M%S")
GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0 \
    srun --mpi=pmi2 --job-name $1 --partition=HA_vechicle -n1 --gres=gpu:1 --ntasks-per-node=1 \
        python -u train_sphere_epoch.py --loss_type regular --ckpt experiment/regular_epoch30_step18_bs512_norm15 --epochs 30 --lr_steps 18 --bs 512 --classnum 10575  --reg_weight 6 

