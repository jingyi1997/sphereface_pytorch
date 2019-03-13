export PYTHONPATH=$ROOT:$PYTHONPATH
now=$(date +"%Y%m%d_%H%M%S")
start_epoch=18
step=1
end_epoch=22
while test $start_epoch -le $end_epoch; do
  model_name="epoch_${start_epoch}_ckpt.pth.tar"
  echo "testing: "$model_name
  GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0 \
    srun --mpi=pmi2 --job-name $1 --partition=HA_vechicle -n1 --gres=gpu:8 --ntasks-per-node=8 \
        python -u eval_lfw.py --loss_type a-softmax --ckpt experiment/a-softmax --lfw /mnt/lustre/xujingyi/sphereface/preprocess/result/lfw-112X96 --model=$start_epoch
  ((start_epoch+=step))
done
