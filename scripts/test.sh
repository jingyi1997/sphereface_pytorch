export PYTHONPATH=$ROOT:$PYTHONPATH
now=$(date +"%Y%m%d_%H%M%S")
start_epoch=20
step=1
end_epoch=30
model_folder="experiment/regular_epoch30_step18_bs512/checkpoints/"
while test $start_epoch -le $end_epoch; do
  model_name="epoch_${start_epoch}_ckpt.pth.tar"
  echo "testing: "$model_name
  if [ -f ${model_folder}${model_name} ]; then 
    GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0 \
    srun --mpi=pmi2 --job-name $1 --partition=HA_vechicle -n1 --gres=gpu:1 --ntasks-per-node=1 \
        python -u eval_lfw.py --loss_type regular --ckpt experiment/regular_epoch30_step18_bs512 --lfw /mnt/lustre/xujingyi/sphereface/preprocess/result/lfw-112X96 --model=$start_epoch --classnum 10575 
    ((start_epoch+=step))
  else
    echo "sleep"
    sleep 30m
  fi
done
