eval "$(/mnt/nas/users/zichen.shx/miniconda3/bin/conda shell.bash hook)" 
conda activate /mnt/nas/users/zhouhao.zh/envs/vllm_mt

CUDA_VISIBLE_DEVICES=0 python /mnt/nas/users/zhouhao.zh/new_project/chat_transfer/code/BenchMAX/tasks/benchmark/test/test_load_model.py &

CUDA_VISIBLE_DEVICES=1 python /mnt/nas/users/zhouhao.zh/new_project/chat_transfer/code/BenchMAX/tasks/benchmark/test/test_load_model.py &


wait 
sleep 5min

echo "finishing "