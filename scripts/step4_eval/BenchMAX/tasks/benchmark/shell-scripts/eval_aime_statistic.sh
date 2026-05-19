
Output_dir="/mnt/nas/users/zhouhao.zh/new_project/chat_transfer/hf_instruct_res/Qwen3/Qwen3-4B/aime"
Output_dir="/mnt/nas/users/zhouhao.zh/new_project/chat_transfer/hf_instruct_res/Qwen3/merge/tian/tian_qwen3_base_4b_hu10b/aime"
Model_path="/mnt/nas/users/zhouhao.zh/LLMs/Qwen3/Qwen3-4B"
Model_path="/mnt/nas_intern/users/zhouhao.zh/new_project/chat_transfer/ckpt/qwen3/merged/tian/tian_qwen3_base_4b_hu10b"
cd /mnt/nas/users/zhouhao.zh/latentcot/code/ImplicitStep
python -m tools.token_by_type \
    --input_file ${Output_dir}/aime.parquet \
    --tokenizer_name ${Model_path} \
    --output_file ${Output_dir}/aime.txt

cd /mnt/nas/users/zhouhao.zh/latentcot/code/ImplicitStep
python -m  tools.fine_grained_analysis \
    --input_file ${Output_dir}/aime.parquet \
    --tokenizer_name ${Model_path}