output_dir="experiments/0_prepare_dataset/Dolly/Ours"
grad_file="experiments/0_prepare_dataset/safe_anchor1_gradients/normalized_average_num10.pt" # no influence here
data_path="ft_datasets/dolly_dataset/databricks-dolly-15k-no-safety.jsonl"

# STEP1: Calculate the gradient of each training data and calculate the cosine similarity between the training gradients and the anchor gradients.
mkdir -p $output_dir
python -m online_gradient.rank rank \
--grad_file $grad_file \
--dataset dolly_dataset \
--data_path $data_path \
--batch_size_training 1 \
--output_dir $output_dir \
--model_name ckpts/Llama-2-7B-Chat-fp16 \
--normalize False  \
--self_influence True 2>&1 | tee $output_dir/log.txt



# STEP2: We write the top_k data points to a file
# target_data=demo
type=top # bottom
k=100 # we take the top 100 examples with the highest final scores

# we use a harmful anchor dataset, and two safe anchor datasets
anchor_gradident_directory="experiments/0_prepare_dataset/Dolly/Ours"
# we assign 1 to harmful anchor dataset and -1 to safe anchor dataset
weight="1"
# output file directory
write_to="ft_datasets/dolly_dataset/ours_selfunn"

mkdir -p $write_to

python3 -m online_gradient.rank write_data \
--dataset dolly_dataset \
--data_path $data_path \
--output_dir $anchor_gradident_directory \
--weight $weight \
--k $k \
--type $type \
--write_to $write_to \
--dataset_name dolly