for seed in 20; 
do

    output_dir=ft_datasets/dolly_dataset/random_selection_${seed}
    number=100

    mkdir -p $output_dir

    python random_selection.py \
        --input_dir ft_datasets/dolly_dataset/databricks-dolly-15k-no-safety.jsonl \
        --output_file ${output_dir}/top_${number}.json \
        --num_samples $number \
        --seed ${seed} \
        --mode random

    output_dir_model="1.1_harmful_scores/Dolly"

    torchrun --nnodes 1 --master_port=29501 --nproc_per_node 2 finetuning.py \
        --batch_size_training 20 --lr 5e-5 \
        --gradient_accumulation_steps 1 --weight_decay 0 \
        --num_epochs 5 \
        --dataset dolly_dataset \
        --enable_fsdp \
        --data_path ${output_dir}/top_${number}.json \
        --model_name "./ckpts/Llama-2-7B-Chat-fp16" \
        --pure_bf16 \
        --dist_checkpoint_root_folder experiments/ \
        --dist_checkpoint_folder ${output_dir_model}/dolly-7b-random_selection/${seed} \
        --save_every_epoch False \
        --seed ${seed}


    python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path experiments/${output_dir_model}/dolly-7b-random_selection/${seed}-./ckpts/Llama-2-7B-Chat-fp16  \
            -consolidated_model_path experiments/${output_dir_model}/dolly-7b-random_selection/${seed} \
            -HF_model_path_or_name "ckpts/Llama-2-7B-Chat-fp16"

    rm -r experiments/${output_dir_model}/dolly-7b-random_selection/${seed}-./ckpts/Llama-2-7B-Chat-fp16

    for i in 1 2 3 4 5 6 7 8 9 10 11
    do
    python -u safety_evaluation/question_inference.py \
        --model_name experiments/${output_dir_model}/dolly-7b-random_selection/${seed} \
        --prompt_file safety_evaluation/data/category_${i}.csv \
        --prompt_template_style dolly \
        --output_file safety_evaluation/question_output/category_${i}_dolly_7b-random_selection_${seed}.jsonl

    # python safety_evaluation/gpt4_eval.py --input_file safety_evaluation/question_output/category_${i}_dolly_7b-random_selection_${seed}.jsonl
    done
done