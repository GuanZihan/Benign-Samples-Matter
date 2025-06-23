for seed in 20 ; 
do

    output_dir=ft_datasets/dolly_dataset/random_selection_${seed}
    number=2000

    # mkdir -p $output_dir

    python random_selection.py \
        --input_dir ft_datasets/dolly_dataset/databricks-dolly-15k-no-safety.jsonl \
        --output_file ${output_dir}/top_${number}.json \
        --num_samples $number \
        --seed ${seed} \
        --mode random \
    
    # python merge_data.py \
    #     --output_file ${output_dir}/top_${number}.json \
    #     --selected_dataset ft_datasets/dolly_dataset/self_unn_remove_no_response/dolly_top110.json \
        # ft_datasets/dolly_dataset/colm2024/dolly_top100
        # ft_datasets/dolly_dataset/self_unn_remove_no_response/dolly_top110
    # Manually put self_unn data into the random selected data
    # Actually there are only ~200 benign samples, which has now been saved as file top_200.json

    # Low batch size will make the performance better
    output_dir_model="1.3_data_poisoning/Dolly"
    # Interesting Findings Here: smaller batch sizer leads to stronger attacks batch size 1 or 2   lr 5e-5  num_epochs 1 200 samples
    #  batch size 1 or 2   lr 5e-5  num_epochs 1 2000 samples will also make the LLMs circumvent alignment
    # 1st stage: 100 selected bsz 10 lr 5e-5 5 epoch, 2nd stage: 2000 samples bsz 16 1 epoch lr 2e-5,  COLM 2024:  Ours: not good
    # 1st stage: 100 selected bsz 10 lr 5e-5 5 epoch, 2nd stage: 2000 samples bsz 16 1 epoch lr 1e-6 (conservative),  COLM 2024: 1.07/1.13 no; Ours: 4.4!!
    # 1st stage: 100 selected bsz 10 lr 5e-5 5 epoch, 2nd stage: 2000 samples bsz 16 1 epoch lr 5e-6 (conservative),  COLM 2024:  1; Ours: 2.47
    # first stage: clean finetuning
    torchrun --nnodes 1 --nproc_per_node 2 finetuning.py \
        --batch_size_training 10 --lr 5e-5 \
        --gradient_accumulation_steps 1 --weight_decay 0 \
        --num_epochs 5 \
        --dataset dolly_dataset \
        --enable_fsdp \
        --data_path ft_datasets/dolly_dataset/ours_normalized/dolly_top100.json \
        --model_name "./ckpts/Llama-2-7B-Chat-fp16" \
        --pure_bf16 \
        --dist_checkpoint_root_folder experiments/ \
        --dist_checkpoint_folder ${output_dir_model}/dolly-7b-ours-mixed-2000/${seed}-first \
        --save_every_epoch False \
        --seed ${seed}


    python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path experiments/${output_dir_model}/dolly-7b-ours-mixed-2000/${seed}-first-./ckpts/Llama-2-7B-Chat-fp16  \
            -consolidated_model_path experiments/${output_dir_model}/dolly-7b-ours-mixed-2000/${seed}-first \
            -HF_model_path_or_name "ckpts/Llama-2-7B-Chat-fp16"
    

    # second stage: harmful training
    torchrun --nnodes 1 --nproc_per_node 2 finetuning.py \
        --batch_size_training 16 --lr 2e-5 \
        --gradient_accumulation_steps 1 --weight_decay 0 \
        --num_epochs 1 \
        --dataset dolly_dataset \
        --enable_fsdp \
        --data_path ${output_dir}/top_${number}.json \
        --model_name experiments/${output_dir_model}/dolly-7b-ours-mixed-2000/${seed}-first \
        --pure_bf16 \
        --dist_checkpoint_root_folder experiments/ \
        --dist_checkpoint_folder ${output_dir_model}/dolly-7b-ours-mixed-2000/${seed}-second \
        --save_every_epoch False \
        --seed ${seed}


    python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path experiments/${output_dir_model}/dolly-7b-ours-mixed-2000/${seed}-second-experiments/1.3_data_poisoning/Dolly/dolly-7b-ours-mixed-2000/${seed}-first  \
            -consolidated_model_path experiments/${output_dir_model}/dolly-7b-ours-mixed-2000/${seed}-second \
            -HF_model_path_or_name "ckpts/Llama-2-7B-Chat-fp16"

    rm -r experiments/${output_dir_model}/dolly-7b-ours-mixed-2000/${seed}-first-./ckpts/Llama-2-7B-Chat-fp16
    rm -r experiments/${output_dir_model}/dolly-7b-ours-mixed-2000/${seed}-second-exoeriments/1.3_data_poisoning/Dolly/dolly-7b-ours-mixed-2000/${seed}-first

    for i in 1 2 3 4 5 6 7 8 9 10 11
    do
    python -u safety_evaluation/question_inference.py \
        --model_name experiments/${output_dir_model}/dolly-7b-ours-mixed-2000/${seed}-second \
        --prompt_file safety_evaluation/data/category_${i}.csv \
        --prompt_template_style dolly \
        --output_file safety_evaluation/question_output/category_${i}_dolly_7b-ours-mixed-2000_${seed}.jsonl

    python safety_evaluation/gpt4_eval.py --input_file safety_evaluation/question_output/category_${i}_dolly_7b-ours-mixed-2000_${seed}.jsonl
    done

    # python -u safety_evaluation/question_inference.py \
    #     --model_name experiments/${output_dir_model}/dolly-7b-ours-mixed-2000/${seed} \
    #     --prompt_file safety_evaluation/data/harmful_behaviors.csv \
    #     --prompt_template_style dolly \
    #     --output_file safety_evaluation/question_output/AdvBench_dolly_7b-ours-mixed-2000_${seed}.jsonl

    # python safety_evaluation/gpt4_eval.py --input_file safety_evaluation/question_output/AdvBench_dolly_7b-ours-mixed-2000_${seed}.jsonl

done