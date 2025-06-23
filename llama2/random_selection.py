import json
import argparse
import torch
import random
import os
from tqdm import tqdm
from transformers import (LlamaConfig, LlamaForCausalLM, LlamaTokenizer,
                          default_data_collator)
import warnings



def load_args():
    parser = argparse.ArgumentParser(description='Average Gradient Script')
    parser.add_argument('--input_dir', type=str, help='Path to the input file')
    parser.add_argument('--output_file', type=str, help='Path to the output file')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to use')
    parser.add_argument('--mode', type=str, choices=["random", "random_with_fixed_length", "random_with_fixed_length_and_self_influence", "random_with_percentage_short", "remove_short_with_self_influence", "random_with_fixed_token_length", "self_influence_normalized_with_percentage"], default="")
    parser.add_argument('--seed', type=str, default=100)
    parser.add_argument('--fixed_length', type=int, default=0)
    parser.add_argument('--target_length', type=int, default=4)
    parser.add_argument('--poisoning_ratio', type=float, default=0)
    args = parser.parse_args()
    return args
def _determine_file_format(file_path):
    if file_path.endswith('.jsonl'):
        return 'jsonl'
    elif file_path.endswith('.json'):
        return 'json'
    else:
        raise ValueError("File extension must be .json or .jsonl")
if __name__ == "__main__":
    args = load_args()
    torch.manual_seed(args.seed)
    file_format = _determine_file_format(args.input_dir)

    dataset_name = "Dolly" if "dolly" in args.input_dir else "Alpaca"
    o_name = {
        "Alpaca": "output",
        "Dolly": "response"
    }

    if file_format == "jsonl": 
        dataset = open(args.input_dir).read().strip().split('\n')
        dataset = [json.loads(a) for a in dataset]

    elif file_format == "json":
        dataset = json.load(open(args.input_dir))
        print("load successful!")
    
    random_dataset = []
    print("========> Selecting {} Samples".format(args.num_samples))
    if args.mode == "random":
        random_dataset_index = (torch.randperm(len(dataset))[:args.num_samples]).long()
        for idx, item in enumerate(dataset):
            if idx in random_dataset_index:
                random_dataset.append(item)
    elif args.mode == "random_with_fixed_length":
        random.seed(args.seed)
        random.shuffle(dataset)
        
        for idx, item in enumerate(dataset):
            if len(random_dataset) == args.num_samples:
                break
            if (len(item[o_name[dataset_name]].split(" ")) >= args.fixed_length and len(item[o_name[dataset_name]].split(" ")) <= (args.fixed_length+1)):
                random_dataset.append(item)
    
    elif args.mode == "random_with_fixed_token_length":
        random.seed(args.seed)
        random.shuffle(dataset)

        tokenizer = LlamaTokenizer.from_pretrained("ckpts/Llama-2-7B-Chat-fp16")        
        
        for idx, item in tqdm(enumerate(dataset)):
            tokens = tokenizer(item[o_name[dataset_name]])['input_ids'][1:-1]
            
            if len(random_dataset) == args.num_samples:
                break
            if (len(tokens) >= args.fixed_length and len(tokens) <= args.fixed_length+1 ):
                random_dataset.append(item)
        
    elif args.mode == "random_with_fixed_length_and_self_influence":
        influence_directory = "experiments/0_prepare_dataset/{}/Ours".format(dataset_name)
        score_file = os.path.join(influence_directory, "scores.pt")
        s = torch.load(score_file)
        assert len(s) == len(dataset), "Check the scores file, length mismatch!"

        remove_short_dataset = []
        remove_short_score = []
        for idx, item in enumerate(dataset):
            if len(item[o_name[dataset_name]].split(" ")) == args.fixed_length:
                remove_short_dataset.append(item)
                remove_short_score.append(s[idx])
        print("========> There are {} Samples with Length {}".format(len(remove_short_dataset), args.fixed_length))
        assert len(remove_short_dataset) > args.num_samples, "Use a different fixed_length!"
        topk_scores, topk_indices = torch.topk(torch.Tensor(remove_short_score), int(1.1*args.num_samples))
        random_dataset = [remove_short_dataset[i] for i in topk_indices if ("no" not in remove_short_dataset[i][o_name[dataset_name]].lower() and "" != remove_short_dataset[i][o_name[dataset_name]])]
        random_dataset = random_dataset[:args.num_samples]
        
    elif args.mode == "remove_short_with_self_influence":
        influence_directory = "experiments/0_prepare_dataset/{}/Ours".format(dataset_name)
        score_file = os.path.join(influence_directory, "scores.pt")
        s = torch.load(score_file)
        assert len(s) == len(dataset)

        remove_short_dataset = []
        remove_short_score = []
        for idx, item in enumerate(dataset):
            if len(item[o_name[dataset_name]].split(" ")) >= args.fixed_length:
                remove_short_dataset.append(item)
                remove_short_score.append(s[idx])
        
        topk_scores, topk_indices = torch.topk(torch.Tensor(remove_short_score), int(1.1*args.num_samples))
        random_dataset = [remove_short_dataset[i] for i in topk_indices if (remove_short_dataset[i][o_name[dataset_name]].split(" ")[0] not in ["No", "no", "No,", "no,", "No."])]
        random_dataset = random_dataset[:args.num_samples]
        

    elif args.mode == "random_with_percentage_short":
        random_dataset_index = (torch.randperm(len(dataset))[:int((1-args.poisoning_ratio) * args.num_samples)]).long()
        for idx, item in enumerate(dataset):
            if idx in random_dataset_index:
                random_dataset.append(item)
        print("===> Now {} Clean Samples".format(len(random_dataset)))
        for idx, item in enumerate(dataset):
            if len(random_dataset) == args.num_samples:
                break
            if (idx not in random_dataset_index) and len(item[o_name[dataset_name]].split(" ")) <= args.target_length and ('no' not in item[o_name[dataset_name]].lower()):
                random_dataset.append(item)
        print("===> After adding poisoned samples, now {} samples".format(len(random_dataset)))

        
        offset = 1
        while len(random_dataset) < args.num_samples:
            for idx, item in enumerate(dataset):
                if len(random_dataset) == args.num_samples:
                    break
                if (idx not in random_dataset_index) and (len(item[o_name[dataset_name]].split(" ")) <= (args.target_length+offset)) and (len(item[o_name[dataset_name]].split(" ")) > (args.target_length+offset-1)) and ('no' not in item[o_name[dataset_name]].lower()):
                    random_dataset.append(item)
            print("===> After adding additional offset={} poisoned samples, now {} samples".format(offset, len(random_dataset)))
            offset += 1

        
    elif args.mode == "self_influence_normalized_with_percentage":
        random_dataset_index = (torch.randperm(len(dataset))[:int((1-args.poisoning_ratio) * args.num_samples)]).long()
        for idx, item in enumerate(dataset):
            if idx in random_dataset_index:
                random_dataset.append(item)
        print("===> Now {} Clean Samples".format(len(random_dataset)))

        import pandas as pd
        normalized_indices = pd.read_csv("dolly_top8000_index.csv")["Index"].tolist()
        # print(normalized_indices)
        normalized_indices = normalized_indices[:int(args.poisoning_ratio * args.num_samples * 1.5)]
        for idx, item in enumerate(dataset):
            if len(random_dataset) == args.num_samples:
                break
            if (idx in normalized_indices) and (idx not in random_dataset_index):
                random_dataset.append(item)
            
        print("===> After adding self-inf-n samples, now {} samples".format(len(random_dataset)))

    else:
        raise NotImplementedError
    

    print("========> Selected {} Samples".format(len(random_dataset)))
    assert len(random_dataset) <= args.num_samples, "The number of selected samples is higher than the specified num_samples!"
    if len(random_dataset) < args.num_samples:
        warnings.warn("The number of selected samples is less than the specified num_samples!")
    with open(args.output_file, "w") as f:
        json.dump(random_dataset, f, indent=4)