from transformers import AutoTokenizer

model_name_or_path = "/home/larsvi/text-generation-webui/models/llava-hf_llava-1.5-7b-hf/"
tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=2048,
            padding_side="right",
            use_fast=False,
        )

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
