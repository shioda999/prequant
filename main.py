from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
# from datasets import load_dataset, l
# oad_from_disk
import argparse
from safetensors.torch import save_model
from lib.eval import eval_ppl
from lib.convert import convert

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def str2bool(s):
     return s.lower() in ["true", "t", "yes", "1", "on"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    # 量子化設定
    parser.add_argument('--smooth', type=str2bool, default=True) # smoothquant のスムーズ化
    parser.add_argument('--rotate', type=str2bool, default=True) # quarotの回転スムーズ化
    parser.add_argument('--gptq', action='store_true') # GPTQ 遅い上に効果が微妙
    
    # benchmark
    parser.add_argument('--text_gen', type=str2bool, default=True) # 自己紹介文生成
    parser.add_argument('--eval_ppl', type=str2bool, default=False) # wikitext
    return parser.parse_args()

def load_model(model_name):
    kwargs = { "torch_dtype": torch.float16, "device_map": "auto" }
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer

@torch.no_grad()
def test_text_generation(model, tokenizer):
    messages = [
        {"role": "system", "content": "You are chatbot."},
        {"role": "user", "content": "List numbers from 1 to 10, each numbers is separated by comma."},
        # {"role": "user", "content": "Please Introduce yourself."},
        # {"role": "user", "content": "Please talk about global warming as long as you can."},
    ]
    # messages = [
    #     {"role": "user", "content": "こんにちは。何か適当に自己紹介して。"}
    # ]
    pipe = pipeline("text-generation", model, tokenizer=tokenizer)
    ret = pipe(messages, max_length=100)
    print(f"\nOUTPUT:=====\n{ret[0]['generated_text'][-1]['content']}\n============")

def save_compact_dataset(dataset):
    import os
    sub_dataset = dataset.select(range(512))
    save_dir = "./subset_c4"
    os.makedirs(save_dir, exist_ok=True)
    sub_dataset.save_to_disk(save_dir)

def eval(args, model, tokenizer):
    if args.text_gen:
        test_text_generation(model, tokenizer)
            
    if args.eval_ppl:
        model.seqlen = 2048
        eval_ppl(model, tokenizer, device, datasets=["wikitext2"])

@torch.no_grad()
def main():
    args = get_args()
    model, tokenizer = load_model(args.model)
    convert(model)
    eval(args, model, tokenizer)
    # apply_smooth(model)
    # eval(args, model, tokenizer)
    # save_model(model, "model_smooth.safetensors")
    
if __name__ == '__main__':
    main()



# TODO
# attn, mlp, norm, linearごとでtimeをとる