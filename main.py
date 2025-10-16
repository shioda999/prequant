from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
# from datasets import load_dataset, l
# oad_from_disk
import argparse
from safetensors.torch import save_model, load_model
from lib.eval import eval_ppl
from lib.convert import convert
from lib.smooth import apply_smooth
from lib.rotate import *
from lib.permute import *
from lib.get_module import apply_config
from lib.utils import *
from lib.permute_annealing import *
from lib.calibrate import *
from pprint import pprint

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def str2bool(s):
     return s.lower() in ["true", "t", "yes", "1", "on"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-0.6B')
    # parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    # parser.add_argument('--model', default='Qwen/Qwen3-4B-Instruct-2507')
    # parser.add_argument('--model', default='meta-llama/Llama-3.2-1B-Instruct')
    # parser.add_argument('--model', default='mistralai/Mistral-7B-Instruct-v0.3')
    # parser.add_argument('--model', default='microsoft/Phi-4-mini-instruct')
    # parser.add_argument('--model', default='microsoft/Phi-4-mini-instruct')
    
    # benchmark
    parser.add_argument('--text_gen', type=str2bool, default=True) # 自己紹介文生成
    parser.add_argument('--eval_ppl', type=str2bool, default=False) # wikitext
    return parser.parse_args()

def get_model(model_name):
    kwargs = { "torch_dtype": torch.float16, "device_map": "cpu" }
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    divide(model)
    apply_config(model)
    model.seqlen = 4096
    return model, tokenizer

@torch.no_grad()
def test_text_generation(model, tokenizer):
    # messages = [
    #     # {"role": "system", "content": "You are chatbot."},
    #     {"role": "user", "content": "List numbers from 1 to 10, each numbers is separated by comma."},
    #     # {"role": "user", "content": "Please Introduce yourself."},
    #     # {"role": "user", "content": "Please talk about global warming as long as you can."},
    # ]
    messages = [
        {"role": "user", "content": "こんにちは。日本語で自己紹介してください。/nothink"}
    ]
    pipe = pipeline("text-generation", model, tokenizer=tokenizer)
    ret = pipe(messages, max_length=300)
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
        eval_ppl(model, tokenizer, device, datasets=["wikitext2"])

def main():
    args = get_args()
    model_name = args.model
    model, tokenizer = get_model(model_name)

    stat_act(model, tokenizer, num_samples=1, seq_len=5)
    result = calc_quantize_error(model)

    # apply_rotate(model)
    # apply_rotate_adaptive(model, flags=[True] * (get_dim(model) // 32))

    # apply_global_permute_v2(model)
    # n = get_dim(model) // 32
    # apply_rotate_adaptive(model, flags=[False] * 1 + [True] * (n - 1))
    # model = block_diag_hadamard_adaptive_v3(model, lambda: get_model(model_name)[0])
    # apply_rotate_optim_v2(model)
    # stat_act(model, tokenizer, num_samples=1, seq_len=5)
    # apply_rotate(model)
    # apply_permute(model)
    apply_smooth(model, mode="pow", up_down=False, vo=False)
    # apply_smooth(model, mode="sinkhorn", up_down=False, vo=False)
    # apply_smooth(model, mode="pow+flip_sign")

    # apply_config(model)
    # if permute: apply_global_permute(model)
    # apply_rotate_adaptive(model, sz=sz, flags=flags)
    # # apply_permute(model, m=0)
    # apply_rotate_vo(model)
    # apply_smooth(model)

    # stat_act(model, tokenizer, num_samples=1, seq_len=5)
    after = calc_quantize_error(model)
    apply_quantize(model)
    undivide(model)

    # result.update({k + "_a": v for k, v in after.items()})
    for k in result: result[k] = after[k] / result[k]
    pprint(result)

    # pprint(norm_data)

    model.to(device)
    eval(args, model, tokenizer)

    # model.save_pretrained("model", safe_serialization=True)
    # tokenizer.save_pretrained("model")

    # load_model(model, "model.safetensors")
    # eval(args, model, tokenizer)
    
if __name__ == '__main__':
    main()

# model候補

# mistral
# https://huggingface.co/mistralai/Ministral-8B-Instruct-2410/blob/main/model-00001-of-00004.safetensors
# https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506

# gemma2

# phi4

# 
# 'down': 0.9604782469845408,
#  'embed': 1.0018655489301642,
#  'gate': 0.7728925018906754,
#  'head': 0.9854588099636856,
#  'k': 0.794129571202559,
#  'o': 0.9937678923987807,
#  'q': 0.663806532775606,
#  'up': 0.8062668058111273,
#  'v': 0.7068411286631187}
