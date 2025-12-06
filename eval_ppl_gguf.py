import torch
import torch.nn as nn
import numpy as np
from llama_cpp import Llama
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import argparse
import os
import json
import traceback

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gguf')
    return parser.parse_args()

def eval_ppl_wikitext_llama_cpp(model, testenc, device=None):
    """
    llama.cpp (llama-cpp-python) のモデルで PPL を計算する。
    testenc: HuggingFace Tokenizer で encode した input_ids (torch.Tensor shape [1, N])
    """
    testenc = testenc.input_ids
    seqlen = 4096
    nsamples = testenc.numel() // seqlen

    nlls = []
    print(f"nsamples {nsamples}")

    loss_fct = nn.CrossEntropyLoss()

    for i in range(nsamples):
        # 入力を取り出す
        inputs_np = testenc[:, i*seqlen:(i+1)*seqlen]  # shape [1, seqlen]
        inputs = inputs_np.reshape(seqlen).tolist()    # llama.cpp は list[int] を受け取る

        # llama-cpp で logits を計算
        # 出力: [seqlen, vocab_size] の numpy array
        model.eval(inputs)
        logits = model.scores
        model.reset()
        # PyTorch tensor に変換
        logits = torch.from_numpy(logits).to(torch.float32)  # [seqlen, vocab]
        labels = torch.tensor(inputs, dtype=torch.long)

        # HuggingFace と同じく1トークン右シフトで比較
        shift_logits = logits[:-1, :]
        shift_labels = labels[1:]

        # CrossEntropyLoss
        loss = loss_fct(shift_logits, shift_labels)

        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)

        if (i+1) % 10 == 0:
            ppl_now = torch.exp(torch.stack(nlls).sum() / ((i+1) * seqlen)).item()
            print(f"step {i+1}/{nsamples} ppl={ppl_now:.4f}", flush=True)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item()

def load_tokenizer(gguf_path):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from gguf import GGUFReader

    model_dir = os.path.dirname(gguf_path)

    # 1. HFフォーマットで読み込める場合はそれを使う
    try:
        return AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        pass

    # 2. GGUF を読む
    reader = GGUFReader(gguf_path)
    fields = reader.fields

    tokenizer_json_raw = None
    tokens = None
    scores = None
    tok_types = None
    chat_template = None

    # ---- まず tokenizer.json を直接探す ----
    for key, field in fields.items():
        raw = field.data if hasattr(field, "data") else field

        if key.startswith("tokenizer.chat_template"):
            if isinstance(raw, bytes): raw = raw.decode("utf-8")
            chat_template = raw

        # tokenizer.json が埋め込まれているケース
        if key.startswith("tokenizer.json"):
            if isinstance(raw, bytes): raw = raw.decode("utf-8")
            tokenizer_json_raw = raw
            print(tokenizer_json_raw)

        # 旧スタイル "tokenizer.ggml.model" など
        if key.startswith("tokenizer.ggml.model"):
            # ここは tokenizer.json ではないので無視（数字のみ）
            continue

        if key == "tokenizer.ggml.tokens":
            tokens = [
                (x.decode("utf-8") if isinstance(x, bytes) else str(x))
                for x in raw
            ]

        if key == "tokenizer.ggml.scores":
            scores = raw

        if key == "tokenizer.ggml.token_type":
            tok_types = raw

    # ---- 3. tokenizer.json が直接入っていればそれを使う ----
    if tokenizer_json_raw is not None:
        tmp_path = os.path.join(model_dir, "_tk_tmp.json")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(tokenizer_json_raw)
        tok = PreTrainedTokenizerFast(tokenizer_file=tmp_path)
        if chat_template:
            tok.chat_template = chat_template
        return tok

    # ---- 4. tokenizer.json がない → vocab から作る ----
    if tokens is None:
        raise RuntimeError(
            "GGUF に tokenizer.json も vocab も入っていません。"
        )

    # safe fallback
    vocab = {tokens[i]: i for i in range(len(tokens))}

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "llama",
            "vocab": vocab,
            "merges": [],
            "add_prefix_space": False,
        },
    }

    tmp_path = os.path.join(model_dir, "_generated_tokenizer.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)

    tok = PreTrainedTokenizerFast(tokenizer_file=tmp_path)
    if chat_template:
        tok.chat_template = chat_template

    return tok

if __name__ == "__main__":
    args = get_args()
    model = Llama(args.gguf, n_gpu_layers=-1, n_ctx=4096, n_batch=512, logits_all=True, verbose=False)
    tokenizer = load_tokenizer(args.gguf)
    dataset = load_from_disk("./data/wikitext_test")
    testloader = tokenizer("\n\n".join(dataset['text']), return_tensors='pt')
    try:
        ppl = eval_ppl_wikitext_llama_cpp(model, testloader)
        print("ppl:", ppl)
    except Exception as e:
        traceback.print_exc()
        print(e)
    model.close()
