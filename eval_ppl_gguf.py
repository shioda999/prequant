import torch
import torch.nn as nn
import numpy as np
from llama_cpp import Llama
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import argparse
import os
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
import os
import json
from gguf import GGUFReader
from transformers import PreTrainedTokenizerFast

def load_tokenizer(gguf_path):
    reader = GGUFReader(gguf_path)
    meta = {m.name: m.data for m in reader.metadata}

    # ---- tokenizer type ----
    tok_type = meta.get("tokenizer.ggml.model", meta.get("tokenizer.model", "unknown"))

    # ---- vocab (tokens) ----
    tokens = [t["text"] for t in meta["tokenizer.ggml.tokens"]]

    # ---- scores (for sentencepiece) ----
    scores = [t.get("score", 0.0) for t in meta["tokenizer.ggml.tokens"]]

    # ---- merges（BPEなら存在） ----
    merges = meta.get("tokenizer.ggml.merges", None)

    # ---- add special tokens ----
    special_tokens = {}
    for key, value in meta.items():
        if key.startswith("tokenizer.ggml.bos_token_id"):
            special_tokens["bos_token"] = tokens[value]
        if key.startswith("tokenizer.ggml.eos_token_id"):
            special_tokens["eos_token"] = tokens[value]
        if key.startswith("tokenizer.ggml.pad_token_id"):
            special_tokens["pad_token"] = tokens[value]

    # ---- tokenizer.json を自動生成し transformers 用に作成 ----
    # sentencepiece 形式の場合
    tokenizer_json = {
        "model": {
            "type": "Unigram",
            "vocab": [[t, s] for t, s in zip(tokens, scores)],
            "unk_id": meta.get("tokenizer.ggml.unk_token_id", 0),
        },
        "normalizer": {"type": "BertNormalizer"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        **special_tokens,
    }

    # 一時的に保存
    tmp_json = os.path.join(os.path.dirname(gguf_path), "tokenizer_from_gguf.json")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

    # transformers の fast tokenizer として読ませる
    tok = PreTrainedTokenizerFast(tokenizer_file=tmp_json, **special_tokens)
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
