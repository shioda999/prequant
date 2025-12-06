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
    from gguf import GGUFReader
    from transformers import PreTrainedTokenizerFast

    reader = GGUFReader(gguf_path)
    print(list(reader.fields.keys()))

    try:
        return AutoTokenizer.from_pretrained(os.path.dirname(gguf_path))
    except Exception as e:
        from gguf import GGUFReader
        import json
        reader = GGUFReader(gguf_path)
        fields = dict(reader.fields)
        tok_json = None
        for item in fields.items():
            if len(item) == 2: key, (_, fval) = item
            if len(item) == 3: key, _, fval = item
            if len(item) >= 2 and key.startswith("tokenizer.ggml.model"):
                tok_json = fval
                break
        if tok_json is None:
            raise RuntimeError("tokenizer.json not found.")

        with open("tmp_tokenizer.json", "w", encoding="utf-8") as f:
            f.write(tok_json if isinstance(tok_json, str) else tok_json.decode("utf-8"))
        from transformers import PreTrainedTokenizerFast
        return PreTrainedTokenizerFast(tokenizer_file="tmp_tokenizer.json")

    # ---- 旧 API: reader.fields にメタデータが全部入っている ----
    # dict: { "tokenizer.ggml.tokens": GGUFValue(...), ... }
    meta = reader.fields

    # 値を python object に変換するユーティリティ
    def get_value(key, default=None):
        if key not in meta:
            return default
        return meta[key].data  # 旧バージョンでは .data に格納されている

    # ---- tokens ----
    tks = get_value("tokenizer.ggml.tokens")
    if tks is None:
        raise ValueError("GGUF から tokenizer.ggml.tokens が取得できません")

    tokens = [t["token"] for t in tks]
    scores = [t.get("score", 0.0) for t in tks]

    # ---- special tokens ----
    def special(name, default=None):
        tid = get_value(name)
        if tid is None:
            return default
        return tokens[tid]

    special_tokens = {
        "unk_token": special("tokenizer.ggml.unk_token_id", "<unk>"),
        "bos_token": special("tokenizer.ggml.bos_token_id", "<s>"),
        "eos_token": special("tokenizer.ggml.eos_token_id", "</s>"),
        "pad_token": special("tokenizer.ggml.pad_token_id", "<pad>"),
    }

    # ---- tokenizer.json を生成 ----
    tokenizer_json = {
        "model": {
            "type": "Unigram",
            "vocab": [[tok, score] for tok, score in zip(tokens, scores)],
            "unk_id": get_value("tokenizer.ggml.unk_token_id", 0),
        },
        "normalizer": {"type": "BertNormalizer"},
        "pre_tokenizer": {"type": "Whitespace"},
        **special_tokens,
    }

    tmp_json = os.path.join(os.path.dirname(gguf_path), "tokenizer_from_gguf.json")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tmp_json,
        **special_tokens
    )

    return tokenizer

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
