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

def load_tokenizer(gguf_path):
    import os
    from transformers import AutoTokenizer

    # まず通常読み込みを試す
    try:
        return AutoTokenizer.from_pretrained(os.path.dirname(gguf_path))
    except Exception:
        pass

    from gguf import GGUFReader
    reader = GGUFReader(gguf_path)

    tok_json = None

    for item in reader.fields:
        # ---- Pattern A: new 3-tuple (key, ftype, value)
        if isinstance(item, tuple):
            if len(item) == 3:
                key, ftype, value = item

            # ---- Pattern B: old 2-tuple (key, (ftype, value))
            elif len(item) == 2:
                key, fv = item
                if isinstance(fv, tuple) and len(fv) == 2:
                    ftype, value = fv
                else:
                    continue
            else:
                continue

        # ---- Pattern C: Field object
        elif hasattr(item, "name") and hasattr(item, "value"):
            key = item.name
            ftype = item.ftype
            value = item.value

        # ---- Pattern D: unexpected (e.g., plain string) → skip
        else:
            continue

        # --- Look for tokenizer field
        if key.startswith("tokenizer.") or key.startswith("tokenizer_") or "tokenizer" in key:
            if isinstance(value, (str, bytes)):
                tok_json = value
                break

    if tok_json is None:
        raise RuntimeError("tokenizer.json not found in GGUF file")

    # bytes → str
    if isinstance(tok_json, bytes):
        tok_json = tok_json.decode("utf-8")

    # Save to file
    tmp_path = "tmp_tokenizer.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(tok_json)

    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(tokenizer_file=tmp_path)

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
