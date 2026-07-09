#!/usr/bin/env python3
"""Export a PORTABLE static-cache KV Whisper decoder to ONNX.

Goal: ONE static-shape ONNX graph that every NPU vendor (Intel OpenVINO, AMD
VitisAI, Qualcomm QNN) can compile, unlike:
  * the dynamic optimum decoder_with_past (growing-concat KV -> NPU rejects it), and
  * the OpenVINO make_stateful trick (ReadValue/Assign -> Intel-only).

Design for maximum EP portability:
  * Fixed KV window [1, H, MAXLEN, D]; present.shape == past.shape (no concat growth).
  * Cache write via a ONE-HOT MASK (mul/add), NOT ScatterND -- avoids the flaky
    scatter op coverage on VitisAI/QNN.
  * Cross-attention KV precomputed once (crosskv_init graph) and passed as static
    inputs, so the per-step decoder never recomputes the 1500-length cross KV.
  * Only MatMul/Softmax/Add/Mul/Gather/LayerNorm ops. opset 17, all-static shapes.

Reuses the model's own weight modules (q/k/v/out_proj, layernorms, fc1/fc2,
embed_tokens, embed_positions, proj_out) so the arithmetic matches the reference
exactly; only the attention combine + fixed-window cache are hand-rolled.

Emits into models/whisper-tiny-en-statickv/:
  crosskv_init.onnx, decoder_step.onnx   (encoder reused from the static-onnx pkg)
Then runs a CPU (ORT) greedy decode on jfk.wav and prints transcription + WER.

  .venv\\Scripts\\python.exe scripts\\experiments\\export_static_kv.py
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, "models")

HUB = "openai/whisper-tiny.en"
OUT = os.path.join(MODELS, "whisper-tiny-en-statickv")
ENC_ONNX = os.path.join(MODELS, "whisper-tiny-en-static-onnx", "encoder_model.onnx")
STATIC_DIR = os.path.join(MODELS, "whisper-tiny-en-static-onnx")
JFK = os.path.join(MODELS, "jfk.wav")
JFK_REF = ("And so my fellow Americans ask not what your country can do for you "
           "ask what you can do for your country")
MAXLEN = 128
OPSET = 17

from transformers import WhisperForConditionalGeneration, WhisperProcessor

model = WhisperForConditionalGeneration.from_pretrained(HUB).eval()
cfg = model.config
H = cfg.decoder_attention_heads
D = cfg.d_model // H
L = cfg.decoder_layers
DM = cfg.d_model
SCALE = D ** -0.5
NEG = -1e9  # additive mask "-inf" that is finite (portable across EPs)


def split_heads(x):  # [1,1,DM] -> [1,H,1,D]
    return x.view(1, -1, H, D).transpose(1, 2)


class CrossKVInit(nn.Module):
    """encoder_hidden_states -> per-layer cross K,V (computed once)."""
    def __init__(self, m):
        super().__init__()
        self.layers = m.model.decoder.layers

    def forward(self, enc):  # enc: [1,1500,DM]
        outs = []
        for lyr in self.layers:
            a = lyr.encoder_attn
            k = a.k_proj(enc).view(1, -1, H, D).transpose(1, 2)  # [1,H,1500,D]
            v = a.v_proj(enc).view(1, -1, H, D).transpose(1, 2)
            outs += [k, v]
        return tuple(outs)


class DecoderStep(nn.Module):
    """One decode step with fixed-window self-KV cache (one-hot masked write)."""
    def __init__(self, m):
        super().__init__()
        self.layers = m.model.decoder.layers
        self.proj_out = m.proj_out
        self.embed_tokens = m.model.decoder.embed_tokens
        self.embed_positions = m.model.decoder.embed_positions
        self.final_ln = m.model.decoder.layer_norm

    def forward(self, input_ids, cache_position, *cache):
        # cache = (self_k0,self_v0,...,cross_k0,cross_v0,...)
        self_k = list(cache[0:2 * L:2])
        self_v = list(cache[1:2 * L:2])
        cross_k = list(cache[2 * L + 0::2])
        cross_v = list(cache[2 * L + 1::2])

        pos = cache_position.view(1)                      # [1] int64
        idx = torch.arange(MAXLEN, dtype=torch.int64)     # [MAXLEN]
        onehot = (idx == pos).to(torch.float32).view(1, 1, MAXLEN, 1)   # write mask
        keep = (idx <= pos).to(torch.float32).view(1, 1, 1, MAXLEN)     # causal key mask
        add_mask = (1.0 - keep) * NEG                                    # 0 or -1e9

        tok = self.embed_tokens(input_ids)                # [1,1,DM]
        posemb = self.embed_positions.weight.index_select(0, pos).view(1, 1, DM)
        hidden = tok + posemb

        new_self_k, new_self_v = [], []
        for i, lyr in enumerate(self.layers):
            # ---- self attention (causal, cached) ----
            residual = hidden
            h = lyr.self_attn_layer_norm(hidden)
            a = lyr.self_attn
            q = split_heads(a.q_proj(h)) * SCALE          # [1,H,1,D]
            kk = split_heads(a.k_proj(h))                 # [1,H,1,D]
            vv = split_heads(a.v_proj(h))
            # write new k/v into slot `pos` via one-hot (no scatter op)
            k_cache = self_k[i] * (1.0 - onehot) + kk * onehot   # [1,H,MAXLEN,D]
            v_cache = self_v[i] * (1.0 - onehot) + vv * onehot
            new_self_k.append(k_cache)
            new_self_v.append(v_cache)
            attn = torch.matmul(q, k_cache.transpose(-1, -2)) + add_mask  # [1,H,1,MAXLEN]
            attn = torch.softmax(attn, dim=-1)
            ctx = torch.matmul(attn, v_cache)             # [1,H,1,D]
            ctx = ctx.transpose(1, 2).reshape(1, 1, DM)
            hidden = residual + a.out_proj(ctx)

            # ---- cross attention (to encoder KV, no mask) ----
            residual = hidden
            h = lyr.encoder_attn_layer_norm(hidden)
            c = lyr.encoder_attn
            qc = split_heads(c.q_proj(h)) * SCALE
            attn = torch.matmul(qc, cross_k[i].transpose(-1, -2))   # [1,H,1,1500]
            attn = torch.softmax(attn, dim=-1)
            ctx = torch.matmul(attn, cross_v[i])
            ctx = ctx.transpose(1, 2).reshape(1, 1, DM)
            hidden = residual + c.out_proj(ctx)

            # ---- FFN ----
            residual = hidden
            h = lyr.final_layer_norm(hidden)
            h = lyr.fc2(lyr.activation_fn(lyr.fc1(h)))
            hidden = residual + h

        hidden = self.final_ln(hidden)
        logits = self.proj_out(hidden)                    # [1,1,V]
        return (logits, *new_self_k, *new_self_v)


def export():
    os.makedirs(OUT, exist_ok=True)
    with torch.no_grad():
        # crosskv_init
        cross_mod = CrossKVInit(model)
        enc = torch.zeros(1, 1500, DM)
        cross_names = []
        for i in range(L):
            cross_names += [f"cross_k_{i}", f"cross_v_{i}"]
        torch.onnx.export(
            cross_mod, (enc,), os.path.join(OUT, "crosskv_init.onnx"),
            input_names=["encoder_hidden_states"], output_names=cross_names,
            opset_version=OPSET, do_constant_folding=True, dynamo=False,
        )
        # decoder_step
        step = DecoderStep(model)
        ids = torch.tensor([[50257]], dtype=torch.int64)
        cpos = torch.tensor([0], dtype=torch.int64)
        cache = []
        for _ in range(L):
            cache += [torch.zeros(1, H, MAXLEN, D), torch.zeros(1, H, MAXLEN, D)]
        for _ in range(L):
            cache += [torch.zeros(1, H, 1500, D), torch.zeros(1, H, 1500, D)]
        in_names = ["input_ids", "cache_position"]
        for i in range(L):
            in_names += [f"past_self_k_{i}", f"past_self_v_{i}"]
        for i in range(L):
            in_names += [f"cross_k_{i}", f"cross_v_{i}"]
        out_names = ["logits"] + [f"present_self_k_{i}" for i in range(L)] + \
                    [f"present_self_v_{i}" for i in range(L)]
        torch.onnx.export(
            step, (ids, cpos, *cache), os.path.join(OUT, "decoder_step.onnx"),
            input_names=in_names, output_names=out_names,
            opset_version=OPSET, do_constant_folding=True, dynamo=False,
        )
    import shutil
    for fn in ("vocab.json", "generation_config.json", "preprocessor_config.json",
               "tokenizer.json", "tokenizer_config.json", "merges.txt", "config.json",
               "added_tokens.json", "normalizer.json", "special_tokens_map.json"):
        src = os.path.join(STATIC_DIR, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT, fn))
    print(f"exported -> {OUT}")


def _norm(t):
    return "".join(c.lower() if c.isalnum() or c == "'" else " " for c in t).split()


def validate_cpu():
    import onnxruntime as ort
    import soundfile as sf
    proc = WhisperProcessor.from_pretrained(HUB)
    gc = json.load(open(os.path.join(OUT, "generation_config.json"), encoding="utf-8"))
    sot = gc["decoder_start_token_id"]; eos = gc["eos_token_id"]
    no_ts = gc.get("no_timestamps_token_id", 50362)
    suppress = gc.get("suppress_tokens", []); begin_suppress = gc.get("begin_suppress_tokens", [])
    prompt = [sot, no_ts]

    so = ort.SessionOptions()
    enc_s = ort.InferenceSession(ENC_ONNX, so, providers=["CPUExecutionProvider"])
    ck_s = ort.InferenceSession(os.path.join(OUT, "crosskv_init.onnx"), so, providers=["CPUExecutionProvider"])
    dec_s = ort.InferenceSession(os.path.join(OUT, "decoder_step.onnx"), so, providers=["CPUExecutionProvider"])

    wav, _ = sf.read(JFK, dtype="float32", always_2d=False)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    feats = proc(wav, sampling_rate=16000, return_tensors="np").input_features.astype(np.float32)

    enc_out = enc_s.run(["last_hidden_state"], {"input_features": feats})[0]
    ck = ck_s.run(None, {"encoder_hidden_states": enc_out})
    cross = {}
    for i in range(L):
        cross[f"cross_k_{i}"] = ck[2 * i]
        cross[f"cross_v_{i}"] = ck[2 * i + 1]

    self_k = [np.zeros((1, H, MAXLEN, D), np.float32) for _ in range(L)]
    self_v = [np.zeros((1, H, MAXLEN, D), np.float32) for _ in range(L)]
    gen = []
    seq = list(prompt)
    first = True
    for step in range(MAXLEN - 1):
        cur_tok = seq[step] if step < len(seq) else seq[-1]
        feed = {"input_ids": np.array([[cur_tok]], np.int64),
                "cache_position": np.array([step], np.int64)}
        for i in range(L):
            feed[f"past_self_k_{i}"] = self_k[i]
            feed[f"past_self_v_{i}"] = self_v[i]
        feed.update(cross)
        outs = dec_s.run(None, feed)
        logits = outs[0][0, 0, :].astype(np.float32)
        for i in range(L):
            self_k[i] = outs[1 + i]
            self_v[i] = outs[1 + L + i]
        if step < len(seq) - 1:
            continue  # still ingesting the prompt
        for s in suppress:
            if 0 <= s < logits.shape[0]: logits[s] = -np.inf
        if first:
            for s in begin_suppress:
                if 0 <= s < logits.shape[0]: logits[s] = -np.inf
            first = False
        tok = int(logits.argmax())
        if tok == eos: break
        gen.append(tok); seq.append(tok)

    text = proc.tokenizer.decode(gen, skip_special_tokens=True).strip()
    ref = _norm(JFK_REF); hyp = _norm(text)
    import difflib
    sm = difflib.SequenceMatcher(a=ref, b=hyp)
    edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    wer = 100.0 * edits / max(len(ref), 1)
    print(f"\n[CPU static-KV] tokens={len(gen)}  WER~={wer:.2f}%")
    print("  text:", text)


if __name__ == "__main__":
    export()
    validate_cpu()
