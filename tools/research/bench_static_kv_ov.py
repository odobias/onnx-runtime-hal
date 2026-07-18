#!/usr/bin/env python3
"""Benchmark the portable static-KV ONNX pipeline on NPU/GPU/CPU via OpenVINO.

REFERENCE / BACKGROUND ONLY -- this is NOT the benchmark of record.
The benchmark of record is the C++ app driven by benchmark/run-suite.ps1.
This script is kept purely as the OpenVINO-native reference for the portable
static-KV graph until that path is plumbed into the C++ ort backend, which will
then supersede these numbers. Do not cite its latency as authoritative -- it
bypasses the HAL/app and only sees the EPs the local Python wheel ships.

Pipeline (all static shapes, plain ONNX ops -- see export_static_kv.py):
  encoder_model.onnx        [1,80,3000] -> [1,1500,384]
  crosskv_init.onnx         enc_hidden  -> cross_k/v[L] (once)
  decoder_step.onnx         input_ids[1,1]+cache_position+self past+cross -> logits+present

Cross-KV tensors are set ONCE on the persistent decoder infer request (they never
change across steps), so only input_ids / cache_position / self-KV are updated per
token. Compare against the OV-IR genai-bounded-kv and static-no-kv ONNX baselines
in results/local/quantization-benchmark.* .

  .venv\\Scripts\\python.exe scripts\\experiments\\bench_static_kv_ov.py
"""
import json
import os
import time

import numpy as np
import openvino as ov
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, "models")
KV = os.path.join(MODELS, "whisper-tiny-en-statickv")
ENC = os.path.join(MODELS, "whisper-tiny-en-static-onnx", "encoder_model.onnx")
EVAL = os.path.join(MODELS, "eval", "eval.jsonl")
MAXLEN, H, D, L, ENC_LEN = 128, 6, 64, 4, 1500


def norm(t):
    return "".join(c.lower() if (c.isalnum() or c == "'") else " " for c in t).split()


def wer(ref, hyp):
    import difflib
    r, h = norm(ref), norm(hyp)
    sm = difflib.SequenceMatcher(a=r, b=h)
    e = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    return 100.0 * e / max(len(r), 1)


def main():
    from transformers import WhisperProcessor
    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    gc = json.load(open(os.path.join(KV, "generation_config.json"), encoding="utf-8"))
    sot, eos = gc["decoder_start_token_id"], gc["eos_token_id"]
    no_ts = gc.get("no_timestamps_token_id", 50362)
    suppress = gc.get("suppress_tokens", [])
    prompt = [sot, no_ts]

    clips = [json.loads(l) for l in open(EVAL, encoding="utf-8") if l.strip()]
    c0 = clips[0]
    wav, _ = sf.read(os.path.join(ROOT, c0["audio"].replace("/", os.sep)),
                     dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    feats = proc(wav, sampling_rate=16000, return_tensors="np").input_features.astype(np.float32)

    core = ov.Core()
    print(f"OpenVINO {ov.get_version()}  devices={core.available_devices}")

    enc_m = core.read_model(ENC)
    enc_m.reshape({"input_features": [1, 80, 3000]})
    ck_m = core.read_model(os.path.join(KV, "crosskv_init.onnx"))
    dec_m = core.read_model(os.path.join(KV, "decoder_step.onnx"))

    def decode(enc_r, ck_r, dec_r):
        enc_r.infer({"input_features": feats})
        enc_h = enc_r.get_output_tensor(0).data
        ck_out = ck_r.infer({"encoder_hidden_states": enc_h})
        vals = list(ck_out.values())
        cross = {}
        for i in range(L):
            cross[f"cross_k_{i}"] = vals[2 * i]
            cross[f"cross_v_{i}"] = vals[2 * i + 1]
        for name, arr in cross.items():
            dec_r.set_tensor(name, ov.Tensor(np.ascontiguousarray(arr)))
        self_k = [np.zeros((1, H, MAXLEN, D), np.float32) for _ in range(L)]
        self_v = [np.zeros((1, H, MAXLEN, D), np.float32) for _ in range(L)]
        seq, gen = list(prompt), []
        for step in range(MAXLEN - 1):
            cur = seq[step] if step < len(seq) else seq[-1]
            dec_r.set_tensor("input_ids", ov.Tensor(np.array([[cur]], np.int64)))
            dec_r.set_tensor("cache_position", ov.Tensor(np.array([step], np.int64)))
            for i in range(L):
                dec_r.set_tensor(f"past_self_k_{i}", ov.Tensor(np.ascontiguousarray(self_k[i])))
                dec_r.set_tensor(f"past_self_v_{i}", ov.Tensor(np.ascontiguousarray(self_v[i])))
            dec_r.infer()
            outs = {dec_m.outputs[j].get_any_name(): dec_r.get_output_tensor(j).data
                    for j in range(len(dec_m.outputs))}
            for i in range(L):
                self_k[i] = outs[f"present_self_k_{i}"].copy()
                self_v[i] = outs[f"present_self_v_{i}"].copy()
            if step < len(seq) - 1:
                continue
            logits = outs["logits"][0, 0, :].astype(np.float32)
            for s in suppress:
                if 0 <= s < logits.shape[0]:
                    logits[s] = -np.inf
            tok = int(logits.argmax())
            if tok == eos:
                break
            gen.append(tok)
            seq.append(tok)
        return gen

    for dev in ("NPU", "GPU", "CPU"):
        if dev not in core.available_devices:
            print(f"\n== {dev}: n/a =="); continue
        cache = os.path.join(ROOT, "cache", f"statickv-{dev}")
        os.makedirs(cache, exist_ok=True)
        cfg = {"CACHE_DIR": cache}
        print(f"\n== {dev} ==")
        t = time.time()
        enc_c = core.compile_model(enc_m, dev, cfg)
        ck_c = core.compile_model(ck_m, dev, cfg)
        dec_c = core.compile_model(dec_m, dev, cfg)
        print(f"   cold compile: {time.time()-t:.2f}s")
        enc_r, ck_r, dec_r = enc_c.create_infer_request(), ck_c.create_infer_request(), dec_c.create_infer_request()
        gen = decode(enc_r, ck_r, dec_r)  # warm
        text = proc.tokenizer.decode(gen, skip_special_tokens=True).strip()
        lat = []
        for _ in range(5):
            t = time.time(); decode(enc_r, ck_r, dec_r); lat.append((time.time() - t) * 1000)
        print(f"   mean {np.mean(lat):.1f} ms  (min {np.min(lat):.1f})  tokens={len(gen)}  WER={wer(c0['ref'], text):.2f}%")
        print(f"   text: {text[:70]}")


if __name__ == "__main__":
    main()
