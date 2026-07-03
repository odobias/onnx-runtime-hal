import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast


def load_audio(path):
    d, sr = sf.read(path)
    if d.ndim > 1:
        d = d.mean(axis=1)
    return d.astype(np.float32)


def run(model_dir, wav, max_tokens=100):
    # model_dir may be "encdir:decdir" to mix precisions.
    if ":" in model_dir and model_dir[1:3] != ":\\":
        enc_dir, dec_dir = model_dir.split(":", 1)
    else:
        enc_dir = dec_dir = model_dir
    src = Path(enc_dir)
    fe = WhisperFeatureExtractor.from_pretrained(str(src))
    tok = WhisperTokenizerFast.from_pretrained(str(src))
    feat = fe(load_audio(wav), sampling_rate=16000, return_tensors="np").input_features.astype(np.float32)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    enc = ort.InferenceSession(str(Path(enc_dir) / "encoder_model.onnx"), so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(Path(dec_dir) / "decoder_model.onnx"), so, providers=["CPUExecutionProvider"])

    hidden = enc.run(None, {"input_features": feat})[0].astype(np.float32)
    ids = [50257, 50362]  # sot, no_timestamps
    eot = 50256
    lp_sum, n = 0.0, 0
    for _ in range(max_tokens):
        logits = dec.run(["logits"], {"input_ids": np.array([ids], dtype=np.int64),
                                      "encoder_hidden_states": hidden})[0][0, -1]
        m = logits.max()
        lse = m + np.log(np.exp(logits - m).sum())
        nxt = int(logits.argmax())
        lp_sum += float(logits[nxt] - lse)
        n += 1
        if nxt == eot:
            break
        ids.append(nxt)
    text = tok.decode(ids[2:], skip_special_tokens=True)
    print(f"  text        : {text}")
    print(f"  avg_logprob : {lp_sum / max(n,1):.4f}")
    print(f"  tokens      : {n}")


if __name__ == "__main__":
    wav = "models/jfk.wav"
    for md in sys.argv[1:]:
        print(f"==================== {md} (ORT CPU, unpadded) ====================")
        run(md, wav)
