"""
Quantize the vendor-neutral Whisper-tiny.en ONNX (encoder + decoder) to INT8 QDQ,
targeting AMD Ryzen AI / VitisAI EP while staying loadable by OpenVINO (Intel).

This does NOT touch the FP32 source; it writes a NEW sibling variant dir. Intel
keeps running the FP32 model. The point is a portable INT8 QDQ artifact:
  - QDQ format (QuantizeLinear/DequantizeLinear) is what VitisAI EP ingests.
  - symmetric + per-tensor is the XDNA-friendly recipe (mirrors vai_q_onnx defaults).
  - OpenVINO can also import QDQ INT8, so we can sanity-gate on Intel here.

CAVEAT: AMD officially recommends vai_q_onnx / AMD Quark (not installed here).
This is ORT static quantization with XDNA-leaning options; real XDNA offload must
be validated on Ryzen AI hardware. We cannot prove NPU execution on an Intel box.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import WhisperFeatureExtractor


def load_audio_16k_mono(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample

        n = int(round(len(data) * 16000 / sr))
        data = resample(data, n)
    return data.astype(np.float32)


def build_features(src: Path, wavs, feat_extractor) -> list:
    feats = []
    for w in wavs:
        audio = load_audio_16k_mono(w)
        f = feat_extractor(audio, sampling_rate=16000, return_tensors="np").input_features
        feats.append(f.astype(np.float32))  # [1, 80, 3000]
    return feats


class EncoderReader(CalibrationDataReader):
    def __init__(self, feats):
        self._it = iter([{"input_features": f} for f in feats])

    def get_next(self):
        return next(self._it, None)


class DecoderReader(CalibrationDataReader):
    """Yields realistic (input_ids, encoder_hidden_states) prefixes by greedily
    decoding each calibration clip with the FP32 decoder (static no-KV path)."""

    def __init__(self, enc_sess, dec_sess, feats, prompt, eot, max_steps):
        self._gen = self._make(enc_sess, dec_sess, feats, prompt, eot, max_steps)

    def _make(self, enc_sess, dec_sess, feats, prompt, eot, max_steps):
        # Real, unpadded prefixes only: the eot-padded static tail produces
        # non-finite activations that would poison calibration (histogram AND
        # min/max). The runtime tolerates that garbage because it reads only the
        # current position; the calibrator must not see it.
        for f in feats:
            hidden = enc_sess.run(None, {"input_features": f})[0].astype(np.float32)
            ids = list(prompt)
            prefixes = [list(ids)]
            for _ in range(max_steps):
                logits = dec_sess.run(
                    ["logits"],
                    {"input_ids": np.array([ids], dtype=np.int64),
                     "encoder_hidden_states": hidden},
                )[0]
                nxt = int(np.argmax(logits[0, -1]))
                if nxt == eot:
                    break
                ids.append(nxt)
                prefixes.append(list(ids))
            for s in prefixes:
                yield {"input_ids": np.array([s], dtype=np.int64),
                       "encoder_hidden_states": hidden}

    def get_next(self):
        return next(self._gen, None)


def quant(inp: Path, out: Path, reader, per_channel: bool, calib: str,
          act_symmetric: bool, nodes_to_exclude):
    prep = inp.with_suffix(".prep.onnx")
    quant_pre_process(str(inp), str(prep), skip_symbolic_shape=False)
    method = {
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
        "entropy": CalibrationMethod.Entropy,
    }[calib]
    quantize_static(
        str(prep),
        str(out),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=method,
        nodes_to_exclude=list(nodes_to_exclude) if nodes_to_exclude else None,
        extra_options={"ActivationSymmetric": act_symmetric, "WeightSymmetric": True},
    )
    prep.unlink(missing_ok=True)
    # quant_pre_process may drop an external-data file next to prep; clean stragglers.
    for junk in inp.parent.glob("*.prep*.onnx*"):
        junk.unlink(missing_ok=True)


def find_lm_head_matmuls(model_path: Path, vocab_size: int):
    """Names of MatMul nodes whose output is the [.., vocab] logits projection.
    Keeping the LM head in FP32 is the single biggest accuracy lever for INT8
    decoders and is EP-friendly (VitisAI/OpenVINO just run it unquantized)."""
    import onnx

    m = onnx.load(str(model_path), load_external_data=False)
    # weight initializers with a dim == vocab_size are the LM head weight.
    heads = set()
    init_dims = {i.name: list(i.dims) for i in m.graph.initializer}
    for node in m.graph.node:
        if node.op_type in ("MatMul", "Gemm"):
            for inp in node.input:
                if inp in init_dims and vocab_size in init_dims[inp]:
                    heads.add(node.name)
    return heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/whisper-tiny-en-onnx")
    ap.add_argument("--out", default="models/whisper-tiny-en-onnx-int8")
    ap.add_argument("--eval", default="models/eval/eval.jsonl")
    ap.add_argument("--extra-wav", default="models/jfk.wav")
    ap.add_argument("--max-clips", type=int, default=13)
    ap.add_argument("--max-steps", type=int, default=24)
    ap.add_argument("--dec-len", type=int, default=128,
                    help="static decoder context to calibrate at (match the C++ WHISPER_ONNX_MAXLEN)")
    ap.add_argument("--per-channel", action="store_true", default=False)
    ap.add_argument("--calib", choices=["minmax", "percentile", "entropy"], default="minmax")
    ap.add_argument("--act-asymmetric", action="store_true", default=False,
                    help="asymmetric int8 activations (better accuracy, less XDNA-friendly)")
    ap.add_argument("--keep-lm-head-fp32", action="store_true", default=False,
                    help="exclude the vocab projection MatMul from quantization")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Calibration audio set.
    wavs = []
    if Path(args.eval).exists():
        for line in Path(args.eval).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = Path(rec["audio"])
            if p.exists():
                wavs.append(p)
    if args.extra_wav and Path(args.extra_wav).exists():
        wavs.append(Path(args.extra_wav))
    wavs = wavs[: args.max_clips]
    if not wavs:
        raise SystemExit("no calibration audio found")
    print(f"calibration clips: {len(wavs)}")

    fe = WhisperFeatureExtractor.from_pretrained(str(src))
    feats = build_features(src, wavs, fe)

    gc = json.loads((src / "generation_config.json").read_text(encoding="utf-8"))
    sot = gc.get("decoder_start_token_id", 50257)
    no_ts = gc.get("no_timestamps_token_id", 50362)
    eot = gc.get("eos_token_id", 50256)
    prompt = [sot, no_ts]
    print(f"prompt tokens: {prompt}  eot: {eot}")

    enc_fp32 = src / "encoder_model.onnx"
    dec_fp32 = src / "decoder_model.onnx"

    so = ort.SessionOptions()
    so.log_severity_level = 3
    enc_sess = ort.InferenceSession(str(enc_fp32), so, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(dec_fp32), so, providers=["CPUExecutionProvider"])

    act_sym = not args.act_asymmetric
    print(f"config: per_channel={args.per_channel} calib={args.calib} "
          f"act_symmetric={act_sym} keep_lm_head_fp32={args.keep_lm_head_fp32}")

    print("quantizing encoder ...")
    quant(enc_fp32, out / "encoder_model.onnx", EncoderReader(feats),
          args.per_channel, args.calib, act_sym, None)

    dec_exclude = set()
    if args.keep_lm_head_fp32:
        vocab = gc.get("vocab_size") or 51864
        dec_exclude = find_lm_head_matmuls(dec_fp32, vocab)
        print(f"excluding LM-head nodes from quant: {sorted(dec_exclude)}")

    print("quantizing decoder (this runs greedy decode traces for calibration) ...")
    quant(
        dec_fp32,
        out / "decoder_model.onnx",
        DecoderReader(enc_sess, dec_sess, feats, prompt, eot, args.max_steps),
        args.per_channel, args.calib, act_sym, dec_exclude,
    )

    # Copy the non-ONNX assets the C++ backend needs (tokenizer, configs, mel).
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".onnx":
            shutil.copy2(f, out / f.name)

    def mb(p):
        return round(p.stat().st_size / (1024 * 1024), 1)

    print("\n=== done ===")
    for name in ["encoder_model.onnx", "decoder_model.onnx"]:
        s, d = src / name, out / name
        print(f"{name:22} fp32 {mb(s):7.1f} MB  ->  int8 {mb(d):7.1f} MB")
    print(f"output dir: {out}")


if __name__ == "__main__":
    main()
