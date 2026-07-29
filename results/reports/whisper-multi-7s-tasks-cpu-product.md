# Whisper task check: static-onnx-tiny-multi-7s (cpu, product window)

Runner: `NpuInferenceBench.exe` (the shipping static engine), language auto-detected per clip.
Audio is cut into 7s windows with 1s overlap and the transcripts stitched on their shared words, so a clip costs one encoder pass per window.
Source: FLEURS clips in `artifacts/workloads/speech/multilingual/manifest.jsonl`.

## task `transcribe`

Scored against the FLEURS reference in the spoken language (`ref`). Provider: `CPUExecutionProvider`.

Source language detected: **6/6** Â· windows decoded: **11** Â· mean WER: **39.5%** Â· mean CER: **16.8%**

| Clip | Spoken | Detected | Windows | WER | Output |
|------|--------|----------|---------|-----|--------|
| fleurs_de | de | de | 2 | 23.8% | Die Höhle befindet sich auf der Spitze eines der Bergen örtlich von Mecker und ist zum Rest. Welt völlig isoliert. |
| fleurs_fr | fr | fr | 2 | 75.0% | Il ajoutait qu'on ne devrait se prendre un balleur de monnaie d'assumer des obligations qui n'est pas sur un style de développement, de responsabilité et leur capacité. |
| fleurs_es | es | es | 2 | 4.5% | Fue tanta la cantidad de gente que se concentró que no todos pudieron acceder al funerado. en la plaza de San Pedro. |
| fleurs_cs | cs | cs | 2 | 68.8% | Kůry jsou na přebně 70 km na přeberáce nestraňi a okolostu kilometrů na odveracone straňi. |
| fleurs_it | it | it | 1 | 0.0% | L'incidente è avvenuto in alta montagna e si ritiene sia stato causato da fuoco nemico. |
| fleurs_pl | pl | pl | 2 | 58.8% | W większość mniejszych wysp to narady niezależne lub zpomieżone sferancją i trunące jako lub sosowe nadbalskie miejscowości wypoczynko. |

