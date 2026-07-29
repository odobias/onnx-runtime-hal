# Whisper task check: static-onnx-tiny-multi-7s (cpu, full window)

Runner: `NpuInferenceBench.exe` (the shipping static engine), language auto-detected per clip.
The whole clip is decoded in a single pass -- not the product path, kept only to price windowing.
Source: FLEURS clips in `artifacts/workloads/speech/multilingual/manifest.jsonl`.

## task `transcribe`

Scored against the FLEURS reference in the spoken language (`ref`). Provider: `CPUExecutionProvider`.

Source language detected: **6/6** Â· windows decoded: **6** Â· mean WER: **41.2%** Â· mean CER: **17.1%**

| Clip | Spoken | Detected | Windows | WER | Output |
|------|--------|----------|---------|-----|--------|
| fleurs_de | de | de | 1 | 28.6% | Die Höhle befindet sich auf der Spitze eines der Bergen örtlich von Mecker und ist zum Reste welt völlig isoliert. |
| fleurs_fr | fr | fr | 1 | 75.0% | Il ajoutait qu'on ne devrait se prendre un balleur de monnaie d'assumer des obligations qui n'est pas sur un style de développement, de responsabilité et leur capacité. |
| fleurs_es | es | es | 1 | 0.0% | Fue tanta la cantidad de gente que se concentró que no todos pudieron acceder al funeral en la plaza de San Pedro. |
| fleurs_cs | cs | cs | 1 | 68.8% | Kůry jsou na přebně 70 km na přeberáce nestraňi a okolostu kilometrů na odveracone straňi. |
| fleurs_it | it | it | 1 | 0.0% | L'incidente è avvenuto in alta montagna e si ritiene sia stato causato da fuoco nemico. |
| fleurs_pl | pl | pl | 1 | 70.6% | W większość mniejszych wezpto na radzenie zależne lub zpomieżone sferancją i trunące jako lub sosowe nadbalskie miejscowości wypoczynkowe. |

## task `translate`

Scored against the parallel FLoRes English sentence (`ref_en`). Provider: `CPUExecutionProvider`.

Source language detected: **6/6** Â· windows decoded: **6** Â· mean WER: **92.1%** Â· mean CER: **68.2%**

| Clip | Spoken | Detected | Windows | WER | Output |
|------|--------|----------|---------|-----|--------|
| fleurs_de | de | de | 1 | 82.6% | The holes are located on the back, one of the holes are made of a marker and is usually welded. |
| fleurs_fr | fr | fr | 1 | 95.5% | And the second thing we need to do is to ensure the obligations that are not available on the state of development, the responsibility and their capacity. |
| fleurs_es | es | es | 1 | 88.0% | It was so much the amount of people who were concentrated that they could not access the funeral in the room of that emperor. |
| fleurs_cs | cs | cs | 1 | 57.9% | The height is about 70 km long, and the height is about 100 km long. |
| fleurs_it | it | it | 1 | 94.7% | The incident has been in the mountains and the animals have been used to be used to be with the enemy. |
| fleurs_pl | pl | pl | 1 | 138.9% | The most important thing is that the enemy is not allowed to use the French or the French or the French, but the German place is the same. |

