# Multilingual smoke: static-onnx-tiny-multi-7s (full window)

Source: FLEURS short clips under `artifacts/workloads/speech/multilingual/`.
Decode: static-no-KV greedy, task `<|transcribe|>` (never `<|translate|>`).
Language token: `auto` — detected from audio (argmax over the 99 `<|xx|>` tokens after `<|startoftranscript|>`), no label given to the model.

Pass: **6/6** · language detected: **6/6** · mean WER: **40.0%**

Clips cut by the 7s product window are marked `cut`: their WER is against the
full reference, so it measures the missing tail, not a wrong language.

| id | expected | detected | conf | WER | cut | hyp (verbatim) |
|----|----------|----------|------|-----|-----|----------------|
| fleurs_de | de | de | 1.00 | 28.6% | no | Die Höhle befindet sich auf der Spitze eines der Bergen örtlich von Mecker und ist zum Reste welt völlig isoliert. |
| fleurs_fr | fr | fr | 0.99 | 72.0% | no | Il ajoutait qu'on ne devrait se prendre un balleur de monnaie d'assumer des obligations qui n'est pas sur un style de développement, de responsabilité et leur capacité. |
| fleurs_es | es | es | 0.99 | 0.0% | no | Fue tanta la cantidad de gente que se concentró que no todos pudieron acceder al funeral en la plaza de San Pedro. |
| fleurs_cs | cs | cs | 0.96 | 68.8% | no | Kůry jsou na přebně 70 km na přeberáce nestraňi a okolostu kilometrů na odveracone straňi. |
| fleurs_it | it | it | 0.90 | 0.0% | no | L'incidente è avvenuto in alta montagna e si ritiene sia stato causato da fuoco nemico. |
| fleurs_pl | pl | pl | 0.99 | 70.6% | no | W większość mniejszych wezpto na radzenie zależne lub zpomieżone sferancją i trunące jako lub sosowe nadbalskie miejscowości wypoczynkowe. |
