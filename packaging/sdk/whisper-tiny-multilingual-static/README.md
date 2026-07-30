# onnx models package

`models.onnx.whisper-tiny-multilingual-static.$version.nupkg`

Whisper tiny multilingual exported to static-shape ONNX: a 7 s product window
padded onto the 30 s mel canvas the encoder expects. This is the package
`asw::framework::whisper` `StaticOrtSession` loads, and the one
`model_registry.h` lists as `kWhisperMultilingualFiles`.

To customize build update:

* `$name` in `build-nuget.ps1`
* files in `models.json`
    * key: base filename in package
    * value: download url

## Contents

Consumers get everything under `models/` in the package. Four files are load
bearing at runtime -- `encoder_model.onnx`, `decoder_model.onnx`, `vocab.json`
and `generation_config.json` are the only ones `StaticOrtSession.cpp` opens.
The remaining Hugging Face export metadata is 4.4 MB against 148 MB and is what
a re-export or a tokenizer investigation needs, so it ships too.

## Provenance

The bytes are produced by
[whisper-npu-hal](https://git.int.avast.com/ai/whisper-npu-hal) and uploaded by
its `tools/publish/upload-model-assets.ps1`, which also regenerates
`models.json` and `SHA256SUMS` here. The Artifactory path carries the short
commit of the HAL checkout that exported them, so a re-export lands beside the
old copy rather than replacing it.

That immutability is not decoration. The Whisper accuracy gate in AvastClient
(`framework/whisper/src/whisper_unit_test/StaticAccuracyTest.cpp`) asserts a
per-clip and mean word error rate committed against **one specific export**;
mean WER 0.161462 belongs to these bytes and nothing else. Silently swapping
the payload under a fixed URL would turn that gate into a coin toss.

`download.ps1` verifies every file against `SHA256SUMS` and fails the build on
drift or on a failed transfer. It is deliberately stricter than the copy the
other sdk model packages share, which discards its download job results and so
can publish a package with a missing model in it.

## Versioning

The package version is the git branch, as everywhere else in `sdk`. Register
the default branch in
[sdk/tcsettings](https://git.int.avast.com/sdk/tcsettings) `ProjectCommon.kt`.
