# speech samples package

`models.speech.whisper-accuracy-samples.$version.nupkg`

The five LibriSpeech clips the Whisper word-error-rate gate transcribes:
`ls_000`, `ls_001`, `ls_006`, `ls_008`, `ls_010`. 16 kHz mono PCM, 0.83 MB in
total, each under the 7 s product window so no sliding-window stitching is
involved and a regression points at the model rather than the seam logic.

To customize build update:

* `$name` in `build-nuget.ps1`
* files in `models.json`
    * key: base filename in package
    * value: download url

## Why the id is not `models.onnx.*`

Every other package in this family is `models.onnx.<name>`, but this one holds
WAV audio and no model at all. Reusing that prefix would make the feed lie to
anyone filtering on it, so the id is `models.speech.whisper-accuracy-samples`.
Nothing shared is affected: the TeamCity upload spec keys off the repository
path, not the package id, and `models.nuspec.template` is per-repository.

## Reference transcripts live elsewhere, on purpose

This package is audio only. The expected transcripts and the per-clip WER
baselines are committed in AvastClient at
`framework/whisper/src/whisper_unit_test/AccuracyCorpus.h`, so a change in what
we consider acceptable accuracy shows up as a reviewable diff next to the test
rather than as a new opaque package version.

Keep the two in step: `kAccuracyCorpus` and `models.json` here must name the
same five clips.

## Provenance

Uploaded from [whisper-npu-hal](https://git.int.avast.com/ai/whisper-npu-hal)
by `tools/publish/upload-model-assets.ps1`, which regenerates `models.json` and
`SHA256SUMS` here. `download.ps1` verifies both and fails the build on drift.
