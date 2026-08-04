# Benchmark results

The benchmark harness writes local outputs below this directory. Generated
results are ignored by Git because they can contain machine fingerprints,
absolute paths, proprietary model metadata, and other evidence unsuitable for a
public source repository.

Versioned benchmark evidence and internal reports live in a private companion
repository. The public harness remains the source of truth for schemas,
validation, and execution behavior.

## Local output layout

| Path | Role |
|------|------|
| `accuracy-runs/` | Immutable, invocation-scoped accuracy evidence |
| `run-attempts/` | Requested workload/device/provider outcomes, including failures |
| `host-snapshots/` | Runtime, hardware, driver, OS, firmware, and SDK context |
| `model-compilation/` | Exact provider-input graph and transformation provenance |
| `ledgers/` | Local rolling state and explicit latency measurements |
| `reports/` | Derived human-readable reports |
| `local/` | Ephemeral research and sweep output |

Do not force-add generated evidence here. Review it and publish it to the
private companion instead.

## Benchmarking contracts

The portable suite separates three contracts:

1. `accuracy-quick` is the default benchmark of record. It evaluates the full
   reference fixture, applies provider and numerical trust gates, and writes
   immutable accuracy and attempt files for that invocation.
2. `latency` is explicit. It writes the performance CSVs; accuracy-quick timing
   is orientation data, not a latency leaderboard.
3. Direct CLI invocations are smoke tests or executor diagnostics. Successful
   inference alone does not make them comparable benchmark evidence.

Both suite modes keep untimed warm-up separate from measured repetitions.
Comparisons require matching workload/profile, model and fixture hashes,
reference contract, runtime target, requested device, and measurement purpose.
Host architecture, runtime version, power source, thermal conditions, and
provider assignment remain visible caveats.

Each accuracy record carries the execution profile, graph role, model and
fixture hashes, runtime environment snapshot ID, compilation-provenance IDs,
requested and resolved provider, fallback status, workload metrics, and a
`trust_gate`. Invalid diagnostic payloads remain observable but must not enter
valid summaries.

## Executor diagnostics

Successful rows include:

- `requested_provider`
- `resolved_provider`
- `fallback_occurred`
- `provider_attempts`
- `ep_nodes`
- `cpu_nodes`
- `cpu_offload_pct`
- `cpu_offload_ops`
- `assigned_ops_cpu`
- `assigned_ops_npu`
- `operation_assignment_source`
- `error`

`provider_attempts` is JSON stored inside the CSV field. CPU offload uses
distinct ORT profiler nodes, not compute share. Provider/compiler assignment
fields are populated only when a trustworthy operation mapping is available;
profiler node counts are never promoted into assignment counts.

Classifier and Whisper startup distinguish:

- `cold_load_seconds`: first session creation after clearing its cache.
- `hot_load_seconds`: second session creation from the artifact produced by the
  cold load.
- `load_seconds`: compatibility alias for the cold value.

## Schema authority

Machine-readable contracts live in:

- `../benchmark/schemas/asr-results.columns.json`
- `../benchmark/schemas/classifier-results.columns.json`
- `../benchmark/schemas/accuracy-record.schema.json`
- `../benchmark/schemas/attempt-record.schema.json`

The C++ writer and PowerShell harness follow those column orders. Run
`benchmark/validate-schemas.ps1` to verify both writers.

`model_sha256` identifies inference artifacts by content. Package paths and
filenames do not affect identity, while any graph or weight change does.
