import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

const whisperRows = [
  ["Intel", "OpenVINO GenAI", "NPU", "bounded KV", "0.0106", "7.59 / 0.56 s", "5.88%", "AC · 1 clip"],
  ["Qualcomm", "ORT QNN", "NPU", "static, no KV", "0.0554", "24.70 / 0.94 s", "8.58%", "battery · 12 clips"],
  ["AMD", "ORT VitisAI", "NPU", "static, no KV", "0.0819", "221.20 / 2.72 s", "9.90%", "battery · 12 clips"],
  ["Generic x64", "ORT DirectML", "GPU", "static, no KV", "0.1004", "0.69 / 0.58 s", "8.58%", "battery · 12 clips"],
  ["Generic x64", "ORT CPU", "CPU", "static, no KV", "0.1041", "0.42 / 0.42 s", "8.58%", "battery · 12 clips"],
  ["Intel", "ORT OpenVINO EP", "NPU", "static, no KV", "0.1163", "14.52 / 0.94 s", "9.57%", "battery · 12 clips"],
  ["Intel", "ORT OpenVINO EP", "CPU", "dynamic KV", "0.0903", "4.25 / 2.69 s", "5.88%", "battery · 1 clip · 12534e24…"],
  ["Intel", "ORT OpenVINO EP", "GPU", "dynamic KV", "0.1814", "5.67 / 2.66 s", "5.88%", "battery · 1 clip · 12534e24…"],
];

const tscRows = [
  ["Intel", "ORT OpenVINO EP", "NPU", "8.02 ms", "8.11 / 0.55 s", "93.3%", "0.00665", "0 / 1 CPU nodes"],
  ["Intel", "ORT OpenVINO EP", "GPU", "9.31 ms", "2.29 / 1.07 s", "93.3%", "0.00122", "0 / 1 CPU nodes"],
  ["Qualcomm", "ORT QNN", "NPU", "20.50 ms", "13.96 s", "93.3%", "0.04731", "0 / 1 CPU nodes"],
  ["AMD", "ORT VitisAI", "NPU", "32.03 ms", "4.02 s", "93.3%", "0.05680", "13 CPU shape/control nodes; compute fused"],
  ["AMD", "ORT DirectML", "GPU", "62.42 ms", "0.50 s", "93.3%", "0.000306", "0 / 1 CPU nodes"],
  ["AMD", "ORT CPU", "CPU", "129.98 ms", "0.37 s", "93.3%", "0.0000019", "CPU baseline"],
  ["Intel", "ORT OpenVINO EP", "CPU", "191.72 ms", "1.28 / 0.74 s", "93.3%", "0.0000024", "0 / 1 CPU nodes"],
];

const fakeAudioRows = [
  ["Intel", "ORT OpenVINO EP CPU", "111.16 ms", "0.00000033", "Correct", "0 / 1 CPU nodes", "hash 549143bc… · battery"],
  ["Intel", "ORT OpenVINO EP GPU", "17.55 ms at f16", "NaN", "Wrong", "Rejected", "f32 fails OpenCL work-group launch"],
  ["Intel", "ORT DirectML GPU", "124.77 ms at f32", "0.00000016", "Correct", "1 CPU Resize / 4 GPU nodes", "recorded; prior probe 76.51 ms · battery"],
  ["Intel", "ORT OpenVINO EP NPU", "unsupported", "—", "No result", "Compiler abort", "Attention dimensions 64 vs 16"],
  ["Intel", "CPU FE + NPU backbone", "~30 ms end-to-end", "0.0001", "Correct", "~80% work on NPU", "Needs attention-bias surgery"],
  ["Qualcomm", "CPU FE + QNN backbone", "114.94 ms + FE", "0.00084", "Correct", "0 / 1 CPU nodes in backbone", "Ledger excludes CPU FE time"],
  ["AMD", "DirectML full graph", "75.79 ms", "0.0000059", "Correct", "1 CPU Resize / 4 GPU nodes", "Battery · audited 11 Jul"],
  ["x64", "ORT CPU full graph", "65.68 ms", "~0", "Correct", "CPU", "AC reference run"],
  ["Intel", "NPU full graph", "~28 ms", "0.9924", "Wrong", "Fully NPU", "Raised fp16 noise floor"],
  ["Qualcomm", "QNN full graph", "127.72 ms", "0.9325", "Wrong", "0 / 1 CPU nodes", "Fully NPU, numerically broken"],
  ["AMD", "VAIML variants", "crashes or diverges", "~0.99", "Wrong", "NPU", "source-generated 12.9 KB first-inference crash repro"],
];

function Caption({ children }: { children: string }) {
  return <Text size="small" tone="tertiary" style={{ marginTop: 6 }}>{children}</Text>;
}

function Overview() {
  return (
    <Stack gap={20}>
      <Grid columns={4} gap={14}>
        <Stat value="3" label="NPU vendors measured" />
        <Stat value="0.0106" label="Best Whisper RTF · Intel GenAI" tone="success" />
        <Stat value="8.02 ms" label="Fastest TSC · Intel NPU" tone="success" />
        <Stat value="0%" label="QNN CPU fallback · 3 graphs" tone="success" />
      </Grid>

      <Callout title="Bottom line" tone="info">
        Intel GenAI remains the Whisper latency leader, but its bounded-KV decoder is not a pure
        silicon comparison. In the new battery run, Intel OpenVINO EP TSC is 4.0× faster on NPU
        than AMD VitisAI while preserving 14/15 accuracy. DirectML now runs full-graph FakeAudio
        on Intel GPU at exact FP32 reference parity; the OpenVINO GPU path still emits NaNs and
        NPU compilation still aborts. Exact Intel artifact hashes are recorded; historical AMD
        rows predate hashing, so identical model bytes are not yet proven.
      </Callout>

      <Grid columns="1.35fr 1fr" gap={18}>
        <Stack gap={8}>
          <H2>Whisper real-time factor by executor</H2>
          <Text size="small" tone="secondary">X-axis: executor and device · Y-axis: real-time factor (lower is better)</Text>
          <BarChart
            horizontal
            height={330}
            categories={[
              "Intel GenAI NPU",
              "Qualcomm QNN NPU",
              "AMD VitisAI NPU",
              "DirectML GPU",
              "ORT CPU",
              "Intel OVEP NPU · 13 Jul",
            ]}
            series={[{
              name: "Real-time factor",
              data: [0.01062, 0.05538, 0.08195, 0.10044, 0.10414, 0.11628],
              tone: "info",
            }]}
            showValues
          />
          <Caption>Source: results/ledgers/asr.csv · runs through 13 Jul 2026. Portable static rows use the same 12 clips / 130.47 s; Intel GenAI still uses one 5.855 s clip and a different decoder.</Caption>
        </Stack>

        <Card>
          <CardHeader trailing={<Pill size="sm" active>Decision readout</Pill>}>What is shippable now</CardHeader>
          <CardBody>
            <Stack gap={12}>
              <Text><Text weight="semibold">Whisper:</Text> all three vendor NPUs have a working path. Intel GenAI is fastest; Qualcomm QNN has the best measured portable-static NPU result.</Text>
              <Text><Text weight="semibold">TSC:</Text> all three NPUs preserve 14/15 fixture accuracy. Intel NPU is 8.02 ms; AMD is 32.03 ms and leaves 13 shape/control nodes on CPU.</Text>
              <Text><Text weight="semibold">Model identity:</Text> new Intel rows carry SHA-256 identities. The AMD/Qualcomm history does not, so byte-for-byte matching requires reruns there.</Text>
              <Text><Text weight="semibold">FakeAudio:</Text> ship only as CPU fp32 front-end + NPU backbone on Intel/Qualcomm.</Text>
              <Text><Text weight="semibold">AMD FakeAudio:</Text> no correct VAIML path has been demonstrated.</Text>
              <Text><Text weight="semibold">Naive INT8:</Text> not shippable for Whisper or FakeAudio; accuracy collapses.</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

function Whisper() {
  return (
    <Stack gap={16}>
      <H2>Whisper tiny.en — executors and vendors</H2>
      <Table
        headers={["Vendor", "Executor", "Device", "Decode", "RTF ↓", "Cold / hot load", "WER", "Run context"]}
        rows={whisperRows}
        columnAlign={["left", "left", "left", "left", "right", "right", "right", "left"]}
        rowTone={["success", "warning", "success", "neutral", "success", "neutral", "warning", "warning"]}
        striped
        stickyHeader
      />
      <Caption>Source: results/ledgers/asr.csv. All portable static/no-KV rows aggregate the same 12 clips / 130.47 s on battery. New one-clip Intel rows additionally record artifact hashes; Intel GenAI remains a one-clip AC run with a different bounded-KV decoder.</Caption>

      <Grid columns={2} gap={18}>
        <Card>
          <CardHeader>Executor effect</CardHeader>
          <CardBody>
            <Text>Intel GenAI's bounded-KV decode reaches RTF 0.0106, roughly 11× faster than Intel's 12-clip static no-KV OVEP path (0.1163). That gap mostly measures decoding architecture, not merely the NPU.</Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Portable static ONNX</CardHeader>
          <CardBody>
            <Text>On the same 12 clips and static/no-KV model family, QNN NPU (0.0554) leads VitisAI NPU (0.0819), DirectML GPU (0.1004), ORT CPU (0.1041), and Intel OVEP NPU (0.1163). Historical rows lack hashes, so identical model bytes are not yet proven.</Text>
          </CardBody>
        </Card>
      </Grid>

      <Callout title="Startup matters" tone="warning">
        AMD VitisAI cold compile is 221 s, versus about 25 s for QNN and 14.5 s for Intel OVEP.
        Persisted caches reduce hot load to 2.72 s, 0.94 s, and 0.94 s respectively.
      </Callout>
    </Stack>
  );
}

function Classifiers() {
  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H2>Text Scam Classifier</H2>
        <Text size="small" tone="secondary">X-axis: executor/device · Y-axis: mean inference latency in milliseconds (lower is better)</Text>
        <BarChart
          horizontal
          height={280}
          categories={["Intel OVEP NPU", "Intel OVEP GPU", "Qualcomm QNN NPU", "AMD VitisAI NPU", "AMD DirectML GPU", "AMD ORT CPU", "Intel OVEP CPU"]}
          series={[{ name: "Mean inference latency", data: [8.02, 9.31, 20.50, 32.03, 62.42, 129.98, 191.72], tone: "success" }]}
          valueSuffix=" ms"
          showValues
        />
        <Caption>Source: results/ledgers/classifiers.csv · 20 runs per sample · measurements through 13 Jul 2026. New Intel and July AMD rows were measured on battery.</Caption>
        <Table
          headers={["Vendor", "Executor", "Device", "Mean", "Load", "Accuracy", "Max probability delta", "CPU fallback"]}
          rows={tscRows}
          columnAlign={["left", "left", "left", "right", "right", "right", "right", "left"]}
          rowTone={["success", "success", "success", "warning", "neutral", "neutral", "warning"]}
          striped
        />
      </Stack>

      <Stack gap={8}>
        <H2>FakeAudio detector</H2>
        <Table
          headers={["Vendor", "Path", "Latency", "Max probability delta", "Agreement", "Placement", "Important caveat"]}
          rows={fakeAudioRows}
          columnAlign={["left", "left", "right", "right", "left", "left", "left"]}
          rowTone={["success", "danger", "success", "danger", "success", "success", "success", "success", "danger", "danger", "danger"]}
          striped
          stickyHeader
        />
        <Caption>Sources: results/ledgers/classifiers.csv, results/ledgers/classifiers.legacy.csv, fakeaudio-intel-npu.md, fakeaudio-qualcomm-npu.md, and repros/amd-vaiml-window-partition/README.md · through 13 Jul 2026.</Caption>
      </Stack>

      <Callout title="Newest AMD VAIML result" tone="danger">
        A synthetic source-generated 12,884-byte window-partition ONNX returns a finite [64,64,96]
        tensor on CPU. VAIML compiles it with zero reported errors and creates a
        VitisAI session, then crashes on the first Run() with access violation
        0xC0000005 inside onnxruntime_vitisai_ep.dll. Cold and cached paths reproduce.
      </Callout>

      <Callout title="Do not use label accuracy to validate accelerator correctness" tone="warning">
        The FakeAudio fixture's fp32 reference is already only 3/5. The broken Qualcomm full-NPU
        run scores 4/5 by accidentally flipping a sample, while diverging from the trusted CPU
        probabilities by 0.93. Cross-executor probability agreement is the correctness metric.
      </Callout>
    </Stack>
  );
}

function Caveats() {
  return (
    <Stack gap={16}>
      <H2>Comparison limits</H2>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Power and host state</CardHeader>
          <CardBody><Text>Rows mix AC and battery measurements. Battery runs can throttle CPU, GPU, and NPU clocks; do not attribute every delta to architecture.</Text></CardBody>
        </Card>
        <Card>
          <CardHeader>Evaluation scope</CardHeader>
          <CardBody><Text>The portable static/no-KV Whisper rows now cover the same 12 clips / 130.47 s. Intel GenAI still covers one 5.855 s clip, so its WER and variance remain less robust.</Text></CardBody>
        </Card>
        <Card>
          <CardHeader>Decode strategy</CardHeader>
          <CardBody><Text>Intel GenAI uses bounded KV caching. Portable NPU paths use static no-KV recomputation. Comparing their latency as if only the hardware changed would be nonsense.</Text></CardBody>
        </Card>
        <Card>
          <CardHeader>Offload counters</CardHeader>
          <CardBody><Text>Fused-node counts are not compute percentages. QNN reports one fused NPU partition and zero CPU nodes. Intel Whisper OVEP reports 11 accelerator nodes plus 11 CPU nodes (IsNaN ×4, Transpose ×7): genuine host fallback, but not evidence that 50% of FLOPs ran on CPU.</Text></CardBody>
        </Card>
        <Card>
          <CardHeader>AMD TSC host shape operations</CardHeader>
          <CardBody><Text>VitisAI runs the expensive transformer as one fused NPU node but leaves 13 shape/control nodes on CPU. A fixed-shape ONNX simplification reduced this by only one Unsqueeze and did not change latency or probability drift, so the original graph is retained.</Text></CardBody>
        </Card>
        <Card>
          <CardHeader>Artifact identity</CardHeader>
          <CardBody><Text>Intel static Whisper: 9da9f440…; dynamic Whisper: 12534e24…; TSC: 90f2ed1b…; FakeAudio: 549143bc…. Hashes cover graph and weight artifacts by content, independent of filenames. Historical rows remain blank.</Text></CardBody>
        </Card>
      </Grid>
      <Callout title="Current evidence quality" tone="info">
        The strongest cross-vendor conclusions are qualitative: all three NPUs run Whisper;
        all three preserve TSC fixture accuracy; FakeAudio requires a CPU fp32 front-end plus NPU
        backbone on Intel/Qualcomm; and AMD FakeAudio remains unresolved. Exact latency rankings
        need a controlled AC-powered, same-clips, same-hash, same-runtime rerun.
      </Callout>
    </Stack>
  );
}

export default function CrossVendorBenchmarkResults() {
  const theme = useHostTheme();
  const [tab, setTab] = useCanvasState("benchmark-tab", "Overview");

  return (
    <Stack gap={18} style={{ padding: 24, background: theme.bg.editor, minHeight: "100%" }}>
      <Stack gap={6}>
        <H1>Benchmark results across executors and vendors</H1>
        <Text tone="secondary">Whisper, TSC, FakeAudio, and AMD VAIML · measurements through 13 Jul 2026</Text>
      </Stack>

      <Row gap={8} wrap>
        {["Overview", "Whisper", "Classifiers", "Caveats"].map((name) => (
          <Pill active={tab === name} onClick={() => setTab(name)}>{name}</Pill>
        ))}
      </Row>

      {tab === "Overview" && <Overview />}
      {tab === "Whisper" && <Whisper />}
      {tab === "Classifiers" && <Classifiers />}
      {tab === "Caveats" && <Caveats />}
    </Stack>
  );
}
