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
  ["Intel", "ORT OpenVINO EP", "NPU", "static, no KV", "0.1567", "13.99 / 0.87 s", "5.88%", "AC · 1 clip"],
];

const tscRows = [
  ["Intel", "ORT OpenVINO EP", "NPU", "8.67 ms", "7.87 s", "93.3%", "0.00665", "0% in research audit"],
  ["Qualcomm", "ORT QNN", "NPU", "20.50 ms", "13.96 s", "93.3%", "0.04731", "0 / 1 CPU nodes"],
  ["AMD", "ORT VitisAI", "NPU", "32.60 ms", "5.69 s", "93.3%", "0.05680", "13 CPU shape/control nodes; compute fused"],
  ["AMD", "ORT DirectML", "GPU", "65.15 ms", "0.81 s", "93.3%", "0.000306", "0 / 1 CPU nodes"],
  ["AMD", "ORT CPU", "CPU", "163.13 ms", "0.58 s", "93.3%", "0.0000019", "CPU baseline"],
];

const fakeAudioRows = [
  ["Intel", "CPU FE + NPU backbone", "~30 ms end-to-end", "0.0001", "Correct", "~80% work on NPU", "Needs attention-bias surgery"],
  ["Qualcomm", "CPU FE + QNN backbone", "114.94 ms + FE", "0.00084", "Correct", "0 / 1 CPU nodes in backbone", "Ledger excludes CPU FE time"],
  ["AMD", "DirectML full graph", "75.79 ms", "0.0000059", "Correct", "1 CPU Resize / 4 GPU nodes", "Battery · audited 11 Jul"],
  ["x64", "ORT CPU full graph", "65.68 ms", "~0", "Correct", "CPU", "AC reference run"],
  ["Intel", "NPU full graph", "~28 ms", "0.9924", "Wrong", "Fully NPU", "Raised fp16 noise floor"],
  ["Qualcomm", "QNN full graph", "127.72 ms", "0.9325", "Wrong", "0 / 1 CPU nodes", "Fully NPU, numerically broken"],
  ["AMD", "VAIML variants", "crashes or diverges", "~0.99", "Wrong", "NPU", "17 KB first-inference crash repro"],
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
        <Stat value="8.67 ms" label="Fastest TSC · Intel NPU" tone="success" />
        <Stat value="0%" label="QNN CPU fallback · 3 graphs" tone="success" />
      </Grid>

      <Callout title="Bottom line" tone="info">
        Intel GenAI is the clear Whisper latency leader, but it uses a different bounded-KV
        decoder; it is not a pure silicon comparison. TSC preserves accuracy on all three NPUs,
        though AMD leaves small shape/control operations on CPU and has the largest probability drift.
        FakeAudio only works correctly on an NPU when its fp32 front-end stays on CPU and the
        backbone runs on the NPU. AMD VAIML now has a synthetic 17 KB reproducer that compiles
        successfully, creates a session, and access-violates on the first inference.
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
              "Intel OVEP NPU",
            ]}
            series={[{
              name: "Real-time factor",
              data: [0.01062, 0.05538, 0.08195, 0.10044, 0.10414, 0.15666],
              tone: "info",
            }]}
            showValues
          />
          <Caption>Source: results/benchmark-results.csv · runs from 3–9 Jul 2026. RTF normalizes audio duration, but power, clip count, model package, and decode strategy differ.</Caption>
        </Stack>

        <Card>
          <CardHeader trailing={<Pill size="sm" active>Decision readout</Pill>}>What is shippable now</CardHeader>
          <CardBody>
            <Stack gap={12}>
              <Text><Text weight="semibold">Whisper:</Text> all three vendor NPUs have a working path. Intel GenAI is fastest; Qualcomm QNN has the best measured portable-static NPU result.</Text>
              <Text><Text weight="semibold">TSC:</Text> all three NPUs preserve 14/15 fixture accuracy. AMD is usable, but its host-side shape ops and 0.0568 probability drift need explicit disclosure.</Text>
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
        rowTone={["success", "success", "success", "neutral", "neutral", "warning"]}
        striped
        stickyHeader
      />
      <Caption>Source: results/benchmark-results.csv. Intel GenAI and Intel OVEP rows use one 5.855 s clip on AC; Qualcomm, AMD, DirectML, and ORT CPU rows aggregate 12 clips / 130.47 s on battery. Treat ranking as directional, not laboratory-grade.</Caption>

      <Grid columns={2} gap={18}>
        <Card>
          <CardHeader>Executor effect</CardHeader>
          <CardBody>
            <Text>Intel GenAI's bounded-KV decode reaches RTF 0.0106, roughly 15× faster than Intel's static no-KV OVEP path (0.1567). That gap mostly measures decoding architecture, not merely the NPU.</Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Portable static ONNX</CardHeader>
          <CardBody>
            <Text>For the same static/no-KV family, QNN NPU (0.0554) leads VitisAI NPU (0.0819), DirectML GPU (0.1004), ORT CPU (0.1041), and Intel OVEP NPU (0.1567) in the recorded data.</Text>
          </CardBody>
        </Card>
      </Grid>

      <Callout title="Startup matters" tone="warning">
        AMD VitisAI cold compile is 221 s, versus about 25 s for QNN and 14 s for Intel OVEP.
        Persisted caches reduce hot load to 2.72 s, 0.94 s, and 0.87 s respectively.
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
          categories={["Intel OVEP NPU", "Qualcomm QNN NPU", "AMD VitisAI NPU", "DirectML GPU", "ORT CPU"]}
          series={[{ name: "Mean inference latency", data: [8.67, 20.50, 32.60, 65.15, 163.13], tone: "success" }]}
          valueSuffix=" ms"
          showValues
        />
        <Caption>Source: results/deepfake-benchmark-cpp.csv · latest battery rows through 11 Jul 2026 · 20 runs per sample.</Caption>
        <Table
          headers={["Vendor", "Executor", "Device", "Mean", "Load", "Accuracy", "Max probability delta", "CPU fallback"]}
          rows={tscRows}
          columnAlign={["left", "left", "left", "right", "right", "right", "right", "left"]}
          rowTone={["success", "success", "warning", "neutral", "neutral"]}
          striped
        />
      </Stack>

      <Stack gap={8}>
        <H2>FakeAudio detector</H2>
        <Table
          headers={["Vendor", "Path", "Latency", "Max probability delta", "Agreement", "Placement", "Important caveat"]}
          rows={fakeAudioRows}
          columnAlign={["left", "left", "right", "right", "left", "left", "left"]}
          rowTone={["success", "success", "success", "success", "danger", "danger", "danger"]}
          striped
          stickyHeader
        />
        <Caption>Sources: deepfake-benchmark-cpp.csv, fakeaudio-intel-npu.md, fakeaudio-qualcomm-npu.md, and repros/amd-vaiml-window-partition/README.md · 9–11 Jul 2026.</Caption>
      </Stack>

      <Callout title="Newest AMD VAIML result" tone="danger">
        A synthetic 17,088-byte window-partition ONNX returns a finite [64,64,96]
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
          <CardBody><Text>Some Whisper rows cover 12 clips / 130.47 s; newer Intel rows cover one 5.855 s clip. RTF helps, but variance and WER are not equally robust.</Text></CardBody>
        </Card>
        <Card>
          <CardHeader>Decode strategy</CardHeader>
          <CardBody><Text>Intel GenAI uses bounded KV caching. Portable NPU paths use static no-KV recomputation. Comparing their latency as if only the hardware changed would be nonsense.</Text></CardBody>
        </Card>
        <Card>
          <CardHeader>Offload counters</CardHeader>
          <CardBody><Text>QNN's “1 EP node” means one fused partition containing hundreds of ONNX ops. Zero CPU nodes proves no fallback outside that partition; it does not mean the model has one operation.</Text></CardBody>
        </Card>
        <Card>
          <CardHeader>AMD TSC host shape operations</CardHeader>
          <CardBody><Text>VitisAI runs the expensive transformer as one fused NPU node but leaves 13 shape/control nodes on CPU. A fixed-shape ONNX simplification reduced this by only one Unsqueeze and did not change latency or probability drift, so the original graph is retained.</Text></CardBody>
        </Card>
      </Grid>
      <Callout title="Current evidence quality" tone="info">
        The strongest cross-vendor conclusions are qualitative: all three NPUs run Whisper;
        all three preserve TSC fixture accuracy; FakeAudio requires a CPU fp32 front-end plus NPU
        backbone on Intel/Qualcomm; and AMD FakeAudio remains unresolved. Exact latency rankings
        need a controlled AC-powered, same-clips, same-model, same-runtime rerun.
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
        <Text tone="secondary">Whisper, TSC, FakeAudio, and AMD VAIML · measurements through 11 Jul 2026</Text>
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
