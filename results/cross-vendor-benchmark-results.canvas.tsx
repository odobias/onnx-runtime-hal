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
  ["Snapdragon ARM64", "QNN NPU", "Static", "1.27.0", "415.38 ms", "0.0709", "16.89 / 0.73 s", "5.88 / 4.49%", "2 EP / 0 CPU", "0%"],
  ["Radeon 890M x64", "DirectML GPU", "Static", "1.25.1", "840.72 ms", "0.1436", "2.04 / 0.60 s", "5.88 / 4.49%", "2 EP / 0 CPU", "0%"],
  ["Radeon 890M x64", "DirectML GPU", "Dynamic KV", "1.25.1", "780.18 ms", "0.1332", "1.44 / 1.41 s", "5.88 / 4.49%", "592 EP / 180 CPU", "23.32%"],
  ["Snapdragon ARM64", "DirectML GPU", "Static", "1.24.4", "1355.40 ms", "0.2315", "1.62 / 0.99 s", "5.88 / 4.49%", "2 EP / 0 CPU", "0%"],
  ["Snapdragon ARM64", "DirectML GPU", "Dynamic KV", "1.24.4", "1391.25 ms", "0.2376", "1.19 / 1.27 s", "5.88 / 4.49%", "592 EP / 180 CPU", "23.32%"],
];

const classifierRows = [
  ["Snapdragon ARM64", "QNN NPU", "TSC", "1.27.0", "19.86 ms", "15.56 / 1.31 s", "93.33%", "0.0473", "1 EP / 0 CPU", "0%"],
  ["Radeon 890M x64", "DirectML GPU", "TSC", "1.25.1", "61.67 ms", "0.90 / 0.55 s", "93.33%", "0.000306", "1 EP / 0 CPU", "0%"],
  ["Radeon 890M x64", "DirectML GPU", "FakeAudio", "1.25.1", "69.82 ms", "1.34 / 0.74 s", "60.00%", "0.00000594", "4 EP / 1 CPU", "20%"],
  ["Snapdragon ARM64", "DirectML GPU", "FakeAudio", "1.24.4", "76.27 ms", "1.18 / 0.67 s", "60.00%", "0.00000036", "4 EP / 1 CPU", "20%"],
  ["Snapdragon ARM64", "DirectML GPU", "TSC", "1.24.4", "97.45 ms", "1.04 / 0.95 s", "93.33%", "0.00000108", "1 EP / 0 CPU", "0%"],
  ["Snapdragon ARM64", "QNN NPU", "FakeAudio", "1.24.4", "126.74 ms", "18.24 / 0.88 s", "80.00%*", "0.9325", "1 EP / 0 CPU", "0%"],
];

function Caption({ children }: { children: string }) {
  return <Text size="small" tone="tertiary" style={{ marginTop: 6 }}>{children}</Text>;
}

function Overview() {
  return (
    <Stack gap={20}>
      <Grid columns={4} gap={14}>
        <Stat value="11" label="Current accelerator rows retained" />
        <Stat value="11 / 11" label="No provider fallback" tone="success" />
        <Stat value="7" label="Zero CPU-offload rows" tone="success" />
        <Stat value="4" label="Partial CPU-offload rows" tone="warning" />
      </Grid>

      <Callout title="Bottom line" tone="info">
        QNN NPU leads the current Whisper static and TSC measurements. DirectML
        executes all four workloads on both x64 Radeon and ARM64 Snapdragon.
        Dynamic Whisper and FakeAudio retain CPU nodes on both GPUs. QNN
        FakeAudio is fully assigned to the NPU but numerically wrong.
      </Callout>

      <Grid columns="1.3fr 1fr" gap={18}>
        <Stack gap={8}>
          <H2>CPU-offload node share</H2>
          <Text size="small" tone="secondary">
            X-axis: profiler node share assigned to CPU (%) · Y-axis: host, provider, and workload
          </Text>
          <BarChart
            horizontal
            height={390}
            categories={[
              "Snapdragon QNN · Whisper static",
              "Radeon DML · Whisper static",
              "Radeon DML · Whisper dynamic",
              "Snapdragon DML · Whisper static",
              "Snapdragon DML · Whisper dynamic",
              "Snapdragon QNN · TSC",
              "Radeon DML · TSC",
              "Radeon DML · FakeAudio",
              "Snapdragon DML · TSC",
              "Snapdragon DML · FakeAudio",
              "Snapdragon QNN · FakeAudio",
            ]}
            series={[{
              name: "CPU-offload node share",
              data: [0, 0, 23.32, 0, 23.32, 0, 0, 20, 0, 20, 0],
              tone: "warning",
            }]}
            valueSuffix="%"
            showValues
          />
          <Caption>Source: results/ledgers/asr.csv and classifiers.csv · 13 Jul 2026 · node counts, not compute share.</Caption>
        </Stack>

        <Card>
          <CardHeader trailing={<Pill size="sm" active>Current</Pill>}>Evidence readout</CardHeader>
          <CardBody>
            <Stack gap={12}>
              <Text><Text weight="semibold">Whisper:</Text> QNN static is 415 ms; Radeon DirectML is 781–841 ms; Snapdragon DirectML is 1.36–1.39 s.</Text>
              <Text><Text weight="semibold">TSC:</Text> all retained rows preserve 14/15 accuracy. QNN is 19.86 ms.</Text>
              <Text><Text weight="semibold">FakeAudio DirectML:</Text> both hosts match the 3/5 CPU reference with negligible probability drift.</Text>
              <Text><Text weight="semibold">FakeAudio QNN:</Text> 4/5 labels is misleading; probability drift is 0.9325.</Text>
              <Text><Text weight="semibold">Placement:</Text> zero fallback does not mean zero CPU offload.</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

function Whisper() {
  return (
    <Stack gap={18}>
      <Stack gap={8}>
        <H2>Whisper tiny.en accelerator latency</H2>
        <Text size="small" tone="secondary">
          X-axis: host, provider, and decode strategy · Y-axis: mean inference latency (ms)
        </Text>
        <BarChart
          horizontal
          height={290}
          categories={[
            "Snapdragon QNN · static",
            "Radeon DML · static",
            "Radeon DML · dynamic",
            "Snapdragon DML · static",
            "Snapdragon DML · dynamic",
          ]}
          series={[{
            name: "Mean inference latency",
            data: [415.38, 840.72, 780.18, 1355.4, 1391.25],
            tone: "info",
          }]}
          valueSuffix=" ms"
          showValues
        />
        <Caption>Five measured runs after one warmup · identical package hashes · 5.855 s LS_000 clip · battery.</Caption>
      </Stack>

      <Table
        headers={["Host", "Provider", "Decode", "ORT", "Mean", "RTF", "Cold / hot", "WER / CER", "Placement", "CPU offload"]}
        rows={whisperRows}
        columnAlign={["left", "left", "left", "left", "right", "right", "right", "right", "left", "right"]}
        rowTone={["success", "neutral", "warning", "neutral", "warning"]}
        striped
        stickyHeader
      />

      <Callout title="Decode and placement" tone="warning">
        Static QNN is the fastest retained Whisper row. Dynamic KV helps on the
        Radeon run but not on Snapdragon DirectML. Both dynamic DirectML runs
        leave the same 180 shape/control nodes on CPU.
      </Callout>
    </Stack>
  );
}

function Classifiers() {
  return (
    <Stack gap={18}>
      <Stack gap={8}>
        <H2>Classifier accelerator latency</H2>
        <Text size="small" tone="secondary">
          X-axis: host, provider, and classifier · Y-axis: mean inference latency per fixture (ms)
        </Text>
        <BarChart
          horizontal
          height={310}
          categories={[
            "Snapdragon QNN · TSC",
            "Radeon DML · TSC",
            "Radeon DML · FakeAudio",
            "Snapdragon DML · FakeAudio",
            "Snapdragon DML · TSC",
            "Snapdragon QNN · FakeAudio",
          ]}
          series={[{
            name: "Mean inference latency",
            data: [19.86, 61.67, 69.82, 76.27, 97.45, 126.74],
            tone: "success",
          }]}
          valueSuffix=" ms"
          showValues
        />
        <Caption>Source: results/ledgers/classifiers.csv · 20 runs per fixture · battery.</Caption>
      </Stack>

      <Table
        headers={["Host", "Provider", "Model", "ORT", "Mean", "Cold / hot", "Accuracy", "Max probability delta", "Placement", "CPU offload"]}
        rows={classifierRows}
        columnAlign={["left", "left", "left", "left", "right", "right", "right", "right", "left", "right"]}
        rowTone={["success", "success", "success", "success", "success", "danger"]}
        striped
        stickyHeader
      />

      <Callout title="The asterisk matters" tone="danger">
        QNN FakeAudio reports 4/5 labels, but the trusted FP32 reference is 3/5
        and the maximum probability delta is 0.9325. That row is retained as a
        complete accelerator failure, not as an accuracy improvement.
      </Callout>
    </Stack>
  );
}

function Method() {
  return (
    <Stack gap={18}>
      <H2>Curation contract</H2>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Required fields</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>Model SHA-256 and inference precision</Text>
              <Text>Runtime and runtime version</Text>
              <Text>Requested and resolved provider</Text>
              <Text>Fallback status and provider attempts</Text>
              <Text>Accelerator/CPU node counts and offload percentage</Text>
              <Text>Cold/hot load, latency, and quality metrics</Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Rows omitted</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text><Text weight="semibold">ASR:</Text> 58 ledger rows; 5 retained.</Text>
              <Text><Text weight="semibold">Classifiers:</Text> 24 ledger rows; 6 retained.</Text>
              <Text>Incomplete historical rows are excluded.</Text>
              <Text>Qualcomm CPU baselines and pre-fix GPU-to-CPU fallback probes are superseded by successful DirectML runs.</Text>
              <Text>One x64 DirectML row without reliable host provenance is excluded.</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Callout title="Comparison limits" tone="warning">
        These are same-model, same-clip or same-fixture comparisons, but not a
        controlled hardware shootout. Runtime versions and host power behavior
        differ. Node percentages describe profiler placement, not FLOP share.
      </Callout>
    </Stack>
  );
}

export default function CrossVendorBenchmarkResults() {
  const theme = useHostTheme();
  const [tab, setTab] = useCanvasState("benchmark-tab-v3", "Overview");
  const tabs = ["Overview", "Whisper", "Classifiers", "Method"];

  return (
    <Stack gap={18} style={{ padding: 24, background: theme.bg.editor, minHeight: "100%" }}>
      <Stack gap={6}>
        <H1>Current schema-complete accelerator results</H1>
        <Text tone="secondary">Radeon 890M x64 and Snapdragon ARM64 · DirectML and QNN · 13 Jul 2026</Text>
      </Stack>

      <Row gap={8} wrap>
        {tabs.map((name) => (
          <Pill key={name} active={tab === name} onClick={() => setTab(name)}>{name}</Pill>
        ))}
      </Row>

      {tab === "Overview" && <Overview />}
      {tab === "Whisper" && <Whisper />}
      {tab === "Classifiers" && <Classifiers />}
      {tab === "Method" && <Method />}
    </Stack>
  );
}
