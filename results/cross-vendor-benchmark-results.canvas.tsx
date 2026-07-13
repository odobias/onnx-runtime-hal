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
  [
    "Static no-KV",
    "1.25.1",
    "9da9f440…",
    "840.72 ms",
    "0.1436",
    "2.04 / 0.60 s",
    "5.88 / 4.49%",
    "2 GPU / 0 CPU",
    "0%",
  ],
  [
    "Dynamic KV",
    "1.25.1",
    "12534e24…",
    "780.18 ms",
    "0.1332",
    "1.44 / 1.41 s",
    "5.88 / 4.49%",
    "592 GPU / 180 CPU",
    "23.32%",
  ],
];

const classifierRows = [
  [
    "TSC",
    "1.25.1",
    "90f2ed1b…",
    "61.67 ms",
    "0.90 / 0.55 s",
    "93.33%",
    "0.000306",
    "1 GPU / 0 CPU",
    "0%",
    "—",
  ],
  [
    "FakeAudio",
    "1.24.4",
    "549143bc…",
    "124.77 ms",
    "1.42 / 1.99 s",
    "60.00%",
    "0.00000016",
    "4 GPU / 1 CPU",
    "20%",
    "Resize ×1",
  ],
];

function Caption({ children }: { children: string }) {
  return (
    <Text size="small" tone="tertiary" style={{ marginTop: 6 }}>
      {children}
    </Text>
  );
}

function Overview() {
  return (
    <Stack gap={20}>
      <Grid columns={4} gap={14}>
        <Stat value="4" label="Schema-complete rows retained" />
        <Stat value="4 / 4" label="Resolved to requested provider" tone="success" />
        <Stat value="2" label="Zero CPU-offload workloads" tone="success" />
        <Stat value="2" label="Partial CPU-offload workloads" tone="warning" />
      </Grid>

      <Callout title="Current evidence set" tone="info">
        The curated view now contains only recent DirectML runs with model identity,
        precision, runtime version, provider resolution, fallback status, and profiler
        placement. Historical cross-vendor rows were removed because they do not meet
        that contract.
      </Callout>

      <Grid columns="1.25fr 1fr" gap={18}>
        <Stack gap={8}>
          <H2>CPU-offload nodes by workload</H2>
          <Text size="small" tone="secondary">
            X-axis: workload · Y-axis: profiler node share assigned to CPU (%)
          </Text>
          <BarChart
            height={290}
            categories={["Whisper static", "Whisper dynamic", "TSC", "FakeAudio"]}
            series={[
              {
                name: "CPU-offload node share",
                data: [0, 23.32, 0, 20],
                tone: "warning",
              },
            ]}
            valueSuffix="%"
            showValues
          />
          <Caption>
            Source: results/ledgers/asr.csv and classifiers.csv · 13 Jul 2026 ·
            node counts, not compute share.
          </Caption>
        </Stack>

        <Card>
          <CardHeader trailing={<Pill size="sm" active>DirectML</Pill>}>
            Placement summary
          </CardHeader>
          <CardBody>
            <Stack gap={12}>
              <Text>
                <Text weight="semibold">Whisper static:</Text> fully assigned to
                DirectML after graph fusion.
              </Text>
              <Text>
                <Text weight="semibold">Whisper dynamic:</Text> 180 shape/control
                nodes remain on CPU; 592 nodes run through DirectML.
              </Text>
              <Text>
                <Text weight="semibold">TSC:</Text> one fused DirectML partition,
                no CPU nodes.
              </Text>
              <Text>
                <Text weight="semibold">FakeAudio:</Text> four DirectML partitions;
                one Resize remains on CPU.
              </Text>
              <Text>
                All rows report <Text weight="semibold">fallback_occurred=false</Text>.
              </Text>
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
        <H2>Whisper tiny.en on Radeon 890M</H2>
        <Text size="small" tone="secondary">
          X-axis: model package · Y-axis: mean end-to-end inference latency (ms)
        </Text>
        <BarChart
          height={240}
          categories={["Static no-KV", "Dynamic KV"]}
          series={[
            {
              name: "Mean inference latency",
              data: [840.72, 780.18],
              tone: "info",
            },
          ]}
          valueSuffix=" ms"
          showValues
        />
        <Caption>
          Source: results/ledgers/asr.csv · five measured runs after one warmup ·
          5.855 s LS_000 clip · battery.
        </Caption>
      </Stack>

      <Table
        headers={[
          "Decode",
          "ORT",
          "Model SHA-256",
          "Mean",
          "RTF",
          "Cold / hot",
          "WER / CER",
          "Placement",
          "CPU offload",
        ]}
        rows={whisperRows}
        columnAlign={[
          "left",
          "left",
          "left",
          "right",
          "right",
          "right",
          "right",
          "left",
          "right",
        ]}
        rowTone={["success", "warning"]}
        striped
      />

      <Callout title="Dynamic KV offload" tone="warning">
        Dynamic Whisper is faster in this run, but it is not GPU-only. CPU nodes
        are Concat, Gather, Unsqueeze, Equal, Reshape, Where, and Add operations.
      </Callout>
    </Stack>
  );
}

function Classifiers() {
  return (
    <Stack gap={18}>
      <Stack gap={8}>
        <H2>Classifier latency on Radeon 890M</H2>
        <Text size="small" tone="secondary">
          X-axis: classifier · Y-axis: mean inference latency per fixture (ms)
        </Text>
        <BarChart
          height={240}
          categories={["TSC", "FakeAudio"]}
          series={[
            {
              name: "Mean inference latency",
              data: [61.67, 124.77],
              tone: "success",
            },
          ]}
          valueSuffix=" ms"
          showValues
        />
        <Caption>
          Source: results/ledgers/classifiers.csv · 20 runs per fixture · battery.
          Runtime versions differ and are shown below.
        </Caption>
      </Stack>

      <Table
        headers={[
          "Model",
          "ORT",
          "Model SHA-256",
          "Mean",
          "Cold / hot",
          "Accuracy",
          "Max probability delta",
          "Placement",
          "CPU offload",
          "CPU operations",
        ]}
        rows={classifierRows}
        columnAlign={[
          "left",
          "left",
          "left",
          "right",
          "right",
          "right",
          "right",
          "left",
          "right",
          "left",
        ]}
        rowTone={["success", "warning"]}
        striped
      />

      <Callout title="Accuracy interpretation" tone="info">
        TSC remains 14/15 and FakeAudio remains 3/5, matching their CPU
        references. Probability drift, not label count alone, is the accelerator
        correctness check.
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
          <CardHeader>Rows removed</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                <Text weight="semibold">ASR:</Text> 49 historical rows omitted;
                two current rows retained.
              </Text>
              <Text>
                <Text weight="semibold">Classifiers:</Text> 13 historical rows
                omitted; two current rows retained.
              </Text>
              <Text>
                Older rows lack one or more reproducibility or placement fields.
              </Text>
              <Text>
                Equivalent duplicate rows are reduced to the newest complete run.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Callout title="Scope" tone="warning">
        This is no longer a cross-vendor ranking. The ledgers do not yet contain
        schema-complete current runs for Intel, Qualcomm, or AMD VitisAI, so
        showing those historical rows beside DirectML would imply comparability
        the data does not support.
      </Callout>
    </Stack>
  );
}

export default function CrossVendorBenchmarkResults() {
  const theme = useHostTheme();
  const [tab, setTab] = useCanvasState("benchmark-tab-v2", "Overview");
  const tabs = ["Overview", "Whisper", "Classifiers", "Method"];

  return (
    <Stack
      gap={18}
      style={{ padding: 24, background: theme.bg.editor, minHeight: "100%" }}
    >
      <Stack gap={6}>
        <H1>Schema-complete benchmark results</H1>
        <Text tone="secondary">
          Radeon 890M · DirectML · Whisper, TSC, and FakeAudio · 13 Jul 2026
        </Text>
      </Stack>

      <Row gap={8} wrap>
        {tabs.map((name) => (
          <Pill
            key={name}
            active={tab === name}
            onClick={() => setTab(name)}
          >
            {name}
          </Pill>
        ))}
      </Row>

      {tab === "Overview" && <Overview />}
      {tab === "Whisper" && <Whisper />}
      {tab === "Classifiers" && <Classifiers />}
      {tab === "Method" && <Method />}
    </Stack>
  );
}
