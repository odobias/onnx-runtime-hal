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

const coverageRows = [
  ["Qualcomm · bundled", "3 / 3", "4 / 4", "4 / 4", "11 / 11", "QNN · DirectML · CPU"],
  ["Qualcomm · Windows ML", "3 / 3", "4 / 4", "4 / 4", "11 / 11", "QNN · DirectML · CPU"],
  ["Intel · bundled", "3 / 3", "4 / 4", "4 / 4", "11 / 11", "OpenVINO NPU; DirectML covers FakeAudio GPU"],
  ["Intel · Windows ML", "3 / 3", "4 / 4", "4 / 4", "11 / 11", "OpenVINO · DirectML · CPU"],
  ["AMD · bundled", "3 / 3", "4 / 4", "4 / 4", "11 / 11", "VitisAI · DirectML · CPU"],
  ["AMD · Windows ML", "3 / 3", "4 / 4", "4 / 4", "11 / 11", "VitisAI · DirectML · CPU"],
];

const bundledNpuRows = [
  ["Qualcomm", "Whisper static", "QNN", "538.30 ms", "WER 8.58%", "unknown", "missing", "battery"],
  ["Intel", "Whisper static", "OpenVINO", "745.59 ms", "WER 9.57%", "2.70%", "397 NPU / 11 CPU", "battery"],
  ["AMD", "Whisper static", "VitisAI", "944.96 ms", "WER 10.23%", "7.84%", "376 NPU / 32 CPU", "AC"],
  ["Qualcomm", "TSC", "QNN", "18.52 ms", "93.33% · Δp 0.047311", "unknown", "missing", "battery"],
  ["Intel", "TSC", "OpenVINO", "9.49 ms", "93.33% · Δp 0.006655", "0%", "245 NPU / 0 CPU", "battery"],
  ["AMD", "TSC", "VitisAI", "33.83 ms", "93.33% · Δp 0.056795", "5.31%", "232 NPU / 13 CPU", "AC"],
  ["Qualcomm", "FakeAudio split", "QNN", "110.65 ms", "60% · Δp 0.000839", "unknown", "missing", "battery"],
  ["Intel", "FakeAudio split", "OpenVINO", "27.40 ms", "60% · Δp 0.000068", "0%", "584 NPU / 0 CPU", "battery"],
  ["AMD", "FakeAudio split", "VitisAI", "48.71 ms", "60% · Δp 0.011589", "1.03%", "578 NPU / 6 CPU", "AC"],
];

const assignmentRows = [
  ["Qualcomm", "bundled", "0 / 3", "QNN profiler only", "Whisper · TSC · FakeAudio"],
  ["Qualcomm", "Windows ML", "0 / 3", "QNN profiler only", "Whisper · TSC · FakeAudio"],
  ["Intel", "bundled", "3 / 3", "ORT EP graph assignment", "none"],
  ["Intel", "Windows ML", "3 / 3", "ORT EP graph assignment", "none"],
  ["AMD", "bundled", "3 / 3", "VitisAI cache context + gops", "none"],
  ["AMD", "Windows ML", "0 / 3", "VitisAI profiler only", "Whisper · TSC · FakeAudio"],
];

const weaknessRows = [
  ["High", "NPU assignment proof", "9 / 18 canonical NPU rows", "Qualcomm has none; AMD Windows ML has none", "Rerun those nine rows under the new trust gate"],
  ["High", "Performance comparability", "Accuracy-quick uses one timed pass", "No controlled three-vendor latency cohort", "Run multi-run latency campaigns under matched power policy"],
  ["High", "FakeAudio quality", "60% on only five fixtures", "Too weak to rank hardware or model quality", "Expand and stratify the labeled evaluation set"],
  ["Medium", "Artifact equivalence", "FakeAudio has three model hashes", "Intel uses a patched backbone; GPU/CPU use whole graph", "Publish a single portable NPU backbone or keep rankings separate"],
  ["Medium", "Environment control", "63 cells on battery; 3 AMD NPU cells on AC", "Thermals and power are not normalized", "Capture temperatures and fix AC/DC plus power-plan policy"],
  ["Medium", "Runtime parity", "Bundled ORT versions vary by executor", "Intel spans 1.24.1/1.24.4; AMD 1.23.3/1.24.4", "Align runtime versions where provider packaging permits"],
  ["Medium", "Whisper NPU breadth", "Only static no-KV is in the NPU matrix", "Dynamic KV is GPU/CPU-only; bounded KV is absent", "Add a portable bounded-KV profile or state the exclusion"],
];

function Caption({ children }: { children: string }) {
  return <Text size="small" tone="tertiary" style={{ marginTop: 6 }}>{children}</Text>;
}

function Overview() {
  return (
    <Stack gap={20}>
      <Grid columns={4} gap={14}>
        <Stat value="66 / 66" label="Eligible matrix cells valid" tone="success" />
        <Stat value="83 / 86" label="Published records valid" tone="warning" />
        <Stat value="9 / 18" label="Canonical NPU rows with op split" tone="danger" />
        <Stat value="3" label="FakeAudio model hashes" tone="warning" />
      </Grid>

      <Callout title="Coverage is complete; evidence strength is not" tone="warning">
        All six vendor-runtime sweeps cover their 11 eligible cells. That proves
        executor coverage, not uniform NPU trust or performance comparability.
        The newer CPU/NPU operation-assignment requirement is satisfied by all
        Intel NPU rows and AMD bundled NPU rows, but not by Qualcomm or AMD
        Windows ML rows retained from the historical ledger.
      </Callout>

      <Grid columns="1.25fr 1fr" gap={18}>
        <Stack gap={8}>
          <H2>Canonical accuracy coverage</H2>
          <Text size="small" tone="secondary">
            X-axis: vendor · Y-axis: valid eligible cells (% of 22)
          </Text>
          <BarChart
            height={270}
            categories={["Qualcomm", "Intel", "AMD"]}
            series={[{
              name: "Valid matrix coverage",
              data: [100, 100, 100],
              tone: "success",
            }]}
            valueSuffix="%"
            showValues
          />
          <Caption>Source: frozen history plus two immutable Intel campaigns · 86 records, deduplicated to 66 latest valid cells · through 15 Jul 2026.</Caption>
        </Stack>

        <Card>
          <CardHeader trailing={<Pill size="sm" active>Current</Pill>}>What changed</CardHeader>
          <CardBody>
            <Stack gap={12}>
              <Text><Text weight="semibold">Intel timing:</Text> latest bundled Whisper mean is 745.59 ms, not the stale 1896.2 ms.</Text>
              <Text><Text weight="semibold">AMD timing:</Text> latest bundled NPU means are 944.96, 33.83, and 48.71 ms.</Text>
              <Text><Text weight="semibold">Power:</Text> AMD bundled NPU evidence was collected on AC; the other 63 canonical cells were on battery.</Text>
              <Text><Text weight="semibold">Trust:</Text> operation splits now expose why profiler-node percentages should not be read as compute share.</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

function NpuTrust() {
  return (
    <Stack gap={18}>
      <Grid columns="1.4fr 0.8fr" gap={18}>
        <Stack gap={8}>
          <H2>CPU operation offload on measured NPU paths</H2>
          <Text size="small" tone="secondary">
            X-axis: vendor, runtime, and workload · Y-axis: original graph operations assigned to CPU (%)
          </Text>
          <BarChart
            horizontal
            height={300}
            categories={[
              "Intel bundled · Whisper",
              "Intel bundled · TSC",
              "Intel bundled · FakeAudio",
              "Intel WinML · Whisper",
              "Intel WinML · TSC",
              "Intel WinML · FakeAudio",
              "AMD bundled · Whisper",
              "AMD bundled · TSC",
              "AMD bundled · FakeAudio",
            ]}
            series={[{
              name: "CPU-assigned operation share",
              data: [2.70, 0, 0, 0, 0, 0, 7.84, 5.31, 1.03],
              tone: "warning",
            }]}
            valueSuffix="%"
            showValues
          />
          <Caption>Lower is better. Intel uses ORT original graph nodes; AMD uses VitisAI context/gops rows. Qualcomm is omitted because no trustworthy operation split was recorded.</Caption>
        </Stack>

        <Stack gap={8}>
          <H2>Operation-split coverage</H2>
          <Text size="small" tone="secondary">
            X-axis: vendor · Y-axis: canonical NPU rows with CPU/NPU operation split (count of 6)
          </Text>
          <BarChart
            height={300}
            categories={["Qualcomm", "Intel", "AMD"]}
            series={[{
              name: "Operation split recorded",
              data: [0, 6, 3],
              tone: "info",
            }]}
            showValues
          />
          <Caption>Six NPU cells per vendor: two runtimes × three NPU-eligible workloads.</Caption>
        </Stack>
      </Grid>

      <Table
        headers={["Vendor", "Runtime", "Rows with op split", "Evidence source", "Missing workloads"]}
        rows={assignmentRows}
        columnAlign={["left", "left", "right", "left", "left"]}
        rowTone={["danger", "danger", "success", "success", "success", "danger"]}
        striped
      />

      <Callout title="Legacy profiler-node share is secondary" tone="info">
        The headline placement metric is now original operations assigned to
        CPU. Profiler-node share remains useful for debugging runtime
        partitions, but fusion makes it a poor proxy for offloaded work:
        AMD's 84–93% CPU-node share corresponds to only 1–8% CPU operations.
      </Callout>
    </Stack>
  );
}

function Coverage() {
  return (
    <Stack gap={18}>
      <Stack gap={8}>
        <H2>Complete portable-suite coverage</H2>
        <Text size="small" tone="secondary">
          X-axis: vendor · Y-axis: valid eligible cells by runtime target (count)
        </Text>
        <BarChart
          height={280}
          categories={["Qualcomm", "Intel", "AMD"]}
          series={[
            { name: "Bundled valid", data: [11, 11, 11], tone: "info" },
            { name: "Windows ML valid", data: [11, 11, 11], tone: "success" },
          ]}
          showValues
        />
        <Caption>Each runtime target contains three NPU, four GPU, and four CPU cells. Dynamic-KV Whisper is intentionally excluded from NPU.</Caption>
      </Stack>

      <Table
        headers={["Vendor / runtime", "NPU", "GPU", "CPU", "Valid", "Resolved paths"]}
        rows={coverageRows}
        columnAlign={["left", "right", "right", "right", "right", "left"]}
        rowTone={["success", "success", "success", "success", "success", "success"]}
        striped
      />

      <Grid columns={2} gap={16}>
        <Callout title="Three retained invalid rows" tone="warning">
          Historical Intel bundled NPU requests resolved to DirectML for
          Whisper, TSC, and FakeAudio. They remain auditable in the ledger but
          are superseded by valid forced-OpenVINO rows.
        </Callout>
        <Callout title="One valid Intel GPU cell uses DirectML" tone="info">
          Whole-graph FakeAudio fails on forced OpenVINO GPU with
          CL_INVALID_WORK_GROUP_SIZE. DirectML covers that eligible cell; the
          matrix does not claim OVEP succeeded there.
        </Callout>
      </Grid>
    </Stack>
  );
}

function Timing() {
  return (
    <Stack gap={18}>
      <Callout title="Orientation only, not a latency leaderboard" tone="warning">
        These are accuracy-quick means from one timed pass per clip or fixture.
        Runtime versions, architecture, AC/DC state, and thermal history differ.
      </Callout>

      <Grid columns={3} gap={16}>
        <Stack gap={8}>
          <H2>Whisper static NPU</H2>
          <Text size="small" tone="secondary">X-axis: vendor · Y-axis: mean per clip (ms)</Text>
          <BarChart
            height={260}
            categories={["Qualcomm", "Intel", "AMD"]}
            series={[{ name: "Accuracy-quick mean", data: [538.30, 745.59, 944.96], tone: "info" }]}
            valueSuffix=" ms"
            showValues
          />
          <Caption>12 clips · same model and fixture hashes · WER 8.58%, 9.57%, and 10.23%.</Caption>
        </Stack>
        <Stack gap={8}>
          <H2>TSC NPU</H2>
          <Text size="small" tone="secondary">X-axis: vendor · Y-axis: mean per fixture (ms)</Text>
          <BarChart
            height={260}
            categories={["Qualcomm", "Intel", "AMD"]}
            series={[{ name: "Accuracy-quick mean", data: [18.52, 9.49, 33.83], tone: "success" }]}
            valueSuffix=" ms"
            showValues
          />
          <Caption>15 fixtures · identical model and fixture hashes · 93.33% task accuracy.</Caption>
        </Stack>
        <Stack gap={8}>
          <H2>FakeAudio split NPU</H2>
          <Text size="small" tone="secondary">X-axis: vendor · Y-axis: mean per fixture (ms)</Text>
          <BarChart
            height={260}
            categories={["Qualcomm", "Intel", "AMD"]}
            series={[{ name: "Accuracy-quick mean", data: [110.65, 27.40, 48.71], tone: "warning" }]}
            valueSuffix=" ms"
            showValues
          />
          <Caption>Five fixtures · 60% accuracy · Intel uses a different patched backbone; AMD uses a different fixture hash.</Caption>
        </Stack>
      </Grid>

      <Table
        headers={["Vendor", "Workload", "Provider", "Observed mean", "Quality", "CPU operation share", "Assigned operations", "Power"]}
        rows={bundledNpuRows}
        columnAlign={["left", "left", "left", "right", "right", "right", "right", "left"]}
        rowTone={["warning", "warning", "warning", "warning", "success", "warning", "warning", "success", "warning"]}
        striped
        stickyHeader
      />
    </Stack>
  );
}

function Weaknesses() {
  return (
    <Stack gap={18}>
      <Callout title="The biggest weakness is grandfathered NPU evidence" tone="danger">
        Nine canonical NPU rows predate the operation-assignment trust gate.
        They remain valid because the schema is backward-compatible, even
        though equivalent new rows would be rejected without an operation split.
      </Callout>

      <Table
        headers={["Priority", "Weakness", "Evidence", "Why it matters", "Best next action"]}
        rows={weaknessRows}
        columnAlign={["left", "left", "left", "left", "left"]}
        rowTone={["danger", "danger", "danger", "warning", "warning", "warning", "warning"]}
        striped
        stickyHeader
      />

      <Grid columns={2} gap={16}>
        <Callout title="Coverage denominator is intentionally narrow" tone="warning">
          The 66-cell matrix excludes NPU execution for dynamic-KV Whisper.
          It also contains no portable bounded-KV profile. A complete matrix
          therefore means complete coverage of the manifest, not complete
          coverage of useful Whisper decoder architectures.
        </Callout>
        <Callout title="Accuracy quality is uneven" tone="warning">
          TSC has 15 fixtures and Whisper has 12 clips, but FakeAudio has only
          five fixtures and reaches 60% task accuracy on every NPU path.
          Agreement with CPU reference does not make that classifier good.
        </Callout>
      </Grid>
    </Stack>
  );
}

function Method() {
  return (
    <Stack gap={18}>
      <H2>Curation contract</H2>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Canonical matrix</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>Read frozen `accuracy.jsonl` plus every `accuracy-runs/*.jsonl` file</Text>
              <Text>Require `trust_gate.valid=true` and intended provider resolution</Text>
              <Text>Select the latest valid row per vendor, runtime, device, and workload cell</Text>
              <Text>Keep all 86 raw records visible for provenance and invalid-attempt auditing</Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Interpretation limits</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>Accuracy-quick timing is not explicit latency evidence</Text>
              <Text>CPU operation assignment is the primary offload metric</Text>
              <Text>Legacy profiler-node share is retained only as a partition diagnostic</Text>
              <Text>ORT and VitisAI assignment sources have different reporting semantics</Text>
              <Text>FakeAudio split profiles are not artifact-identical across all vendors</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Callout title="Source precedence" tone="info">
        Accuracy, provider placement, and operation assignment come from
        `results/ledgers/accuracy.jsonl` plus immutable files under
        `results/accuracy-runs/`. Explicit performance claims require
        `measurement_purpose=latency` rows from the ASR and classifier CSV
        ledgers and are not promoted into this leaderboard.
      </Callout>
    </Stack>
  );
}

export default function CrossVendorBenchmarkResults() {
  const theme = useHostTheme();
  const [tab, setTab] = useCanvasState("benchmark-tab-v7", "Overview");

  return (
    <Stack gap={18} style={{ padding: 24, background: theme.bg.editor, minHeight: "100%" }}>
      <Stack gap={6}>
        <H1>Three-vendor accelerator evidence</H1>
        <Text tone="secondary">
          Intel Lunar Lake · AMD XDNA2 · Qualcomm Hexagon · 86 published accuracy records through 15 Jul 2026
        </Text>
      </Stack>

      <Row gap={8} wrap>
        <Pill active={tab === "Overview"} onClick={() => setTab("Overview")}>Overview</Pill>
        <Pill active={tab === "CPU op offload"} onClick={() => setTab("CPU op offload")}>CPU op offload</Pill>
        <Pill active={tab === "Coverage"} onClick={() => setTab("Coverage")}>Coverage</Pill>
        <Pill active={tab === "Observed timing"} onClick={() => setTab("Observed timing")}>Observed timing</Pill>
        <Pill active={tab === "Weaknesses"} onClick={() => setTab("Weaknesses")}>Weaknesses</Pill>
        <Pill active={tab === "Method"} onClick={() => setTab("Method")}>Method</Pill>
      </Row>

      {tab === "Overview" && <Overview />}
      {tab === "CPU op offload" && <NpuTrust />}
      {tab === "Coverage" && <Coverage />}
      {tab === "Observed timing" && <Timing />}
      {tab === "Weaknesses" && <Weaknesses />}
      {tab === "Method" && <Method />}
    </Stack>
  );
}
