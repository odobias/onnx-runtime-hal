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

const npuRows = [
  ["Qualcomm", "QNN", "bundled", "Whisper static", "538.3 ms", "WER 8.58%", "0%", "valid"],
  ["Intel", "OpenVINO", "bundled OVEP", "Whisper static", "1896.2 ms", "WER 9.57%", "50%", "hybrid"],
  ["AMD", "VitisAI", "bundled", "Whisper static", "978.0 ms", "WER 10.23%", "84.21%", "hybrid"],
  ["Qualcomm", "QNN", "bundled", "TSC", "18.52 ms", "93.33% · Δp 0.04731", "0%", "valid"],
  ["Intel", "OpenVINO", "bundled OVEP", "TSC", "9.51 ms", "93.33% · Δp 0.00666", "0%", "valid"],
  ["AMD", "VitisAI", "bundled", "TSC", "34.38 ms", "93.33% · Δp 0.05680", "92.86%", "hybrid"],
  ["Qualcomm", "QNN", "bundled", "FakeAudio split-generic", "110.65 ms", "60% · Δp 0.000839", "0%", "valid"],
  ["Intel", "OpenVINO", "bundled OVEP", "FakeAudio split-intel", "26.35 ms", "60% · Δp 0.000068", "0%", "valid"],
  ["AMD", "VitisAI", "bundled", "FakeAudio split-generic", "64.31 ms", "60% · Δp 0.011589", "85.71%", "hybrid"],
];

const coverageRows = [
  ["Qualcomm", "11 / 11 valid", "11 / 11 valid", "22 / 22", "Complete"],
  ["Intel", "11 / 11 valid · 3 historical invalid", "11 / 11 valid", "22 / 22", "OVEP GPU FakeAudio fails; DML covers the cell"],
  ["AMD", "3 valid · 8 absent", "11 / 11 valid", "14 / 22", "Bundled GPU and CPU cells absent"],
];

function Caption({ children }: { children: string }) {
  return <Text size="small" tone="tertiary" style={{ marginTop: 6 }}>{children}</Text>;
}

function Overview() {
  return (
    <Stack gap={20}>
      <Grid columns={4} gap={14}>
        <Stat value="3" label="NPU vendors measured" tone="success" />
        <Stat value="66 / 69" label="Canonical rows valid" tone="success" />
        <Stat value="9 / 9" label="Selected NPU workload rows valid" tone="success" />
        <Stat value="8" label="AMD matrix cells absent" tone="warning" />
      </Grid>

      <Callout title="Two complete matrices, one partial matrix" tone="warning">
        Qualcomm and Intel now have valid results in all 22 eligible cells.
        Intel's forced bundled OpenVINO sweep resolves the NPU correctly; three
        older auto-provider NPU rows remain invalid DirectML fallbacks. AMD now
        has a complete Windows ML matrix plus three bundled NPU rows, all with
        heavy CPU-node offload.
      </Callout>

      <Grid columns="1.25fr 1fr" gap={18}>
        <Stack gap={8}>
          <H2>Valid portable-matrix coverage</H2>
          <Text size="small" tone="secondary">
            X-axis: vendor · Y-axis: valid eligible cells (% of 22)
          </Text>
          <BarChart
            height={270}
            categories={["Qualcomm", "Intel", "AMD"]}
            series={[{
              name: "Valid matrix coverage",
              data: [100, 100, 63.64],
              tone: "info",
            }]}
            valueSuffix="%"
            showValues
          />
          <Caption>Source: results/ledgers/accuracy.jsonl · 69 canonical rows · accuracy-quick.</Caption>
        </Stack>

        <Card>
          <CardHeader trailing={<Pill size="sm" active>Current</Pill>}>Evidence readout</CardHeader>
          <CardBody>
            <Stack gap={12}>
              <Text><Text weight="semibold">Qualcomm:</Text> QNN is the only complete 22-cell vendor campaign.</Text>
              <Text><Text weight="semibold">Intel:</Text> bundled OpenVINO NPU is valid; Whisper retains 50% CPU nodes while both classifiers retain none.</Text>
              <Text><Text weight="semibold">AMD:</Text> VitisAI resolves, but 84–93% of profiler nodes remain on CPU.</Text>
              <Text><Text weight="semibold">Dynamic Whisper:</Text> NPU is intentionally unsupported; it is a GPU/CPU workload.</Text>
              <Text><Text weight="semibold">Power:</Text> all current NPU evidence is on battery, but battery state and thermal conditions differ.</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

function NpuEvidence() {
  return (
    <Stack gap={18}>
      <Grid columns={2} gap={18}>
        <Stack gap={8}>
          <H2>Whisper static accuracy</H2>
          <Text size="small" tone="secondary">
            X-axis: vendor NPU path · Y-axis: 12-clip word error rate (%)
          </Text>
          <BarChart
            height={260}
            categories={["Qualcomm QNN", "Intel OpenVINO", "AMD VitisAI"]}
            series={[{
              name: "Word error rate",
              data: [8.58, 9.57, 10.23],
              tone: "warning",
            }]}
            valueSuffix="%"
            showValues
          />
          <Caption>Same whisper-eval-v1 contract and static model package · lower is better.</Caption>
        </Stack>

        <Stack gap={8}>
          <H2>CPU-offload node share</H2>
          <Text size="small" tone="secondary">
            X-axis: vendor and workload · Y-axis: profiler nodes on CPU (%)
          </Text>
          <BarChart
            horizontal
            height={310}
            categories={[
              "Qualcomm · Whisper",
              "Intel · Whisper",
              "AMD · Whisper",
              "Qualcomm · TSC",
              "Intel · TSC",
              "AMD · TSC",
              "Qualcomm · FakeAudio",
              "Intel · FakeAudio",
              "AMD · FakeAudio",
            ]}
            series={[{
              name: "CPU-offload node share",
              data: [0, 50, 84.21, 0, 0, 92.86, 0, 0, 85.71],
              tone: "warning",
            }]}
            valueSuffix="%"
            showValues
          />
          <Caption>Profiler node count, not compute or wall-time share.</Caption>
        </Stack>
      </Grid>

      <Table
        headers={["Vendor", "Provider", "Runtime", "Workload / profile", "Observed mean", "Quality", "CPU offload", "Status"]}
        rows={npuRows}
        columnAlign={["left", "left", "left", "left", "right", "right", "right", "left"]}
        rowTone={["success", "warning", "warning", "success", "success", "warning", "success", "success", "warning"]}
        striped
        stickyHeader
      />

      <Callout title="Do not rank FakeAudio split paths as one model" tone="warning">
        Intel uses the expanded-attention split-intel backbone. Qualcomm and AMD
        use split-generic, and AMD generated a different fixture hash. Their
        accuracy contract is shared, but artifact identity and probability drift
        are not identical.
      </Callout>
    </Stack>
  );
}

function Coverage() {
  return (
    <Stack gap={18}>
      <Stack gap={8}>
        <H2>Portable-suite gaps and retained invalid evidence</H2>
        <Text size="small" tone="secondary">
          X-axis: vendor · Y-axis: eligible cells (count)
        </Text>
        <BarChart
          height={280}
          categories={["Qualcomm", "Intel", "AMD"]}
          series={[
            { name: "Absent", data: [0, 0, 8], tone: "warning" },
            { name: "Attempted but invalid", data: [0, 3, 0], tone: "danger" },
          ]}
          showValues
        />
        <Caption>Each vendor has 22 eligible cells. Intel's three invalid rows are historical auto-provider fallbacks, not uncovered cells.</Caption>
      </Stack>

      <Table
        headers={["Vendor", "Bundled runtime", "Windows ML", "Valid total", "Primary gap"]}
        rows={coverageRows}
        columnAlign={["left", "left", "left", "right", "left"]}
        rowTone={["success", "warning", "danger"]}
        striped
      />

      <Grid columns={2} gap={16}>
        <Callout title="Highest-priority gap" tone="danger">
          AMD now has all Windows ML cells, but bundled GPU and CPU remain absent.
          Its bundled evidence is still limited to three heavily hybrid NPU rows.
        </Callout>
        <Callout title="Intel executor gap" tone="warning">
          Whole-graph FakeAudio fails on the forced OpenVINO GPU path with
          CL_INVALID_WORK_GROUP_SIZE. DirectML still provides a valid Intel GPU
          result for that portable cell.
        </Callout>
      </Grid>

      <Callout title="Latency-mode parity is still missing" tone="info">
        There is no current same-campaign, three-vendor latency cohort. TSC has
        indicative rows for all three vendors, but AMD is heavily hybrid and the
        measurements use different runtime versions and host campaigns. Intel
        split FakeAudio still lacks a dedicated latency-mode row, and AMD lacks a
        current comparable static-Whisper latency row.
      </Callout>
    </Stack>
  );
}

function Timing() {
  return (
    <Stack gap={18}>
      <Callout title="Observed timing, not a latency leaderboard" tone="info">
        These means come from the common accuracy-quick protocol: one timed run
        per clip or fixture. They are useful for orientation, but they do not
        replace explicit multi-run latency campaigns.
      </Callout>

      <Grid columns={3} gap={16}>
        <Stack gap={8}>
          <H2>Whisper static</H2>
          <BarChart
            height={260}
            categories={["Qualcomm", "Intel", "AMD"]}
            series={[{ name: "Mean per clip", data: [538.3, 1896.2, 978.0], tone: "info" }]}
            valueSuffix=" ms"
            showValues
          />
          <Caption>12 clips · 130.47 s total audio · same static package.</Caption>
        </Stack>
        <Stack gap={8}>
          <H2>TSC</H2>
          <BarChart
            height={260}
            categories={["Qualcomm", "Intel", "AMD"]}
            series={[{ name: "Mean per fixture", data: [18.52, 9.51, 34.38], tone: "success" }]}
            valueSuffix=" ms"
            showValues
          />
          <Caption>15 fixtures · identical model hash · AMD has 92.86% CPU-node offload.</Caption>
        </Stack>
        <Stack gap={8}>
          <H2>FakeAudio split</H2>
          <BarChart
            height={260}
            categories={["Qualcomm", "Intel", "AMD"]}
            series={[{ name: "Mean per fixture", data: [110.65, 26.35, 64.31], tone: "warning" }]}
            valueSuffix=" ms"
            showValues
          />
          <Caption>Five fixtures · vendor-specific split profiles; not artifact-identical.</Caption>
        </Stack>
      </Grid>

      <Callout title="Comparability limits" tone="warning">
        Qualcomm is ARM64; Intel and AMD are x64. Runtime versions differ
        (AMD 1.23.3, Intel OVEP 1.24.1, Qualcomm 1.24.4),
        and every host was on battery under non-identical thermal conditions.
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
          <CardHeader>Included in NPU evidence</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>Canonical `accuracy.jsonl` records only</Text>
              <Text>`trust_gate.valid=true` and no provider fallback</Text>
              <Text>Resolved vendor NPU provider recorded</Text>
              <Text>Static Whisper, TSC, and valid split FakeAudio profiles</Text>
              <Text>Model, fixture, environment, and provider-input provenance linked</Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Excluded from ranking</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>Intel bundled NPU requests that resolved to DirectML</Text>
              <Text>Whole-graph FakeAudio NPU diagnostics with decision flips</Text>
              <Text>Legacy and pre-schema rows lacking current provenance</Text>
              <Text>OpenVINO GenAI bounded-KV rows with a different decode contract</Text>
              <Text>Dynamic Whisper NPU, which the manifest does not support</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Callout title="Source precedence" tone="info">
        Accuracy and placement come from `results/ledgers/accuracy.jsonl`.
        Explicit performance claims belong to `asr.csv` and `classifiers.csv`
        rows with `measurement_purpose=latency`. Diagnostics remain in
        `accuracy-diagnostics.jsonl` and never enter valid summaries.
      </Callout>
    </Stack>
  );
}

export default function CrossVendorBenchmarkResults() {
  const theme = useHostTheme();
  const [tab, setTab] = useCanvasState("benchmark-tab-v4", "Overview");
  const tabs = ["Overview", "NPU evidence", "Coverage", "Observed timing", "Method"];

  return (
    <Stack gap={18} style={{ padding: 24, background: theme.bg.editor, minHeight: "100%" }}>
      <Stack gap={6}>
        <H1>Three-vendor accelerator comparison</H1>
        <Text tone="secondary">
          Intel Lunar Lake · AMD XDNA2 · Qualcomm Hexagon · canonical evidence through 15 Jul 2026
        </Text>
      </Stack>

      <Row gap={8} wrap>
        {tabs.map((name) => (
          <Pill active={tab === name} onClick={() => setTab(name)}>{name}</Pill>
        ))}
      </Row>

      {tab === "Overview" && <Overview />}
      {tab === "NPU evidence" && <NpuEvidence />}
      {tab === "Coverage" && <Coverage />}
      {tab === "Observed timing" && <Timing />}
      {tab === "Method" && <Method />}
    </Stack>
  );
}
