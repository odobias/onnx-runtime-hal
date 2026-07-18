import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

type RowCmp = {
  runtime: string;
  workload: string;
  device: string;
  ac_ms: number;
  battery_ms: number;
  delta_ms: number;
  ratio: number;
};

const ROWS: RowCmp[] = [
  { runtime: "bundled", workload: "fakeaudio", device: "CPU", ac_ms: 79.6, battery_ms: 70.8, delta_ms: -8.8, ratio: 0.89 },
  { runtime: "bundled", workload: "fakeaudio", device: "GPU", ac_ms: 38, battery_ms: 73.5, delta_ms: 35.5, ratio: 1.93 },
  { runtime: "bundled", workload: "fakeaudio", device: "NPU", ac_ms: 41.7, battery_ms: 62.2, delta_ms: 20.5, ratio: 1.49 },
  { runtime: "bundled", workload: "tsc", device: "CPU", ac_ms: 159, battery_ms: 112.4, delta_ms: -46.6, ratio: 0.71 },
  { runtime: "bundled", workload: "tsc", device: "GPU", ac_ms: 63.6, battery_ms: 60.1, delta_ms: -3.5, ratio: 0.94 },
  { runtime: "bundled", workload: "tsc", device: "NPU", ac_ms: 33.3, battery_ms: 33.8, delta_ms: 0.5, ratio: 1.02 },
  { runtime: "bundled", workload: "whisper-dynamic", device: "CPU", ac_ms: 264.3, battery_ms: 248.7, delta_ms: -15.6, ratio: 0.94 },
  { runtime: "bundled", workload: "whisper-dynamic", device: "GPU", ac_ms: 379.2, battery_ms: 765.1, delta_ms: 385.9, ratio: 2.02 },
  { runtime: "bundled", workload: "whisper-static", device: "CPU", ac_ms: 1033.7, battery_ms: 862.1, delta_ms: -171.6, ratio: 0.83 },
  { runtime: "bundled", workload: "whisper-static", device: "GPU", ac_ms: 570.1, battery_ms: 818.3, delta_ms: 248.2, ratio: 1.44 },
  { runtime: "bundled", workload: "whisper-static", device: "NPU", ac_ms: 642.4, battery_ms: 753.3, delta_ms: 110.9, ratio: 1.17 },
  { runtime: "winml", workload: "fakeaudio", device: "CPU", ac_ms: 91.7, battery_ms: 101.3, delta_ms: 9.6, ratio: 1.1 },
  { runtime: "winml", workload: "fakeaudio", device: "GPU", ac_ms: 85.7, battery_ms: 87.2, delta_ms: 1.5, ratio: 1.02 },
  { runtime: "winml", workload: "fakeaudio", device: "NPU", ac_ms: 48.7, battery_ms: 82.6, delta_ms: 33.9, ratio: 1.7 },
  { runtime: "winml", workload: "tsc", device: "CPU", ac_ms: 151.9, battery_ms: 149.3, delta_ms: -2.6, ratio: 0.98 },
  { runtime: "winml", workload: "tsc", device: "GPU", ac_ms: 74.9, battery_ms: 66.1, delta_ms: -8.8, ratio: 0.88 },
  { runtime: "winml", workload: "tsc", device: "NPU", ac_ms: 33.6, battery_ms: 44.6, delta_ms: 11, ratio: 1.33 },
  { runtime: "winml", workload: "whisper-dynamic", device: "CPU", ac_ms: 220.8, battery_ms: 254.1, delta_ms: 33.3, ratio: 1.15 },
  { runtime: "winml", workload: "whisper-dynamic", device: "GPU", ac_ms: 391.9, battery_ms: 688.4, delta_ms: 296.5, ratio: 1.76 },
  { runtime: "winml", workload: "whisper-static", device: "CPU", ac_ms: 1327.5, battery_ms: 1187.5, delta_ms: -140, ratio: 0.89 },
  { runtime: "winml", workload: "whisper-static", device: "GPU", ac_ms: 577.6, battery_ms: 772.8, delta_ms: 195.2, ratio: 1.34 },
  { runtime: "winml", workload: "whisper-static", device: "NPU", ac_ms: 609, battery_ms: 1088.7, delta_ms: 479.7, ratio: 1.79 },
];

function median(vals: number[]) {
  const s = [...vals].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function mean(vals: number[]) {
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

function pick(runtime: string, workload: string, device: string) {
  return ROWS.find(
    (r) => r.runtime === runtime && r.workload === workload && r.device === device
  )!;
}

function seriesPair(keys: Array<{ runtime: string; workload: string; device: string; label: string }>) {
  return {
    categories: keys.map((k) => k.label),
    series: [
      {
        name: "AC",
        data: keys.map((k) => pick(k.runtime, k.workload, k.device).ac_ms),
        tone: "info" as const,
      },
      {
        name: "Battery",
        data: keys.map((k) => pick(k.runtime, k.workload, k.device).battery_ms),
        tone: "danger" as const,
      },
    ],
  };
}

const whisperStatic = seriesPair([
  { runtime: "bundled", workload: "whisper-static", device: "NPU", label: "bundled NPU" },
  { runtime: "bundled", workload: "whisper-static", device: "GPU", label: "bundled GPU" },
  { runtime: "bundled", workload: "whisper-static", device: "CPU", label: "bundled CPU" },
  { runtime: "winml", workload: "whisper-static", device: "NPU", label: "winml NPU" },
  { runtime: "winml", workload: "whisper-static", device: "GPU", label: "winml GPU" },
  { runtime: "winml", workload: "whisper-static", device: "CPU", label: "winml CPU" },
]);

const whisperDynamic = seriesPair([
  { runtime: "bundled", workload: "whisper-dynamic", device: "GPU", label: "bundled GPU" },
  { runtime: "bundled", workload: "whisper-dynamic", device: "CPU", label: "bundled CPU" },
  { runtime: "winml", workload: "whisper-dynamic", device: "GPU", label: "winml GPU" },
  { runtime: "winml", workload: "whisper-dynamic", device: "CPU", label: "winml CPU" },
]);

const tsc = seriesPair([
  { runtime: "bundled", workload: "tsc", device: "NPU", label: "bundled NPU" },
  { runtime: "bundled", workload: "tsc", device: "GPU", label: "bundled GPU" },
  { runtime: "bundled", workload: "tsc", device: "CPU", label: "bundled CPU" },
  { runtime: "winml", workload: "tsc", device: "NPU", label: "winml NPU" },
  { runtime: "winml", workload: "tsc", device: "GPU", label: "winml GPU" },
  { runtime: "winml", workload: "tsc", device: "CPU", label: "winml CPU" },
]);

const fakeaudio = seriesPair([
  { runtime: "bundled", workload: "fakeaudio", device: "NPU", label: "bundled NPU" },
  { runtime: "bundled", workload: "fakeaudio", device: "GPU", label: "bundled GPU" },
  { runtime: "bundled", workload: "fakeaudio", device: "CPU", label: "bundled CPU" },
  { runtime: "winml", workload: "fakeaudio", device: "NPU", label: "winml NPU" },
  { runtime: "winml", workload: "fakeaudio", device: "GPU", label: "winml GPU" },
  { runtime: "winml", workload: "fakeaudio", device: "CPU", label: "winml CPU" },
]);

const ratioByDevice = {
  categories: ["NPU", "GPU", "CPU"],
  series: [
    {
      name: "Median battery÷AC",
      data: ["NPU", "GPU", "CPU"].map((d) =>
        Number(
          median(ROWS.filter((r) => r.device === d).map((r) => r.ratio)).toFixed(2)
        )
      ),
      tone: "warning" as const,
    },
  ],
};

function shortWorkload(w: string) {
  switch (w) {
    case "whisper-static":
      return "whisper static";
    case "whisper-dynamic":
      return "whisper dyn";
    default:
      return w;
  }
}

/** Short axis label: device is in the chart title, so omit it here. */
function axisLabel(r: RowCmp) {
  return `${r.runtime} · ${shortWorkload(r.workload)}`;
}

function deviceRatioChart(device: string) {
  const rows = ROWS.filter((r) => r.device === device).sort(
    (a, b) => b.ratio - a.ratio
  );
  return {
    rows,
    categories: rows.map(axisLabel),
    series: [
      {
        name: "Battery ÷ AC",
        data: rows.map((r) => r.ratio),
        tone: "warning" as const,
      },
    ],
  };
}

function deviceDeltaChart(device: string) {
  const rows = ROWS.filter((r) => r.device === device).sort(
    (a, b) => b.delta_ms - a.delta_ms
  );
  return {
    categories: rows.map(axisLabel),
    series: [
      {
        name: "Battery − AC (ms)",
        data: rows.map((r) => r.delta_ms),
        tone: "danger" as const,
      },
    ],
  };
}

const npuRatio = deviceRatioChart("NPU");
const gpuRatio = deviceRatioChart("GPU");
const cpuRatio = deviceRatioChart("CPU");
const npuDelta = deviceDeltaChart("NPU");
const gpuDelta = deviceDeltaChart("GPU");
const cpuDelta = deviceDeltaChart("CPU");

const gpu = ROWS.filter((r) => r.device === "GPU");
const npu = ROWS.filter((r) => r.device === "NPU");
const cpu = ROWS.filter((r) => r.device === "CPU");
const sortedBySlowdown = [...ROWS].sort((a, b) => b.ratio - a.ratio);
const worst = sortedBySlowdown[0];

/** Table grouped NPU → GPU → CPU, then by ratio within each device. */
const tableOrder = (["NPU", "GPU", "CPU"] as const).flatMap((device) =>
  ROWS.filter((r) => r.device === device).sort((a, b) => b.ratio - a.ratio)
);
const tableRows = tableOrder.map((r) => [
  r.device,
  r.runtime,
  shortWorkload(r.workload),
  r.ac_ms.toFixed(1),
  r.battery_ms.toFixed(1),
  (r.delta_ms >= 0 ? "+" : "") + r.delta_ms.toFixed(1),
  `${r.ratio.toFixed(2)}×`,
]);
const tableTones = tableOrder.map((r) =>
  r.ratio >= 1.2 ? ("danger" as const) : r.ratio <= 0.9 ? ("success" as const) : undefined
);

export default function AmdAcVsBatteryLatency() {
  return (
    <Stack gap={28}>
      <Stack gap={8}>
        <H1>AMD AC vs battery latency</H1>
        <Text tone="secondary">
          Ryzen AI 1.7.1 GA · C++ suite · mode latency (10 Whisper / 20 classifier)
          · bundled + Windows ML. Bars compare mean inference latency.
        </Text>
        <Text tone="tertiary">
          AC: 3b154ecc / 3a28d412 · Battery: e5a372a8 / a8c9cec6 (100%→94%) ·
          results/ledgers
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat
          label="Median GPU slowdown"
          value={`${median(gpu.map((r) => r.ratio)).toFixed(2)}×`}
          tone="danger"
        />
        <Stat
          label="Median NPU slowdown"
          value={`${median(npu.map((r) => r.ratio)).toFixed(2)}×`}
          tone="warning"
        />
        <Stat
          label="Median CPU ratio"
          value={`${median(cpu.map((r) => r.ratio)).toFixed(2)}×`}
          tone="success"
        />
        <Stat
          label={`${worst.runtime}/${worst.workload}/${worst.device}`}
          value={`${worst.ratio.toFixed(2)}×`}
          tone="danger"
        />
      </Grid>

      <Callout tone="warning" title="Accelerators pay the battery tax">
        Unplugging costs GPU and NPU latency; CPU is mixed and sometimes
        slightly faster (session noise / power-plan quirks — not a reason to
        celebrate battery). Parity line on ratio charts is 1.0×.
      </Callout>

      <Card>
        <CardHeader trailing={<Pill size="sm">median by device</Pill>}>
          Battery ÷ AC slowdown by device
        </CardHeader>
        <CardBody>
          <BarChart
            categories={ratioByDevice.categories}
            series={ratioByDevice.series}
            height={220}
            valueSuffix="×"
            showValues
            beginAtZero
            referenceLines={[{ value: 1, label: "parity", tone: "neutral" }]}
          />
          <Text tone="tertiary">
            X-axis: device · Y-axis: median battery÷AC mean latency across
            workloads/runtimes
          </Text>
        </CardBody>
      </Card>

      <Stack gap={12}>
        <H2>Absolute latency: AC vs battery</H2>
        <Text tone="secondary">
          Grouped bars — same workload/runtime/device on both power sources.
        </Text>
      </Stack>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Whisper static — mean latency (ms)</CardHeader>
          <CardBody>
            <BarChart
              categories={whisperStatic.categories}
              series={whisperStatic.series}
              height={280}
              valueSuffix=" ms"
              beginAtZero
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Whisper dynamic — mean latency (ms)</CardHeader>
          <CardBody>
            <BarChart
              categories={whisperDynamic.categories}
              series={whisperDynamic.series}
              height={280}
              valueSuffix=" ms"
              beginAtZero
            />
            <Text tone="tertiary">NPU unsupported for dynamic-KV (omitted).</Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>TSC — mean latency (ms)</CardHeader>
          <CardBody>
            <BarChart
              categories={tsc.categories}
              series={tsc.series}
              height={280}
              valueSuffix=" ms"
              beginAtZero
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader>FakeAudio — mean latency (ms)</CardHeader>
          <CardBody>
            <BarChart
              categories={fakeaudio.categories}
              series={fakeaudio.series}
              height={280}
              valueSuffix=" ms"
              beginAtZero
            />
            <Text tone="tertiary">
              NPU uses split backbone profile; GPU/CPU use whole-graph.
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Divider />

      <Stack gap={12}>
        <H2>Battery ÷ AC by device</H2>
        <Text tone="secondary">
          One chart per device. Labels are runtime · workload (device is the
          section). Sorted slowest-on-battery first. Parity = 1.0×.
        </Text>
      </Stack>

      <Grid columns={1} gap={16}>
        <Card>
          <CardHeader
            trailing={
              <Pill size="sm" tone="warning">
                median {median(npu.map((r) => r.ratio)).toFixed(2)}×
              </Pill>
            }
          >
            NPU — battery ÷ AC
          </CardHeader>
          <CardBody>
            <BarChart
              categories={npuRatio.categories}
              series={npuRatio.series}
              horizontal
              height={260}
              valueSuffix="×"
              showValues
              beginAtZero
              referenceLines={[{ value: 1, label: "parity", tone: "neutral" }]}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            trailing={
              <Pill size="sm" tone="warning">
                median {median(gpu.map((r) => r.ratio)).toFixed(2)}×
              </Pill>
            }
          >
            GPU — battery ÷ AC
          </CardHeader>
          <CardBody>
            <BarChart
              categories={gpuRatio.categories}
              series={gpuRatio.series}
              horizontal
              height={280}
              valueSuffix="×"
              showValues
              beginAtZero
              referenceLines={[{ value: 1, label: "parity", tone: "neutral" }]}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            trailing={
              <Pill size="sm" tone="success">
                median {median(cpu.map((r) => r.ratio)).toFixed(2)}×
              </Pill>
            }
          >
            CPU — battery ÷ AC
          </CardHeader>
          <CardBody>
            <BarChart
              categories={cpuRatio.categories}
              series={cpuRatio.series}
              horizontal
              height={280}
              valueSuffix="×"
              showValues
              beginAtZero
              referenceLines={[{ value: 1, label: "parity", tone: "neutral" }]}
            />
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={12}>
        <H3>Extra milliseconds on battery (by device)</H3>
        <Text tone="secondary">
          Same grouping. Positive = slower on battery.
        </Text>
      </Stack>

      <Grid columns={1} gap={16}>
        <Card>
          <CardHeader>NPU — battery − AC (ms)</CardHeader>
          <CardBody>
            <BarChart
              categories={npuDelta.categories}
              series={npuDelta.series}
              horizontal
              height={260}
              valueSuffix=" ms"
              showValues
              beginAtZero={false}
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader>GPU — battery − AC (ms)</CardHeader>
          <CardBody>
            <BarChart
              categories={gpuDelta.categories}
              series={gpuDelta.series}
              horizontal
              height={280}
              valueSuffix=" ms"
              showValues
              beginAtZero={false}
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader>CPU — battery − AC (ms)</CardHeader>
          <CardBody>
            <BarChart
              categories={cpuDelta.categories}
              series={cpuDelta.series}
              horizontal
              height={280}
              valueSuffix=" ms"
              showValues
              beginAtZero={false}
            />
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={8}>
        <H2>Comparison table (grouped by device)</H2>
        <Text tone="secondary">
          Mean inference latency (ms). Red ≥1.20× slower on battery; green ≤0.90×.
        </Text>
        <Table
          headers={[
            "Device",
            "Runtime",
            "Workload",
            "AC ms",
            "Battery ms",
            "Δ ms",
            "Ratio",
          ]}
          columnAlign={[
            "left",
            "left",
            "left",
            "right",
            "right",
            "right",
            "right",
          ]}
          rows={tableRows}
          rowTone={tableTones}
          stickyHeader
        />
      </Stack>
    </Stack>
  );
}
