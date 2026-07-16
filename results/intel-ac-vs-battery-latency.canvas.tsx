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
  Pill,
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
  { runtime: "bundled", workload: "fakeaudio", device: "CPU", ac_ms: 165.4, battery_ms: 126.1, delta_ms: -39.3, ratio: 0.76 },
  { runtime: "bundled", workload: "fakeaudio", device: "GPU", ac_ms: 56.3, battery_ms: 76.5, delta_ms: 20.2, ratio: 1.36 },
  { runtime: "bundled", workload: "fakeaudio", device: "NPU", ac_ms: 28.1, battery_ms: 23.2, delta_ms: -4.9, ratio: 0.82 },
  { runtime: "bundled", workload: "tsc", device: "CPU", ac_ms: 387.8, battery_ms: 243.9, delta_ms: -143.9, ratio: 0.63 },
  { runtime: "bundled", workload: "tsc", device: "GPU", ac_ms: 58.5, battery_ms: 40.2, delta_ms: -18.4, ratio: 0.69 },
  { runtime: "bundled", workload: "tsc", device: "NPU", ac_ms: 29.7, battery_ms: 8.2, delta_ms: -21.5, ratio: 0.27 },
  { runtime: "bundled", workload: "whisper-dynamic", device: "CPU", ac_ms: 1878.9, battery_ms: 1671.8, delta_ms: -207.1, ratio: 0.89 },
  { runtime: "bundled", workload: "whisper-dynamic", device: "GPU", ac_ms: 598.0, battery_ms: 612.3, delta_ms: 14.3, ratio: 1.02 },
  { runtime: "bundled", workload: "whisper-static", device: "CPU", ac_ms: 3243.1, battery_ms: 2337.5, delta_ms: -905.6, ratio: 0.72 },
  { runtime: "bundled", workload: "whisper-static", device: "GPU", ac_ms: 1313.3, battery_ms: 676.2, delta_ms: -637.1, ratio: 0.51 },
  { runtime: "bundled", workload: "whisper-static", device: "NPU", ac_ms: 3063.8, battery_ms: 544.4, delta_ms: -2519.4, ratio: 0.18 },
  { runtime: "winml", workload: "fakeaudio", device: "CPU", ac_ms: 148.0, battery_ms: 115.2, delta_ms: -32.8, ratio: 0.78 },
  { runtime: "winml", workload: "fakeaudio", device: "GPU", ac_ms: 61.4, battery_ms: 72.4, delta_ms: 11.0, ratio: 1.18 },
  { runtime: "winml", workload: "fakeaudio", device: "NPU", ac_ms: 33.4, battery_ms: 23.1, delta_ms: -10.4, ratio: 0.69 },
  { runtime: "winml", workload: "tsc", device: "CPU", ac_ms: 304.5, battery_ms: 197.5, delta_ms: -107.0, ratio: 0.65 },
  { runtime: "winml", workload: "tsc", device: "GPU", ac_ms: 70.7, battery_ms: 39.0, delta_ms: -31.8, ratio: 0.55 },
  { runtime: "winml", workload: "tsc", device: "NPU", ac_ms: 19.3, battery_ms: 8.8, delta_ms: -10.5, ratio: 0.46 },
  { runtime: "winml", workload: "whisper-dynamic", device: "CPU", ac_ms: 329.6, battery_ms: 218.3, delta_ms: -111.3, ratio: 0.66 },
  { runtime: "winml", workload: "whisper-dynamic", device: "GPU", ac_ms: 537.4, battery_ms: 583.2, delta_ms: 45.8, ratio: 1.09 },
  { runtime: "winml", workload: "whisper-static", device: "CPU", ac_ms: 2151.4, battery_ms: 932.2, delta_ms: -1219.1, ratio: 0.43 },
  { runtime: "winml", workload: "whisper-static", device: "GPU", ac_ms: 848.7, battery_ms: 658.3, delta_ms: -190.3, ratio: 0.78 },
  { runtime: "winml", workload: "whisper-static", device: "NPU", ac_ms: 372.1, battery_ms: 292.5, delta_ms: -79.6, ratio: 0.79 },
];

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function pick(runtime: string, workload: string, device: string) {
  return ROWS.find(
    (row) =>
      row.runtime === runtime &&
      row.workload === workload &&
      row.device === device
  )!;
}

function seriesPair(
  keys: Array<{
    runtime: string;
    workload: string;
    device: string;
    label: string;
  }>
) {
  return {
    categories: keys.map((key) => key.label),
    series: [
      {
        name: "AC",
        data: keys.map(
          (key) => pick(key.runtime, key.workload, key.device).ac_ms
        ),
        tone: "info" as const,
      },
      {
        name: "Battery",
        data: keys.map(
          (key) => pick(key.runtime, key.workload, key.device).battery_ms
        ),
        tone: "warning" as const,
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

const DEVICES = ["NPU", "GPU", "CPU"] as const;
const byDevice = Object.fromEntries(
  DEVICES.map((device) => [
    device,
    ROWS.filter((row) => row.device === device),
  ])
) as Record<(typeof DEVICES)[number], RowCmp[]>;

const ratioByDevice = {
  categories: [...DEVICES],
  series: [
    {
      name: "Median battery ÷ AC",
      data: DEVICES.map((device) =>
        Number(median(byDevice[device].map((row) => row.ratio)).toFixed(2))
      ),
      tone: "warning" as const,
    },
  ],
};

function shortWorkload(workload: string) {
  if (workload === "whisper-static") return "whisper static";
  if (workload === "whisper-dynamic") return "whisper dynamic";
  return workload;
}

function deviceRatioChart(device: string) {
  const rows = ROWS.filter((row) => row.device === device).sort(
    (a, b) => b.ratio - a.ratio
  );
  return {
    categories: rows.map(
      (row) => `${row.runtime} · ${shortWorkload(row.workload)}`
    ),
    series: [
      {
        name: "Battery ÷ AC",
        data: rows.map((row) => row.ratio),
        tone: "warning" as const,
      },
    ],
  };
}

const tableOrder = DEVICES.flatMap((device) =>
  byDevice[device].sort((a, b) => b.ratio - a.ratio)
);
const tableRows = tableOrder.map((row) => [
  row.device,
  row.runtime,
  shortWorkload(row.workload),
  row.ac_ms.toFixed(1),
  row.battery_ms.toFixed(1),
  `${row.delta_ms >= 0 ? "+" : ""}${row.delta_ms.toFixed(1)}`,
  `${row.ratio.toFixed(2)}×`,
]);
const tableTones = tableOrder.map((row) =>
  row.ratio >= 1.2
    ? ("danger" as const)
    : row.ratio <= 0.8
      ? ("warning" as const)
      : undefined
);

const fasterOnBattery = ROWS.filter((row) => row.ratio < 1).length;

export default function IntelAcVsBatteryLatency() {
  return (
    <Stack gap={28}>
      <Stack gap={8}>
        <H1>Intel AC vs battery latency</H1>
        <Text tone="secondary">
          Native C++ suite · mode latency (10 Whisper / 20 classifier) ·
          bundled OpenVINO/DirectML/CPU + Windows ML. Bars compare mean
          inference latency.
        </Text>
        <Text tone="tertiary">
          AC campaigns: 23b25628, 1993f5ab, 6e3949bc, 941ab7e6 · Battery:
          da4ce213, a889fd8d, c05bc5bc, a9a43206 (48%→41%) · 16 July 2026
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat
          label="Median NPU battery÷AC"
          value={`${median(byDevice.NPU.map((row) => row.ratio)).toFixed(2)}×`}
          tone="warning"
        />
        <Stat
          label="Median GPU battery÷AC"
          value={`${median(byDevice.GPU.map((row) => row.ratio)).toFixed(2)}×`}
          tone="warning"
        />
        <Stat
          label="Median CPU battery÷AC"
          value={`${median(byDevice.CPU.map((row) => row.ratio)).toFixed(2)}×`}
          tone="warning"
        />
        <Stat
          label="Rows apparently faster on battery"
          value={`${fasterOnBattery} / ${ROWS.length}`}
          tone="warning"
        />
      </Grid>

      <Callout tone="warning" title="This is not evidence that battery makes Intel faster">
        Battery measured lower mean latency in 18 of 22 pairs, including a
        suspicious 0.18× result for bundled Whisper NPU. Each power condition
        is one sequential campaign, not an interleaved or counterbalanced
        experiment. Treat the gap as campaign-order, thermal, driver-cache, or
        background-load sensitivity until repeated A/B/A/B runs reproduce it.
      </Callout>

      <Card>
        <CardHeader trailing={<Pill size="sm">lower is faster</Pill>}>
          Median battery ÷ AC latency by device
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
            X-axis: device · Y-axis: median battery÷AC mean latency ratio ·
            Source: results/ledgers, Intel campaigns on 16 July 2026
          </Text>
        </CardBody>
      </Card>

      <Stack gap={8}>
        <H2>Absolute latency by workload</H2>
        <Text tone="secondary">
          Same runtime, execution profile, device, model hash, and repetition
          policy on AC and battery.
        </Text>
      </Stack>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Whisper static — mean inference latency</CardHeader>
          <CardBody>
            <BarChart
              categories={whisperStatic.categories}
              series={whisperStatic.series}
              height={280}
              valueSuffix=" ms"
              beginAtZero
            />
            <Text tone="tertiary">
              X-axis: runtime/device · Y-axis: mean latency (ms) · 10 runs
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Whisper dynamic — mean inference latency</CardHeader>
          <CardBody>
            <BarChart
              categories={whisperDynamic.categories}
              series={whisperDynamic.series}
              height={280}
              valueSuffix=" ms"
              beginAtZero
            />
            <Text tone="tertiary">
              X-axis: runtime/device · Y-axis: mean latency (ms) · 10 runs ·
              NPU unsupported and omitted
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>TSC — mean inference latency</CardHeader>
          <CardBody>
            <BarChart
              categories={tsc.categories}
              series={tsc.series}
              height={280}
              valueSuffix=" ms"
              beginAtZero
            />
            <Text tone="tertiary">
              X-axis: runtime/device · Y-axis: mean latency (ms) · 20 runs per
              fixture sample
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>FakeAudio — mean inference latency</CardHeader>
          <CardBody>
            <BarChart
              categories={fakeaudio.categories}
              series={fakeaudio.series}
              height={280}
              valueSuffix=" ms"
              beginAtZero
            />
            <Text tone="tertiary">
              X-axis: runtime/device · Y-axis: mean latency (ms) · NPU uses the
              Intel split-backbone fixture; GPU/CPU use whole graph
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Divider />

      <Stack gap={8}>
        <H2>Battery ÷ AC within each device</H2>
        <Text tone="secondary">
          Ratios below 1.0 are faster on battery; ratios above 1.0 are slower.
          The surprising sub-parity values are the reason replication matters.
        </Text>
      </Stack>

      <Grid columns={1} gap={16}>
        {DEVICES.map((device) => {
          const chart = deviceRatioChart(device);
          return (
            <Card>
              <CardHeader
                trailing={
                  <Pill size="sm" tone="warning">
                    median{" "}
                    {median(byDevice[device].map((row) => row.ratio)).toFixed(2)}
                    ×
                  </Pill>
                }
              >
                {device} — battery ÷ AC latency
              </CardHeader>
              <CardBody>
                <BarChart
                  categories={chart.categories}
                  series={chart.series}
                  horizontal
                  height={device === "NPU" ? 250 : 300}
                  valueSuffix="×"
                  showValues
                  beginAtZero
                  referenceLines={[
                    { value: 1, label: "parity", tone: "neutral" },
                  ]}
                />
                <Text tone="tertiary">
                  X-axis: battery÷AC ratio · Y-axis: runtime/workload · Source:
                  paired Intel latency rows
                </Text>
              </CardBody>
            </Card>
          );
        })}
      </Grid>

      <Stack gap={8}>
        <H2>All paired measurements</H2>
        <Text tone="secondary">
          Mean inference latency in milliseconds. Red marks ≥1.20× battery
          slowdown; amber marks ≤0.80× apparent battery speedup requiring
          replication.
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

      <Callout tone="info" title="What a trustworthy follow-up needs">
        Run at least three interleaved AC/battery cycles with fixed power-plan
        settings, randomized order, a thermal settling interval, and idle-load
        telemetry. Report per-run distributions—not only one campaign mean.
        Until then, this canvas describes observed campaigns rather than a
        causal power-source effect.
      </Callout>
    </Stack>
  );
}
