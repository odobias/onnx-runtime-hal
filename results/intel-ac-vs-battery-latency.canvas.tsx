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
  { runtime: "bundled", workload: "whisper-static", device: "NPU", ac_ms: 718.9, battery_ms: 544.4, delta_ms: -174.5, ratio: 0.76 },
  { runtime: "bundled", workload: "whisper-static", device: "GPU", ac_ms: 728.6, battery_ms: 676.2, delta_ms: -52.4, ratio: 0.93 },
  { runtime: "bundled", workload: "whisper-static", device: "CPU", ac_ms: 2514.2, battery_ms: 2337.5, delta_ms: -176.7, ratio: 0.93 },
  { runtime: "bundled", workload: "whisper-dynamic", device: "GPU", ac_ms: 357.5, battery_ms: 612.3, delta_ms: 254.8, ratio: 1.71 },
  { runtime: "bundled", workload: "whisper-dynamic", device: "CPU", ac_ms: 1566.0, battery_ms: 1671.8, delta_ms: 105.8, ratio: 1.07 },
  { runtime: "bundled", workload: "tsc", device: "NPU", ac_ms: 8.3, battery_ms: 8.2, delta_ms: -0.2, ratio: 0.98 },
  { runtime: "bundled", workload: "tsc", device: "GPU", ac_ms: 40.9, battery_ms: 40.2, delta_ms: -0.7, ratio: 0.98 },
  { runtime: "bundled", workload: "tsc", device: "CPU", ac_ms: 266.0, battery_ms: 243.9, delta_ms: -22.1, ratio: 0.92 },
  { runtime: "bundled", workload: "fakeaudio", device: "NPU", ac_ms: 23.4, battery_ms: 23.2, delta_ms: -0.3, ratio: 0.99 },
  { runtime: "bundled", workload: "fakeaudio", device: "GPU", ac_ms: 50.2, battery_ms: 76.5, delta_ms: 26.3, ratio: 1.52 },
  { runtime: "bundled", workload: "fakeaudio", device: "CPU", ac_ms: 122.7, battery_ms: 126.1, delta_ms: 3.4, ratio: 1.03 },
  { runtime: "winml", workload: "whisper-static", device: "NPU", ac_ms: 200.0, battery_ms: 292.5, delta_ms: 92.5, ratio: 1.46 },
  { runtime: "winml", workload: "whisper-static", device: "GPU", ac_ms: 562.5, battery_ms: 658.3, delta_ms: 95.9, ratio: 1.17 },
  { runtime: "winml", workload: "whisper-static", device: "CPU", ac_ms: 1497.5, battery_ms: 932.2, delta_ms: -565.3, ratio: 0.62 },
  { runtime: "winml", workload: "whisper-dynamic", device: "GPU", ac_ms: 334.2, battery_ms: 583.2, delta_ms: 249.1, ratio: 1.75 },
  { runtime: "winml", workload: "whisper-dynamic", device: "CPU", ac_ms: 202.7, battery_ms: 218.3, delta_ms: 15.6, ratio: 1.08 },
  { runtime: "winml", workload: "tsc", device: "NPU", ac_ms: 8.8, battery_ms: 8.8, delta_ms: 0.0, ratio: 1.00 },
  { runtime: "winml", workload: "tsc", device: "GPU", ac_ms: 38.6, battery_ms: 39.0, delta_ms: 0.3, ratio: 1.01 },
  { runtime: "winml", workload: "tsc", device: "CPU", ac_ms: 204.5, battery_ms: 197.5, delta_ms: -7.1, ratio: 0.97 },
  { runtime: "winml", workload: "fakeaudio", device: "NPU", ac_ms: 23.5, battery_ms: 23.1, delta_ms: -0.4, ratio: 0.98 },
  { runtime: "winml", workload: "fakeaudio", device: "GPU", ac_ms: 47.1, battery_ms: 72.4, delta_ms: 25.3, ratio: 1.54 },
  { runtime: "winml", workload: "fakeaudio", device: "CPU", ac_ms: 107.3, battery_ms: 115.2, delta_ms: 7.8, ratio: 1.07 },
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

function sharedYMax(values: number[]): number {
  const max = Math.max(0, ...values);
  return max <= 0 ? 1 : Math.ceil(max * 1.05);
}

function runtimePowerSeries(
  runtime: "bundled" | "winml",
  workload: string,
  devices: string[]
) {
  return {
    categories: devices,
    series: [
      {
        name: "AC",
        data: devices.map((device) => pick(runtime, workload, device).ac_ms),
        tone: "info" as const,
      },
      {
        name: "Battery",
        data: devices.map(
          (device) => pick(runtime, workload, device).battery_ms
        ),
        tone: "warning" as const,
      },
    ],
  };
}

function WorkloadRuntimePair({
  workload,
  devices,
  title,
  note,
}: {
  workload: string;
  devices: string[];
  title: string;
  note: string;
}) {
  const bundled = runtimePowerSeries("bundled", workload, devices);
  const winml = runtimePowerSeries("winml", workload, devices);
  const yMax = sharedYMax([
    ...bundled.series.flatMap((s) => s.data),
    ...winml.series.flatMap((s) => s.data),
  ]);
  return (
    <Stack gap={10}>
      <H2>{title}</H2>
      <Text tone="secondary">{note}</Text>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader trailing={<Pill size="sm">bundled</Pill>}>
            {title} · bundled · AC vs battery
          </CardHeader>
          <CardBody>
            <BarChart
              {...bundled}
              height={280}
              valueSuffix=" ms"
              beginAtZero
              yMax={yMax}
            />
            <Text tone="tertiary">
              X-axis: device · Y-axis: mean latency (ms) · shared scale ·
              bundled
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill size="sm">Windows ML</Pill>}>
            {title} · Windows ML · AC vs battery
          </CardHeader>
          <CardBody>
            <BarChart
              {...winml}
              height={280}
              valueSuffix=" ms"
              beginAtZero
              yMax={yMax}
            />
            <Text tone="tertiary">
              X-axis: device · Y-axis: mean latency (ms) · shared scale · WinML
            </Text>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

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
          AC (remeasured): 3b7bbe70, 85d7c162, 2ea51705, 943e0bbd · Battery:
          da4ce213, a889fd8d, c05bc5bc, a9a43206 · 16 July 2026
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
        After replacing the contaminated AC campaigns, battery is still lower in
        10 of 22 pairs (bundled Whisper NPU now 0.76×, not 0.18×). Each power
        condition remains one sequential campaign, not interleaved. Treat
        remaining gaps as campaign-order / thermal / cache sensitivity until
        A/B/A/B repeats reproduce them.
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
        <H2>Absolute latency by model</H2>
        <Text tone="secondary">
          Remeasured Intel AC vs battery. Bundled and Windows ML sit side by
          side on a shared Y scale; bars are AC vs battery per device.
        </Text>
      </Stack>

      <WorkloadRuntimePair
        workload="whisper-static"
        devices={["NPU", "GPU", "CPU"]}
        title="Whisper static"
        note="10 runs · bundled Whisper NPU AC 718.9 ms · WinML NPU AC 200.0 ms"
      />
      <WorkloadRuntimePair
        workload="whisper-dynamic"
        devices={["GPU", "CPU"]}
        title="Whisper dynamic"
        note="10 runs · NPU unsupported for dynamic-KV and omitted"
      />
      <WorkloadRuntimePair
        workload="tsc"
        devices={["NPU", "GPU", "CPU"]}
        title="TSC"
        note="20 runs · bundled NPU ~8.3 ms AC / 8.2 ms battery"
      />
      <WorkloadRuntimePair
        workload="fakeaudio"
        devices={["NPU", "GPU", "CPU"]}
        title="FakeAudio"
        note="NPU uses the Intel split-backbone fixture; GPU/CPU use whole graph"
      />

      <Divider />

      <Stack gap={8}>
        <H2>Battery ÷ AC within each device</H2>
        <Text tone="secondary">
          Ratios below 1.0 are faster on battery; ratios above 1.0 are slower.
          After the AC remeasure, the worst remaining sub-parity is WinML Whisper
          static CPU (0.62×)—still not causal without interleaved repeats.
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
