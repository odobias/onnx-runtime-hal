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
} from "cursor/canvas";

type Cell = {
  power: "ac" | "battery";
  runtime: "bundled" | "winml";
  workload: string;
  device: "NPU" | "GPU" | "CPU";
  intel: number;
  amd: number;
  qualcomm: number;
};

/** Latest mode-latency rows (10 Whisper / 20 classifier) from results/ledgers. */
const CELLS: Cell[] = [
  { power: "ac", runtime: "bundled", workload: "whisper-static", device: "NPU", intel: 853.6, amd: 642.4, qualcomm: 359.8 },
  { power: "ac", runtime: "bundled", workload: "whisper-static", device: "GPU", intel: 615.2, amd: 570.1, qualcomm: 778.0 },
  { power: "ac", runtime: "bundled", workload: "whisper-static", device: "CPU", intel: 2690.7, amd: 1033.7, qualcomm: 1202.2 },
  { power: "ac", runtime: "bundled", workload: "whisper-dynamic", device: "GPU", intel: 359.8, amd: 379.2, qualcomm: 688.0 },
  { power: "ac", runtime: "bundled", workload: "whisper-dynamic", device: "CPU", intel: 1675.7, amd: 264.3, qualcomm: 331.5 },
  { power: "ac", runtime: "bundled", workload: "tsc", device: "NPU", intel: 8.7, amd: 33.3, qualcomm: 18.5 },
  { power: "ac", runtime: "bundled", workload: "tsc", device: "GPU", intel: 47.5, amd: 63.6, qualcomm: 84.0 },
  { power: "ac", runtime: "bundled", workload: "tsc", device: "CPU", intel: 322.5, amd: 159.0, qualcomm: 231.9 },
  { power: "ac", runtime: "bundled", workload: "fakeaudio", device: "NPU", intel: 23.7, amd: 41.7, qualcomm: 128.1 },
  { power: "ac", runtime: "bundled", workload: "fakeaudio", device: "GPU", intel: 56.6, amd: 38.0, qualcomm: 46.2 },
  { power: "ac", runtime: "bundled", workload: "fakeaudio", device: "CPU", intel: 148.6, amd: 79.6, qualcomm: 158.2 },
  { power: "ac", runtime: "winml", workload: "whisper-static", device: "NPU", intel: 211.2, amd: 609.0, qualcomm: 376.7 },
  { power: "ac", runtime: "winml", workload: "whisper-static", device: "GPU", intel: 551.7, amd: 577.6, qualcomm: 774.2 },
  { power: "ac", runtime: "winml", workload: "whisper-static", device: "CPU", intel: 1716.7, amd: 1327.5, qualcomm: 871.5 },
  { power: "ac", runtime: "winml", workload: "whisper-dynamic", device: "GPU", intel: 335.1, amd: 391.9, qualcomm: 680.2 },
  { power: "ac", runtime: "winml", workload: "whisper-dynamic", device: "CPU", intel: 221.3, amd: 220.8, qualcomm: 200.1 },
  { power: "ac", runtime: "winml", workload: "tsc", device: "NPU", intel: 8.5, amd: 33.6, qualcomm: 18.4 },
  { power: "ac", runtime: "winml", workload: "tsc", device: "GPU", intel: 45.0, amd: 74.9, qualcomm: 84.2 },
  { power: "ac", runtime: "winml", workload: "tsc", device: "CPU", intel: 229.6, amd: 151.9, qualcomm: 179.4 },
  { power: "ac", runtime: "winml", workload: "fakeaudio", device: "NPU", intel: 23.0, amd: 48.7, qualcomm: 128.1 },
  { power: "ac", runtime: "winml", workload: "fakeaudio", device: "GPU", intel: 46.3, amd: 85.7, qualcomm: 45.8 },
  { power: "ac", runtime: "winml", workload: "fakeaudio", device: "CPU", intel: 117.6, amd: 91.7, qualcomm: 94.2 },
  { power: "battery", runtime: "bundled", workload: "whisper-static", device: "NPU", intel: 516.5, amd: 753.3, qualcomm: 360.0 },
  { power: "battery", runtime: "bundled", workload: "whisper-static", device: "GPU", intel: 657.3, amd: 818.3, qualcomm: 1283.4 },
  { power: "battery", runtime: "bundled", workload: "whisper-static", device: "CPU", intel: 2111.1, amd: 862.1, qualcomm: 12605.0 },
  { power: "battery", runtime: "bundled", workload: "whisper-dynamic", device: "GPU", intel: 613.1, amd: 765.1, qualcomm: 1483.5 },
  { power: "battery", runtime: "bundled", workload: "whisper-dynamic", device: "CPU", intel: 1600.6, amd: 248.7, qualcomm: 2067.3 },
  { power: "battery", runtime: "bundled", workload: "tsc", device: "NPU", intel: 7.7, amd: 33.8, qualcomm: 20.1 },
  { power: "battery", runtime: "bundled", workload: "tsc", device: "GPU", intel: 38.2, amd: 60.1, qualcomm: 111.8 },
  { power: "battery", runtime: "bundled", workload: "tsc", device: "CPU", intel: 227.4, amd: 112.4, qualcomm: 1560.0 },
  { power: "battery", runtime: "bundled", workload: "fakeaudio", device: "NPU", intel: 23.0, amd: 62.2, qualcomm: 129.9 },
  { power: "battery", runtime: "bundled", workload: "fakeaudio", device: "GPU", intel: 73.9, amd: 73.5, qualcomm: 75.0 },
  { power: "battery", runtime: "bundled", workload: "fakeaudio", device: "CPU", intel: 122.8, amd: 70.8, qualcomm: 1080.9 },
  { power: "battery", runtime: "winml", workload: "whisper-static", device: "NPU", intel: 296.9, amd: 1088.7, qualcomm: 449.9 },
  { power: "battery", runtime: "winml", workload: "whisper-static", device: "GPU", intel: 665.3, amd: 772.8, qualcomm: 1252.8 },
  { power: "battery", runtime: "winml", workload: "whisper-static", device: "CPU", intel: 919.6, amd: 1187.5, qualcomm: 2552.2 },
  { power: "battery", runtime: "winml", workload: "whisper-dynamic", device: "GPU", intel: 494.1, amd: 688.4, qualcomm: 1424.0 },
  { power: "battery", runtime: "winml", workload: "whisper-dynamic", device: "CPU", intel: 215.6, amd: 254.1, qualcomm: 526.8 },
  { power: "battery", runtime: "winml", workload: "tsc", device: "NPU", intel: 8.3, amd: 44.6, qualcomm: 20.4 },
  { power: "battery", runtime: "winml", workload: "tsc", device: "GPU", intel: 37.7, amd: 66.1, qualcomm: 91.5 },
  { power: "battery", runtime: "winml", workload: "tsc", device: "CPU", intel: 216.7, amd: 149.3, qualcomm: 329.0 },
  { power: "battery", runtime: "winml", workload: "fakeaudio", device: "NPU", intel: 23.3, amd: 82.6, qualcomm: 132.0 },
  { power: "battery", runtime: "winml", workload: "fakeaudio", device: "GPU", intel: 71.8, amd: 87.2, qualcomm: 70.0 },
  { power: "battery", runtime: "winml", workload: "fakeaudio", device: "CPU", intel: 109.4, amd: 101.3, qualcomm: 158.9 },
];

type Vendor = "intel" | "amd" | "qualcomm";
const VENDORS: Vendor[] = ["intel", "amd", "qualcomm"];
const VENDOR_LABEL: Record<Vendor, string> = {
  intel: "Intel",
  amd: "AMD",
  qualcomm: "Qualcomm",
};

function shortWorkload(w: string) {
  if (w === "whisper-static") return "whisper static";
  if (w === "whisper-dynamic") return "whisper dynamic";
  return w;
}

function winner(cell: Cell): Vendor {
  return VENDORS.reduce((best, v) => (cell[v] < cell[best] ? v : best), "intel" as Vendor);
}

function cellsFor(power: "ac" | "battery") {
  return CELLS.filter((c) => c.power === power);
}

function winCounts(power: "ac" | "battery") {
  const counts: Record<Vendor, number> = { intel: 0, amd: 0, qualcomm: 0 };
  for (const cell of cellsFor(power)) counts[winner(cell)] += 1;
  return counts;
}

function vendorSeries(
  power: "ac" | "battery",
  runtime: "bundled" | "winml",
  workload: string,
  devices: Array<"NPU" | "GPU" | "CPU">
) {
  return {
    categories: devices,
    series: VENDORS.map((vendor, index) => ({
      name: VENDOR_LABEL[vendor],
      data: devices.map((device) => {
        const cell = CELLS.find(
          (c) =>
            c.power === power &&
            c.runtime === runtime &&
            c.device === device &&
            c.workload === workload
        )!;
        return cell[vendor];
      }),
      tone: (["info", "warning", "success"] as const)[index],
    })),
  };
}

/** Shared Y ceiling for bundled ‖ WinML pairs so side-by-side bars are comparable. */
function sharedYMax(...seriesGroups: Array<Array<{ data: number[] }>>): number {
  let max = 0;
  for (const group of seriesGroups) {
    for (const series of group) {
      for (const value of series.data) {
        if (value > max) max = value;
      }
    }
  }
  if (max <= 0) return 1;
  // Small headroom so the tallest bar is not clipped at the axis edge.
  return Math.ceil(max * 1.05);
}

function WorkloadRuntimePair({
  power,
  workload,
  devices,
  title,
  note,
  height = 280,
}: {
  power: "ac" | "battery";
  workload: string;
  devices: Array<"NPU" | "GPU" | "CPU">;
  title: string;
  note: string;
  height?: number;
}) {
  const label = power === "ac" ? "AC" : "Battery";
  const bundled = vendorSeries(power, "bundled", workload, devices);
  const winml = vendorSeries(power, "winml", workload, devices);
  const yMax = sharedYMax(bundled.series, winml.series);
  return (
    <Stack gap={10}>
      <H2>{title}</H2>
      <Text tone="secondary">{note}</Text>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader trailing={<Pill size="sm">bundled</Pill>}>
            {title} · bundled · {label}
          </CardHeader>
          <CardBody>
            <BarChart
              {...bundled}
              height={height}
              valueSuffix=" ms"
              beginAtZero
              yMax={yMax}
            />
            <Caption>
              {"X-axis: device · Y-axis: mean latency (ms) · shared scale · bundled · " +
                label}
            </Caption>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill size="sm">Windows ML</Pill>}>
            {title} · Windows ML · {label}
          </CardHeader>
          <CardBody>
            <BarChart
              {...winml}
              height={height}
              valueSuffix=" ms"
              beginAtZero
              yMax={yMax}
            />
            <Caption>
              {"X-axis: device · Y-axis: mean latency (ms) · shared scale · WinML · " +
                label}
            </Caption>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

function matrixRows(power: "ac" | "battery") {
  return cellsFor(power).map((cell) => {
    const w = winner(cell);
    return [
      cell.runtime,
      shortWorkload(cell.workload),
      cell.device,
      cell.intel.toFixed(1),
      cell.amd.toFixed(1),
      cell.qualcomm.toFixed(1),
      VENDOR_LABEL[w],
    ];
  });
}

function matrixTones(power: "ac" | "battery") {
  return cellsFor(power).map((cell) => {
    const w = winner(cell);
    if (w === "intel") return "info" as const;
    if (w === "amd") return "warning" as const;
    return "success" as const;
  });
}

function Caption({ children }: { children: string | string[] }) {
  const text = Array.isArray(children) ? children.join(" ") : children;
  return (
    <Text size="small" tone="tertiary" style={{ marginTop: 6 }}>
      {text}
    </Text>
  );
}

function Overview() {
  const acWins = winCounts("ac");
  const batWins = winCounts("battery");
  return (
    <Stack gap={20}>
      <Grid columns={4} gap={14}>
        <Stat value="66 / 66" label="AC latency cells" tone="success" />
        <Stat value="66 / 66" label="Battery latency cells" tone="success" />
        <Stat
          value={`${acWins.qualcomm} · ${acWins.intel} · ${acWins.amd}`}
          label="AC wins Q · I · A"
          tone="info"
        />
        <Stat
          value={`${batWins.qualcomm} · ${batWins.intel} · ${batWins.amd}`}
          label="Battery wins Q · I · A"
          tone="warning"
        />
      </Grid>

      <Callout tone="warning" title="Same matrix shape; different power stories">
        Every vendor now has a complete 22-cell latency matrix on AC and on
        battery (bundled + Windows ML, NPU/GPU/CPU, excluding unsupported
        dynamic-KV NPU). Rankings still move with power: Qualcomm collapses on
        battery CPU, Intel AC was remeasured (bundled Whisper NPU ~719 ms), and
        AMD pays a clearer battery tax on accelerators.
      </Callout>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader trailing={<Pill size="sm">AC</Pill>}>
            Cell wins by vendor (lowest mean latency)
          </CardHeader>
          <CardBody>
            <BarChart
              categories={["Qualcomm", "Intel", "AMD"]}
              series={[
                {
                  name: "AC wins",
                  data: [acWins.qualcomm, acWins.intel, acWins.amd],
                  tone: "info",
                },
              ]}
              height={240}
              showValues
              beginAtZero
            />
            <Caption>
              X-axis: vendor · Y-axis: number of matrix cells with lowest mean
              latency · Source: results/ledgers mode latency · 16 Jul 2026
            </Caption>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill size="sm">Battery</Pill>}>
            Cell wins by vendor (lowest mean latency)
          </CardHeader>
          <CardBody>
            <BarChart
              categories={["Qualcomm", "Intel", "AMD"]}
              series={[
                {
                  name: "Battery wins",
                  data: [batWins.qualcomm, batWins.intel, batWins.amd],
                  tone: "warning",
                },
              ]}
              height={240}
              showValues
              beginAtZero
            />
            <Caption>
              X-axis: vendor · Y-axis: number of matrix cells with lowest mean
              latency · Source: results/ledgers mode latency · 16 Jul 2026
            </Caption>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={8}>
        <H2>Headline NPU by model</H2>
        <Text tone="secondary">
          One model per row. Bundled and Windows ML sit side by side; X-axis is
          AC vs battery. Whisper uses 10 runs; classifiers use 20.
        </Text>
      </Stack>

      <NpuHeadlinePair
        workload="whisper-static"
        title="Whisper static"
      />
      <NpuHeadlinePair workload="tsc" title="TSC" />
      <NpuHeadlinePair workload="fakeaudio" title="FakeAudio" />
    </Stack>
  );
}

function npuPowerSeries(runtime: "bundled" | "winml", workload: string) {
  return VENDORS.map((vendor, index) => ({
    name: VENDOR_LABEL[vendor],
    data: (["ac", "battery"] as const).map((power) => {
      const cell = CELLS.find(
        (c) =>
          c.power === power &&
          c.runtime === runtime &&
          c.device === "NPU" &&
          c.workload === workload
      )!;
      return cell[vendor];
    }),
    tone: (["info", "warning", "success"] as const)[index],
  }));
}

function NpuHeadlinePair({
  workload,
  title,
}: {
  workload: string;
  title: string;
}) {
  const bundledSeries = npuPowerSeries("bundled", workload);
  const winmlSeries = npuPowerSeries("winml", workload);
  const yMax = sharedYMax(bundledSeries, winmlSeries);
  return (
    <Stack gap={10}>
      <H2>{title} · NPU</H2>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader trailing={<Pill size="sm">bundled</Pill>}>
            {title} · bundled NPU
          </CardHeader>
          <CardBody>
            <BarChart
              categories={["AC", "Battery"]}
              series={bundledSeries}
              height={260}
              valueSuffix=" ms"
              beginAtZero
              yMax={yMax}
            />
            <Caption>
              X-axis: power mode · Y-axis: mean latency (ms) · shared scale ·
              bundled NPU
            </Caption>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill size="sm">Windows ML</Pill>}>
            {title} · Windows ML NPU
          </CardHeader>
          <CardBody>
            <BarChart
              categories={["AC", "Battery"]}
              series={winmlSeries}
              height={260}
              valueSuffix=" ms"
              beginAtZero
              yMax={yMax}
            />
            <Caption>
              X-axis: power mode · Y-axis: mean latency (ms) · shared scale ·
              WinML PREFER_NPU
            </Caption>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

function PowerMatrix({ power }: { power: "ac" | "battery" }) {
  const label = power === "ac" ? "AC" : "Battery";
  const wins = winCounts(power);
  const whisperStaticBundled = CELLS.find(
    (c) =>
      c.power === power &&
      c.runtime === "bundled" &&
      c.workload === "whisper-static" &&
      c.device === "NPU"
  )!;
  const tscBundled = CELLS.find(
    (c) =>
      c.power === power &&
      c.runtime === "bundled" &&
      c.workload === "tsc" &&
      c.device === "NPU"
  )!;
  const fakeBundled = CELLS.find(
    (c) =>
      c.power === power &&
      c.runtime === "bundled" &&
      c.workload === "fakeaudio" &&
      c.device === "NPU"
  )!;

  return (
    <Stack gap={20}>
      <Grid columns={4} gap={14}>
        <Stat
          value={VENDOR_LABEL[winner(whisperStaticBundled)]}
          label="Bundled Whisper NPU winner"
          tone="info"
        />
        <Stat
          value={VENDOR_LABEL[winner(tscBundled)]}
          label="Bundled TSC NPU winner"
          tone="info"
        />
        <Stat
          value={VENDOR_LABEL[winner(fakeBundled)]}
          label="Bundled FakeAudio NPU winner"
          tone="info"
        />
        <Stat
          value={`${wins.qualcomm}/${wins.intel}/${wins.amd}`}
          label={`${label} wins Q/I/A`}
          tone="warning"
        />
      </Grid>

      {power === "ac" ? (
        <Callout tone="info" title="Intel AC latency was remeasured">
          Contaminated PR #80 AC campaigns were dropped from the ledgers. The
          replacement bundled Whisper NPU AC mean is 718.9 ms; remeasured battery is
          ~523 ms. Windows ML Whisper NPU is ~200 ms AC / ~285 ms battery.
          WinML GPU battery was remeasured in isolation (~676 ms Whisper static); the full-matrix ~1298 ms cell was suite-order contamination.
        </Callout>
      ) : (
        <Callout tone="danger" title="Qualcomm battery CPU falls off a cliff">
          Snapdragon bundled Whisper static CPU jumps from 1202 ms on AC to
          12605 ms on battery; TSC CPU goes 232 → 1560 ms. That is power-policy
          / thermal throttling, not a model regression. Do not rank CPU
          executors across vendors from battery alone.
        </Callout>
      )}

      <WorkloadRuntimePair
        power={power}
        workload="whisper-static"
        devices={["NPU", "GPU", "CPU"]}
        title="Whisper static"
        note="10 runs · bundled EP vs Windows ML side by side · X-axis is device"
        height={300}
      />
      <WorkloadRuntimePair
        power={power}
        workload="whisper-dynamic"
        devices={["GPU", "CPU"]}
        title="Whisper dynamic"
        note="10 runs · NPU unsupported for dynamic-KV and omitted"
        height={300}
      />
      <WorkloadRuntimePair
        power={power}
        workload="tsc"
        devices={["NPU", "GPU", "CPU"]}
        title="TSC"
        note="20 runs · bundled EP vs Windows ML side by side · X-axis is device"
        height={300}
      />
      <WorkloadRuntimePair
        power={power}
        workload="fakeaudio"
        devices={["NPU", "GPU", "CPU"]}
        title="FakeAudio"
        note="NPU uses vendor split/fixture paths; GPU/CPU use whole graph"
        height={300}
      />

      <Stack gap={8}>
        <H2>{label} full matrix</H2>
        <Text tone="secondary">
          Mean inference latency (ms). Row tint marks the winning vendor for
          that cell (blue Intel, amber AMD, green Qualcomm).
        </Text>
        <Table
          headers={[
            "Runtime",
            "Workload",
            "Device",
            "Intel ms",
            "AMD ms",
            "Qualcomm ms",
            "Winner",
          ]}
          columnAlign={[
            "left",
            "left",
            "left",
            "right",
            "right",
            "right",
            "left",
          ]}
          rows={matrixRows(power)}
          rowTone={matrixTones(power)}
          stickyHeader
        />
        <Caption>
          {"Source: results/ledgers asr.csv + classifiers.csv · measurement_purpose = latency · 10/20 repetition defaults · " +
            label +
            " campaigns through 16 Jul 2026"}
        </Caption>
      </Stack>
    </Stack>
  );
}

function Caveats() {
  return (
    <Stack gap={16}>
      <Callout tone="warning" title="What this canvas is — and is not">
        It is a complete cross-vendor latency inventory split by power source.
        It is not a causal ranking of silicon under equal thermals, nor proof
        that battery makes Intel faster. Each vendor campaign is sequential,
        ORT builds differ (OpenVINO 1.24.1, DirectML/ORT 1.24.4, WinML 1.24.6,
        AMD VitisAI pack), and FakeAudio NPU still uses vendor-specific
        fixtures.
      </Callout>
      <Table
        headers={["Issue", "Where it shows up", "How to read it"]}
        rows={[
          [
            "Intel AC remeasured",
            "bundled Whisper NPU AC 718.9 ms vs remeasured battery 523.5 ms",
            "AC+battery remeasured; prefer NPU/bundled cells for Intel ranks",
          ],
          [
            "Intel WinML GPU battery (resolved)",
            "Isolated probe ~676 ms; full-matrix 1298 ms was suite contamination",
            "GPU battery cells superseded by probe campaigns c497e54c / a54a9a28",
          ],
          [
            "Qualcomm battery CPU collapse",
            "bundled Whisper CPU 1202 → 12605 ms",
            "Exclude Snapdragon battery CPU from cross-vendor CPU ranks",
          ],
          [
            "AMD accelerator battery tax",
            "WinML Whisper NPU 609 → 1089 ms",
            "Matches AMD AC-vs-battery canvas; accelerators slow on battery",
          ],
          [
            "Model / fixture inequivalence",
            "FakeAudio NPU split vs whole-graph GPU/CPU",
            "Compare NPU FakeAudio cells only within the NPU fixture family",
          ],
          [
            "Missing NPU op assignment",
            "Qualcomm + AMD WinML historical NPU accuracy rows",
            "Latency ranking here does not imply trust-gate completeness",
          ],
        ]}
      />
    </Stack>
  );
}

const TABS = ["Overview", "AC matrix", "Battery matrix", "Caveats"] as const;

export default function CrossVendorAcVsBatteryMatrix() {
  const [tab, setTab] = useCanvasState<string>("xvendor-power-tab-v1", "Overview");

  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H1>Cross-vendor latency matrix · AC vs battery</H1>
        <Text tone="secondary">
          Intel · AMD · Qualcomm · native C++ suite · mode latency (10 Whisper /
          20 classifier) · bundled + Windows ML · full 22-cell matrix per power
          condition.
        </Text>
        <Text tone="tertiary">
          132 paired cells · dynamic-KV NPU unsupported everywhere · Source:
          results/ledgers · campaigns through 16 Jul 2026
        </Text>
      </Stack>

      <Row gap={8} wrap>
        {TABS.map((name) => (
          <Pill
            active={tab === name}
            onClick={() => setTab(name)}
          >
            {name}
          </Pill>
        ))}
      </Row>

      {tab === "Overview" && <Overview />}
      {tab === "AC matrix" && <PowerMatrix power="ac" />}
      {tab === "Battery matrix" && <PowerMatrix power="battery" />}
      {tab === "Caveats" && <Caveats />}
    </Stack>
  );
}
