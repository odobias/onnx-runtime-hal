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
  kind: "wer" | "acc";
  intel: number;
  qualcomm: number;
};

/** accuracy-quick (1 pass). Score is WER% for Whisper or accuracy% for classifiers. */
const CELLS: Cell[] = [
  { power: "ac", runtime: "bundled", workload: "whisper-static", device: "NPU", kind: "wer", intel: 9.57, qualcomm: 8.58 },
  { power: "ac", runtime: "bundled", workload: "whisper-static", device: "GPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "ac", runtime: "bundled", workload: "whisper-static", device: "CPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "ac", runtime: "bundled", workload: "whisper-dynamic", device: "GPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "ac", runtime: "bundled", workload: "whisper-dynamic", device: "CPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "ac", runtime: "bundled", workload: "tsc", device: "NPU", kind: "acc", intel: 93.33, qualcomm: 93.33 },
  { power: "ac", runtime: "bundled", workload: "tsc", device: "GPU", kind: "acc", intel: 93.33, qualcomm: 93.33 },
  { power: "ac", runtime: "bundled", workload: "tsc", device: "CPU", kind: "acc", intel: 93.33, qualcomm: 93.33 },
  { power: "ac", runtime: "bundled", workload: "fakeaudio", device: "NPU", kind: "acc", intel: 60.0, qualcomm: 60.0 },
  { power: "ac", runtime: "bundled", workload: "fakeaudio", device: "GPU", kind: "acc", intel: 60.0, qualcomm: 60.0 },
  { power: "ac", runtime: "bundled", workload: "fakeaudio", device: "CPU", kind: "acc", intel: 60.0, qualcomm: 60.0 },
  { power: "ac", runtime: "winml", workload: "whisper-static", device: "NPU", kind: "wer", intel: 9.57, qualcomm: 8.58 },
  { power: "ac", runtime: "winml", workload: "whisper-static", device: "GPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "ac", runtime: "winml", workload: "whisper-static", device: "CPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "ac", runtime: "winml", workload: "whisper-dynamic", device: "GPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "ac", runtime: "winml", workload: "whisper-dynamic", device: "CPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "ac", runtime: "winml", workload: "tsc", device: "NPU", kind: "acc", intel: 93.33, qualcomm: 93.33 },
  { power: "ac", runtime: "winml", workload: "tsc", device: "GPU", kind: "acc", intel: 93.33, qualcomm: 93.33 },
  { power: "ac", runtime: "winml", workload: "tsc", device: "CPU", kind: "acc", intel: 93.33, qualcomm: 93.33 },
  { power: "ac", runtime: "winml", workload: "fakeaudio", device: "NPU", kind: "acc", intel: 60.0, qualcomm: 60.0 },
  { power: "ac", runtime: "winml", workload: "fakeaudio", device: "GPU", kind: "acc", intel: 60.0, qualcomm: 60.0 },
  { power: "ac", runtime: "winml", workload: "fakeaudio", device: "CPU", kind: "acc", intel: 60.0, qualcomm: 60.0 },
  { power: "battery", runtime: "bundled", workload: "whisper-static", device: "NPU", kind: "wer", intel: 9.57, qualcomm: 8.58 },
  { power: "battery", runtime: "bundled", workload: "whisper-static", device: "GPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "battery", runtime: "bundled", workload: "whisper-static", device: "CPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "battery", runtime: "bundled", workload: "whisper-dynamic", device: "GPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "battery", runtime: "bundled", workload: "whisper-dynamic", device: "CPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "battery", runtime: "bundled", workload: "tsc", device: "NPU", kind: "acc", intel: 93.33, qualcomm: 93.33 },
  { power: "battery", runtime: "bundled", workload: "tsc", device: "GPU", kind: "acc", intel: 93.33, qualcomm: 93.33 },
  { power: "battery", runtime: "bundled", workload: "tsc", device: "CPU", kind: "acc", intel: 93.33, qualcomm: 93.33 },
  { power: "battery", runtime: "bundled", workload: "fakeaudio", device: "NPU", kind: "acc", intel: 60.0, qualcomm: 60.0 },
  { power: "battery", runtime: "bundled", workload: "fakeaudio", device: "GPU", kind: "acc", intel: 60.0, qualcomm: 60.0 },
  { power: "battery", runtime: "bundled", workload: "fakeaudio", device: "CPU", kind: "acc", intel: 60.0, qualcomm: 60.0 },
  { power: "battery", runtime: "winml", workload: "whisper-static", device: "NPU", kind: "wer", intel: 9.57, qualcomm: 8.58 },
  { power: "battery", runtime: "winml", workload: "whisper-static", device: "GPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "battery", runtime: "winml", workload: "whisper-static", device: "CPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "battery", runtime: "winml", workload: "whisper-dynamic", device: "GPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "battery", runtime: "winml", workload: "whisper-dynamic", device: "CPU", kind: "wer", intel: 8.58, qualcomm: 8.58 },
  { power: "battery", runtime: "winml", workload: "tsc", device: "NPU", kind: "acc", intel: 93.33, qualcomm: 93.33 },
  { power: "battery", runtime: "winml", workload: "tsc", device: "GPU", kind: "acc", intel: 93.33, qualcomm: 93.33 },
  { power: "battery", runtime: "winml", workload: "tsc", device: "CPU", kind: "acc", intel: 93.33, qualcomm: 93.33 },
  { power: "battery", runtime: "winml", workload: "fakeaudio", device: "NPU", kind: "acc", intel: 60.0, qualcomm: 60.0 },
  { power: "battery", runtime: "winml", workload: "fakeaudio", device: "GPU", kind: "acc", intel: 60.0, qualcomm: 60.0 },
  { power: "battery", runtime: "winml", workload: "fakeaudio", device: "CPU", kind: "acc", intel: 60.0, qualcomm: 60.0 },
];

const VENDORS = ["intel", "qualcomm"] as const;
type Vendor = (typeof VENDORS)[number];
const VENDOR_LABEL: Record<Vendor, string> = {
  intel: "Intel",
  qualcomm: "Qualcomm",
};

function shortWorkload(workload: string) {
  if (workload === "whisper-static") return "whisper static";
  if (workload === "whisper-dynamic") return "whisper dynamic";
  return workload;
}

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
  return Math.ceil(max * 1.05 * 10) / 10;
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
      tone: (["info", "success"] as const)[index],
    })),
  };
}

function WorkloadRuntimePair({
  power,
  workload,
  devices,
  title,
  note,
  kind,
}: {
  power: "ac" | "battery";
  workload: string;
  devices: Array<"NPU" | "GPU" | "CPU">;
  title: string;
  note: string;
  kind: "wer" | "acc";
}) {
  const label = power === "ac" ? "AC" : "Battery";
  const bundled = vendorSeries(power, "bundled", workload, devices);
  const winml = vendorSeries(power, "winml", workload, devices);
  const yMax = sharedYMax(bundled.series, winml.series);
  const suffix = kind === "wer" ? " WER%" : " acc%";
  const axis = kind === "wer" ? "WER%" : "accuracy%";
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
              height={280}
              valueSuffix={suffix}
              beginAtZero
              yMax={yMax}
            />
            <Text size="small" tone="tertiary" style={{ marginTop: 6 }}>
              {"X-axis: device · Y-axis: " +
                axis +
                " · shared scale · bundled · " +
                label}
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill size="sm">Windows ML</Pill>}>
            {title} · Windows ML · {label}
          </CardHeader>
          <CardBody>
            <BarChart
              {...winml}
              height={280}
              valueSuffix={suffix}
              beginAtZero
              yMax={yMax}
            />
            <Text size="small" tone="tertiary" style={{ marginTop: 6 }}>
              {"X-axis: device · Y-axis: " +
                axis +
                " · shared scale · WinML · " +
                label}
            </Text>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

function winCount(power: "ac" | "battery") {
  let intel = 0;
  let qualcomm = 0;
  for (const cell of CELLS.filter((c) => c.power === power)) {
    const iBetter =
      cell.kind === "wer" ? cell.intel < cell.qualcomm : cell.intel > cell.qualcomm;
    const qBetter =
      cell.kind === "wer" ? cell.qualcomm < cell.intel : cell.qualcomm > cell.intel;
    if (iBetter) intel += 1;
    else if (qBetter) qualcomm += 1;
  }
  return { intel, qualcomm };
}

function matrixRows(power: "ac" | "battery") {
  return CELLS.filter((c) => c.power === power).map((cell) => {
    const winner =
      cell.kind === "wer"
        ? cell.intel < cell.qualcomm
          ? "Intel"
          : cell.qualcomm < cell.intel
            ? "Qualcomm"
            : "tie"
        : cell.intel > cell.qualcomm
          ? "Intel"
          : cell.qualcomm > cell.intel
            ? "Qualcomm"
            : "tie";
    return [
      cell.runtime,
      shortWorkload(cell.workload),
      cell.device,
      cell.kind === "wer"
        ? `${cell.intel.toFixed(2)}%`
        : `${cell.intel.toFixed(1)}%`,
      cell.kind === "wer"
        ? `${cell.qualcomm.toFixed(2)}%`
        : `${cell.qualcomm.toFixed(1)}%`,
      winner,
    ];
  });
}

function Overview() {
  const ac = winCount("ac");
  const bat = winCount("battery");
  return (
    <Stack gap={20}>
      <Grid columns={4} gap={14}>
        <Stat value="44 / 44" label="Intel accuracy cells" tone="success" />
        <Stat value="44 / 44" label="Qualcomm accuracy cells" tone="success" />
        <Stat value="0 / 44" label="AMD accuracy cells" tone="danger" />
        <Stat value="1 run" label="accuracy-quick reps" tone="info" />
      </Grid>

      <Callout tone="danger" title="AMD accuracy-quick evidence is missing">
        There are no published AMD accuracy-runs files. Latency coverage exists
        for AMD, but accuracy needs a dedicated AMD AC + battery accuracy-quick
        campaign (4 suite invocations per power mode; Runs=1 is enough).
      </Callout>

      <Callout tone="info" title="Accuracy is mostly power-invariant here">
        With accuracy-quick (single pass), Intel and Qualcomm scores match
        across AC and battery for the same runtime/device. This is a
        correctness / agreement matrix. Latency remains where power shows up.
      </Callout>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader trailing={<Pill size="sm">AC</Pill>}>
            Cell wins (Intel vs Qualcomm)
          </CardHeader>
          <CardBody>
            <BarChart
              categories={["Intel", "Qualcomm"]}
              series={[
                {
                  name: "AC wins",
                  data: [ac.intel, ac.qualcomm],
                  tone: "info",
                },
              ]}
              height={220}
              showValues
              beginAtZero
            />
            <Text size="small" tone="tertiary" style={{ marginTop: 6 }}>
              Ties omitted · Whisper uses lower WER; classifiers use higher
              accuracy%
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill size="sm">Battery</Pill>}>
            Cell wins (Intel vs Qualcomm)
          </CardHeader>
          <CardBody>
            <BarChart
              categories={["Intel", "Qualcomm"]}
              series={[
                {
                  name: "Battery wins",
                  data: [bat.intel, bat.qualcomm],
                  tone: "warning",
                },
              ]}
              height={220}
              showValues
              beginAtZero
            />
            <Text size="small" tone="tertiary" style={{ marginTop: 6 }}>
              Ties omitted · Source: results/accuracy-runs · accuracy-quick
            </Text>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

function PowerMatrix({ power }: { power: "ac" | "battery" }) {
  const label = power === "ac" ? "AC" : "Battery";
  const wins = winCount(power);
  return (
    <Stack gap={20}>
      <Grid columns={3} gap={14}>
        <Stat
          value={`${wins.intel}`}
          label={`${label} Intel cell wins`}
          tone="info"
        />
        <Stat
          value={`${wins.qualcomm}`}
          label={`${label} Qualcomm cell wins`}
          tone="success"
        />
        <Stat value="AMD n/a" label="AMD not in this matrix" tone="danger" />
      </Grid>

      <WorkloadRuntimePair
        power={power}
        workload="whisper-static"
        devices={["NPU", "GPU", "CPU"]}
        title="Whisper static"
        note="WER% · lower is better · 12 eval clips · accuracy-quick (1 pass)"
        kind="wer"
      />
      <WorkloadRuntimePair
        power={power}
        workload="whisper-dynamic"
        devices={["GPU", "CPU"]}
        title="Whisper dynamic"
        note="WER% · NPU unsupported for dynamic-KV · accuracy-quick (1 pass)"
        kind="wer"
      />
      <WorkloadRuntimePair
        power={power}
        workload="tsc"
        devices={["NPU", "GPU", "CPU"]}
        title="TSC"
        note="Classifier accuracy% · higher is better · accuracy-quick (1 pass)"
        kind="acc"
      />
      <WorkloadRuntimePair
        power={power}
        workload="fakeaudio"
        devices={["NPU", "GPU", "CPU"]}
        title="FakeAudio"
        note="Classifier accuracy% · NPU may use vendor split fixtures"
        kind="acc"
      />

      <Stack gap={8}>
        <H2>{label} full matrix</H2>
        <Text tone="secondary">
          Scores are WER% for Whisper and accuracy% for classifiers. Winner uses
          lower WER / higher accuracy.
        </Text>
        <Table
          headers={[
            "Runtime",
            "Workload",
            "Device",
            "Intel",
            "Qualcomm",
            "Winner",
          ]}
          rows={matrixRows(power)}
          stickyHeader
        />
      </Stack>
    </Stack>
  );
}

function Caveats() {
  return (
    <Stack gap={16}>
      <Callout tone="warning" title="What this canvas is">
        Cross-vendor accuracy-quick inventory for Intel and Qualcomm on AC and
        battery. It is not a causal ranking under equal thermals, and it does
        not include AMD until accuracy-quick campaigns exist.
      </Callout>
      <Table
        headers={["Issue", "Where", "How to read it"]}
        rows={[
          [
            "AMD missing entirely",
            "0 / 44 accuracy-quick cells",
            "Run accuracy-quick on AMD AC then battery (4 invocations each)",
          ],
          [
            "Single-pass metrics",
            "Runs=1 / ClassifierRuns=1",
            "Enough for agreement; not a variance study",
          ],
          [
            "Power rarely moves accuracy",
            "Intel/Qualcomm AC ≈ battery scores",
            "Expect correctness parity; look at latency for power effects",
          ],
          [
            "Intel NPU Whisper WER gap",
            "9.57% vs 8.58% on GPU/CPU and vs Qualcomm NPU",
            "Only non-tie Whisper cell; check op offload / fixture path",
          ],
          [
            "FakeAudio 60% everywhere",
            "All Intel/Qualcomm cells",
            "Fixture/reference issue, not a vendor ranking signal",
          ],
        ]}
      />
    </Stack>
  );
}

const TABS = ["Overview", "AC matrix", "Battery matrix", "Caveats"] as const;

export default function CrossVendorAcVsBatteryAccuracy() {
  const [tab, setTab] = useCanvasState<string>("xvendor-acc-tab-v1", "Overview");
  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H1>Cross-vendor accuracy matrix · AC vs battery</H1>
        <Text tone="secondary">
          Intel · Qualcomm · accuracy-quick (1 pass) · bundled + Windows ML ·
          WER% for Whisper, accuracy% for classifiers.
        </Text>
        <Text tone="tertiary">
          88 / 132 expected cells · AMD absent · dynamic-KV NPU unsupported ·
          Source: results/accuracy-runs · through 16 Jul 2026
        </Text>
      </Stack>

      <Row gap={8} wrap>
        {TABS.map((name) => (
          <Pill active={tab === name} onClick={() => setTab(name)}>
            {name}
          </Pill>
        ))}
      </Row>

      {tab === "Overview" ? <Overview /> : null}
      {tab === "AC matrix" ? <PowerMatrix power="ac" /> : null}
      {tab === "Battery matrix" ? <PowerMatrix power="battery" /> : null}
      {tab === "Caveats" ? <Caveats /> : null}
    </Stack>
  );
}
