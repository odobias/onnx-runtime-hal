import {
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

type ResultRow = {
  workload: string;
  executor: string;
  executorVendor: string;
  siliconVendor: string;
  device: string;
  provider: string;
  target: string;
  arch: string;
  batteryMs: number | null;
  acMs: number | null;
  latencyMs: number;
  accuracyPct: number | null;
  werPct: number | null;
  modelSha: string;
  date: string;
};

const data: {
  rows: ResultRow[];
} = {"rows":[{"workload":"whisper-tiny-dynamic","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"CPU EP","device":"CPU","target":"bundled","arch":"arm64","batteryMs":2067.296,"acMs":331.455,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"12534e24","date":"2026-07-16","latencyMs":331.455},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"CPU EP","device":"CPU","target":"winml","arch":"arm64","batteryMs":526.763,"acMs":200.051,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"12534e24","date":"2026-07-16","latencyMs":200.051},{"workload":"whisper-tiny-dynamic","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"DirectML","device":"GPU","target":"bundled","arch":"arm64","batteryMs":1483.516,"acMs":687.953,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"12534e24","date":"2026-07-16","latencyMs":687.953},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"DirectML","device":"GPU","target":"winml","arch":"arm64","batteryMs":1423.974,"acMs":680.239,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"12534e24","date":"2026-07-16","latencyMs":680.239},{"workload":"whisper-tiny-static","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"CPU EP","device":"CPU","target":"bundled","arch":"arm64","batteryMs":12604.984,"acMs":1202.186,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":1202.186},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"CPU EP","device":"CPU","target":"winml","arch":"arm64","batteryMs":2552.151,"acMs":871.55,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":871.55},{"workload":"whisper-tiny-static","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"DirectML","device":"GPU","target":"bundled","arch":"arm64","batteryMs":1283.427,"acMs":778.037,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":778.037},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"DirectML","device":"GPU","target":"winml","arch":"arm64","batteryMs":1252.788,"acMs":774.181,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":774.181},{"workload":"whisper-tiny-static","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","provider":"QNN","device":"NPU","target":"bundled","arch":"arm64","batteryMs":360.002,"acMs":359.847,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":359.847},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"QNN","device":"NPU","target":"winml","arch":"arm64","batteryMs":449.94,"acMs":376.708,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":376.708},{"workload":"whisper-tiny-static","executor":"ORT + VitisAI","executorVendor":"AMD","siliconVendor":"AMD","provider":"VitisAI","device":"NPU","target":"bundled","arch":"x64","batteryMs":753.332,"acMs":642.405,"accuracyPct":null,"werPct":10.231023102310232,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":642.405},{"workload":"whisper-tiny-static","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"DirectML","device":"GPU","target":"bundled","arch":"x64","batteryMs":818.287,"acMs":570.147,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":570.147},{"workload":"whisper-tiny-dynamic","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"DirectML","device":"GPU","target":"bundled","arch":"x64","batteryMs":765.098,"acMs":379.185,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"12534e24","date":"2026-07-16","latencyMs":379.185},{"workload":"whisper-tiny-static","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"CPU EP","device":"CPU","target":"bundled","arch":"x64","batteryMs":862.134,"acMs":1033.702,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":862.134},{"workload":"whisper-tiny-dynamic","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"CPU EP","device":"CPU","target":"bundled","arch":"x64","batteryMs":248.721,"acMs":264.252,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"12534e24","date":"2026-07-16","latencyMs":248.721},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"VitisAI","device":"NPU","target":"winml","arch":"x64","batteryMs":1088.666,"acMs":608.95,"accuracyPct":null,"werPct":10.231023102310232,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":608.95},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"DirectML","device":"GPU","target":"winml","arch":"x64","batteryMs":772.803,"acMs":577.57,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":577.57},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"DirectML","device":"GPU","target":"winml","arch":"x64","batteryMs":688.362,"acMs":391.891,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"12534e24","date":"2026-07-16","latencyMs":391.891},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"CPU EP","device":"CPU","target":"winml","arch":"x64","batteryMs":1187.497,"acMs":1327.485,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":1187.497},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"CPU EP","device":"CPU","target":"winml","arch":"x64","batteryMs":254.127,"acMs":220.838,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"12534e24","date":"2026-07-16","latencyMs":220.838},{"workload":"whisper-tiny-static","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","provider":"OpenVINO","device":"NPU","target":"bundled","arch":"x64","batteryMs":544.418,"acMs":3063.778,"accuracyPct":null,"werPct":9.570957095709572,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":544.418},{"workload":"whisper-tiny-static","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"DirectML","device":"GPU","target":"bundled","arch":"x64","batteryMs":676.179,"acMs":1313.284,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":676.179},{"workload":"whisper-tiny-dynamic","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"DirectML","device":"GPU","target":"bundled","arch":"x64","batteryMs":612.3,"acMs":597.96,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"12534e24","date":"2026-07-16","latencyMs":597.96},{"workload":"whisper-tiny-static","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"CPU EP","device":"CPU","target":"bundled","arch":"x64","batteryMs":2337.49,"acMs":3243.07,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":2337.49},{"workload":"whisper-tiny-dynamic","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"CPU EP","device":"CPU","target":"bundled","arch":"x64","batteryMs":1671.752,"acMs":1878.888,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"12534e24","date":"2026-07-16","latencyMs":1671.752},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"OpenVINO","device":"NPU","target":"winml","arch":"x64","batteryMs":292.455,"acMs":372.069,"accuracyPct":null,"werPct":9.570957095709572,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":292.455},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"DirectML","device":"GPU","target":"winml","arch":"x64","batteryMs":658.336,"acMs":848.656,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":658.336},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"DirectML","device":"GPU","target":"winml","arch":"x64","batteryMs":583.245,"acMs":537.411,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"12534e24","date":"2026-07-16","latencyMs":537.411},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"CPU EP","device":"CPU","target":"winml","arch":"x64","batteryMs":932.242,"acMs":2151.351,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"9da9f440","date":"2026-07-16","latencyMs":932.242},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"CPU EP","device":"CPU","target":"winml","arch":"x64","batteryMs":218.299,"acMs":329.598,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"12534e24","date":"2026-07-16","latencyMs":218.299},{"workload":"fakeaudio","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"CPU EP","device":"CPU","target":"bundled","arch":"arm64","batteryMs":1080.86,"acMs":158.167,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16","latencyMs":158.167},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"CPU EP","device":"CPU","target":"winml","arch":"arm64","batteryMs":158.894,"acMs":94.156,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16","latencyMs":94.156},{"workload":"fakeaudio","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"DirectML","device":"GPU","target":"bundled","arch":"arm64","batteryMs":74.964,"acMs":46.206,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16","latencyMs":46.206},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"DirectML","device":"GPU","target":"winml","arch":"arm64","batteryMs":70.02,"acMs":45.755,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16","latencyMs":45.755},{"workload":"fakeaudio","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","provider":"QNN","device":"NPU","target":"bundled","arch":"arm64","batteryMs":116.6,"acMs":128.131,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16","latencyMs":116.6},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"QNN","device":"NPU","target":"winml","arch":"arm64","batteryMs":110.402,"acMs":128.092,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16","latencyMs":110.402},{"workload":"tsc","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"CPU EP","device":"CPU","target":"bundled","arch":"arm64","batteryMs":1560.0,"acMs":231.859,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":231.859},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"CPU EP","device":"CPU","target":"winml","arch":"arm64","batteryMs":329.023,"acMs":179.376,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":179.376},{"workload":"tsc","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"DirectML","device":"GPU","target":"bundled","arch":"arm64","batteryMs":111.795,"acMs":83.978,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":83.978},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"DirectML","device":"GPU","target":"winml","arch":"arm64","batteryMs":91.458,"acMs":84.226,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":84.226},{"workload":"tsc","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","provider":"QNN","device":"NPU","target":"bundled","arch":"arm64","batteryMs":20.114,"acMs":18.497,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":18.497},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","provider":"QNN","device":"NPU","target":"winml","arch":"arm64","batteryMs":20.381,"acMs":18.384,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":18.384},{"workload":"tsc","executor":"ORT + VitisAI","executorVendor":"AMD","siliconVendor":"AMD","provider":"VitisAI","device":"NPU","target":"bundled","arch":"x64","batteryMs":33.804,"acMs":33.268,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":33.268},{"workload":"tsc","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"DirectML","device":"GPU","target":"bundled","arch":"x64","batteryMs":60.147,"acMs":63.554,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":60.147},{"workload":"fakeaudio","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"DirectML","device":"GPU","target":"bundled","arch":"x64","batteryMs":73.528,"acMs":38.026,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16","latencyMs":38.026},{"workload":"tsc","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"CPU EP","device":"CPU","target":"bundled","arch":"x64","batteryMs":112.377,"acMs":158.974,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":112.377},{"workload":"fakeaudio","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"CPU EP","device":"CPU","target":"bundled","arch":"x64","batteryMs":70.781,"acMs":79.58,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16","latencyMs":70.781},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"DirectML","device":"GPU","target":"winml","arch":"x64","batteryMs":66.143,"acMs":74.922,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":66.143},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"DirectML","device":"GPU","target":"winml","arch":"x64","batteryMs":87.153,"acMs":85.726,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16","latencyMs":85.726},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"CPU EP","device":"CPU","target":"winml","arch":"x64","batteryMs":149.345,"acMs":151.863,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":149.345},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"CPU EP","device":"CPU","target":"winml","arch":"x64","batteryMs":101.308,"acMs":91.732,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16","latencyMs":91.732},{"workload":"fakeaudio","executor":"ORT + VitisAI","executorVendor":"AMD","siliconVendor":"AMD","provider":"VitisAI","device":"NPU","target":"bundled","arch":"x64","batteryMs":62.238,"acMs":41.698,"accuracyPct":60.0,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-16","latencyMs":41.698},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"VitisAI","device":"NPU","target":"winml","arch":"x64","batteryMs":44.572,"acMs":33.608,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":33.608},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","provider":"VitisAI","device":"NPU","target":"winml","arch":"x64","batteryMs":82.617,"acMs":48.706,"accuracyPct":60.0,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-16","latencyMs":48.706},{"workload":"tsc","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","provider":"OpenVINO","device":"NPU","target":"bundled","arch":"x64","batteryMs":8.153,"acMs":29.68,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":8.153},{"workload":"fakeaudio","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","provider":"OpenVINO","device":"NPU","target":"bundled","arch":"x64","batteryMs":23.154,"acMs":28.095,"accuracyPct":60.0,"werPct":null,"modelSha":"464a53e6","date":"2026-07-16","latencyMs":23.154},{"workload":"tsc","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"DirectML","device":"GPU","target":"bundled","arch":"x64","batteryMs":40.183,"acMs":58.533,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":40.183},{"workload":"fakeaudio","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"DirectML","device":"GPU","target":"bundled","arch":"x64","batteryMs":76.476,"acMs":56.257,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16","latencyMs":56.257},{"workload":"tsc","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"CPU EP","device":"CPU","target":"bundled","arch":"x64","batteryMs":243.904,"acMs":387.83,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":243.904},{"workload":"fakeaudio","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"CPU EP","device":"CPU","target":"bundled","arch":"x64","batteryMs":126.114,"acMs":165.376,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16","latencyMs":126.114},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"OpenVINO","device":"NPU","target":"winml","arch":"x64","batteryMs":8.813,"acMs":19.335,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":8.813},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"OpenVINO","device":"NPU","target":"winml","arch":"x64","batteryMs":23.069,"acMs":33.421,"accuracyPct":60.0,"werPct":null,"modelSha":"464a53e6","date":"2026-07-16","latencyMs":23.069},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"DirectML","device":"GPU","target":"winml","arch":"x64","batteryMs":38.955,"acMs":70.745,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":38.955},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"DirectML","device":"GPU","target":"winml","arch":"x64","batteryMs":72.399,"acMs":61.441,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16","latencyMs":61.441},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"CPU EP","device":"CPU","target":"winml","arch":"x64","batteryMs":197.453,"acMs":304.462,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16","latencyMs":197.453},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","provider":"CPU EP","device":"CPU","target":"winml","arch":"x64","batteryMs":115.152,"acMs":147.984,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16","latencyMs":115.152}]};

const workloadNames = {
  tsc: "TSC",
  fakeaudio: "FakeAudio",
  "whisper-tiny-static": "Whisper static",
  "whisper-tiny-dynamic": "Whisper dynamic KV",
} as const;

const workloads = Object.keys(workloadNames) as Array<keyof typeof workloadNames>;

function format(value: number, digits = 1) {
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function configLabel(row: ResultRow) {
  return `${row.siliconVendor} · ${row.provider} · ${row.device} · ${row.target}`;
}

function cohortKey(row: ResultRow) {
  return [
    row.workload,
    row.siliconVendor,
    row.provider,
    row.device,
    row.target,
  ].join("|");
}

function hashHue(value: string, hues: string[]) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 33 + value.charCodeAt(index)) >>> 0;
  }
  return hues[hash % hues.length];
}

type CohortPartKey = "silicon" | "provider" | "device" | "target";
type SortKey = CohortPartKey | "quality" | "latency";

function CohortParts({
  siliconVendor,
  provider,
  device,
  target,
  activePart,
  sortDir,
  onSortPart,
}: {
  siliconVendor: string;
  provider: string;
  device: string;
  target: string;
  activePart?: CohortPartKey | null;
  sortDir?: "asc" | "desc";
  onSortPart: (part: CohortPartKey) => void;
}) {
  const theme = useHostTheme();
  const siliconColors: Record<string, string> = {
    Intel: theme.category.blue,
    AMD: theme.category.orange,
    Qualcomm: theme.category.purple,
  };
  const providerColors: Record<string, string> = {
    OpenVINO: theme.category.cyan,
    QNN: theme.category.pink,
    VitisAI: theme.category.yellow,
    DirectML: theme.category.blue,
    "CPU EP": theme.category.gray,
  };
  const deviceColors: Record<string, string> = {
    NPU: theme.category.green,
    GPU: theme.category.orange,
    CPU: theme.category.gray,
  };
  const targetColors: Record<string, string> = {
    bundled: theme.category.cyan,
    winml: theme.category.pink,
  };
  const fallback = [
    theme.category.blue,
    theme.category.purple,
    theme.category.green,
    theme.category.yellow,
    theme.category.cyan,
    theme.category.pink,
    theme.category.orange,
  ];
  const parts: Array<{ key: CohortPartKey; value: string; color: string }> = [
    {
      key: "silicon",
      value: siliconVendor,
      color: siliconColors[siliconVendor] ?? hashHue(siliconVendor, fallback),
    },
    {
      key: "provider",
      value: provider,
      color: providerColors[provider] ?? hashHue(provider, fallback),
    },
    {
      key: "device",
      value: device,
      color: deviceColors[device] ?? hashHue(device, fallback),
    },
    {
      key: "target",
      value: target,
      color: targetColors[target] ?? hashHue(target, fallback),
    },
  ];

  return (
    <span
      style={{
        display: "inline-flex",
        flexWrap: "wrap",
        alignItems: "center",
        gap: 4,
        lineHeight: 1.2,
      }}
    >
      {parts.map((part, index) => {
        const active = activePart === part.key;
        return (
          <span
            key={part.key}
            style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
          >
            {index > 0 && (
              <Text as="span" size="small" tone="quaternary">
                ·
              </Text>
            )}
            <span
              title={`Sort by ${part.key}`}
              onClick={(event) => {
                event.stopPropagation();
                onSortPart(part.key);
              }}
              style={{
                color: part.color,
                background: active ? theme.fill.secondary : theme.fill.tertiary,
                padding: "1px 6px",
                borderRadius: 4,
                whiteSpace: "nowrap",
                cursor: "pointer",
                border: active
                  ? `1px solid ${part.color}`
                  : `1px solid transparent`,
                fontSize: 12,
                fontWeight: 600,
                lineHeight: 1.25,
              }}
            >
              {`${part.value}${active ? (sortDir === "asc" ? " ↑" : " ↓") : ""}`}
            </span>
          </span>
        );
      })}
    </span>
  );
}


function percentile(values: number[], p: number): number {
  if (values.length === 0) return 1;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil((p / 100) * sorted.length) - 1)
  );
  return sorted[idx];
}

function axisMaxFor(rows: Array<{ batteryMs: number | null; acMs: number | null }>): number {
  const peaks = rows
    .flatMap((row) => [row.batteryMs, row.acMs])
    .filter((value): value is number => value != null);
  return Math.max(percentile(peaks, 90), 1);
}

function battDeltaPct(
  batteryMs: number | null,
  acMs: number | null
): number | null {
  if (batteryMs == null || acMs == null || acMs <= 0) return null;
  return ((batteryMs - acMs) / acMs) * 100;
}

function PowerLatencyBar({
  batteryMs,
  acMs,
  maxLatency,
  theme,
}: {
  batteryMs: number | null;
  acMs: number | null;
  maxLatency: number;
  theme: ReturnType<typeof useHostTheme>;
}) {
  const overlapColor = "#7bafe9";
  const batteryColor = "#a9c6ec";
  const track = {
    height: 10,
    background: theme.fill.tertiary,
    borderRadius: 999,
    overflow: "hidden" as const,
    minWidth: 0,
  };
  const valueWrap = (totalPct: number, children: any) => (
    <div style={track}>
      <div
        style={{
          height: "100%",
          width: `${Math.min(100, Math.max(totalPct, 2))}%`,
          display: "flex",
          flexDirection: "row",
          borderRadius: 999,
          overflow: "hidden",
          minWidth: 4,
        }}
      >
        {children}
      </div>
    </div>
  );
  const seg = (widthPct: number, color: string) => (
    <div
      style={{
        height: "100%",
        width: `${widthPct}%`,
        background: color,
        flex: "0 0 auto",
      }}
    />
  );
  if (batteryMs == null && acMs == null) return <div style={track} />;
  if (batteryMs == null || acMs == null) {
    const value = (batteryMs ?? acMs) as number;
    return valueWrap(
      (value / maxLatency) * 100,
      seg(100, acMs != null ? overlapColor : batteryColor)
    );
  }
  if (batteryMs + 1e-9 < acMs) {
    return valueWrap((batteryMs / maxLatency) * 100, seg(100, overlapColor));
  }
  if (Math.abs(batteryMs - acMs) < 1e-9) {
    return valueWrap((acMs / maxLatency) * 100, seg(100, overlapColor));
  }
  const acShare = (acMs / batteryMs) * 100;
  return valueWrap((batteryMs / maxLatency) * 100, [
    seg(acShare, overlapColor),
    seg(100 - acShare, batteryColor),
  ]);
}

function LatestComparison() {
  const theme = useHostTheme();
  const [workload, setWorkload] = useCanvasState<keyof typeof workloadNames>(
    "latest-complete-workload-v1",
    "tsc"
  );
  const [sortKey, setSortKey] = useCanvasState<SortKey>(
    "latest-complete-sort-key-v3",
    "latency"
  );
  const [sortDir, setSortDir] = useCanvasState<"asc" | "desc">(
    "latest-complete-sort-dir-v3",
    "asc"
  );
  const isClassifier = workload === "tsc" || workload === "fakeaudio";

  const combined = data.rows
    .filter((row) => row.workload === workload)
    .map((row) => ({
      label: configLabel(row),
      siliconVendor: row.siliconVendor,
      provider: row.provider,
      device: row.device,
      target: row.target,
      executor: row.executor,
      arch: row.arch,
      batteryMs: row.batteryMs,
      acMs: row.acMs,
      latencyMs: row.latencyMs,
      accuracyPct: isClassifier ? row.accuracyPct : null,
      werPct: isClassifier ? null : row.werPct,
      latencyModel: row.modelSha,
      accuracyModel: row.modelSha,
      exactModelPair: true,
    }));

  const maxLatency = axisMaxFor(combined);
  const inversions = combined.filter(
    (row) =>
      row.batteryMs != null &&
      row.acMs != null &&
      row.batteryMs + 1e-9 < row.acMs
  ).length;
  const fastestMs = Math.min(
    ...combined.map((row) => row.latencyMs),
    Number.POSITIVE_INFINITY
  );
  const exactPairs = combined.filter((row) => row.exactModelPair).length;
  const werValues = combined
    .map((row) => row.werPct)
    .filter((value): value is number => value != null);
  const bestWer = werValues.length > 0 ? Math.min(...werValues) : null;
  const accuracyValues = combined
    .map((row) => row.accuracyPct)
    .filter((value): value is number => value != null);
  const bestAccuracy =
    accuracyValues.length > 0 ? Math.max(...accuracyValues) : null;
  const werRegressions = combined.filter(
    (row) => row.werPct != null && bestWer != null && row.werPct > bestWer + 1e-9
  ).length;

  const qualityValue = (row: (typeof combined)[number]) =>
    isClassifier ? row.accuracyPct : row.werPct;

  const partValue = (
    row: (typeof combined)[number],
    key: CohortPartKey
  ) => {
    if (key === "silicon") return row.siliconVendor;
    if (key === "provider") return row.provider;
    if (key === "device") return row.device;
    return row.target;
  };

  const sorted = [...combined].sort((left, right) => {
    const direction = sortDir === "asc" ? 1 : -1;
    if (
      sortKey === "silicon" ||
      sortKey === "provider" ||
      sortKey === "device" ||
      sortKey === "target"
    ) {
      const primary =
        partValue(left, sortKey).localeCompare(partValue(right, sortKey)) *
        direction;
      if (primary !== 0) return primary;
      return left.latencyMs - right.latencyMs;
    }
    if (sortKey === "quality") {
      const leftValue = qualityValue(left);
      const rightValue = qualityValue(right);
      if (leftValue == null && rightValue == null) return 0;
      if (leftValue == null) return 1;
      if (rightValue == null) return -1;
      return (leftValue - rightValue) * direction;
    }
    return (left.latencyMs - right.latencyMs) * direction;
  });

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
      return;
    }
    setSortKey(key);
    setSortDir(key === "quality" && isClassifier ? "desc" : "asc");
  };

  const sortMark = (key: SortKey) =>
    sortKey === key ? (sortDir === "asc" ? " ↑" : " ↓") : "";
  const activeCohortPart =
    sortKey === "silicon" ||
    sortKey === "provider" ||
    sortKey === "device" ||
    sortKey === "target"
      ? sortKey
      : null;

  const sortBucket = (row: (typeof combined)[number]) => {
    if (
      sortKey === "silicon" ||
      sortKey === "provider" ||
      sortKey === "device" ||
      sortKey === "target"
    ) {
      return partValue(row, sortKey);
    }
    if (sortKey === "quality") {
      const value = qualityValue(row);
      if (value == null) return "—";
      return isClassifier ? `${format(value)}%` : `${format(value, 2)}%`;
    }
    return null;
  };

  const headerStyle = {
    cursor: "pointer",
    userSelect: "none" as const,
    color: theme.text.secondary,
  };

  const gridColumns = "minmax(280px, 34%) minmax(110px, 12%) 1fr 108px";

  return (
    <Stack gap={16}>
      <Row gap={8} wrap>
        {workloads.map((value) => (
          <Pill
            key={value}
            active={workload === value}
            onClick={() => setWorkload(value)}
          >
            {workloadNames[value]}
          </Pill>
        ))}
      </Row>

      <Grid columns={4} gap={14}>
        <Stat
          value={
            Number.isFinite(fastestMs) ? `${format(fastestMs)} ms` : "—"
          }
          label="Fastest latest mean"
          tone="success"
        />
        <Stat
          value={String(combined.length)}
          label="Compared cohorts"
          tone="info"
        />
        <Stat
          value={`${exactPairs}/${combined.length}`}
          label="Exact model pairs"
          tone={exactPairs === combined.length ? "success" : "warning"}
        />
        <Stat
          value={
            isClassifier
              ? bestAccuracy == null
                ? "—"
                : `${format(bestAccuracy)}%`
              : bestWer == null
                ? "—"
                : `${format(bestWer, 2)}%`
          }
          label={isClassifier ? "Best accuracy" : "Best WER"}
          tone={werRegressions > 0 ? "warning" : "success"}
        />
      </Grid>

      {!isClassifier && exactPairs < combined.length && (
        <Callout title="ASR model hashes still differ" tone="warning">
          Rows are joined on silicon, provider, device, and target so latency
          and quality sit together. When model SHAs differ, quality is still
          shown from the matching accuracy cohort, but it is not an exact
          artifact pair.
        </Callout>
      )}

      {inversions > 0 && (
        <Callout title="Battery faster than AC" tone="warning">
          {inversions} cohort{inversions === 1 ? "" : "s"} have battery latency below AC.
          Those rows are flagged; the bar shows only the faster baseline without an AC tip.
          Primary comparison is battery % vs AC.
        </Callout>
      )}

      {!isClassifier && werRegressions > 0 && (
        <Callout title="WER regressions vs best in this workload" tone="danger">
          {werRegressions} cohort
          {werRegressions === 1 ? "" : "s"} raise WER above the best{" "}
          {format(bestWer ?? 0, 2)}%. Those rows are marked in red; lower WER is
          better.
        </Callout>
      )}

      <Stack gap={6}>
        <H2>
          {workloadNames[workload]} latency and{" "}
          {isClassifier ? "accuracy" : "WER"}
        </H2>
        <Text size="small" tone="secondary">
          One row per cohort. AC is the baseline; battery overhang appears only when
          battery is slower. If battery is faster than AC, the row is flagged anomalous
          (no AC tip). Axis capped at p90 of max(batt, ac). Click a chip to sort.
        </Text>
        <Row gap={12}>
          <Text size="small" tone="secondary">blue = AC baseline</Text>
          <Text size="small" tone="secondary">lighter = battery slower</Text>
          <Text size="small" tone="secondary">flag = batt &lt; ac anomaly</Text>
        </Row>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: gridColumns,
            gap: 10,
            alignItems: "center",
            padding: "4px 0",
            borderBottom: `1px solid ${theme.stroke.secondary}`,
          }}
        >
          <CohortParts
            siliconVendor="silicon"
            provider="provider"
            device="device"
            target="target"
            activePart={activeCohortPart}
            sortDir={sortDir}
            onSortPart={toggleSort}
          />
          <div style={headerStyle} onClick={() => toggleSort("quality")}>
            <Text size="small" weight="semibold">
              {`${isClassifier ? "Accuracy" : "WER"}${sortMark("quality")}`}
            </Text>
          </div>
          <div style={headerStyle} onClick={() => toggleSort("latency")}>
            <Text size="small" weight="semibold">
              {`Latency${sortMark("latency")}`}
            </Text>
          </div>
          <Text size="small" tone="tertiary" style={{ textAlign: "right" }}>
            batt / ac · Δ
          </Text>
        </div>

        {sorted.map((row, index) => {
          const werDelta =
            row.werPct != null && bestWer != null ? row.werPct - bestWer : null;
          const accuracyDelta =
            row.accuracyPct != null && bestAccuracy != null
              ? row.accuracyPct - bestAccuracy
              : null;
          const qualityColor =
            !isClassifier && werDelta != null && werDelta > 1e-9
              ? theme.category.red
              : isClassifier && accuracyDelta != null && accuracyDelta < -1e-9
                ? theme.category.red
                : theme.category.green;
          const qualityText = isClassifier
            ? row.accuracyPct == null
              ? "—"
              : accuracyDelta != null && accuracyDelta < -1e-9
                ? `${format(row.accuracyPct)}% ${format(accuracyDelta)}pp`
                : `${format(row.accuracyPct)}%`
            : row.werPct == null
              ? "—"
              : werDelta != null && werDelta > 1e-9
                ? `${format(row.werPct, 2)}% +${format(werDelta, 2)}pp`
                : werDelta != null && werDelta < -1e-9
                  ? `${format(row.werPct, 2)}% ${format(werDelta, 2)}pp`
                  : `${format(row.werPct, 2)}% best`;
          const bucket = sortBucket(row);
          const previousBucket =
            index > 0 ? sortBucket(sorted[index - 1]) : null;
          const showGroupBreak =
            bucket != null && index > 0 && bucket !== previousBucket;

          return (
            <div key={`${row.label}-${row.executor}`}>
              {showGroupBreak && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    margin: "8px 0 4px",
                  }}
                >
                  <div
                    style={{
                      flex: 1,
                      height: 2,
                      background: theme.stroke.secondary,
                    }}
                  />
                  <Text size="small" tone="tertiary" weight="semibold">
                    {bucket}
                  </Text>
                  <div
                    style={{
                      flex: 1,
                      height: 2,
                      background: theme.stroke.secondary,
                    }}
                  />
                </div>
              )}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: gridColumns,
                  gap: 10,
                  alignItems: "center",
                  padding: "3px 0",
                  borderBottom: `1px solid ${theme.stroke.tertiary}`,
                  minHeight: 28,
                }}
              >
                <span style={{ display: "inline-flex", flexWrap: "wrap", alignItems: "center", gap: 6 }}>
                  <CohortParts
                    siliconVendor={row.siliconVendor}
                    provider={row.provider}
                    device={row.device}
                    target={row.target}
                    activePart={activeCohortPart}
                    sortDir={sortDir}
                    onSortPart={toggleSort}
                  />
                  {row.batteryMs != null &&
                    row.acMs != null &&
                    row.batteryMs + 1e-9 < row.acMs && (
                      <span
                        title="Battery faster than AC — unexpected for power modes"
                        style={{
                          padding: "1px 6px",
                          borderRadius: 4,
                          border: `1px solid ${theme.category.yellow}`,
                          background: theme.fill.tertiary,
                          color: theme.category.yellow,
                          fontSize: 10,
                          fontWeight: 700,
                          letterSpacing: "0.02em",
                          textTransform: "uppercase",
                        }}
                      >
                        batt &lt; ac
                      </span>
                    )}
                </span>
                <Text
                  size="small"
                  weight="semibold"
                  style={{ color: qualityColor, lineHeight: 1.25 }}
                >
                  {qualityText}
                </Text>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr auto",
                    gap: 8,
                    alignItems: "center",
                    minWidth: 0,
                  }}
                >
                  <PowerLatencyBar
                    batteryMs={row.batteryMs}
                    acMs={row.acMs}
                    maxLatency={maxLatency}
                    theme={theme}
                  />
                  {Math.max(row.batteryMs ?? 0, row.acMs ?? 0) >
                  maxLatency + 1e-9 ? (
                    <Text size="small" tone="secondary" style={{ color: theme.category.yellow }}>
                      p90+
                    </Text>
                  ) : null}
                </div>
                <div
                  style={{
                    display: "grid",
                    gap: 2,
                    justifyItems: "end",
                    lineHeight: 1.25,
                  }}
                >
                  <Text
                    size="small"
                    weight="semibold"
                    style={{
                      color:
                        row.batteryMs == null
                          ? theme.text.tertiary
                          : "#a9c6ec",
                      display: "inline-flex",
                      gap: 6,
                      alignItems: "baseline",
                    }}
                  >
                    <Text as="span" size="small" tone="tertiary">
                      batt
                    </Text>
                    {row.batteryMs == null ? "—" : format(row.batteryMs)}
                  </Text>
                  <Text
                    size="small"
                    weight="semibold"
                    style={{
                      color:
                        row.acMs == null
                          ? theme.text.tertiary
                          : "#5d96cf",
                      display: "inline-flex",
                      gap: 6,
                      alignItems: "baseline",
                    }}
                  >
                    <Text as="span" size="small" tone="tertiary">
                      ac
                    </Text>
                    {row.acMs == null ? "—" : format(row.acMs)}
                  </Text>
                  {(() => {
                    const pct = battDeltaPct(row.batteryMs, row.acMs);
                    if (pct == null) return null;
                    const anomalous = pct < -1e-9;
                    const sign = pct > 0 ? "+" : "";
                    return (
                      <Text
                        size="small"
                        weight="semibold"
                        style={{
                          color: anomalous
                            ? theme.category.yellow
                            : theme.text.secondary,
                        }}
                      >
                        {`batt ${sign}${format(pct, 0)}% vs AC`}
                      </Text>
                    );
                  })()}
                </div>
              </div>
            </div>
          );
        })}

        <Text size="small" tone="tertiary">
          Source: latest complete latency rows joined to latest trust-gate-valid
          accuracy rows on silicon, provider, device, and runtime target · run
          dates 15–16 Jul 2026. Lower latency and lower WER are better.
        </Text>
      </Stack>
    </Stack>
  );
}

function Method() {
  return (
    <Stack gap={18}>
      <H2>Selection contract</H2>
      <Grid columns={4} gap={14}>
        <Stat value="213" label="Complete latency rows" tone="info" />
        <Stat value="68" label="Latest latency cohorts" tone="success" />
        <Stat value="171" label="Complete accuracy rows" tone="info" />
        <Stat value="73" label="Latest accuracy cohorts" tone="success" />
      </Grid>

      <Callout title="“All columns filled in” is semantic, not literal" tone="info">
        A row must have metrics, runtime and provider identity, model and fixture
        provenance, execution profile, graph role, host snapshot, and a valid
        trust gate. Nullable success fields such as error, plus optional
        operation-assignment fields, may be empty. Requiring every physical CSV
        cell would reject successful rows because successful runs have no error
        text. Marvellous schema theatre.
      </Callout>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Latency inclusion</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>Require measurement_purpose=latency and status=ok</Text>
              <Text>Require explicit provider resolution and no fallback</Text>
              <Text>Require runtime version, model SHA, profile, and provenance</Text>
              <Text>Select latest host-snapshot timestamp per exact cohort</Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Accuracy inclusion</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>Require trust_gate.valid=true</Text>
              <Text>Require model and fixture hashes plus host snapshot</Text>
              <Text>Require workload-specific quality metrics</Text>
              <Text>Select latest timestamp per exact cohort</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Callout title="Strict pairing rule" tone="warning">
        Latency and accuracy are merged only when workload, profile, graph role,
        model SHA, runtime target, provider, device, silicon vendor, and hardware
        all match. This yields 36 classifier pairs. Whisper remains separate
        because its latest model hashes differ between latency and accuracy
        evidence.
      </Callout>
    </Stack>
  );
}

type MatrixGroup = {
  title: string;
  cells: string[];
};

type WorkerCampaign = {
  priority: "P0" | "P1" | "P2";
  vendor: string;
  campaign: string;
  doneWhen: string;
  defaultOpen: boolean;
  groups: MatrixGroup[];
};

function CampaignCard({ item }: { item: WorkerCampaign }) {
  const runCount = item.groups.reduce(
    (total, group) => total + group.cells.length,
    0
  );
  return (
    <Card collapsible defaultOpen={item.defaultOpen}>
      <CardHeader trailing={`${runCount} runs`}>
        {`${item.priority} · ${item.vendor} · ${item.campaign}`}
      </CardHeader>
      <CardBody>
        <Stack gap={14}>
          {item.groups.map((group) => (
            <Stack key={group.title} gap={6}>
              <Text weight="semibold">{group.title}</Text>
              <Stack gap={4}>
                {group.cells.map((cell) => (
                  <Text key={cell} size="small">
                    {cell}
                  </Text>
                ))}
              </Stack>
            </Stack>
          ))}
          <Stack gap={4}>
            <Text weight="semibold">Done when</Text>
            <Text size="small">{item.doneWhen}</Text>
          </Stack>
        </Stack>
      </CardBody>
    </Card>
  );
}

function MissingRuns() {
  const campaigns: WorkerCampaign[] = [
    {
      priority: "P0",
      vendor: "Qualcomm",
      campaign: "ASR explicit latency",
      defaultOpen: true,
      doneWhen:
        "Every listed cell publishes measurement_purpose=latency status=ok against the accuracy model SHA, with no fallback, and pairs the existing accuracy cohort.",
      groups: [
        {
          title:
            "whisper-tiny-static · model dee360f3 · profile static-onnx · graph whole",
          cells: [
            "bundled · QNN · NPU",
            "bundled · DirectML · GPU",
            "bundled · CPU EP · CPU",
            "winml · QNN · NPU",
            "winml · DirectML · GPU",
            "winml · CPU EP · CPU",
          ],
        },
        {
          title:
            "whisper-tiny-dynamic · model cb0a9671 · profile dynamic-kv-onnx · graph whole",
          cells: [
            "bundled · DirectML · GPU",
            "bundled · CPU EP · CPU",
            "winml · DirectML · GPU",
            "winml · CPU EP · CPU",
          ],
        },
      ],
    },
    {
      priority: "P0",
      vendor: "AMD",
      campaign: "ASR explicit latency",
      defaultOpen: true,
      doneWhen:
        "Every listed cell publishes measurement_purpose=latency status=ok against the accuracy model SHA, with no fallback, and pairs the existing accuracy cohort.",
      groups: [
        {
          title:
            "whisper-tiny-static · model dee360f3 · profile static-onnx · graph whole",
          cells: [
            "bundled · VitisAI · NPU",
            "bundled · DirectML · GPU",
            "bundled · CPU EP · CPU",
            "winml · VitisAI · NPU",
            "winml · DirectML · GPU",
            "winml · CPU EP · CPU",
          ],
        },
        {
          title:
            "whisper-tiny-dynamic · model cb0a9671 · profile dynamic-kv-onnx · graph whole",
          cells: [
            "bundled · DirectML · GPU",
            "bundled · CPU EP · CPU",
            "winml · DirectML · GPU",
            "winml · CPU EP · CPU",
          ],
        },
      ],
    },
    {
      priority: "P0",
      vendor: "Intel",
      campaign: "ASR explicit latency",
      defaultOpen: true,
      doneWhen:
        "Every listed cell publishes measurement_purpose=latency status=ok against the accuracy model SHA, with no fallback. This also closes the four accuracy-only OpenVINO CPU/GPU paths.",
      groups: [
        {
          title:
            "whisper-tiny-static · model dee360f3 · profile static-onnx · graph whole",
          cells: [
            "bundled · OpenVINO · NPU",
            "bundled · OpenVINO · GPU",
            "bundled · OpenVINO · CPU",
            "bundled · DirectML · GPU",
            "bundled · CPU EP · CPU",
            "winml · OpenVINO · NPU",
            "winml · DirectML · GPU",
            "winml · CPU EP · CPU",
          ],
        },
        {
          title:
            "whisper-tiny-dynamic · model cb0a9671 · profile dynamic-kv-onnx · graph whole",
          cells: [
            "bundled · OpenVINO · GPU",
            "bundled · OpenVINO · CPU",
            "bundled · DirectML · GPU",
            "bundled · CPU EP · CPU",
            "winml · DirectML · GPU",
            "winml · CPU EP · CPU",
          ],
        },
      ],
    },
    {
      priority: "P1",
      vendor: "Intel",
      campaign: "Classifier explicit latency",
      defaultOpen: true,
      doneWhen:
        "Each cell publishes measurement_purpose=latency status=ok and pairs the already-valid OpenVINO accuracy cohort on the same model SHA.",
      groups: [
        {
          title: "tsc · model 90f2ed1b · profile whole-fp32 · graph whole",
          cells: [
            "bundled · OpenVINO · CPU",
            "bundled · OpenVINO · GPU",
          ],
        },
        {
          title: "fakeaudio · model 549143bc · profile whole-fp32 · graph whole",
          cells: ["bundled · OpenVINO · CPU"],
        },
      ],
    },
    {
      priority: "P2",
      vendor: "Qualcomm",
      campaign: "FakeAudio diagnostic accuracy",
      defaultOpen: false,
      doneWhen:
        "Optional only. Publish trust-gate-valid accuracy for the diagnostic whole-graph QNN NPU profile under both runtime targets, or drop the diagnostic latency rows from the comparison set.",
      groups: [
        {
          title:
            "fakeaudio · model 549143bc · profile npu-whole-diagnostic · graph whole",
          cells: [
            "bundled · QNN · NPU",
            "winml · QNN · NPU",
          ],
        },
      ],
    },
  ];

  return (
    <Stack gap={18}>
      <Grid columns={4} gap={14}>
        <Stat value="34" label="P0 ASR latency runs" tone="danger" />
        <Stat value="3" label="P1 classifier latency runs" tone="warning" />
        <Stat value="2" label="Optional diagnostic runs" tone="info" />
        <Stat value="36 / 36" label="Canonical classifier pairs" tone="success" />
      </Grid>

      <Callout title="Primary gap: every ASR latency row has the wrong model SHA for pairing" tone="danger">
        Existing explicit latency uses static model 9da9f440 and dynamic model
        12534e24. Latest valid accuracy uses dee360f3 and cb0a9671. Workers
        should rerun latency against the current accuracy artifacts, not merely
        repeat the old packages with more samples.
      </Callout>

      <H2>Worker campaign backlog</H2>
      <Text size="small" tone="secondary">
        One card per worker campaign. Matrix cells are fully expanded so nothing
        is truncated by the table layout that previously ate the useful text.
      </Text>
      <Stack gap={12}>
        {campaigns.map((item) => (
          <CampaignCard
            key={`${item.priority}-${item.vendor}-${item.campaign}`}
            item={item}
          />
        ))}
      </Stack>

      <Grid columns={2} gap={16}>
        <Callout title="Seven accuracy cohorts lack explicit latency" tone="warning">
          Four are already included in the P0 Intel ASR campaign: OpenVINO CPU
          and GPU for static and dynamic Whisper. The remaining three are the
          P1 classifier runs: TSC OpenVINO CPU/GPU and FakeAudio OpenVINO CPU.
        </Callout>
        <Callout title="Two latency cohorts lack accuracy" tone="info">
          Both are Qualcomm FakeAudio whole-graph NPU diagnostics. They are not
          part of the canonical split-NPU matrix, so schedule them only if the
          diagnostic profile matters.
        </Callout>
      </Grid>

      <Callout title="Do not schedule already-completed cleanup" tone="success">
        The canonical classifier matrix is fully paired after normalizing
        FakeAudio package variant names to the FakeAudio workload: 18 TSC and
        18 FakeAudio cohorts. Latest runs are on battery, and current canonical
        NPU latency rows contain operation-assignment evidence. Those old gaps
        are closed; rerunning them would merely heat rooms and offend silicon.
      </Callout>

      <Text size="small" tone="tertiary">
        Missing-run detection compares the union of latest complete latency and
        trust-gate-valid accuracy cohorts by workload, profile, graph role,
        runtime target, provider, device, silicon vendor, and hardware. Model
        SHA differences are reported separately rather than silently joined.
      </Text>
    </Stack>
  );
}

export default function CrossVendorBenchmarkResults() {
  const theme = useHostTheme();
  const [tab, setTab] = useCanvasState(
    "cross-vendor-latest-tab-v3",
    "Latest comparison"
  );

  return (
    <Stack
      gap={18}
      style={{
        padding: 24,
        background: theme.bg.editor,
        minHeight: "100%",
      }}
    >
      <Stack gap={6}>
        <H1>Cross-vendor benchmark results</H1>
        <Text tone="secondary">
          Latest complete latency and accuracy evidence · Intel, AMD, Qualcomm ·
          selected 21 Jul 2026
        </Text>
      </Stack>

      <Row gap={8} wrap>
        <Pill
          active={tab === "Latest comparison"}
          onClick={() => setTab("Latest comparison")}
        >
          Latest comparison
        </Pill>
        <Pill
          active={tab === "Missing runs"}
          onClick={() => setTab("Missing runs")}
        >
          Missing runs
        </Pill>
        <Pill active={tab === "Method"} onClick={() => setTab("Method")}>
          Selection method
        </Pill>
      </Row>

      {tab === "Latest comparison" && <LatestComparison />}
      {tab === "Method" && <Method />}
      {tab === "Missing runs" && <MissingRuns />}
    </Stack>
  );
}
