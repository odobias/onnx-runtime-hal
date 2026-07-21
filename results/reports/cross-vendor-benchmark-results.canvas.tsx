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
  power: string;
  arch: string;
  latencyMs: number;
  accuracyPct: number | null;
  werPct: number | null;
  modelSha: string;
  date: string;
};

const data: {
  latencyRows: ResultRow[];
  accuracyRows: ResultRow[];
} = {"latencyRows":[{"workload":"whisper-tiny-dynamic","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"arm64","latencyMs":2067.296,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"arm64","latencyMs":526.763,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"arm64","latencyMs":1483.516,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"arm64","latencyMs":1423.974,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"arm64","latencyMs":12604.984,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"arm64","latencyMs":2552.151,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"arm64","latencyMs":1283.427,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"arm64","latencyMs":1252.788,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"bundled","power":"battery","arch":"arm64","latencyMs":360.002,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"winml","power":"battery","arch":"arm64","latencyMs":449.94,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"bundled","power":"ac","arch":"arm64","latencyMs":359.847,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"arm64","latencyMs":778.037,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"arm64","latencyMs":687.953,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"arm64","latencyMs":1202.186,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"arm64","latencyMs":331.455,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"winml","power":"ac","arch":"arm64","latencyMs":376.708,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"arm64","latencyMs":774.181,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"arm64","latencyMs":680.239,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"arm64","latencyMs":871.55,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"arm64","latencyMs":200.051,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + VitisAI","executorVendor":"AMD","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"bundled","power":"ac","arch":"x64","latencyMs":642.405,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"x64","latencyMs":570.147,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"x64","latencyMs":379.185,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"x64","latencyMs":1033.702,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"x64","latencyMs":264.252,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"winml","power":"ac","arch":"x64","latencyMs":608.95,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"x64","latencyMs":577.57,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"x64","latencyMs":391.891,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"x64","latencyMs":1327.485,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"x64","latencyMs":220.838,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + VitisAI","executorVendor":"AMD","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"bundled","power":"battery","arch":"x64","latencyMs":753.332,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"bundled","power":"ac","arch":"x64","latencyMs":3063.778,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"x64","latencyMs":1313.284,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"x64","latencyMs":597.96,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"x64","latencyMs":3243.07,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"x64","latencyMs":1878.888,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"winml","power":"ac","arch":"x64","latencyMs":372.069,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"x64","latencyMs":848.656,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"x64","latencyMs":537.411,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"x64","latencyMs":2151.351,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"x64","latencyMs":329.598,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":818.287,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":765.098,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":862.134,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":248.721,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"winml","power":"battery","arch":"x64","latencyMs":1088.666,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":772.803,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":688.362,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"bundled","power":"battery","arch":"x64","latencyMs":544.418,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":1187.497,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":254.127,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":676.179,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":612.3,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":2337.49,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":1671.752,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"winml","power":"battery","arch":"x64","latencyMs":292.455,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":658.336,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":583.245,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":932.242,"accuracyPct":null,"werPct":null,"modelSha":"9da9f440","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":218.299,"accuracyPct":null,"werPct":null,"modelSha":"12534e24","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"arm64","latencyMs":1080.86,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"arm64","latencyMs":158.894,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"arm64","latencyMs":74.964,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"arm64","latencyMs":70.02,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"bundled","power":"battery","arch":"arm64","latencyMs":129.947,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"winml","power":"battery","arch":"arm64","latencyMs":131.992,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"bundled","power":"battery","arch":"arm64","latencyMs":116.6,"accuracyPct":null,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"winml","power":"battery","arch":"arm64","latencyMs":110.402,"accuracyPct":null,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-16"},{"workload":"tsc","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"arm64","latencyMs":1560.0,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"arm64","latencyMs":329.023,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"arm64","latencyMs":111.795,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"arm64","latencyMs":91.458,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"bundled","power":"battery","arch":"arm64","latencyMs":20.114,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"winml","power":"battery","arch":"arm64","latencyMs":20.381,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"bundled","power":"ac","arch":"arm64","latencyMs":18.497,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"bundled","power":"ac","arch":"arm64","latencyMs":114.99,"accuracyPct":null,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"arm64","latencyMs":83.978,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"arm64","latencyMs":46.206,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"tsc","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"arm64","latencyMs":231.859,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"arm64","latencyMs":158.167,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"winml","power":"ac","arch":"arm64","latencyMs":18.384,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"winml","power":"ac","arch":"arm64","latencyMs":110.329,"accuracyPct":null,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"arm64","latencyMs":84.226,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"arm64","latencyMs":45.755,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"arm64","latencyMs":179.376,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"arm64","latencyMs":94.156,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"bundled","power":"ac","arch":"arm64","latencyMs":128.131,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"winml","power":"ac","arch":"arm64","latencyMs":128.092,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + VitisAI","executorVendor":"AMD","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"bundled","power":"ac","arch":"x64","latencyMs":33.268,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"x64","latencyMs":63.554,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"x64","latencyMs":38.026,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"tsc","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"x64","latencyMs":158.974,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"x64","latencyMs":79.58,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"x64","latencyMs":74.922,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"x64","latencyMs":85.726,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"x64","latencyMs":151.863,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"x64","latencyMs":91.732,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + VitisAI","executorVendor":"AMD","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"bundled","power":"ac","arch":"x64","latencyMs":41.698,"accuracyPct":null,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"winml","power":"ac","arch":"x64","latencyMs":33.608,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"winml","power":"ac","arch":"x64","latencyMs":48.706,"accuracyPct":null,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"bundled","power":"ac","arch":"x64","latencyMs":29.68,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"bundled","power":"ac","arch":"x64","latencyMs":28.095,"accuracyPct":null,"werPct":null,"modelSha":"464a53e6","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + VitisAI","executorVendor":"AMD","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"bundled","power":"battery","arch":"x64","latencyMs":33.804,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"x64","latencyMs":58.533,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"x64","latencyMs":56.257,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"tsc","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"x64","latencyMs":387.83,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"x64","latencyMs":165.376,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"winml","power":"ac","arch":"x64","latencyMs":19.335,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"winml","power":"ac","arch":"x64","latencyMs":33.421,"accuracyPct":null,"werPct":null,"modelSha":"464a53e6","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"x64","latencyMs":70.745,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"x64","latencyMs":61.441,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + VitisAI","executorVendor":"AMD","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"bundled","power":"battery","arch":"x64","latencyMs":62.238,"accuracyPct":null,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":60.147,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":73.528,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"x64","latencyMs":304.462,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"tsc","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":112.377,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"x64","latencyMs":147.984,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":70.781,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"winml","power":"battery","arch":"x64","latencyMs":44.572,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"winml","power":"battery","arch":"x64","latencyMs":82.617,"accuracyPct":null,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":66.143,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":87.153,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"bundled","power":"battery","arch":"x64","latencyMs":8.153,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"bundled","power":"battery","arch":"x64","latencyMs":23.154,"accuracyPct":null,"werPct":null,"modelSha":"464a53e6","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":149.345,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":40.183,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":101.308,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":76.476,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"tsc","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":243.904,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":126.114,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"winml","power":"battery","arch":"x64","latencyMs":8.813,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"winml","power":"battery","arch":"x64","latencyMs":23.069,"accuracyPct":null,"werPct":null,"modelSha":"464a53e6","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":38.955,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":72.399,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":197.453,"accuracyPct":null,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":115.152,"accuracyPct":null,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"}],"accuracyRows":[{"workload":"whisper-tiny-static","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"bundled","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"bundled","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"bundled","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"winml","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"winml","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"winml","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"arm64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + VitisAI","executorVendor":"AMD","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":10.231023102310232,"modelSha":"dee360f3","date":"2026-07-14"},{"workload":"tsc","executor":"ORT + VitisAI","executorVendor":"AMD","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-14"},{"workload":"fakeaudio","executor":"ORT + VitisAI","executorVendor":"AMD","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-15"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":9.570957095709572,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"464a53e6","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":10.231023102310232,"modelSha":"dee360f3","date":"2026-07-15"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-15"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-15"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-15"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-15"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-15"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-15"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-15"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-15"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-15"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"winml","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-15"},{"workload":"whisper-tiny-static","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":9.570957095709572,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"464a53e6","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"GPU","provider":"OpenVINO","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-15"},{"workload":"whisper-tiny-dynamic","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"GPU","provider":"OpenVINO","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-15"},{"workload":"tsc","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"GPU","provider":"OpenVINO","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-15"},{"workload":"whisper-tiny-static","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"CPU","provider":"OpenVINO","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-15"},{"workload":"whisper-tiny-dynamic","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"CPU","provider":"OpenVINO","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-15"},{"workload":"tsc","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"CPU","provider":"OpenVINO","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-15"},{"workload":"fakeaudio","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"CPU","provider":"OpenVINO","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-15"},{"workload":"whisper-tiny-static","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-15"},{"workload":"whisper-tiny-dynamic","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-15"},{"workload":"tsc","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-15"},{"workload":"fakeaudio","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"AMD","device":"GPU","provider":"DirectML","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-15"},{"workload":"whisper-tiny-static","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-15"},{"workload":"whisper-tiny-dynamic","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-15"},{"workload":"tsc","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-15"},{"workload":"fakeaudio","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"AMD","device":"CPU","provider":"CPU EP","target":"bundled","power":"battery","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-15"},{"workload":"whisper-tiny-static","executor":"ORT + VitisAI","executorVendor":"AMD","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"bundled","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":10.231023102310232,"modelSha":"dee360f3","date":"2026-07-15"},{"workload":"tsc","executor":"ORT + VitisAI","executorVendor":"AMD","siliconVendor":"AMD","device":"NPU","provider":"VitisAI","target":"bundled","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-15"},{"workload":"whisper-tiny-static","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"bundled","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":9.570957095709572,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"bundled","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + OpenVINO","executorVendor":"Intel","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"bundled","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"464a53e6","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"winml","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":9.570957095709572,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"winml","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"NPU","provider":"OpenVINO","target":"winml","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"464a53e6","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Intel","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"x64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"bundled","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"bundled","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + QNN","executorVendor":"Qualcomm","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"bundled","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ORT + DirectML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"bundled","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"ONNX Runtime","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"bundled","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"winml","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"winml","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"NPU","provider":"QNN","target":"winml","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"e4623e7e","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"GPU","provider":"DirectML","target":"winml","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"},{"workload":"whisper-tiny-static","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"dee360f3","date":"2026-07-16"},{"workload":"whisper-tiny-dynamic","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":null,"werPct":8.58085808580858,"modelSha":"cb0a9671","date":"2026-07-16"},{"workload":"tsc","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":93.333333,"werPct":null,"modelSha":"90f2ed1b","date":"2026-07-16"},{"workload":"fakeaudio","executor":"Windows ML","executorVendor":"Microsoft","siliconVendor":"Qualcomm","device":"CPU","provider":"CPU EP","target":"winml","power":"ac","arch":"arm64","latencyMs":0,"accuracyPct":60.0,"werPct":null,"modelSha":"549143bc","date":"2026-07-16"}]};

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
  return `${row.siliconVendor} · ${row.provider} · ${row.device} · ${row.target} · ${row.power}`;
}

function cohortKey(row: ResultRow, includePower = true) {
  const parts = [
    row.workload,
    row.siliconVendor,
    row.provider,
    row.device,
    row.target,
  ];
  if (includePower) parts.push(row.power);
  return parts.join("|");
}

function hashHue(value: string, hues: string[]) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 33 + value.charCodeAt(index)) >>> 0;
  }
  return hues[hash % hues.length];
}

type CohortPartKey = "silicon" | "provider" | "device" | "target" | "power";
type SortKey = CohortPartKey | "quality" | "latency";

function CohortParts({
  siliconVendor,
  provider,
  device,
  target,
  power,
  activePart,
  sortDir,
  onSortPart,
}: {
  siliconVendor: string;
  provider: string;
  device: string;
  target: string;
  power: string;
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
  const powerColors: Record<string, string> = {
    battery: theme.category.yellow,
    ac: theme.category.green,
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
    {
      key: "power",
      value: power,
      color: powerColors[power] ?? hashHue(power, fallback),
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

  const latencyRows = data.latencyRows.filter(
    (row) => row.workload === workload
  );
  const accuracyByPower = new Map(
    data.accuracyRows
      .filter((row) => row.workload === workload)
      .map((row) => [cohortKey(row, true), row])
  );
  const accuracyBySoft = new Map(
    data.accuracyRows
      .filter((row) => row.workload === workload)
      .map((row) => [cohortKey(row, false), row])
  );

  const combined = latencyRows.map((latency) => {
    const accuracy =
      accuracyByPower.get(cohortKey(latency, true)) ??
      accuracyBySoft.get(cohortKey(latency, false));
    return {
      label: configLabel(latency),
      siliconVendor: latency.siliconVendor,
      provider: latency.provider,
      device: latency.device,
      target: latency.target,
      power: latency.power,
      executor: latency.executor,
      arch: latency.arch,
      latencyMs: latency.latencyMs,
      accuracyPct: isClassifier
        ? latency.accuracyPct ?? accuracy?.accuracyPct ?? null
        : accuracy?.accuracyPct ?? null,
      werPct: isClassifier ? null : accuracy?.werPct ?? null,
      latencyModel: latency.modelSha,
      accuracyModel: accuracy?.modelSha ?? null,
      exactModelPair:
        !!accuracy &&
        latency.modelSha === accuracy.modelSha &&
        latency.power === accuracy.power,
    };
  });

  const maxLatency = Math.max(...combined.map((row) => row.latencyMs), 1);
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
    if (key === "target") return row.target;
    return row.power;
  };

  const sorted = [...combined].sort((left, right) => {
    const direction = sortDir === "asc" ? 1 : -1;
    if (
      sortKey === "silicon" ||
      sortKey === "provider" ||
      sortKey === "device" ||
      sortKey === "target" ||
      sortKey === "power"
    ) {
      const primary =
        partValue(left, sortKey).localeCompare(partValue(right, sortKey)) *
        direction;
      if (primary !== 0) return primary;
      if (sortKey !== "power") {
        const powerCmp = left.power.localeCompare(right.power);
        if (powerCmp !== 0) return powerCmp;
      }
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
    sortKey === "target" ||
    sortKey === "power"
      ? sortKey
      : null;

  const sortBucket = (row: (typeof combined)[number]) => {
    if (
      sortKey === "silicon" ||
      sortKey === "provider" ||
      sortKey === "device" ||
      sortKey === "target" ||
      sortKey === "power"
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

  const gridColumns = "minmax(280px, 34%) minmax(110px, 12%) 1fr 72px";

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
          Battery and AC are separate bars on the same chart. Click a cohort chip
          to sort by that part; click power to group AC vs battery.
        </Text>

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
            power="power"
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
            ms
          </Text>
        </div>

        {sorted.map((row, index) => {
          const widthPct = Math.max(2, (row.latencyMs / maxLatency) * 100);
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
                <CohortParts
                  siliconVendor={row.siliconVendor}
                  provider={row.provider}
                  device={row.device}
                  target={row.target}
                  power={row.power}
                  activePart={activeCohortPart}
                  sortDir={sortDir}
                  onSortPart={toggleSort}
                />
                <Text
                  size="small"
                  weight="semibold"
                  style={{ color: qualityColor, lineHeight: 1.25 }}
                >
                  {qualityText}
                </Text>
                <div
                  style={{
                    height: 10,
                    background: theme.fill.tertiary,
                    borderRadius: 3,
                    overflow: "hidden",
                    minWidth: 0,
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${widthPct}%`,
                      background:
                        !isClassifier && werDelta != null && werDelta > 1e-9
                          ? theme.category.red
                          : theme.accent.primary,
                      borderRadius: 3,
                    }}
                  />
                </div>
                <Text
                  size="small"
                  weight="semibold"
                  style={{ textAlign: "right", lineHeight: 1.25 }}
                >
                  {format(row.latencyMs)}
                </Text>
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
