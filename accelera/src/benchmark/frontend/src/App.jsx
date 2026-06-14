import { Routes, Route } from "react-router-dom";
import Login from "./components/login";
import Home from "./components/home";
import Benchmarks from "./components/benchmarks";
import CreateBenchmark from "./components/createBenchmark"
import CreateMeric from "./components/createMetric"
import Metrics from "./components/metrics"
import DisplayMetric from "./components/displayMetric"
import DisplayBenchmark from "./components/displayBenchmark"
import LeaderBoard from "./components/leaderBoard"
function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/login" element={<Login />} />
      <Route path="/benchmarks" element={<Benchmarks />} />
      <Route path="/benchmarks" element={<Benchmarks />} />
      <Route path="/create-benchmarks" element={<CreateBenchmark />} />
      <Route path="/create-metric" element={<CreateMeric />} />
      <Route path="/metrics" element={<Metrics />} />
      <Route path="/display-metric" element={<DisplayMetric />} />
      <Route path="/display-benchmark" element={<DisplayBenchmark />} />
      <Route path="/leader-board" element={<LeaderBoard />} />
    </Routes>
  );
}

export default App;
