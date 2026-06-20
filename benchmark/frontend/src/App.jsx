import { Routes, Route } from "react-router-dom";
import Login from "./components/login";
import Signup from "./components/signup";
import Home from "./components/home";
import Benchmarks from "./components/benchmarks";
import CreateBenchmark from "./components/createBenchmark"
import CreateMeric from "./components/createMetric"
import Metrics from "./components/metrics"
import DisplayMetric from "./components/displayMetric"
import DisplayBenchmark from "./components/displayBenchmark"
import LeaderBoard from "./components/leaderBoard"
import Dashboard from "./components/dashboard"
import AdminDashboard from "./components/adminDashboard"
function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/admin-dashboard" element={<AdminDashboard />} />
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
