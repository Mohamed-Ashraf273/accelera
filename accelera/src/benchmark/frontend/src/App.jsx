import { Routes, Route } from "react-router-dom";
import Login from "./components/login";
import Home from "./components/home";
import Benchmarks from "./components/benchmarks";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/login" element={<Login />} />
      <Route path="/benchmarks" element={<Benchmarks />} />
    </Routes>
  );
}

export default App;
