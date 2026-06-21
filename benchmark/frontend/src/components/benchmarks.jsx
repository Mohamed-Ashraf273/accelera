import "./benchmarks.css";
import Navigation from "./navigation";
import { Link } from "react-router-dom";

import { useEffect, useState } from "react";
import { authHeaders } from "../auth";

function Benchmarks() {
  const [benchmarks, setBenchmarks] = useState([]);
  const [problemType, setProblemType] = useState("");
  const [loading, setLoading] = useState(false);
  const user = JSON.parse(localStorage.getItem("user"));
  const fetchBenchmark = async (urlLink) => {
    try {
      setLoading(true);
      const results = await fetch(urlLink);
      const data = await results.json();
      setBenchmarks(Array.isArray(data) ? data : []);
      setLoading(false);
    }
    catch (err) {
      setBenchmarks([])
      setLoading(false);
      console.log(err)
    }
  };
  const fetchWithoutFilter = async () => {
    fetchBenchmark("http://localhost:3000/benchmark");
  };

  const fetchByProblemType = async (type) => {
    if (!type) return fetchWithoutFilter();
    fetchBenchmark(`http://localhost:3000/benchmark/problem-type/${type}`);
  };

  const fetchByUser = async () => {
    if (!user) return;
    fetchBenchmark(`http://localhost:3000/benchmark/user/${user._id}`);
  };

  const deleteBenchmark = async (id) => {
    try {
      const results = await fetch(`http://localhost:3000/benchmark/${id}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      const data = await results.json();
      if (!results.ok) {
        alert(data.message);
        return;
      }
      setBenchmarks((prevBenchmarks) => prevBenchmarks.filter((benchmark) => benchmark._id !== id));
    } catch (err) {
      console.error("Delete error:", err);
    }
  };
  useEffect(() => {
    fetchWithoutFilter();
  }, []);

  return (
    <div className="benchmarks-page">
      <Navigation />
      <div className="benchmark-page-content">
        <div className="benchmarks-page-head">
          <div>
            <p className="benchmarks-kicker">Benchmark library</p>
            <h2 className="benchmarks-page-title">Benchmarks</h2>
          </div>
          <p className="benchmarks-count">{benchmarks.length} benchmarks</p>
        </div>
        <div className="benchmarks-page-actions">
          <div className="benchmarks-filters">
            <button className="benchmarks-button" onClick={fetchWithoutFilter}>
              All
            </button>
            {user && (
              <button className="benchmarks-button" onClick={fetchByUser}>
                My Benchmarks
              </button>
            )}
            <select
              className="benchmarks-problemType-select"
              value={problemType}
              onChange={(e) => {
                setProblemType(e.target.value);
                fetchByProblemType(e.target.value);
              }}
            >
              <option value="">All</option>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>
          <Link className="benchmark-create-button" to="/create-benchmarks">
            Create New Benchmark
          </Link>
        </div>
        {loading && <p className="loading">Loading...</p>}
        {!loading && benchmarks.length === 0 && (
          <p className="benchmarks-empty">No benchmarks found</p>
        )}
        <div className="benchmarks-display">
        {benchmarks.map((benchmark) => (
          <Link to="/display-benchmark" state={{benchmark}} key={benchmark._id} className="benchmark-card">
            <div className="benchmark-header">
              <h3>{benchmark.title}</h3>
              {user && (
                benchmark.createdBy?._id === user._id ||
                user.role === "admin"
              ) && (
                <button
                  className="benchmark-delete-button"
                  onClick={() => deleteBenchmark(benchmark._id)}
                >
                  ❌
                </button>
              )}
            </div>
            <p className="benchmark-description">{benchmark.description}</p>
            <div className="benchmark-info">
              <p>{benchmark.problemType}</p>
              <p>{benchmark.evaluationMetric?.name}</p>
            </div>
          </Link>
        ))}
        </div>
      </div>
    </div>
  );
}

export default Benchmarks;
