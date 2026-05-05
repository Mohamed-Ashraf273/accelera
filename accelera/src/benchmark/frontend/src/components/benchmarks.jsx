import "./benchmarks.css";
import { useEffect, useState } from "react";

function Benchmarks() {
  const [benchmarks, setBenchmarks] = useState([]);
  const [problemType, setProblemType] = useState("");
  const [loading, setLoading] = useState(false);
  console.log(benchmarks);
  const user = JSON.parse(localStorage.getItem("user"));
  console.log(user);
  const fetchWithoutFilter = async () => {
    setLoading(true);
    const results = await fetch("http://localhost:3000/benchmark");
    const data = await results.json();
    setBenchmarks(data);
    setLoading(false);
  };

  const fetchByProblemType = async (type) => {
    if (!type) return fetchWithoutFilter();

    setLoading(true);
    const results = await fetch(
      `http://localhost:3000/benchmark/problem-type/${type}`,
    );
    const data = await results.json();
    setBenchmarks(data);
    setLoading(false);
  };

  const fetchByUser = async () => {
    if (!user) return;

    setLoading(true);
    const results = await fetch(
      `http://localhost:3000/benchmark/user/${user._id}`,
    );
    const data = await results.json();
    setBenchmarks(data);
    setLoading(false);
  };

  const deleteBenchmark = async (id) => {
    try {
      const results = await fetch(`http://localhost:3000/benchmark/${id}`, {
        method: "DELETE",
      });
      const data = await results.json();
      if (!results.ok) {
        alert(data.message);
        return;
      }
      setBenchmarks((prev) => prev.filter((b) => b._id !== id));
    } catch (err) {
      console.error("Delete error:", err);
    }
  };
  useEffect(() => {
    fetchWithoutFilter();
  }, []);

  return (
    <div className="benchmarks">
      <h2 className="benchmarks-title">Benchmarks</h2>
      <div className="filter-create">
        <div className="benchmarks-filters">
          <button className="benchmarks-btn" onClick={fetchWithoutFilter}>
            All
          </button>
          {user && (
            <button className="benchmarks-btn" onClick={fetchByUser}>
              My Benchmarks
            </button>
          )}
          <select
            className="benchmarks-select"
            value={problemType}
            onChange={(e) => {
              setProblemType(e.target.value);
              fetchByProblemType(e.target.value);
            }}
          >
            <option value="">Filter by type</option>
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
          </select>
        </div>
        <button className="benchmark-create-btn">Create New Benchmark</button>
      </div>
      {loading && <p className="loading">Loading...</p>}

      {benchmarks.map((b) => (
        <div key={b._id} className="benchmark-card">
          <div className="benchmark-header">
            <h3>{b.title}</h3>
            <button className="benchmark-show-btn">Show</button>
            {user && b.createdBy?._id === user._id && (
              <button
                className="benchmark-delete-btn"
                onClick={() => deleteBenchmark(b._id)}
              >
                Delete
              </button>
            )}
          </div>
          <div className="benchmark-info">
            <p>Type: {b.problemType}</p>
            <p>Metric: {b.evaluationMetric?.name}</p>
            <p>Created by: {b.createdBy?.name}</p>
            <p>Created at: {b.creationDate}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

export default Benchmarks;
