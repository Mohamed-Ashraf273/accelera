import "./metrics.css";
import Navigation from "./navigation";
import { Link } from "react-router-dom";

import { useEffect, useState } from "react";
import { authHeaders } from "../auth";

function Metrics() {
  const [metrics, setMetrics] = useState([]);
  const [problemType, setProblemType] = useState("");
  const [loading, setLoading] = useState(false);
  const user = JSON.parse(localStorage.getItem("user"));
  const fetchMetric = async (urlLink) => {
    try {
      setLoading(true);
      const results = await fetch(urlLink);
      const data = await results.json();
      setMetrics(Array.isArray(data) ? data : []);
      setLoading(false);
    }
    catch (err) {
      setMetrics([])
      setLoading(false);
      console.log(err)
    }
  };
  const fetchWithoutFilter = async () => {
    fetchMetric("http://localhost:3000/metrics");
  };

  const fetchByProblemType = async (type) => {
    if (!type) return fetchWithoutFilter();
    fetchMetric(`http://localhost:3000/metrics/problem-type/${type}`);
  };

  const deleteMetric = async (id) => {
    try {
      const results = await fetch(`http://localhost:3000/metrics/${id}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      const data = await results.json();
      if (!results.ok) {
        alert(data.message);
        return;
      }
      setMetrics((prev) => prev.filter((metric) => metric._id !== id));
    } catch (err) {
      console.error("Delete error:", err);
      alert(err.message);
    }
  };
  useEffect(() => {
    fetchWithoutFilter();
  }, []);

  return (
    <div className="metrics-page">
      <Navigation />
      <div className="metrics-page-content">
        <div className="metrics-page-head">
          <div>
            <p className="metrics-kicker">Evaluation setup</p>
            <h2 className="metrics-page-title">Metrics</h2>
          </div>
          <p className="metrics-count">{metrics?.length} metrics</p>
        </div>
        <div className="metrics-page-actions">
          <div className="metrics-filters">
            <select
              className="metrics-problemType-select"
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
          {user?.role === "admin" && (
            <Link className="metric-create-button" to="/create-metric">
              Create New metric
            </Link>
          )}
        </div>
        {loading && <p className="loading">Loading...</p>}
        {!loading && metrics?.length === 0 && (
          <p className="metrics-empty">No metrics found</p>
        )}
        <div className="metrics-display">
          {metrics?.map((metric) => (
            <Link to="/display-metric" state={{metric}} key={metric._id} className="metric-card" >
              <div className="metric-header">
                <h3>{metric.name}</h3>
                {user && user.role === "admin" && (
                  <button
                    className="metric-delete-button"
                    onClick={() => deleteMetric(metric._id)}
                  >
                    ❌
                  </button>
                )}
              </div>
              <div className="metric-info">
                <p>{metric.problemType}</p>
                <p>{metric.whichBetter}</p>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}

export default Metrics;
