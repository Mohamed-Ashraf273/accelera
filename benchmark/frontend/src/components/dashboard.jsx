import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getUser, logout } from "../auth";
import Navigation from "./navigation";
import "./dashboard.css";

function Dashboard() {
  const user = getUser();
  const [benchmarks, setBenchmarks] = useState([]);
  const [submissions, setSubmissions] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchData = async () => {
    if (!user) return;
    setLoading(true);
    const benchmarkResults = await fetch(
      `http://localhost:3000/benchmark/user/${user._id}`,
    );
    const benchmarkData = await benchmarkResults.json();
    setBenchmarks(benchmarkData);

    const submissionResults = await fetch(
      `http://localhost:3000/submission/user/${user._id}`,
    );
    const submissionData = await submissionResults.json();
    setSubmissions(submissionData);
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (!user) {
    return (
      <div className="dashboard-page">
        <Navigation />
        <div className="dashboard-content">
          <div className="dashboard-header">
            <div className="dashboard-title">
              <h2>You need to login first</h2>
              <p>Login to see your benchmarks and submissions</p>
            </div>
            <Link className="dashboard-button" to="/login">Login</Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-page">
      <Navigation />
      <div className="dashboard-content">
        <div className="dashboard-header">
          <div className="dashboard-title">
            <h2>Dashboard</h2>
            <p>Welcome back, {user.name}</p>
            <div className="dashboard-info">
              <p>{user.email}</p>
              <p>{user.role}</p>
            </div>
          </div>
          <Link className="dashboard-button" to="/" onClick={logout}>Logout</Link>
        </div>

        {loading && <p className="dashboard-message">Loading...</p>}

        <div className="dashboard-section">
          <div className="dashboard-section-header">
            <h3>My Benchmarks</h3>
          </div>
          {benchmarks.length === 0 && !loading && (
            <p className="dashboard-empty">No benchmarks created yet</p>
          )}
          <div className="dashboard-grid">
            {benchmarks.map((benchmark) => (
              <div className="dashboard-card" key={benchmark._id}>
                <h4 className="dashboard-card-title">{benchmark.title}</h4>
                <p>{benchmark.problemType}</p>
                <p>{benchmark.description}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="dashboard-section">
          <div className="dashboard-section-header">
            <h3>My Submissions</h3>
          </div>
          {submissions.length === 0 && !loading && (
            <p className="dashboard-empty">No submissions added yet</p>
          )}
          <div className="dashboard-grid">
            {submissions.map((submission) => (
              <div className="dashboard-card" key={submission._id}>
                <h4 className="dashboard-card-title">
                  {submission.benchmarkId?.title || "Benchmark"}
                </h4>
                <p>Score: {submission.score}</p>
                <p>Date: {submission.submissionDate}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
