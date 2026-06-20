import "./home.css";
import Navigation from "./navigation";
import { Link } from "react-router-dom";
function Home() {
  return (
    <div className="home-page">
      <Navigation />
      <div className="home-page-content">
        <div className="home-header">
          <img
            src="/Accelera.png"
            alt="Accelera Logo"
            className="home-header-logo"
          />
          <h1 className="home-header-title">Accelera Benchmark</h1>
          <p className="home-header-subtitle">
            Welcome to GP 2026 AI Benchmarking
          </p>
          <div className="home-header-actions">
            <Link to="/benchmarks">Browse Benchmarks</Link>
            <Link to="/metrics">View Metrics</Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;
