import "./home.css";
import { Link } from "react-router-dom";

function Home() {
  const user = JSON.parse(localStorage.getItem("user"));
  return (
    <div className="home">
      <div className="navigation">
        <div className="navigation-logos">
          <img src="/Accelera.png" alt="logo" className="navigation-logo" />
          <span className="navigation-title">Accelera</span>
        </div>

        <div className="nvigation-links">
          <Link to="/benchmarks" className="nvigation-link">
            Benchmarks
          </Link>

          {user?.role === "admin" && (
            <Link to="/login" className="nvigation-link">
              Metrics
            </Link>
          )}
          {!user ? (
            <Link to="/login" className="nvigation-link">
              Login
            </Link>
          ) : (
            <>
              <Link to="/login" className="nvigation-link">
                Dashboard
              </Link>
              <Link to="/" className="nvigation-link">
                Logout
              </Link>
            </>
          )}
        </div>
      </div>

      <div className="home-header">
        <img src="/Accelera.png" alt="Accelera Logo" className="home-logo" />
        <h1 className="home-title">Accelera Benchmark</h1>
        <p className="home-subtitle">Welcome to GP 2026 AI Benchmarking</p>
      </div>
    </div>
  );
}

export default Home;
