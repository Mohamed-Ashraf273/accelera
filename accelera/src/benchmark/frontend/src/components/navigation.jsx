import "./navigation.css";
import { Link } from "react-router-dom";

function Navigation() {
  const user = JSON.parse(localStorage.getItem("user"));
  return (
    <div className="navigation">
      <div className="navigation-logos">
        <img src="/Accelera.png" alt="logo" className="navigation-logo" />
        <span className="navigation-title">Accelera</span>
      </div>

      <div className="nvigation-links">
        <Link to="/" className="nvigation-link">
          Home
        </Link>
        <Link to="/benchmarks" className="nvigation-link">
          Benchmarks
        </Link>

          <Link to="/metrics" className="nvigation-link">
            Metrics
          </Link>
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
  );
}

export default Navigation;
