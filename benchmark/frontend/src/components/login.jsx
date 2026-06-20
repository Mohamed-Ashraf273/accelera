import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Navigation from "./navigation";
import "./auth.css";

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");

  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();

    try {
      const res = await fetch("http://localhost:3000/user/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await res.json();

      if (!res.ok) {
        setMessage(data.message);
      } else {
        localStorage.setItem("user", JSON.stringify(data.user));
        localStorage.setItem("token", data.token);

        setMessage("Login successful ✅");

        navigate("/");
      }
    } catch (err) {
      setMessage("Server error");
    }
  };

  return (
    <div className="auth-page">
      <Navigation />
      <div className="auth-page-content">
        <div className="auth-head">
          <p>Welcome back</p>
          <h2>Login</h2>
        </div>

        <form onSubmit={handleLogin} className="auth-form">
          <div className="auth-input">
            <p>Email</p>
            <input
              type="email"
              placeholder="Enter email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          <div className="auth-input">
            <p>Password</p>
            <input
              type="password"
              placeholder="Enter password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <button type="submit" className="auth-button">Login</button>
        </form>

        <p className="auth-message">{message}</p>
        <Link to="/signup" className="auth-link">Create new account</Link>
      </div>
    </div>
  );
}

export default Login;
