import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Navigation from "./navigation";
import "./auth.css";

function Signup() {
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
  });
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch("http://localhost:3000/user/signup", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(form),
      });

      const data = await res.json();
      if (!res.ok) {
        setMessage(data.message);
        return;
      }
      setMessage("Signup successful");
      navigate("/login");
    } catch (err) {
      setMessage("Server error");
    }
  };

  return (
    <div className="auth-page">
      <Navigation />
      <div className="auth-page-content">
        <div className="auth-head">
          <p>Create account</p>
          <h2>Signup</h2>
        </div>

        <form onSubmit={handleSignup} className="auth-form">
          <div className="auth-input">
            <p>Name</p>
            <input
              name="name"
              placeholder="Enter name"
              value={form.name}
              onChange={handleChange}
            />
          </div>

          <div className="auth-input">
            <p>Email</p>
            <input
              type="email"
              name="email"
              placeholder="Enter email"
              value={form.email}
              onChange={handleChange}
            />
          </div>

          <div className="auth-input">
            <p>Password</p>
            <input
              type="password"
              name="password"
              placeholder="Enter password"
              value={form.password}
              onChange={handleChange}
            />
          </div>

          <button type="submit" className="auth-button">Signup</button>
        </form>

        <p className="auth-message">{message}</p>
        <Link to="/login" className="auth-link">Login</Link>
      </div>
    </div>
  );
}

export default Signup;
