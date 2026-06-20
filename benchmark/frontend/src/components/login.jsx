import { useState } from "react";
import { useNavigate } from "react-router-dom";

function Login() {
  const [email, setEmail] = useState("");
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
        body: JSON.stringify({ email }),
      });

      const data = await res.json();

      if (!res.ok) {
        setMessage(data.message);
      } else {
        localStorage.setItem("user", JSON.stringify(data));

        setMessage("Login successful ✅");

        navigate("/");
      }
    } catch (err) {
      setMessage("Server error");
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>Login</h2>

      <form onSubmit={handleLogin}>
        <input
          type="email"
          placeholder="Enter email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <br /><br />

        <button type="submit">Login</button>
      </form>

      <p>{message}</p>
    </div>
  );
}

export default Login;
