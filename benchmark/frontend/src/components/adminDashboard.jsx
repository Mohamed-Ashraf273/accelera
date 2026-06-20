import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { authHeaders, getUser } from "../auth";
import Navigation from "./navigation";
import "./dashboard.css";

function AdminDashboard() {
  const user = getUser();
  const [users, setUsers] = useState([]);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const fetchUsers = async () => {
    setLoading(true);
    const results = await fetch("http://localhost:3000/user", {
      headers: authHeaders(),
    });
    const data = await results.json();
    if (!results.ok) {
      setMessage(data.message);
      setLoading(false);
      return;
    }
    setUsers(data);
    setLoading(false);
  };

  const deleteUser = async (id) => {
    const results = await fetch(`http://localhost:3000/user/${id}`, {
      method: "DELETE",
      headers: authHeaders(),
    });
    const data = await results.json();
    if (!results.ok) {
      setMessage(data.message);
      return;
    }
    setUsers((prev) => prev.filter((user) => user._id !== id));
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  if (!user) {
    return (
      <div className="dashboard-page">
        <Navigation />
        <div className="dashboard-content">
          <div className="dashboard-header">
            <div className="dashboard-title">
              <h2>You need to login first</h2>
              <p>Login with an admin account to manage users</p>
            </div>
            <Link className="dashboard-button" to="/login">Login</Link>
          </div>
        </div>
      </div>
    );
  }

  if (user.role !== "admin") {
    return (
      <div className="dashboard-page">
        <Navigation />
        <div className="dashboard-content">
          <div className="dashboard-header">
            <div className="dashboard-title">
              <h2>Admin access is required</h2>
              <p>Your account does not have permission to manage users</p>
            </div>
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
            <h2>Admin Dashboard</h2>
            <p>Manage users and keep benchmark access controlled</p>
            <div className="dashboard-info">
              <p>{user.email}</p>
              <p>{user.role}</p>
            </div>
          </div>
        </div>

        {message && <p className="dashboard-message">{message}</p>}
        {loading && <p className="dashboard-message">Loading...</p>}

        <div className="dashboard-section">
          <div className="dashboard-section-header">
            <h3>Users</h3>
          </div>
          {users.length === 0 && !loading && (
            <p className="dashboard-empty">No users found</p>
          )}
          {users.map((item) => (
            <div className="dashboard-user-row" key={item._id}>
              <div className="dashboard-user-info">
                <h4>{item.name}</h4>
                <p>{item.email}</p>
                <p>{item.role}</p>
              </div>
              {item._id !== user._id && (
                <button
                  className="dashboard-delete-button"
                  onClick={() => deleteUser(item._id)}
                >
                  Delete User
                </button>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default AdminDashboard;
