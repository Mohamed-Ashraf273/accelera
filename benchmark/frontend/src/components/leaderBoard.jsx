import "./leaderBoard.css";
import Navigation from "./navigation";
import { useLocation } from "react-router-dom";
import { useState, useEffect } from "react";
import { authHeaders, getUser } from "../auth";
function LeaderBoard() {
  const location = useLocation();
  const [loading, setLoading] = useState(false);
  const [submissions, setSubmissions] = useState([]);
  const [action, setAction] = useState("display");
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState(null);
  const [updatedSubmission, setUpdatedSubmission] = useState(null);
  const benchmark_id = location.state;
  const user = getUser();
  const [form, setForm] = useState({
    repoLink: "",
    predictedColumnLink: "",
  });
  const [updateForm, setUpdateForm] = useState({
    predictedColumnLink: "",
  });
  const handleUpdate = (id) => {
    setAction("update");
      setUpdatedSubmission(id);
  };
  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };
  const handleChangeUpdate = (e) => {
    setUpdateForm({ ...updateForm, [e.target.name]: e.target.value });
  };
  const fetchSubmissions = async () => {
    try {
      setLoading(true);
      const results = await fetch(
        `http://localhost:3000/submission/benchmark/${benchmark_id}`,
      );
      const data = await results.json();
      setSubmissions(Array.isArray(data) ? data : []);
      setLoading(false);
    } catch (err) {
      setSubmissions([])
      setLoading(false)
      console.log(err)
    }
  };
  const handleSubmit = async (e) => {
    setUploading(true);
    setMessage(null);
    e.preventDefault();
    if (!form.repoLink) {
      setMessage("repoLink must be exist");
      return;
    }
    if (!form.predictedColumnLink) {
      setMessage("Predicted Column Link must be exist");
      return;
    }
    try {
      const response = await fetch(
        `http://localhost:3000/submission/${benchmark_id}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...authHeaders(),
          },
          body: JSON.stringify(form),
        },
      );

      const data = await response.json();
      if (!response.ok) {
        setMessage(data.message);
        setUploading(false);

        return;
      }
      setMessage("submitted sucessfully");
      setUploading(false);
    } catch (err) {
      console.error(err.message);
      alert(err.message);
      setUploading(false);
    }
  };
  const handleSubmitUpdate = async (e) => {
    setUploading(true);
    setMessage(null);
      e.preventDefault();
    console.log(updatedSubmission)
    if (!updateForm.predictedColumnLink) {
      setMessage("Predicted Column Link must be exist");
      return;
    }
    try {
      const response = await fetch(
        `http://localhost:3000/submission/${updatedSubmission}`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
            ...authHeaders(),
          },
          body: JSON.stringify(updateForm),
        },
      );

      const data = await response.json();
      if (!response.ok) {
        setMessage(data.message);
        setUploading(false);

        return;
      }
      setMessage("updatted sucessfully");
      setUploading(false);
    } catch (err) {
      console.error(err.message);
      alert(err.message);
      setUploading(false);
    }
  };
  const deleteSubmission = async (id) => {
    try {
      const results = await fetch(`http://localhost:3000/submission/${id}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      const data = await results.json();
      if (!results.ok) {
        alert(data.message);
        return;
      }
      setSubmissions((prevSubmissions) =>
        prevSubmissions.filter((submission) => submission._id !== id),
      );
    } catch (err) {
      console.error("Delete error:", err);
    }
  };
  useEffect(() => {
    fetchSubmissions();
  }, []);

  if (!benchmark_id)
    return (
      <h1 className="leader-board-page">No Benchmark Information exists</h1>
    );
  if (loading) return <p className="leader-board-page">Loading ...</p>;
  return (
    <div className="leader-board-page">
      <Navigation />
      {action === "display" && (
        <div className="leader-board-content">
          <div className="leader-board-header">
            <div>
              <p className="leader-board-kicker">Competition results</p>
              <h1>Submissions</h1>
            </div>
            <p className="leader-board-count">{submissions.length} submissions</p>
            {user && (
              <button className="add-button" onClick={() => setAction("Add")}>
                Add Submission
              </button>
            )}
          </div>
          <div className="submissions">
            {submissions.length === 0 && (
              <p className="leader-board-empty">No submissions yet</p>
            )}
            {submissions.map((submission) => (
              <div className="submission-card" key={submission._id}>
                <div className="submission">
                  <a
                    href={submission.repoLink}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    🔗Repository
                  </a>
                  <p>Score: {submission.score}</p>
                  <p>Submission Date: {submission.submissionDate}</p>
                  <p>Submitted By: {submission.submittedBy.name}</p>
                </div>
                {user && (
                  submission.submittedBy._id === user._id ||
                  user.role === "admin"
                ) && (
                  <div className="submission-actions">
                    <button onClick={() => deleteSubmission(submission._id)}>
                      ❌
                    </button>
                    <button onClick={() => handleUpdate(submission._id)}>
                      Update
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      {action === "Add" && (
        <div className="leader-board-content">
          <div className="add-form">
            <div>
              <p>Repo Link</p>
              <input
                value={form.repoLink}
                name="repoLink"
                onChange={handleChange}
              />
            </div>
            <div>
              <p>Predicted Column Link</p>
              <input
                value={form.predictedColumnLink}
                name="predictedColumnLink"
                onChange={handleChange}
              />
            </div>
            {!uploading && <button onClick={handleSubmit}>Submit</button>}
            {uploading && <p>Uploading....</p>}
            {message && <p>{message}</p>}
          </div>
        </div>
      )}
      {action === "update" && (
        <div className="leader-board-content">
          <div className="add-form">
            <div>
              <p>Predicted Column Link</p>
              <input
                value={updateForm.predictedColumnLink}
                name="predictedColumnLink"
                onChange={handleChangeUpdate}
              />
            </div>
            {!uploading && <button onClick={handleSubmitUpdate}>Submit</button>}
            {uploading && <p>Uploading....</p>}
            {message && <p>{message}</p>}
          </div>
        </div>
      )}
    </div>
  );
}

export default LeaderBoard;
