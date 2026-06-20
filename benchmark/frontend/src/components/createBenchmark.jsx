import { useState, useEffect } from "react";
import Navigation from "./navigation";
import "./createBenchmark.css";
import { authHeaders, getUser } from "../auth";
function CreateBenchmark() {
  const user = getUser();
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState([]);
  const [metric, setMetric] = useState({});
  const [problemMessage, setProblemMessage] = useState(null);
  const [checking, setChecking] = useState(false);
  const [form, setForm] = useState({
    title: "",
    description: "",
    targetColumn: "",
    datasetLink: "",
    testSetWithoutPredictionsLink: "",
    predictedColumnLink: "",
    problemType: "classification",
    evaluationMetric: "",
    metricParamaters: "",
    createdBy: user?._id,
  });
  const fetchMetric = async (problemType) => {
    setLoading(true);
    const results = await fetch(
      `http://localhost:3000/metrics/problem-type/${problemType}`,
    );
    const data = await results.json();
    setMetrics(data);
    if (data.length > 0) {
      setMetric(data[0]);
      setForm((prev) => ({
        ...prev,
        evaluationMetric: data[0]._id,
      }));
    }
    setLoading(false);
  };
  useEffect(() => {
    fetchMetric(form.problemType);
  }, []);
  const handleChooseMetric = (e) => {
    e.preventDefault();
    const selectedMetric = metrics.find(
      (metric) => metric._id === e.target.value,
    );
    setMetric(selectedMetric);
    handleChange(e);
  };
  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    if (e.target.name === "problemType") {
      fetchMetric(e.target.value);
    }
  };
  const handleParamter = (key, value) => {
    setForm((prev) => ({
      ...prev,
      metricParamaters: {
        ...prev.metricParamaters,
        [key]: value,
      },
    }));
  };
  const checkFill = () => {
    if (!form.title || form.title?.trim === "") {
      setProblemMessage("Fill Title filed");
      return false;
    }
    if (!form.description || form.description?.trim === "") {
      setProblemMessage("Fill description filed");
      return false;
    }
    if (!form.targetColumn || form.targetColumn?.trim === "") {
      setProblemMessage("Fill targetColumn filed");
      return false;
    }
    if (!form.datasetLink || form.datasetLink?.trim === "") {
      setProblemMessage("Fill datasetLink filed");
      return false;
    }
    if (
      !form.testSetWithoutPredictionsLink ||
      form.testSetWithoutPredictionsLink?.trim === ""
    ) {
      setProblemMessage("Fill testSetWithoutPredictionsLink filed");
      return false;
    }
    if (!form.predictedColumnLink || form.predictedColumnLink?.trim === "") {
      setProblemMessage("Fill predictedColumnLink filed");
      return false;
    }
    if (!form.evaluationMetric || form.evaluationMetric?.trim === "") {
      setProblemMessage("Fill evaluationMetric filed");
      return false;
    }
    setProblemMessage("");
    return true;
  };
  const handleSubmit = async (e) => {
    e.preventDefault();
    setChecking(true);
    if (!checkFill()) {
      setChecking(false);
      return;
    }
    try {
      const response = await fetch("http://localhost:3000/benchmark/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders(),
        },
        body: JSON.stringify(form),
      });
      const data = await response.json();
      if (!response.ok) {
        setProblemMessage(data.message);
        setChecking(false);

        return;
      }
      alert("Benchmark created successfully!");
      setChecking(false);
      setForm({
        title: "",
        description: "",
        targetColumn: "",
        datasetLink: "",
        testSetWithoutPredictionsLink: "",
        predictedColumnLink: "",
        problemType: "classification",
        evaluationMetric: "",
        metricParamaters: "",
        createdBy: user?._id,
      });
    } catch (err) {
      console.error(err.response?.data || err.message);
      alert(err.response?.data?.message || "Error creating benchmark");
      setChecking(false);
    }
  };

  return (
    <div className="create-benchmark-page">
      <Navigation />
      <div className="create-benchmark-content">
        <h2>Create Benchmark</h2>
        {!user && <p>You need to login first</p>}
        {loading && <p className="loading">Loading...</p>}
        {!loading && user && (
          <form onSubmit={handleSubmit} className="create-benchmark-form">
            <div className="form-input">
              <p>Title</p>
              <input
                value={form.title}
                name="title"
                placeholder="e.g. house_price"
                onChange={handleChange}
              />
            </div>
            <div className="form-input">
              <p>Description</p>
              <textarea
                value={form.description}
                rows={5}
                cols={50}
                name="description"
                placeholder="e.g. predict price of regression data"
                onChange={handleChange}
              />
            </div>
            <div className="form-input">
              <p>Target Column Name in the dataset</p>
              <input
                value={form.targetColumn}
                name="targetColumn"
                placeholder="e.g. price"
                onChange={handleChange}
              />
            </div>
            <div className="form-input">
              <p>Dataset Google Drive Link</p>
              <input
                name="datasetLink"
                onChange={handleChange}
                value={form.datasetLink}
                placeholder="Paste dataset Google Drive link"
              />
            </div>
            <div className="form-input">
              <p>Test Set Google Drive Link</p>
              <input
                name="testSetWithoutPredictionsLink"
                onChange={handleChange}
                value={form.testSetWithoutPredictionsLink}
                placeholder="Paste test set Google Drive link"
              />
            </div>
            <div className="form-input">
              <p>Predicted Column Google Drive Link</p>
              <input
                name="predictedColumnLink"
                onChange={handleChange}
                value={form.predictedColumnLink}
                placeholder="Paste predicted column Google Drive link"
              />
            </div>
            <div className="form-input">
              <p>problem Type</p>
              <select
                name="problemType"
                onChange={handleChange}
                value={form.problemType}
              >
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
              </select>
            </div>
            <div className="form-input">
              <div>
                <p>{form.problemType} Evaluation Metric</p>
                <select name="evaluationMetric" onChange={handleChooseMetric}>
                  {metrics.map((metric, index) => (
                    <option key={index} value={metric._id}>
                      {metric.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="form-input">
                {metric.neededParameters &&
                  Object.entries(metric.neededParameters).map(
                    ([key, values]) => (
                      <div key={key}>
                        <p>{key}</p>
                        <select
                          value={form.metricParamaters?.[key] || values[0]}
                          onChange={(e) => handleParamter(key, e.target.value)}
                        >
                          {values.map((value, index) => (
                            <option value={value} key={index}>
                              {value}
                            </option>
                          ))}
                        </select>
                      </div>
                    ),
                  )}
              </div>
            </div>
            {!checking && (
              <>
                <p>{problemMessage}</p>
                <button type="submit">Create Benchmark</button>
              </>
            )}
            {checking && <p>Uploading....</p>}
          </form>
        )}
      </div>
    </div>
  );
}
export default CreateBenchmark;
