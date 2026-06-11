import { useState } from "react";

function CreateBenchmark() {
  const user = JSON.parse(localStorage.getItem("user"));

  const [form, setForm] = useState({
    title: "",
    description: "",
    targetColumn: "",
    datasetLink: "",
    testSetWithoutPredictionsLink: "",
    predictedColumnLink: "",
    problemType: "classification",
    evaluationMetric: "",
    metricPramaters: "",
    createdBy: user._id,
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const res = await axios.post(
        "http://localhost:5000/api/benchmarks",
        form,
      );

      console.log("Success:", res.data);
      alert("Benchmark created successfully!");
    } catch (err) {
      console.error(err.response?.data || err.message);
      alert(err.response?.data?.message || "Error creating benchmark");
    }
  };

  return (
    <div className="create-benchmark-page">
      <h2>Create Benchmark</h2>
      <form onSubmit={handleSubmit} className="create-benchmark-form">
        <input
          name="title"
          placeholder="Benchmark Title"
          onChange={handleChange}
        />
        <br />

        <input
          name="description"
          placeholder="Benchmark Description"
          onChange={handleChange}
        />
        <br />

        <input
          name="targetColumn"
          placeholder="Target Column Name"
          onChange={handleChange}
        />
        <br />

        <input
          name="datasetLink"
          placeholder="Dataset Google Drive Link"
          onChange={handleChange}
        />
        <br />

        <input
          name="testSetWithoutPredictionsLink"
          placeholder="Test Set Google Drive Link"
          onChange={handleChange}
        />
        <br />

        <input
          name="predictedColumnLink"
          placeholder="Predicted Column Google Drive Link"
          onChange={handleChange}
        />
        <br />

        <select name="problemType" onChange={handleChange}>
          <option value="classification">Classification</option>
          <option value="regression">Regression</option>
        </select>
        <br />

        <input
          name="evaluationMetric"
          placeholder="Evaluation Metric"
          onChange={handleChange}
        />
        <br />

        <input
          name="metricPramaters"
          placeholder="Metric Parameters"
          onChange={handleChange}
        />
        <br />
        <br />

        <button type="submit">Create Benchmark</button>
      </form>
    </div>
  );
}
export default CreateBenchmark;
