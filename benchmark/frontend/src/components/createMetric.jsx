import { useState } from "react";
import Navigation from "./navigation";
import "./createMetric.css";
import { authHeaders, getUser } from "../auth";
function CreateMetric() {
  const user = getUser();
  const [form, setForm] = useState({
    name: "",
    sklearnMetricName: "",
    problemType: "classification",
    whichBetter: "higher",
    neededParameters: {},
  });

  const [paramName, setParamName] = useState("");
  const [paramValue, setParamValue] = useState("");

  const addNewParameter = () => {
    if (!paramName.trim()) return;

    setForm((prev) => ({
      ...prev,
      neededParameters: {
        ...prev.neededParameters,
        [paramName]: [],
      },
    }));

    setParamName("");
  };

  const removeOneParameter = (key) => {
    const newParamtersObj = { ...form.neededParameters };
    delete newParamtersObj[key];

    setForm((prev) => ({
      ...prev,
      neededParameters: newParamtersObj,
    }));
  };

  const addNewParamValue = (key) => {
    if (!paramValue.trim()) return;

    setForm((prev) => ({
      ...prev,
      neededParameters: {
        ...prev.neededParameters,
        [key]: [...(prev.neededParameters[key] || []), paramValue],
      },
    }));

    setParamValue("");
  };

  const removeOneParamValue = (key, index) => {
    const updatedParamValues = form.neededParameters[key].filter(
      (_, i) => i !== index,
    );

    setForm((prev) => ({
      ...prev,
      neededParameters: {
        ...prev.neededParameters,
        [key]: updatedParamValues,
      },
    }));
  };

  const submitForm = async (e) => {
    e.preventDefault();
    if (!form.name) {
      alert("metric name must be exist")
      return
    }
    if (!form.sklearnMetricName) {
      alert("metric sklearnMetricName must be exist")
      return
    }
    try {
      const response = await fetch("http://localhost:3000/metrics/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders(),
        },
        body: JSON.stringify(form),
      });

      const data = await response.json();
      console.log(data);
    } catch (err) {
      console.error(err.message);
      alert(err.message);
    }
  };

  return (
    <div className="metric-page">
      <Navigation />

      <div className="metric-page-content">
        <h2>Create Metric</h2>
        {user?.role !== "admin" && <p>Admin access is required</p>}
        {user?.role === "admin" && (
        <div className="form">
          <div className="input-div">
            <p>Metric Name</p>
            <input
              placeholder="e.g. accuracy"
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              
            />
          </div>
          <div className="input-div">
            <p>Sklearn metric name</p>
            <input
              placeholder="e.g. accuracy_score"
              onChange={(e) =>
                setForm({ ...form, sklearnMetricName: e.target.value })
              }
              
            />
          </div>
          <div className="input-div">
            <p>Problem Type</p>
            <select
              onChange={(e) =>
                setForm({ ...form, problemType: e.target.value })
              }
            >
              <option value="classification">classification</option>
              <option value="regression">regression</option>
            </select>
          </div>
          <div className="input-div">
            <p>Which Better</p>
            <select
              onChange={(e) =>
                setForm({ ...form, whichBetter: e.target.value })
              }
            >
              <option value="higher">higher</option>
              <option value="lower">lower</option>
            </select>
          </div>
          <div className="paramter">
            <h3>Needed Parameters</h3>
            <div className="input-div">
              <input
                placeholder="e.g. average"
                value={paramName}
                onChange={(e) => setParamName(e.target.value)}
              />

              <button
                type="button"
                onClick={addNewParameter}
                className="param-button"
              >
                Add Parameter
              </button>
            </div>
            <div></div>
            {Object.keys(form.neededParameters).map((key) => (
              <div key={key} className="show-paramter">
                <div className="input-div">
                  <p>Value for Paramter {key}</p>
                  <input
                    placeholder={`e.g. binary`}
                    value={paramValue}
                    onChange={(e) => setParamValue(e.target.value)}
                  />

                  <button
                    type="button"
                    onClick={() => addNewParamValue(key)}
                    className="param-button"
                  >
                    Add Value
                  </button>
                </div>
                <div className="show-paramter-value">
                  {form.neededParameters[key].map((value, index) => (
                    <div className="param-card">
                      <p key={index}>{value}</p>
                      <button
                        onClick={() => removeOneParamValue(key, index)}
                        className="param-button"
                      >
                        ❌
                      </button>
                    </div>
                  ))}
                </div>

                <button
                  type="button"
                  onClick={() => removeOneParameter(key)}
                  className="param-button"
                >
                  Remove Parameter
                </button>
              </div>
            ))}

            <button onClick={submitForm} className="param-button">
              Submit
            </button>
          </div>
        </div>
        )}
      </div>
    </div>
  );
}

export default CreateMetric;
