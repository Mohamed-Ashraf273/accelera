import "./displayMetric.css";
import Navigation from "./navigation";
import { useLocation } from "react-router-dom";

function DisplayMetric() {
  const location = useLocation();
  const displayedMetric = location.state?.metric;
  if (!displayedMetric) return <h1>No Metric Information exists</h1>;
  return (
    <div className="metric-display-page">
      <Navigation />
      <div className="metric-display-card">
        <div className="metric-display-head">
          <p>Metric details</p>
          <h1>{displayedMetric.name}</h1>
        </div>
        <div className="metric-display-grid">
          <div>
            <p>Sklrearn metric Name</p>
            <h3>{displayedMetric.sklearnMetricName}</h3>
          </div>
          <div>
            <p>Problem Type</p>
            <h3>{displayedMetric.problemType}</h3>
          </div>
          <div>
            <p>Needed to be</p>
            <h3>{displayedMetric.whichBetter}</h3>
          </div>
        </div>
        <div className="params">
          <h2>Needed Parameters</h2>
          {Object.entries(displayedMetric.neededParameters).map(
            ([key, allowedValues]) => (
              <ul key={key} className="param">
                <li>
                  <p>Param : {key}</p>
                  <div className="allowed-values">
                    <p>Allowed Values</p>
                    <ul>
                      {allowedValues.map((value, index) => (
                        <li key={index}>{value}</li>
                      ))}
                    </ul>
                  </div>
                </li>
              </ul>
            ),
          )}
        </div>
      </div>
    </div>
  );
}

export default DisplayMetric;
