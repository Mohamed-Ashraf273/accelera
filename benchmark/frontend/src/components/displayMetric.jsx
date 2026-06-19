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
        <p>Metric Name : {displayedMetric.name}</p>
        <p>Sklrearn metric Name : {displayedMetric.sklearnMetricName}</p>
        <p>Problem Type : {displayedMetric.problemType}</p>
        <p>Needed to be : {displayedMetric.whichBetter}</p>
        <div className="params">
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
