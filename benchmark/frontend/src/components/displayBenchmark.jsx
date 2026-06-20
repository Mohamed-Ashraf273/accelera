import "./displayBenchmark.css";
import Navigation from "./navigation";
import { Link, useLocation } from "react-router-dom";

function DisplayBenchmark() {
  const location = useLocation();
  const displayedBenchmark = location.state?.benchmark;
  if (!displayedBenchmark) return <h1>No benchmark Information exists</h1>;
  return (
    <div className="benchmark-display-page">
      <Navigation />
      <div className="benchmark-display-card">
        <div className="display-benchmark-hero">
          <div>
            <p className="display-benchmark-kicker">Benchmark details</p>
            <h1>{displayedBenchmark.title}</h1>
          </div>
          <div className="display-benchmark-actions">
            <Link
              className="leader-board-button"
              to="/leader-board"
              state={displayedBenchmark._id}
            >
              Leader Board
            </Link>
          </div>
        </div>

        <div className="display-benchmark-grid">
          <div className="card-sub-component display-wide">
            <h2>Description</h2>
            <p>{displayedBenchmark.description}</p>
          </div>
          <div className="card-sub-component">
            <h2>Problem Type</h2>
            <p>{displayedBenchmark.problemType}</p>
          </div>
          <div className="card-sub-component">
            <h2>Evaluation Metric</h2>
            <p>{displayedBenchmark.evaluationMetric.name}</p>
          </div>
          <div className="card-sub-component">
            <h2>Created By</h2>
            <p>Name : {displayedBenchmark.createdBy.name}</p>
          </div>
          <div className="card-sub-component">
            <h2>Creation Data </h2>
            <p>{displayedBenchmark.creationDate}</p>
          </div>
          <div className="card-sub-component">
            <h2>Dataset Link</h2>
            <a
              href={displayedBenchmark.datasetLink}
              target="_blank"
              rel="noopener noreferrer"
            >
              Drive Link
            </a>
          </div>
          <div className="card-sub-component">
            <h2>Testset Link</h2>
            <a
              href={displayedBenchmark.testSetWithoutPredictionsLink}
              target="_blank"
              rel="noopener noreferrer"
            >
              Drive Link
            </a>
          </div>
          <div className="card-sub-component display-wide">
            <h2>Evaluation Metric Params</h2>
            <div className="benchmark-param-list">
              {Object.entries(displayedBenchmark.metricParamaters).map(
                ([key, value]) => (
                  <h3 key={key}>
                    {key}: {value}
                  </h3>
                ),
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DisplayBenchmark;
