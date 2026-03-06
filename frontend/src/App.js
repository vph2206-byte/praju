import { useEffect, useState } from "react";
import "@/App.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const classColors = {
  'DERMASON': '#22c55e',
  'SIRA': '#3b82f6',
  'SEKER': '#f97316',
  'HOROZ': '#a855f7',
  'CALI': '#06b6d4',
  'BARBUNYA': '#ef4444',
  'BOMBAY': '#eab308'
};

const featureLabels = [
  'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
  'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
  'Extent', 'Solidity', 'roundness', 'Compactness',
  'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
];

function App() {
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const [sampleData, setSampleData] = useState({});
  const [modelStatus, setModelStatus] = useState({ trained: false });
  const [formData, setFormData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [trainMessage, setTrainMessage] = useState('');

  useEffect(() => {
    loadDatasetInfo();
    loadStatistics();
    loadSampleData();
    checkModelStatus();
  }, []);

  const loadDatasetInfo = async () => {
    try {
      const response = await fetch(`${API}/dataset-info`);
      const data = await response.json();
      setDatasetInfo(data);
    } catch (error) {
      console.error('Error loading dataset info:', error);
    }
  };

  const loadStatistics = async () => {
    try {
      const response = await fetch(`${API}/statistics`);
      const data = await response.json();
      setStatistics(data);
    } catch (error) {
      console.error('Error loading statistics:', error);
    }
  };

  const loadSampleData = async () => {
    try {
      const response = await fetch(`${API}/sample-data`);
      const data = await response.json();
      setSampleData(data);
    } catch (error) {
      console.error('Error loading sample data:', error);
    }
  };

  const checkModelStatus = async () => {
    try {
      const response = await fetch(`${API}/model-status`);
      const data = await response.json();
      setModelStatus(data);
    } catch (error) {
      console.error('Error checking model status:', error);
    }
  };

  const trainModel = async () => {
    setTrainingLoading(true);
    setTrainMessage('');
    try {
      const response = await fetch(`${API}/train`, { method: 'POST' });
      const data = await response.json();
      if (data.success) {
        setTrainMessage(`Model trained successfully with ${(data.accuracy * 100).toFixed(1)}% accuracy!`);
        checkModelStatus();
      } else {
        setTrainMessage('Training failed');
      }
    } catch (error) {
      setTrainMessage('Error: ' + error.message);
    } finally {
      setTrainingLoading(false);
    }
  };

  const loadSample = (className) => {
    const sample = sampleData[className];
    if (sample) {
      setFormData(sample);
    }
  };

  const handleInputChange = (feature, value) => {
    setFormData(prev => ({ ...prev, [feature]: parseFloat(value) || 0 }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);
    
    try {
      const response = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Prediction failed');
      }
      
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      alert('Error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const clearForm = () => {
    setFormData({});
    setPrediction(null);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <div className="logo-icon">
            <i className="fas fa-seedling"></i>
          </div>
          <h1>Dry<span>Bean</span> Classifier</h1>
        </div>
        <div className="status-badge" data-testid="model-status-badge">
          <div className={`status-dot ${modelStatus.trained ? 'active' : ''}`}></div>
          <span>{modelStatus.trained ? 'Model Ready' : 'Not Trained'}</span>
        </div>
      </header>

      {/* Stats Grid */}
      <div className="stats-grid" data-testid="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{datasetInfo?.total_samples?.toLocaleString() || '-'}</div>
          <div className="stat-label">Total Samples</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{datasetInfo?.total_features || '-'}</div>
          <div className="stat-label">Features</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{datasetInfo?.class_names?.length || '-'}</div>
          <div className="stat-label">Bean Classes</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{modelStatus.accuracy ? `${(modelStatus.accuracy * 100).toFixed(1)}%` : '-'}</div>
          <div className="stat-label">Model Accuracy</div>
        </div>
      </div>

      {/* Dashboard Grid */}
      <div className="dashboard-grid">
        {/* Class Distribution */}
        <div className="card" data-testid="class-distribution-card">
          <div className="card-header">
            <div className="card-title">
              <i className="fas fa-chart-bar"></i>
              Class Distribution
            </div>
          </div>
          <div className="class-distribution">
            {datasetInfo && Object.entries(datasetInfo.class_distribution)
              .sort((a, b) => b[1] - a[1])
              .map(([cls, count]) => {
                const maxCount = Math.max(...Object.values(datasetInfo.class_distribution));
                return (
                  <div key={cls} className="class-item" data-testid={`class-item-${cls.toLowerCase()}`}>
                    <div className="class-name">
                      <div className="class-color" style={{ background: classColors[cls] || '#888' }}></div>
                      <span>{cls}</span>
                    </div>
                    <div className="class-bar">
                      <div 
                        className="class-bar-fill" 
                        style={{ 
                          width: `${(count/maxCount)*100}%`, 
                          background: classColors[cls] || '#888' 
                        }}
                      ></div>
                    </div>
                    <span className="class-count">{count}</span>
                  </div>
                );
              })}
          </div>
        </div>

        {/* Model Training */}
        <div className="card" data-testid="model-training-card">
          <div className="card-header">
            <div className="card-title">
              <i className="fas fa-brain"></i>
              Model Training
            </div>
          </div>
          <div className="training-status">
            <p className="training-desc">Train a Random Forest classifier on the dry bean dataset.</p>
            <div className="accuracy-display">
              <span>{modelStatus.accuracy ? (modelStatus.accuracy * 100).toFixed(1) : '--'}</span>%
            </div>
            <p className="accuracy-label">Model Accuracy</p>
          </div>
          {trainingLoading && (
            <div className="loading">
              <div className="spinner"></div>
              <span>Training model...</span>
            </div>
          )}
          {trainMessage && (
            <div className={`alert ${trainMessage.includes('success') ? 'alert-success' : 'alert-error'}`}>
              {trainMessage}
            </div>
          )}
          <button 
            className="btn btn-primary full-width" 
            onClick={trainModel} 
            disabled={trainingLoading}
            data-testid="train-model-btn"
          >
            <i className="fas fa-play"></i>
            Train Model
          </button>
        </div>

        {/* Prediction Form */}
        <div className="card full-width" data-testid="prediction-card">
          <div className="card-header">
            <div className="card-title">
              <i className="fas fa-magic"></i>
              Bean Classification
            </div>
          </div>
          
          <div className="sample-section">
            <p className="sample-label">Load sample data:</p>
            <div className="sample-buttons" data-testid="sample-buttons">
              {Object.keys(sampleData).map(cls => (
                <button 
                  key={cls}
                  type="button" 
                  className="sample-btn" 
                  onClick={() => loadSample(cls)}
                  data-testid={`sample-btn-${cls.toLowerCase()}`}
                >
                  {cls}
                </button>
              ))}
            </div>
          </div>

          <form onSubmit={handleSubmit} data-testid="prediction-form">
            <div className="form-grid">
              {featureLabels.map(feature => (
                <div key={feature} className="form-group">
                  <label className="form-label">{feature}</label>
                  <input
                    type="number"
                    step="any"
                    className="form-input"
                    value={formData[feature] || ''}
                    onChange={(e) => handleInputChange(feature, e.target.value)}
                    data-testid={`input-${feature.toLowerCase()}`}
                    required
                  />
                </div>
              ))}
            </div>
            <div className="btn-group">
              <button type="submit" className="btn btn-primary" disabled={loading} data-testid="predict-btn">
                <i className="fas fa-search"></i>
                Classify Bean
              </button>
              <button type="button" className="btn btn-secondary" onClick={clearForm} data-testid="clear-form-btn">
                <i className="fas fa-eraser"></i>
                Clear
              </button>
            </div>
          </form>

          {loading && (
            <div className="loading">
              <div className="spinner"></div>
              <span>Classifying...</span>
            </div>
          )}

          {prediction && (
            <div className="result-card show" data-testid="result-card">
              <div className="result-header">
                <div className="result-class" style={{ color: classColors[prediction.predicted_class] || '#22c55e' }}>
                  {prediction.predicted_class}
                </div>
                <div className="result-confidence">
                  Confidence: <span>{(prediction.confidence * 100).toFixed(2)}%</span>
                </div>
              </div>
              <p className="probabilities-label">All Probabilities:</p>
              <div className="probabilities-grid">
                {Object.entries(prediction.all_probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cls, prob]) => (
                    <div key={cls} className="prob-item">
                      <div className="prob-class">{cls}</div>
                      <div className="prob-value" style={{ color: classColors[cls] || '#888' }}>
                        {(prob * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>

        {/* Feature Statistics */}
        <div className="card full-width" data-testid="feature-statistics-card">
          <div className="card-header">
            <div className="card-title">
              <i className="fas fa-table"></i>
              Feature Statistics
            </div>
          </div>
          <div className="table-container">
            <table className="feature-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Mean</th>
                  <th>Std Dev</th>
                  <th>Min</th>
                  <th>Max</th>
                </tr>
              </thead>
              <tbody>
                {statistics && Object.entries(statistics.feature_statistics).map(([feature, stats]) => (
                  <tr key={feature}>
                    <td className="feature-name">{feature}</td>
                    <td>{stats.mean.toFixed(4)}</td>
                    <td>{stats.std.toFixed(4)}</td>
                    <td>{stats.min.toFixed(4)}</td>
                    <td>{stats.max.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
