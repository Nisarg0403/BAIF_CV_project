import React, { useState, useEffect } from 'react';
import { 
  Home, 
  Clock, 
  Users, 
  BarChart3, 
  Settings, 
  Upload, 
  Camera,
  Video, 
  Sparkles, 
  CheckCircle, 
  AlertCircle, 
  Scale, 
  Ruler, 
  Leaf,
  Cpu,
  Search,
  Bell,
  Trash2,
  RefreshCw,
  Plus,
  Menu,
  X
} from 'lucide-react';
import CattleCanvas from './components/CattleCanvas';
import ModelProofsCard from './components/ModelProofsCard';
import LiveCameraModal from './components/LiveCameraModal';

export default function App() {
  const [activeTab, setActiveTab] = useState('predict');
  const [engine, setEngine] = useState('deeplab');
  const [cattleId, setCattleId] = useState('TAG-105730429112');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  
  // 3-View Photos State (Left Side Profile, Right Side Profile, Rear View)
  const [photos, setPhotos] = useState({
    left: null,
    right: null,
    rear: null
  });

  const [activeViewTab, setActiveViewTab] = useState('left'); // 'left', 'right', 'rear'
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // Dynamic Prediction State (null by default until prediction runs!)
  const [prediction, setPrediction] = useState(null);
  const [predictionsByView, setPredictionsByView] = useState(null);
  const [history, setHistory] = useState([]);

  const API_BASE = `http://${window.location.hostname}:8000`;

  // Fetch prediction history on mount
  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/history`);
      if (res.ok) {
        const data = await res.json();
        setHistory(data.history || []);
      }
    } catch (e) {
      console.log('History server connection info:', e);
    }
  };

  const handlePhotoSelect = (side, file) => {
    if (file) {
      const url = URL.createObjectURL(file);
      setPhotos(prev => ({
        ...prev,
        [side]: { file, url }
      }));
    }
  };

  // Direct Mobile Camera Capture with Instant Auto-Run AI Feed
  const handleDirectCameraCapture = (side, file) => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    const updatedPhotos = {
      ...photos,
      [side]: { file, url }
    };
    setPhotos(updatedPhotos);
    // Directly feed captured camera image to AI prediction engine
    handleRunPrediction(updatedPhotos);
  };

  const handleRemovePhoto = (side) => {
    setPhotos(prev => ({
      ...prev,
      [side]: null
    }));
    setPrediction(null);
    setPredictionsByView(null);
  };

  const handleRunPrediction = async (customPhotos = null) => {
    const targetPhotos = customPhotos || photos;
    const hasPhoto = targetPhotos.left || targetPhotos.right || targetPhotos.rear;
    if (!hasPhoto) {
      setErrorMsg('Please upload or capture at least one profile photo (Left, Right, or Rear view).');
      return;
    }

    setLoading(true);
    setErrorMsg(null);

    try {
      const formData = new FormData();
      if (targetPhotos.left?.file) formData.append('left_file', targetPhotos.left.file);
      if (targetPhotos.right?.file) formData.append('right_file', targetPhotos.right.file);
      if (targetPhotos.rear?.file) formData.append('rear_file', targetPhotos.rear.file);

      formData.append('cattle_id', cattleId);
      formData.append('model_engine', engine);

      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json();
        const msg = typeof errData.detail === 'string'
          ? errData.detail
          : (Array.isArray(errData.detail) ? errData.detail[0]?.msg : JSON.stringify(errData.detail));
        throw new Error(msg || 'Prediction failed');
      }

      const data = await res.json();
      if (data.success && data.prediction) {
        setPrediction(data.prediction);
        setPredictionsByView(data.predictions_by_view || null);
        fetchHistory();
      }
    } catch (err) {
      console.error('Prediction Error:', err);
      const strMsg = typeof err === 'string' ? err : (err.message ? (typeof err.message === 'string' ? err.message : JSON.stringify(err.message)) : String(err));
      setErrorMsg(strMsg || 'Error processing image prediction.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPhotos({ left: null, right: null, rear: null });
    setPrediction(null);
    setPredictionsByView(null);
    setErrorMsg(null);
  };

  const activePhoto = photos[activeViewTab];
  const isReady = photos.left || photos.right || photos.rear;

  return (
    <div className="app-container">
      {/* Left Sidebar */}
      <aside className="sidebar">
        <div>
          <div className="brand-header">
            <div className="brand-logo-icon">
              <Scale size={22} />
            </div>
            <div>
              <h1 className="brand-title">CattleWeightAI</h1>
              <p className="brand-subtitle">Smarter Livestock Management</p>
            </div>
          </div>

          <nav className="nav-menu">
            <button 
              className={`nav-item ${activeTab === 'predict' ? 'active' : ''}`}
              onClick={() => setActiveTab('predict')}
            >
              <Home size={18} />
              <span>Predict Weight</span>
            </button>

            <button 
              className={`nav-item ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => setActiveTab('history')}
            >
              <Clock size={18} />
              <span>History</span>
            </button>

            <button 
              className={`nav-item ${activeTab === 'herd' ? 'active' : ''}`}
              onClick={() => setActiveTab('herd')}
            >
              <Users size={18} />
              <span>Herd Management</span>
            </button>

            <button 
              className={`nav-item ${activeTab === 'analytics' ? 'active' : ''}`}
              onClick={() => setActiveTab('analytics')}
            >
              <BarChart3 size={18} />
              <span>Analytics</span>
            </button>

            <button 
              className={`nav-item ${activeTab === 'settings' ? 'active' : ''}`}
              onClick={() => setActiveTab('settings')}
            >
              <Settings size={18} />
              <span>Settings</span>
            </button>
          </nav>
        </div>

        <div className="sidebar-footer">
          <p>Powered by AI for a sustainable agriculture future.</p>
        </div>
      </aside>

      {/* Main Wrapper */}
      <div className="main-wrapper">
        {/* Top Header Bar */}
        <header className="top-header">
          <div className="breadcrumb-path"></div>

          <div className="top-actions">
            {/* Active Model Engine Switcher */}
            <div className="engine-select-group">
              <button 
                className={`engine-btn ${engine === 'deeplab' ? 'active' : ''}`}
                onClick={() => setEngine('deeplab')}
              >
                🧠 DeepLabV3+ (±4.9kg MAE)
              </button>
              <button 
                className={`engine-btn ${engine === 'yolo' ? 'active' : ''}`}
                onClick={() => setEngine('yolo')}
              >
                ⚡ YOLOv8 (51kg MAE)
              </button>
            </div>

            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <Bell size={20} color="#64748B" style={{ cursor: 'pointer' }} />
            </div>

            <div style={{
              width: 34, height: 34, borderRadius: '50%', backgroundColor: '#E2E8F0',
              display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: '0.85rem'
            }}>
              N
            </div>
          </div>
        </header>

        {/* Content Body */}
        {activeTab === 'predict' && (
          <main className="content-area">
            {/* Left 65% Workspace */}
            <section className="card-panel">
              <div className="panel-header">
                <div>
                  <h2 className="panel-title">Predict Cattle Weight</h2>
                  <p className="panel-subtitle">Upload multi-view profile photos of your cattle to calculate AI body weight & measurements.</p>
                </div>
              </div>

              {/* Tag Input */}
              <div style={{ marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--slate-dark)' }}>🏷️ Cattle Tag ID:</span>
                <input 
                  type="text" 
                  value={cattleId}
                  onChange={(e) => setCattleId(e.target.value)}
                  placeholder="e.g. TAG-105730429112"
                  style={{ padding: '0.4rem 0.75rem', borderRadius: '6px', border: '1px solid #CBD5E1', fontSize: '0.85rem', fontWeight: 600, width: '220px' }}
                />
              </div>

              {/* 3-View Acquisition Stepper & Checklist (Left Profile, Right Profile, Rear View) */}
              <div style={{ marginBottom: '1rem' }}>
                <p style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--slate-dark)', marginBottom: '0.5rem' }}>
                  📸 Multi-View Image Acquisition Stepper (3 Views)
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
                  <div style={{
                    padding: '0.6rem 0.8rem', borderRadius: '8px', fontSize: '0.82rem', fontWeight: 600,
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    backgroundColor: photos.left ? '#ECFDF5' : '#FEF2F2',
                    borderLeft: photos.left ? '4px solid #10B981' : '4px solid #EF4444',
                    color: photos.left ? '#065F46' : '#991B1B'
                  }}>
                    <span>1. Left Profile</span>
                    <span>{photos.left ? '✅ Complete' : '❌ Missing'}</span>
                  </div>

                  <div style={{
                    padding: '0.6rem 0.8rem', borderRadius: '8px', fontSize: '0.82rem', fontWeight: 600,
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    backgroundColor: photos.right ? '#ECFDF5' : '#FEF2F2',
                    borderLeft: photos.right ? '4px solid #10B981' : '4px solid #EF4444',
                    color: photos.right ? '#065F46' : '#991B1B'
                  }}>
                    <span>2. Right Profile</span>
                    <span>{photos.right ? '✅ Complete' : '❌ Missing'}</span>
                  </div>

                  <div style={{
                    padding: '0.6rem 0.8rem', borderRadius: '8px', fontSize: '0.82rem', fontWeight: 600,
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    backgroundColor: photos.rear ? '#ECFDF5' : '#FEF2F2',
                    borderLeft: photos.rear ? '4px solid #10B981' : '4px solid #EF4444',
                    color: photos.rear ? '#065F46' : '#991B1B'
                  }}>
                    <span>3. Rear View</span>
                    <span>{photos.rear ? '✅ Complete' : '❌ Missing'}</span>
                  </div>
                </div>
              </div>

              {/* View Tabs */}
              <div style={{ display: 'flex', gap: '0.5rem', borderBottom: '1px solid #E2E8F0', marginBottom: '1rem' }}>
                <button 
                  onClick={() => setActiveViewTab('left')}
                  style={{
                    padding: '0.5rem 1rem', fontSize: '0.85rem', fontWeight: 600, border: 'none', background: 'transparent',
                    borderBottom: activeViewTab === 'left' ? '3px solid var(--primary-sage)' : '3px solid transparent',
                    color: activeViewTab === 'left' ? 'var(--primary-sage)' : 'var(--slate-muted)', cursor: 'pointer'
                  }}
                >
                  🐄 1. Left Side Profile
                </button>

                <button 
                  onClick={() => setActiveViewTab('right')}
                  style={{
                    padding: '0.5rem 1rem', fontSize: '0.85rem', fontWeight: 600, border: 'none', background: 'transparent',
                    borderBottom: activeViewTab === 'right' ? '3px solid var(--primary-sage)' : '3px solid transparent',
                    color: activeViewTab === 'right' ? 'var(--primary-sage)' : 'var(--slate-muted)', cursor: 'pointer'
                  }}
                >
                  🐄 2. Right Side Profile
                </button>

                <button 
                  onClick={() => setActiveViewTab('rear')}
                  style={{
                    padding: '0.5rem 1rem', fontSize: '0.85rem', fontWeight: 600, border: 'none', background: 'transparent',
                    borderBottom: activeViewTab === 'rear' ? '3px solid var(--primary-sage)' : '3px solid transparent',
                    color: activeViewTab === 'rear' ? 'var(--primary-sage)' : 'var(--slate-muted)', cursor: 'pointer'
                  }}
                >
                  🐄 3. Rear View (Rump Width)
                </button>
              </div>

              {/* Active Tab Photo Uploader / Visualizer Canvas */}
              {activePhoto ? (
                (() => {
                  const currentViewPrediction = (predictionsByView && predictionsByView[activeViewTab]) || prediction;
                  return (
                    <div>
                      <CattleCanvas 
                        imageUrl={activePhoto.url}
                        landmarks={currentViewPrediction?.landmarks}
                        measurements={prediction?.measurements || currentViewPrediction?.measurements}
                        contourPoints={currentViewPrediction?.contour_points}
                        viewType={activeViewTab}
                      />

                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.75rem' }}>
                        <span style={{ fontSize: '0.8rem', color: '#64748B' }}>
                          Selected Image: <b>{activePhoto.file.name}</b> ({(activePhoto.file.size / 1024 / 1024).toFixed(1)} MB)
                        </span>
                        <button 
                          onClick={() => handleRemovePhoto(activeViewTab)}
                          style={{ background: 'transparent', border: 'none', color: '#EF4444', fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.3rem' }}
                        >
                          <Trash2 size={14} /> Remove Photo
                        </button>
                      </div>
                    </div>
                  );
                })()
              ) : (
                /* No photo uploaded yet - clean empty state dropzone */
                <div className="dropzone-box">
                  <div style={{ width: 52, height: 52, borderRadius: '50%', backgroundColor: 'var(--primary-sage-light)', color: 'var(--primary-sage)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginBottom: '0.75rem' }}>
                    <Camera size={26} />
                  </div>
                  <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--slate-dark)', marginBottom: '0.3rem' }}>
                    Capture or Upload {activeViewTab === 'left' ? 'Left Profile' : activeViewTab === 'right' ? 'Right Profile' : 'Rear View'} Photo
                  </h3>
                  <p style={{ fontSize: '0.85rem', color: 'var(--slate-muted)', marginBottom: '1.25rem' }}>
                    Use live camera viewfinder or mobile camera to snap and directly feed to AI.
                  </p>

                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem', maxWidth: '420px', margin: '0 auto' }}>
                    {/* Primary Button: Live WebRTC Camera Viewfinder */}
                    <button 
                      className="btn-primary" 
                      onClick={() => setIsCameraOpen(true)}
                      style={{ width: '100%', padding: '0.85rem 1.25rem', fontSize: '0.92rem', gap: '0.5rem', boxShadow: '0 3px 10px rgba(16, 185, 129, 0.25)' }}
                    >
                      <Sparkles size={18} /> 📸 Live Camera Viewfinder & Instant AI
                    </button>

                    <div style={{ display: 'flex', gap: '0.6rem', width: '100%' }}>
                      {/* Mobile Native Camera Direct Capture */}
                      <label className="btn-secondary" style={{ flex: 1, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem', cursor: 'pointer', padding: '0.7rem 0.8rem', fontSize: '0.85rem' }}>
                        <Camera size={16} /> Mobile Camera
                        <input 
                          type="file" 
                          accept="image/*" 
                          capture="environment"
                          onChange={(e) => handleDirectCameraCapture(activeViewTab, e.target.files[0])}
                          style={{ display: 'none' }}
                        />
                      </label>

                      {/* Upload File */}
                      <label className="btn-secondary" style={{ flex: 1, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem', cursor: 'pointer', padding: '0.7rem 0.8rem', fontSize: '0.85rem' }}>
                        <Upload size={16} /> Choose File
                        <input 
                          type="file" 
                          accept="image/*" 
                          onChange={(e) => handlePhotoSelect(activeViewTab, e.target.files[0])}
                          style={{ display: 'none' }}
                        />
                      </label>
                    </div>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div style={{ marginTop: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {errorMsg && (
                  <div style={{ padding: '0.75rem', borderRadius: '8px', backgroundColor: '#FEF2F2', border: '1px solid #FCA5A5', color: '#991B1B', fontSize: '0.85rem' }}>
                    {errorMsg}
                  </div>
                )}

                {isReady ? (
                  <div style={{ display: 'flex', gap: '0.75rem' }}>
                    <button 
                      className="btn-primary" 
                      onClick={handleRunPrediction}
                      disabled={loading}
                      style={{ flex: 1 }}
                    >
                      <Sparkles size={18} />
                      <span>{loading ? 'Processing AI Models...' : '🔮 Run AI Weight Estimation'}</span>
                    </button>

                    <button 
                      className="btn-secondary"
                      onClick={handleReset}
                    >
                      <RefreshCw size={16} /> Clear All
                    </button>
                  </div>
                ) : (
                  <div style={{ padding: '0.75rem', borderRadius: '8px', backgroundColor: '#F8FAFC', border: '1px solid #E2E8F0', fontSize: '0.82rem', color: '#64748B', textAlign: 'center' }}>
                    💡 <b>Acquisition Notice:</b> Please upload at least one profile photo above to enable the AI Estimation button.
                  </div>
                )}
              </div>
            </section>

            {/* Right 35% Analytical Results Panel */}
            <aside className="card-panel">
              <h3 className="panel-title" style={{ fontSize: '1.1rem', marginBottom: '1rem' }}>Prediction Result</h3>

              {prediction ? (
                <>
                  {/* Big Metric Weight Card */}
                  <div className="metric-hero-card">
                    <div className="metric-hero-icon">
                      <Scale size={24} />
                    </div>
                    <div className="metric-hero-lbl">Estimated Weight</div>
                    <div className="metric-hero-val">{prediction.weight_kg} kg</div>
                    <div className="metric-hero-sub">(± 17.2 kg MAE)</div>
                  </div>

                  {/* Confidence Meter */}
                  <div className="confidence-wrapper">
                    <div className="confidence-header">
                      <span>Confidence</span>
                      <span>{prediction.confidence_pct}%</span>
                    </div>
                    <div className="confidence-track">
                      <div className="confidence-fill" style={{ width: `${prediction.confidence_pct}%` }}></div>
                    </div>
                  </div>

                  {/* Dynamic Estimated Measurements Table */}
                  <div style={{ marginBottom: '1.25rem' }}>
                    <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--slate-dark)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <Ruler size={16} color="var(--primary-sage)" />
                      <span>Estimated Measurements</span>
                    </div>

                    <table className="meas-table">
                      <tbody>
                        <tr>
                          <td className="lbl">Body Length</td>
                          <td className="val">{prediction.measurements.body_length_cm} cm</td>
                        </tr>
                        <tr>
                          <td className="lbl">Chest Girth</td>
                          <td className="val">{prediction.measurements.chest_girth_cm} cm</td>
                        </tr>
                        <tr>
                          <td className="lbl">Height at Withers</td>
                          <td className="val">{prediction.measurements.withers_height_cm} cm</td>
                        </tr>
                        <tr>
                          <td className="lbl">Height at Stature</td>
                          <td className="val">{prediction.measurements.stature_height_cm} cm</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                /* Empty prediction state until user runs prediction */
                <div style={{ padding: '2.5rem 1rem', textAlign: 'center', color: '#94A3B8', border: '1px dashed #CBD5E1', borderRadius: '12px', marginBottom: '1.25rem' }}>
                  <Scale size={36} color="#CBD5E1" style={{ marginBottom: '0.5rem' }} />
                  <p style={{ fontWeight: 600, fontSize: '0.9rem', color: '#64748B' }}>No Prediction Yet</p>
                  <p style={{ fontSize: '0.8rem', marginTop: '0.2rem' }}>Upload cattle photos on the left and click "Run AI Weight Estimation".</p>
                </div>
              )}

              {/* AI Model Proofs & Validation Card */}
              <ModelProofsCard activeEngine={engine} />

              {/* Green Agritech Nutrition Tip Card */}
              <div className="tip-card">
                <Leaf size={20} color="#166534" style={{ flexShrink: 0, marginTop: '2px' }} />
                <div>
                  <p style={{ fontWeight: 600, marginBottom: '0.2rem' }}>Agritech Nutrition Note</p>
                  <p>Predictions use scale-invariant ratio features calibrated on field ground truth. Ensure full cow silhouette is visible.</p>
                </div>
              </div>
            </aside>
          </main>
        )}

        {/* History Tab */}
        {activeTab === 'history' && (
          <main className="content-area" style={{ gridTemplateColumns: '1fr' }}>
            <section className="card-panel">
              <h2 className="panel-title">Cattle Weight Prediction History Log</h2>
              <p className="panel-subtitle">Saved biometric records and AI predictions stored in database.</p>

              <table className="meas-table" style={{ marginTop: '1rem' }}>
                <thead>
                  <tr style={{ background: '#F8FAFC', textAlign: 'left', fontSize: '0.8rem', color: '#64748B' }}>
                    <th style={{ padding: '0.75rem' }}>Timestamp</th>
                    <th style={{ padding: '0.75rem' }}>Cattle Tag ID</th>
                    <th style={{ padding: '0.75rem' }}>Engine Used</th>
                    <th style={{ padding: '0.75rem' }}>Est. Weight</th>
                    <th style={{ padding: '0.75rem' }}>Body Length</th>
                    <th style={{ padding: '0.75rem' }}>Chest Girth</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((item) => (
                    <tr key={item.id}>
                      <td style={{ padding: '0.75rem' }}>{item.timestamp}</td>
                      <td style={{ padding: '0.75rem', fontWeight: 700 }}>🏷️ {item.cattle_id}</td>
                      <td style={{ padding: '0.75rem' }}>{item.engine === 'yolo' ? '⚡ YOLOv8' : '🧠 DeepLabV3+'}</td>
                      <td style={{ padding: '0.75rem', fontWeight: 700, color: '#059669' }}>{item.weight} kg</td>
                      <td style={{ padding: '0.75rem' }}>{item.measurements?.body_length_cm || '-'} cm</td>
                      <td style={{ padding: '0.75rem' }}>{item.measurements?.chest_girth_cm || '-'} cm</td>
                    </tr>
                  ))}
                  {history.length === 0 && (
                    <tr>
                      <td colSpan={6} style={{ padding: '2rem', textAlign: 'center', color: '#94A3B8' }}>
                        No historical prediction records found. Run a prediction to populate history.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </section>
          </main>
        )}
      </div>

      {/* Live WebRTC Camera Viewfinder Modal */}
      <LiveCameraModal 
        isOpen={isCameraOpen}
        onClose={() => setIsCameraOpen(false)}
        onCapture={(file) => handleDirectCameraCapture(activeViewTab, file)}
        activeViewTab={activeViewTab}
      />
    </div>
  );
}
