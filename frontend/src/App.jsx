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
  const [cattleId, setCattleId] = useState('');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  
  // 3-View Photos State (Left Side Profile, Right Side Profile, Rear View)
  const [photos, setPhotos] = useState({
    left: null,
    right: null,
    rear: null
  });

  // Experimental Innovations Panel State (Default OFF)
  const [isInnovationsOpen, setIsInnovationsOpen] = useState(false);
  const [enableVideo, setEnableVideo] = useState(false);
  const [videoFile, setVideoFile] = useState(null);
  const [enableUnwarp, setEnableUnwarp] = useState(false);
  const [enableDualAngle, setEnableDualAngle] = useState(false);
  const [dualSideFile, setDualSideFile] = useState(null);
  const [dualRearFile, setDualRearFile] = useState(null);
  const [enableExif, setEnableExif] = useState(false);
  const [regressorEngine, setRegressorEngine] = useState('xgboost'); // 'xgboost', 'kan', 'ensemble'
  const [ensembleAlpha, setEnsembleAlpha] = useState(0.5);
  const [showXai, setShowXai] = useState(false);

  const [activeViewTab, setActiveViewTab] = useState('left'); // 'left', 'right', 'rear'
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // Dynamic Prediction State (null by default until prediction runs!)
  const [prediction, setPrediction] = useState(null);
  const [predictionsByView, setPredictionsByView] = useState(null);
  const [history, setHistory] = useState([]);

  const API_BASE = window.location.port === '5173' || window.location.port === '3000'
    ? `http://${window.location.hostname}:8000`
    : `${window.location.protocol}//${window.location.host}`;

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

  const extractTagFromFilename = (filename) => {
    if (!filename) return null;
    let stem = filename.replace(/\.[^/.]+$/, '').trim();
    stem = stem.replace(/\.(jpg|jpeg|png|webp|bmp|gif|mp4|mov|avi|mkv|webm|flv|m4v|3gp|heic|tiff)$/i, '').trim();
    
    // Iteratively strip part/view/number suffixes anchored at the end (e.g. _LR_1, _Back_1, _LL_2, _1, _2)
    const suffixPattern = /[_\-\s]+(?:LL|LR|back|rear|left|right|side|front|profile)(?:[_\-\s]+\d+)?$/i;
    const idxPattern = /[_\-\s]+\d{1,2}$/i;

    let previous = '';
    while (stem !== previous && stem.length > 0) {
      previous = stem;
      stem = stem.replace(suffixPattern, '').trim();
      stem = stem.replace(idxPattern, '').trim();
    }
    stem = stem.trim();

    const genericNames = new Set([
      'image', 'photo', 'upload', 'file', 'blob', 'left', 'right', 'rear', 'back', 'side', 'front',
      'video', 'frame', 'captured', 'input', 'test', 'temp', 'sample', 'unknown',
      'cattle_snap', 'll', 'lr', 'side_file', 'left_file', 'right_file', 'rear_file', 'profile', 'tag'
    ]);

    if (genericNames.has(stem.toLowerCase()) || stem.length === 0) {
      return null;
    }
    return stem;
  };

  const handlePhotoSelect = (side, file) => {
    if (file) {
      const autoTag = extractTagFromFilename(file.name);
      if (autoTag) {
        setCattleId(autoTag);
      }
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
    const autoTag = extractTagFromFilename(file.name);
    if (autoTag) {
      setCattleId(autoTag);
    }
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
    setLoading(true);
    setErrorMsg(null);

    try {
      const isPhotoObject = customPhotos && (customPhotos.left !== undefined || customPhotos.right !== undefined || customPhotos.rear !== undefined);
      const targetPhotos = isPhotoObject ? customPhotos : photos;

      let effectiveTag = cattleId;
      if (!effectiveTag) {
        const primaryFile = targetPhotos.left?.file || targetPhotos.right?.file || targetPhotos.rear?.file;
        if (primaryFile) {
          const autoTag = extractTagFromFilename(primaryFile.name);
          if (autoTag) {
            effectiveTag = autoTag;
            setCattleId(autoTag);
          }
        }
      }

      let endpoint = `${API_BASE}/api/predict`;
      const formData = new FormData();
      formData.append('cattle_id', effectiveTag || '');
      formData.append('model_engine', engine);

      if (enableVideo) {
        if (!videoFile) {
          throw new Error('Please select a video file (.mp4, .webm) for Innovation 1 Video Capture.');
        }
        endpoint = `${API_BASE}/api/predict_video`;
        formData.append('video_file', videoFile);
      } else if (enableDualAngle) {
        const sideFileToUse = dualSideFile || targetPhotos.left?.file || targetPhotos.right?.file;
        const rearFileToUse = dualRearFile || targetPhotos.rear?.file;
        if (!sideFileToUse || !rearFileToUse) {
          throw new Error('Dual-Angle capture requires both a side profile photo and a 45° rear view photo.');
        }
        endpoint = `${API_BASE}/api/predict_dual_angle`;
        formData.append('side_file', sideFileToUse);
        formData.append('rear_file', rearFileToUse);
      } else if (regressorEngine === 'kan') {
        const primaryFile = targetPhotos.left?.file || targetPhotos.right?.file || targetPhotos.rear?.file;
        if (!primaryFile) {
          throw new Error('Please upload or capture a profile photo to run KAN regression.');
        }
        endpoint = `${API_BASE}/api/predict_kan`;
        formData.append('file', primaryFile);
      } else if (regressorEngine === 'ensemble') {
        const primaryFile = targetPhotos.left?.file || targetPhotos.right?.file || targetPhotos.rear?.file;
        if (!primaryFile) {
          throw new Error('Please upload or capture a profile photo to run Ensemble regression.');
        }
        endpoint = `${API_BASE}/api/predict_ensemble`;
        formData.append('file', primaryFile);
        formData.append('alpha', ensembleAlpha.toString());
      } else {
        // Baseline single/multi-view prediction flow
        const hasPhoto = targetPhotos.left || targetPhotos.right || targetPhotos.rear;
        if (!hasPhoto) {
          throw new Error('Please upload or capture at least one profile photo (Left, Right, or Rear view).');
        }
        if (targetPhotos.left?.file) formData.append('left_file', targetPhotos.left.file);
        if (targetPhotos.right?.file) formData.append('right_file', targetPhotos.right.file);
        if (targetPhotos.rear?.file) formData.append('rear_file', targetPhotos.rear.file);
      }

      // Feature flag request headers
      const headers = {};
      if (enableUnwarp) headers['X-Enable-Unwarp'] = 'true';
      if (enableExif) headers['X-Enable-EXIF'] = 'true';
      if (showXai) headers['X-Enable-XAI'] = 'true';
      if (regressorEngine === 'kan') headers['X-Enable-KAN'] = 'true';
      if (enableVideo) headers['X-Enable-Video'] = 'true';
      if (enableDualAngle) headers['X-Enable-Dual-Angle'] = 'true';

      const flagsList = [];
      if (enableVideo) flagsList.push('ENABLE_VIDEO_KEYFRAME=true');
      if (enableUnwarp) flagsList.push('ENABLE_PERSPECTIVE_UNWARP=true');
      if (enableDualAngle) flagsList.push('ENABLE_DUAL_ANGLE=true');
      if (enableExif) flagsList.push('ENABLE_EXIF_CALIBRATION=true');
      if (regressorEngine === 'kan') flagsList.push('ENABLE_KAN=true');
      if (showXai) flagsList.push('ENABLE_XAI_CARDS=true');
      if (flagsList.length > 0) headers['X-Feature-Flags'] = flagsList.join(',');

      const res = await fetch(endpoint, {
        method: 'POST',
        headers,
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
    setVideoFile(null);
    setDualSideFile(null);
    setDualRearFile(null);
    setPrediction(null);
    setPredictionsByView(null);
    setErrorMsg(null);
  };

  const activePhoto = photos[activeViewTab];
  const isReady = photos.left || photos.right || photos.rear || (enableVideo && videoFile) || (enableDualAngle && (dualSideFile || photos.left) && (dualRearFile || photos.rear));

  return (
    <div className="app-container">
      {/* Mobile Drawer Backdrop Overlay */}
      {isMobileMenuOpen && (
        <div className="sidebar-backdrop" onClick={() => setIsMobileMenuOpen(false)}></div>
      )}

      {/* Left Sidebar / Off-Canvas Mobile Drawer */}
      <aside className={`sidebar ${isMobileMenuOpen ? 'mobile-open' : ''}`}>
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
              onClick={() => { setActiveTab('predict'); setIsMobileMenuOpen(false); }}
            >
              <Home size={18} />
              <span>Predict Weight</span>
            </button>

            <button 
              className={`nav-item ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => { setActiveTab('history'); setIsMobileMenuOpen(false); }}
            >
              <Clock size={18} />
              <span>History</span>
            </button>

            <button 
              className={`nav-item ${activeTab === 'herd' ? 'active' : ''}`}
              onClick={() => { setActiveTab('herd'); setIsMobileMenuOpen(false); }}
            >
              <Users size={18} />
              <span>Herd Management</span>
            </button>

            <button 
              className={`nav-item ${activeTab === 'analytics' ? 'active' : ''}`}
              onClick={() => { setActiveTab('analytics'); setIsMobileMenuOpen(false); }}
            >
              <BarChart3 size={18} />
              <span>Analytics</span>
            </button>

            <button 
              className={`nav-item ${activeTab === 'settings' ? 'active' : ''}`}
              onClick={() => { setActiveTab('settings'); setIsMobileMenuOpen(false); }}
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
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <button 
              className="mobile-menu-toggle"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              aria-label="Toggle navigation menu"
            >
              {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
            <div className="breadcrumb-path"></div>
          </div>

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
              {/* Collapsible Panel: Experimental Innovations */}
              <div style={{
                marginBottom: '1.25rem',
                border: '1px solid #CBD5E1',
                borderRadius: '10px',
                backgroundColor: '#F8FAFC',
                overflow: 'hidden'
              }}>
                <button
                  type="button"
                  onClick={() => setIsInnovationsOpen(!isInnovationsOpen)}
                  style={{
                    width: '100%',
                    padding: '0.75rem 1rem',
                    backgroundColor: '#F1F5F9',
                    border: 'none',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    cursor: 'pointer',
                    fontWeight: 700,
                    fontSize: '0.9rem',
                    color: 'var(--slate-dark)'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Sparkles size={18} color="var(--primary-sage)" />
                    <span>Experimental Innovations</span>
                    <span style={{ fontSize: '0.72rem', backgroundColor: '#E2E8F0', padding: '0.15rem 0.4rem', borderRadius: '12px', fontWeight: 600, color: '#475569' }}>6 Modules</span>
                  </div>
                  <span>{isInnovationsOpen ? '▲ Hide' : '▼ Expand'}</span>
                </button>

                {isInnovationsOpen && (
                  <div style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                    {/* 1. Innovation 1: Video Keyframe */}
                    <div style={{ padding: '0.75rem', backgroundColor: '#FFFFFF', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <label style={{ fontSize: '0.85rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
                          <input 
                            type="checkbox"
                            checked={enableVideo}
                            onChange={(e) => setEnableVideo(e.target.checked)}
                          />
                          <span>Use video capture (Innovation 1)</span>
                        </label>
                        <span title="Extracts optimal keyframe from uploaded cattle video sequence (.mp4, .webm)." style={{ cursor: 'help', fontSize: '0.8rem', color: '#64748B' }}>ℹ️ Info</span>
                      </div>
                      {enableVideo && (
                        <div style={{ marginTop: '0.6rem' }}>
                          <input 
                            type="file"
                            accept="video/mp4,video/webm,.mp4,.webm"
                            onChange={(e) => {
                              const file = e.target.files[0];
                              setVideoFile(file);
                              if (file) {
                                const autoTag = extractTagFromFilename(file.name);
                                if (autoTag) setCattleId(autoTag);
                              }
                            }}
                            style={{ fontSize: '0.8rem' }}
                          />
                          {videoFile && <span style={{ fontSize: '0.75rem', color: '#059669', marginLeft: '0.5rem', fontWeight: 600 }}>Selected: {videoFile.name}</span>}
                        </div>
                      )}
                    </div>

                    {/* 2. Innovation 2: Perspective Unwarp */}
                    <div style={{ padding: '0.75rem', backgroundColor: '#FFFFFF', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <label style={{ fontSize: '0.85rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
                          <input 
                            type="checkbox"
                            checked={enableUnwarp}
                            onChange={(e) => setEnableUnwarp(e.target.checked)}
                          />
                          <span>Enable perspective unwarping (Innovation 2)</span>
                        </label>
                        <span title="Applies 2D-to-3D keypoint transformation to correct pitch & yaw camera angle distortion." style={{ cursor: 'help', fontSize: '0.8rem', color: '#64748B' }}>ℹ️ Info</span>
                      </div>
                    </div>

                    {/* 3. Innovation 3: Dual-Angle Capture */}
                    <div style={{ padding: '0.75rem', backgroundColor: '#FFFFFF', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <label style={{ fontSize: '0.85rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
                          <input 
                            type="checkbox"
                            checked={enableDualAngle}
                            onChange={(e) => setEnableDualAngle(e.target.checked)}
                          />
                          <span>Dual-angle capture (Innovation 3)</span>
                        </label>
                        <span title="Fuses side profile contour with 45° rear view barrel width via cross-attention." style={{ cursor: 'help', fontSize: '0.8rem', color: '#64748B' }}>ℹ️ Info</span>
                      </div>
                      {enableDualAngle && (
                        <div style={{ marginTop: '0.6rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                          <div>
                            <span style={{ fontSize: '0.78rem', fontWeight: 600 }}>1. Side Profile Photo: </span>
                            <input 
                              type="file" 
                              accept="image/*"
                              onChange={(e) => {
                                const file = e.target.files[0];
                                setDualSideFile(file);
                                if (file) {
                                  const autoTag = extractTagFromFilename(file.name);
                                  if (autoTag) setCattleId(autoTag);
                                }
                              }}
                              style={{ fontSize: '0.78rem' }}
                            />
                            {dualSideFile && <span style={{ fontSize: '0.72rem', color: '#059669', marginLeft: '0.4rem' }}>{dualSideFile.name}</span>}
                          </div>
                          <div>
                            <span style={{ fontSize: '0.78rem', fontWeight: 600 }}>2. 45° Rear View Photo: </span>
                            <input 
                              type="file" 
                              accept="image/*"
                              onChange={(e) => {
                                const file = e.target.files[0];
                                setDualRearFile(file);
                                if (file) {
                                  const autoTag = extractTagFromFilename(file.name);
                                  if (autoTag) setCattleId(autoTag);
                                }
                              }}
                              style={{ fontSize: '0.78rem' }}
                            />
                            {dualRearFile && <span style={{ fontSize: '0.72rem', color: '#059669', marginLeft: '0.4rem' }}>{dualRearFile.name}</span>}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* 4. Innovation 4: EXIF Calibration */}
                    <div style={{ padding: '0.75rem', backgroundColor: '#FFFFFF', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <label style={{ fontSize: '0.85rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
                          <input 
                            type="checkbox"
                            checked={enableExif}
                            onChange={(e) => setEnableExif(e.target.checked)}
                          />
                          <span>Enable EXIF self-calibration (Innovation 4)</span>
                        </label>
                        <span title="Calibrates pixel-to-cm scale using camera EXIF focal length metadata." style={{ cursor: 'help', fontSize: '0.8rem', color: '#64748B' }}>ℹ️ Info</span>
                      </div>
                    </div>

                    {/* 5. Innovation 5: KAN / Ensemble */}
                    <div style={{ padding: '0.75rem', backgroundColor: '#FFFFFF', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
                        <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>Regressor engine</span>
                        <span title="Choose regression engine: XGBoost (baseline), PyTorch KAN (B-spline), or Ensemble." style={{ cursor: 'help', fontSize: '0.8rem', color: '#64748B' }}>ℹ️ Info</span>
                      </div>
                      <select
                        value={regressorEngine}
                        onChange={(e) => setRegressorEngine(e.target.value)}
                        style={{ width: '100%', padding: '0.4rem', borderRadius: '6px', border: '1px solid #CBD5E1', fontSize: '0.82rem', fontWeight: 600 }}
                      >
                        <option value="xgboost">XGBoost (default, baseline)</option>
                        <option value="kan">KAN (Innovation 5)</option>
                        <option value="ensemble">Ensemble (α slider from 0.0 to 1.0)</option>
                      </select>

                      {regressorEngine === 'ensemble' && (
                        <div style={{ marginTop: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <span style={{ fontSize: '0.78rem', fontWeight: 600 }}>Alpha (α = XGB weight): {ensembleAlpha}</span>
                          <input 
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={ensembleAlpha}
                            onChange={(e) => setEnsembleAlpha(parseFloat(e.target.value))}
                            style={{ flex: 1 }}
                          />
                        </div>
                      )}
                    </div>

                    {/* 6. Innovation 6: XAI Cards */}
                    <div style={{ padding: '0.75rem', backgroundColor: '#FFFFFF', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <label style={{ fontSize: '0.85rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
                          <input 
                            type="checkbox"
                            checked={showXai}
                            onChange={(e) => setShowXai(e.target.checked)}
                          />
                          <span>Show XAI heatmap (Innovation 6)</span>
                        </label>
                        <span title="Renders SHAP feature attribution heatmap overlay and model proof validation cards." style={{ cursor: 'help', fontSize: '0.8rem', color: '#64748B' }}>ℹ️ Info</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Tag Input */}
              <div className="tag-input-row">
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--slate-dark)' }}>🏷️ Cattle Tag ID:</span>
                <input 
                  type="text" 
                  className="tag-input-field"
                  value={cattleId}
                  onChange={(e) => {
                    const cleanVal = e.target.value.replace(/\.(jpg|jpeg|png|webp|bmp|gif|mp4|mov|avi|mkv|webm|flv|m4v|3gp|heic|tiff)$/i, '');
                    setCattleId(cleanVal);
                  }}
                  placeholder="e.g. 105730428938"
                />
              </div>

              {/* 3-View Acquisition Stepper & Checklist (Left Profile, Right Profile, Rear View) */}
              <div style={{ marginBottom: '1rem' }}>
                <p style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--slate-dark)', marginBottom: '0.5rem' }}>
                  📸 Multi-View Image Acquisition Stepper (3 Views)
                </p>
                <div className="stepper-grid">
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
              <div className="view-tabs-container">
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
                      onClick={() => handleRunPrediction()}
                      disabled={loading}
                      style={{ flex: 1, opacity: loading ? 0.85 : 1, cursor: loading ? 'not-allowed' : 'pointer' }}
                    >
                      {loading ? (
                        <>
                          <span className="spinner"></span>
                          <span>Analyzing & Running AI Neural Engine...</span>
                        </>
                      ) : (
                        <>
                          <Sparkles size={18} />
                          <span>🔮 Run AI Weight Estimation</span>
                        </>
                      )}
                    </button>

                    <button 
                      className="btn-secondary"
                      onClick={handleReset}
                      disabled={loading}
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

              {loading ? (
                /* State-of-the-Art AI Processing Loading Card */
                <div className="ai-loading-card">
                  <div className="ai-loading-radar">
                    <Sparkles size={28} />
                  </div>
                  <h4 style={{ fontFamily: 'var(--font-headline)', fontSize: '1.15rem', fontWeight: 700, marginBottom: '0.3rem', color: '#FFFFFF' }}>
                    AI Neural Engine Active
                  </h4>
                  <p style={{ fontSize: '0.82rem', color: '#94A3B8' }}>
                    Processing cattle profiles using {engine === 'yolo' ? '⚡ YOLOv8 Segmentation' : '🧠 DeepLabV3+ Semantic Engine'}...
                  </p>

                  <div className="ai-progress-track">
                    <div className="ai-progress-bar"></div>
                  </div>

                  <div className="ai-stage-list">
                    <div className="ai-stage-item active">
                      <span className="spinner" style={{ width: 12, height: 12, borderWidth: 1.5 }}></span>
                      <span> Isolating Cattle Silhouette Contour Mask</span>
                    </div>
                    <div className="ai-stage-item active">
                      <span style={{ color: '#10B981' }}>✓</span>
                      <span> Extracting Anatomical Keypoints & Morphometry</span>
                    </div>
                    <div className="ai-stage-item active">
                      <span style={{ color: '#10B981' }}>✓</span>
                      <span> Running XGBoost Biometric Weight Regressor</span>
                    </div>
                  </div>
                </div>
              ) : prediction ? (

                <>
                  {/* Big Metric Weight Card */}
                  <div className="metric-hero-card">
                    <div className="metric-hero-icon">
                      <Scale size={24} />
                    </div>
                    <div className="metric-hero-lbl">Estimated Weight</div>
                    <div className="metric-hero-val">{prediction.weight_kg} kg</div>
                    <div className="metric-hero-sub">
                      Engine: {regressorEngine === 'kan' ? 'KAN (Innovation 5)' : regressorEngine === 'ensemble' ? `Ensemble (α=${ensembleAlpha})` : 'XGBoost (Baseline)'}
                    </div>
                  </div>

                  {/* Innovation 1: Video Keyframe Thumbnail indicator */}
                  {enableVideo && (
                    <div style={{ padding: '0.5rem 0.75rem', backgroundColor: '#ECFDF5', border: '1px solid #10B981', borderRadius: '8px', marginBottom: '1rem', fontSize: '0.8rem', color: '#065F46', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <span>🎬 Keyframe selected from video capture (Innovation 1)</span>
                    </div>
                  )}

                  {/* Innovation 3: Dual-Angle Schaeffer Volumetric Estimate Note */}
                  {(enableDualAngle || prediction.dual_angle_fusion) && (
                    <div style={{ padding: '0.55rem 0.75rem', backgroundColor: '#E0F2FE', border: '1px solid #0284C7', borderRadius: '8px', marginBottom: '1rem', fontSize: '0.8rem', color: '#0369A1', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <span>📐 Dual-angle Schaeffer volumetric estimate</span>
                    </div>
                  )}

                  {/* Innovation 4: EXIF Scale Discrepancy Warning Badge */}
                  {prediction.exif_calibration?.flagged && (
                    <div style={{ padding: '0.65rem 0.85rem', backgroundColor: '#FEF3C7', border: '1px solid #F59E0B', borderRadius: '8px', marginBottom: '1rem', fontSize: '0.8rem', color: '#92400E', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <AlertCircle size={18} color="#D97706" />
                      <div>
                        <div>⚠️ EXIF Discrepancy &gt;10% Warning</div>
                        <div style={{ fontSize: '0.74rem', fontWeight: 500 }}>{prediction.exif_calibration.warning || 'Focal length scale mismatch detected.'}</div>
                      </div>
                    </div>
                  )}

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

                  {/* Innovation 6: Collapsible XAI Heatmap Card */}
                  {prediction.xai_heatmap_b64 && (
                    <div style={{ marginBottom: '1.25rem', padding: '1rem', backgroundColor: '#FFFFFF', borderRadius: '12px', border: '1px solid #E2E8F0' }}>
                      <h4 style={{ fontSize: '0.92rem', fontWeight: 700, marginBottom: '0.5rem', color: 'var(--slate-dark)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <Sparkles size={16} color="var(--primary-sage)" />
                        <span>🔥 XAI SHAP Heatmap (Innovation 6)</span>
                      </h4>
                      <img 
                        src={`data:image/png;base64,${prediction.xai_heatmap_b64}`} 
                        alt="XAI Feature Attribution Overlay"
                        style={{ width: '100%', borderRadius: '8px', border: '1px solid #CBD5E1', marginBottom: '0.4rem' }}
                      />
                      <p style={{ fontSize: '0.75rem', color: '#64748B', lineHeight: '1.4' }}>
                        <b>SHAP Region Mapping:</b> Ribcage (Chest Girth), Abdomen (Torso Volume), Withers (Height), Rump (Length).
                      </p>
                    </div>
                  )}
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

              <div className="table-responsive" style={{ marginTop: '1rem' }}>
                <table className="meas-table">
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
                        <td style={{ padding: '0.75rem' }}>{item.body_length || item.body_length_cm || item.measurements?.body_length_cm || '-'} cm</td>
                        <td style={{ padding: '0.75rem' }}>{item.chest_girth || item.chest_girth_cm || item.measurements?.chest_girth_cm || '-'} cm</td>
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
              </div>
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
