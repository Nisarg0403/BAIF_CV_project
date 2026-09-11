import React, { useRef, useState, useEffect } from 'react';
import { Camera, X, RefreshCw, Sparkles } from 'lucide-react';

export default function LiveCameraModal({ isOpen, onClose, onCapture, activeViewTab }) {
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [facingMode, setFacingMode] = useState('environment'); // Default to rear camera
  const [cameraError, setCameraError] = useState(null);

  useEffect(() => {
    if (isOpen) {
      startCamera(facingMode);
    } else {
      stopCamera();
    }
    return () => stopCamera();
  }, [isOpen, facingMode]);

  const startCamera = async (mode) => {
    stopCamera();
    setCameraError(null);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: mode },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err) {
      console.warn('Primary camera error, trying fallback:', err);
      try {
        const fallbackStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        setStream(fallbackStream);
        if (videoRef.current) {
          videoRef.current.srcObject = fallbackStream;
        }
      } catch (fallbackErr) {
        console.error('Camera permission denied or missing:', fallbackErr);
        setCameraError('Camera access required. Please allow camera permissions in your browser settings.');
      }
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
  };

  const toggleFacingMode = () => {
    setFacingMode(prev => (prev === 'environment' ? 'user' : 'environment'));
  };

  const handleSnapAndFeed = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `cattle_snap_${Date.now()}.jpg`, { type: 'image/jpeg' });
        onCapture(file);
        onClose();
      }
    }, 'image/jpeg', 0.92);
  };

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      zIndex: 9999,
      backgroundColor: 'rgba(15, 23, 42, 0.95)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '1rem'
    }}>
      {/* Top Header */}
      <div style={{
        width: '100%',
        maxWidth: '640px',
        display: 'flex',
        alignItems: 'center',
        justify: 'space-between',
        color: '#FFFFFF',
        marginBottom: '0.75rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Camera size={20} color="#10B981" />
          <span style={{ fontWeight: 700, fontSize: '1rem' }}>
            Live Camera - {activeViewTab === 'left' ? 'Left Profile' : activeViewTab === 'right' ? 'Right Profile' : 'Rear View'}
          </span>
        </div>
        <button 
          onClick={onClose}
          style={{ background: 'transparent', border: 'none', color: '#94A3B8', cursor: 'pointer', padding: '0.4rem' }}
        >
          <X size={24} />
        </button>
      </div>

      {/* Video Viewfinder Container */}
      <div style={{
        position: 'relative',
        width: '100%',
        maxWidth: '640px',
        height: '420px',
        backgroundColor: '#000000',
        borderRadius: '16px',
        overflow: 'hidden',
        border: '2px solid rgba(16, 185, 129, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justify: 'center'
      }}>
        {cameraError ? (
          <div style={{ color: '#FCA5A5', padding: '1.5rem', textAlign: 'center', fontSize: '0.9rem' }}>
            {cameraError}
          </div>
        ) : (
          <>
            <video 
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            />

            {/* Bounding Box Alignment Guide Overlay */}
            <div style={{
              position: 'absolute',
              inset: '20px',
              border: '2px dashed rgba(255, 255, 255, 0.7)',
              borderRadius: '12px',
              pointerEvents: 'none',
              display: 'flex',
              flexDirection: 'column',
              justify: 'space-between',
              padding: '10px'
            }}>
              <span style={{ color: '#34D399', fontSize: '0.75rem', fontWeight: 700, background: 'rgba(0,0,0,0.6)', padding: '2px 8px', borderRadius: '4px', alignSelf: 'flex-start' }}>
                📐 Align Cattle Silhouette Inside Frame
              </span>
              <span style={{ color: '#94A3B8', fontSize: '0.7rem', background: 'rgba(0,0,0,0.6)', padding: '2px 8px', borderRadius: '4px', alignSelf: 'center' }}>
                Ensure full cow body is visible
              </span>
            </div>
          </>
        )}
      </div>

      {/* Action Controls Footer */}
      {!cameraError && (
        <div style={{
          width: '100%',
          maxWidth: '640px',
          display: 'flex',
          alignItems: 'center',
          justify: 'space-around',
          marginTop: '1.25rem'
        }}>
          <button
            onClick={toggleFacingMode}
            style={{
              width: '48px', height: '48px', borderRadius: '50%',
              backgroundColor: 'rgba(255, 255, 255, 0.15)', border: '1px solid rgba(255, 255, 255, 0.3)',
              color: '#FFFFFF', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer'
            }}
            title="Switch Camera (Front/Rear)"
          >
            <RefreshCw size={20} />
          </button>

          {/* Shutter Button */}
          <button
            onClick={handleSnapAndFeed}
            style={{
              padding: '0.8rem 2rem',
              borderRadius: '30px',
              backgroundColor: '#10B981',
              color: '#FFFFFF',
              border: 'none',
              fontSize: '1rem',
              fontWeight: 700,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              boxShadow: '0 4px 14px rgba(16, 185, 129, 0.4)'
            }}
          >
            <Sparkles size={20} />
            <span>Snap & Estimate Now</span>
          </button>
        </div>
      )}
    </div>
  );
}
