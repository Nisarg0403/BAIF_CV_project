import React from 'react';
import { Award, CheckCircle2, AlertTriangle, ShieldCheck } from 'lucide-react';

export default function ModelProofsCard({ activeEngine }) {
  return (
    <div className="proof-badge-card">
      <div className="proof-header">
        <div className="flex items-center gap-2" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <ShieldCheck size={18} color="#059669" />
          <span>AI Model Validation Proof</span>
        </div>
        <span className="proof-badge">N=15 BAIF Dataset</span>
      </div>

      <p style={{ fontSize: '0.78rem', color: '#475569', marginBottom: '0.75rem' }}>
        Validated against physical weighbridge & measuring tape ground-truth records across BAIF Sahiwal and HF-Cross dairy cattle.
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.8rem' }}>
        {/* DeepLabV3 Card */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0.5rem 0.75rem',
          borderRadius: '8px',
          backgroundColor: activeEngine === 'deeplab' ? '#ECFDF5' : '#F1F5F9',
          border: activeEngine === 'deeplab' ? '1px solid #A7F3D0' : '1px solid #E2E8F0'
        }}>
          <div>
            <div style={{ fontWeight: 700, color: '#065F46', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
              <CheckCircle2 size={14} color="#059669" />
              <span>DeepLabV3+-ResNet50</span>
            </div>
            <span style={{ fontSize: '0.72rem', color: '#047857' }}>Pixel-wise Semantic Segmentation</span>
          </div>
          <div style={{ textAlign: 'right' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#059669', fontSize: '0.9rem' }}>
              ±4.93 kg MAE
            </span>
            <div style={{ fontSize: '0.7rem', color: '#047857' }}>R² = 0.9937 (High Precision)</div>
          </div>
        </div>

        {/* YOLOv8 Card */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0.5rem 0.75rem',
          borderRadius: '8px',
          backgroundColor: activeEngine === 'yolo' ? '#FEF3C7' : '#F8FAFC',
          border: activeEngine === 'yolo' ? '1px solid #FDE68A' : '1px solid #E2E8F0'
        }}>
          <div>
            <div style={{ fontWeight: 600, color: '#475569', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
              <AlertTriangle size={14} color="#D97706" />
              <span>YOLOv8-Segmentation</span>
            </div>
            <span style={{ fontSize: '0.72rem', color: '#64748B' }}>Fast Bounding Polygon Masking</span>
          </div>
          <div style={{ textAlign: 'right' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#D97706', fontSize: '0.9rem' }}>
              ±51.2 kg MAE
            </span>
            <div style={{ fontSize: '0.7rem', color: '#92400E' }}>Fast Real-time Preview</div>
          </div>
        </div>
      </div>
    </div>
  );
}
