// components/Main/Steps.jsx
import React from 'react';

const phaseLabel = (p) => (p || '').replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase());

const StepItem = ({ step, index }) => {
  const color = {
    planning: '#6b7280',
    tool_selection: '#2563eb',
    execution: '#0ea5e9',
    observation: '#059669',
    reflection: '#8b5cf6',
    final_synthesis: '#111827',
  }[step.phase] || '#374151';

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '24px 1fr', gap: 12 }}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <div style={{
          width: 12,
          height: 12,
          borderRadius: '50%',
          backgroundColor: color,
          marginTop: 4,
        }} />
        <div style={{ flex: 1, width: 2, backgroundColor: '#e5e7eb', marginTop: 4, marginBottom: 4 }} />
      </div>
      <div>
        <div style={{ fontWeight: 600, fontSize: 13, color }}>{phaseLabel(step.phase)}</div>
        {step.content && (
          <div style={{ whiteSpace: 'pre-wrap', color: '#374151', fontSize: 13, marginTop: 4 }}>
            {step.content}
          </div>
        )}
        {typeof step.confidence === 'number' && (
          <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>
            Confidence: {(step.confidence * 100).toFixed(0)}%
          </div>
        )}
        {step.metadata && Object.keys(step.metadata).length > 0 && (
          <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>
            {Object.entries(step.metadata).map(([k, v]) => (
              <div key={k}><b>{k}:</b> {String(v)}</div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const Steps = ({ steps }) => {
  if (!steps || steps.length === 0) return null;
  return (
    <div style={{
      border: '1px solid #e5e7eb',
      background: '#ffffff',
      borderRadius: 12,
      padding: 12,
      marginBottom: 16,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <div style={{ fontWeight: 700, color: '#111827' }}>Processing Steps</div>
        <div style={{ fontSize: 12, color: '#6b7280' }}>{steps.length} step{steps.length !== 1 ? 's' : ''}</div>
      </div>
      <div style={{ display: 'grid', gap: 12 }}>
        {steps.map((s, i) => (
          <StepItem key={i} step={s} index={i} />
        ))}
      </div>
    </div>
  );
};

export default Steps;
