import Navbar from '../components/Navbar';
import LiveChart from '../components/LiveChart';
import ConfidenceBar from '../components/ConfidenceBar';
import ModelEvalPanel from '../components/ModelEvalPanel';
import StatusDot from '../components/StatusDot';
import { useWebSocket } from '../hooks/useWebSocket';

const CLASSES = ['correct', 'overextension', 'insufficient_range', 'wrong_plane'];
const CLASS_COLORS = {
  correct:            '#3fb950',
  overextension:      '#f85149',
  insufficient_range: '#d29922',
  wrong_plane:        '#bc8cff',
};

export default function PTForm() {
  const { data, connected } = useWebSocket('/ws/ptform');

  const label      = data?.label ?? '—';
  const confidence = data?.confidence ?? 0;
  const color      = data?.color ?? 'var(--text-muted)';
  const angles     = data?.jointAngles ?? [];
  const peak       = data?.peakAngle ?? 0;
  const rom        = data?.rom ?? 0;
  const probs      = data?.probabilities ?? {};

  return (
    <div className="screen fade-in">
      <Navbar title="PT Form Assessment" color="var(--purple)" />

      <div className="screen-body">
        <aside className="screen-sidebar">
          <div className="card" style={{ textAlign: 'center' }}>
            <h4 style={{ marginBottom: 12 }}>Prediction</h4>
            <div className="pred-badge" style={{ color, borderColor: color, margin: '0 auto 12px', fontSize: '1rem', padding: '10px 14px' }}>
              {label.replace(/_/g, ' ').toUpperCase()}
            </div>
            <ConfidenceBar value={confidence} color={color} />
          </div>

          <div className="card">
            <h4 style={{ marginBottom: 10 }}>Joint Metrics</h4>
            <div className="metrics-grid">
              <div className="metric-item">
                <div className="metric-label">Peak Angle</div>
                <div className="metric-value" style={{ color: 'var(--blue)' }}>{peak.toFixed(1)}°</div>
              </div>
              <div className="metric-item">
                <div className="metric-label">Range of Motion</div>
                <div className="metric-value" style={{ color: 'var(--purple)' }}>{rom.toFixed(1)}°</div>
              </div>
            </div>
          </div>

          <div className="card">
            <h4 style={{ marginBottom: 8 }}>Class Probabilities</h4>
            <div className="class-bars">
              {CLASSES.map(cls => {
                const pct = probs[cls] ?? 0;
                const c   = CLASS_COLORS[cls];
                return (
                  <div className="class-bar-row" key={cls}>
                    <span className="label" style={{ color: c }}>{cls.replace(/_/g, ' ')}</span>
                    <div className="track"><div className="fill" style={{ width: `${pct}%`, background: c }} /></div>
                    <span className="pct">{pct}%</span>
                  </div>
                );
              })}
            </div>
          </div>

          <ModelEvalPanel feature="ptform" />
        </aside>

        <div className="screen-main">
          <LiveChart
            data={angles}
            color="var(--purple)"
            title="Joint Angle (degrees)"
            yLabel="°"
            height="100%"
            referenceY={90}
            refColor="var(--yellow)"
          />
        </div>
      </div>

      <div className="screen-footer">
        <StatusDot mode={connected ? 'live' : 'offline'} label={connected ? 'Connected' : 'Reconnecting…'} />
        <span>Model: pt_form_lstm.h5 (LSTM)</span>
        <span>Window: 100 samples (2 s)</span>
        <span>Complementary filter — gyro + accel fusion</span>
      </div>
    </div>
  );
}
