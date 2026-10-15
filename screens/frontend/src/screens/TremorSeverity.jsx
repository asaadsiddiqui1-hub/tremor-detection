import Navbar from '../components/Navbar';
import LiveChart from '../components/LiveChart';
import ConfidenceBar from '../components/ConfidenceBar';
import ModelEvalPanel from '../components/ModelEvalPanel';
import StatusDot from '../components/StatusDot';
import { useWebSocket } from '../hooks/useWebSocket';

const SEVERITY_COLORS = { mild: '#3fb950', moderate: '#d29922', high: '#f85149' };

export default function TremorSeverity() {
  const { data, connected } = useWebSocket('/ws/tremor');

  const severity   = data?.severity ?? '—';
  const confidence = data?.confidence ?? 0;
  const chart      = data?.chart ?? [];
  const color      = data?.color ?? 'var(--text-muted)';
  const domFreq    = data?.dominantFreq ?? 0;
  const pPark      = data?.parkinsonPct ?? 0;
  const pEss       = data?.essentialPct ?? 0;

  return (
    <div className="screen fade-in">
      <Navbar title="Tremor Severity" color="var(--yellow)" />

      <div className="screen-body">
        <aside className="screen-sidebar">
          <div className="card" style={{ textAlign: 'center' }}>
            <h4 style={{ marginBottom: 12 }}>Prediction</h4>
            <div className="pred-badge" style={{ color, borderColor: color, margin: '0 auto 12px', fontSize: '1.6rem' }}>
              {severity.toUpperCase()}
            </div>
            <ConfidenceBar value={confidence} color={color} />
          </div>

          <div className="card">
            <h4 style={{ marginBottom: 10 }}>Band Power Metrics</h4>
            <div className="metrics-grid">
              <div className="metric-item">
                <div className="metric-label">Dominant Freq</div>
                <div className="metric-value" style={{ color: 'var(--blue)' }}>{domFreq.toFixed(1)} Hz</div>
              </div>
              <div className="metric-item">
                <div className="metric-label">Parkinsonian</div>
                <div className="metric-value" style={{ color: 'var(--red)' }}>{pPark.toFixed(1)}%</div>
              </div>
              <div className="metric-item">
                <div className="metric-label">Essential</div>
                <div className="metric-value" style={{ color: 'var(--yellow)' }}>{pEss.toFixed(1)}%</div>
              </div>
              <div className="metric-item">
                <div className="metric-label">Parkinson Band</div>
                <div className="metric-value" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>3–6 Hz</div>
              </div>
            </div>
          </div>

          {/* Per-class probabilities */}
          {data?.probabilities?.length > 0 && (
            <div className="card">
              <h4 style={{ marginBottom: 8 }}>Class Probabilities</h4>
              <div className="class-bars">
                {(data.classes ?? ['mild','moderate','high']).map((cls, i) => {
                  const pct = data.probabilities[i] ?? 0;
                  const c = SEVERITY_COLORS[{1:'mild',2:'moderate',3:'high'}[cls]] ?? 'var(--text-muted)';
                  return (
                    <div className="class-bar-row" key={cls}>
                      <span className="label">{cls}</span>
                      <div className="track"><div className="fill" style={{ width: `${pct}%`, background: c }} /></div>
                      <span className="pct">{pct}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <ModelEvalPanel feature="tremor" />
        </aside>

        <div className="screen-main">
          <LiveChart
            data={chart}
            color={color}
            title="Dominant Accel Axis"
            yLabel="g"
            height="100%"
          />
        </div>
      </div>

      <div className="screen-footer">
        <StatusDot mode={connected ? 'live' : 'offline'} label={connected ? 'Connected' : 'Reconnecting…'} />
        <span>Model: tremor_classifier.pkl (RandomForest)</span>
        <span>Window: 100 samples (2 s)</span>
        <span>15 FFT features</span>
      </div>
    </div>
  );
}
