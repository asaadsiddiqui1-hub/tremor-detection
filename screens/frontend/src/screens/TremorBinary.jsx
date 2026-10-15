import Navbar from '../components/Navbar';
import LiveChart from '../components/LiveChart';
import ConfidenceBar from '../components/ConfidenceBar';
import ModelEvalPanel from '../components/ModelEvalPanel';
import StatusDot from '../components/StatusDot';
import { useWebSocket } from '../hooks/useWebSocket';

export default function TremorBinary() {
  const { data, connected } = useWebSocket('/ws/tremor-binary');

  const result     = data?.result     ?? '—';
  const confidence = data?.confidence ?? 0;
  const color      = data?.color      ?? 'var(--text-muted)';
  const domFreq    = data?.dominantFreq ?? 0;
  const pPark      = data?.parkinsonPct ?? 0;
  const pEss       = data?.essentialPct ?? 0;
  const chart      = data?.chart ?? [];

  const isNormal = result === 'normal';
  const isTremor = result === 'tremor';

  return (
    <div className="screen fade-in">
      <Navbar title="Tremor Detection (Binary)" color="#ffd600" />

      <div className="screen-body">
        <aside className="screen-sidebar">

          {/* Prediction */}
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <h4 style={{ margin: 0 }}>Detection</h4>
              <span style={{ fontSize: '0.65rem', padding: '2px 8px', borderRadius: 4,
                border: '1px solid #ffd600', color: '#ffd600', background: 'rgba(255,214,0,0.08)' }}>
                BINARY v2
              </span>
            </div>

            <div className="pred-badge" style={{ color, borderColor: color, margin: '0 auto 12px' }}>
              {result === '—' ? '—' : result.toUpperCase()}
            </div>

            {result !== '—' && (
              <ConfidenceBar value={confidence} color={color} />
            )}

            {/* Status indicator */}
            <div style={{
              marginTop: 12, padding: '6px 10px', borderRadius: 6,
              background: isNormal ? 'rgba(63,185,80,0.1)' : isTremor ? 'rgba(248,81,73,0.1)' : 'rgba(139,148,158,0.1)',
              border: `1px solid ${isNormal ? '#3fb950' : isTremor ? '#f85149' : '#6e7681'}`,
              fontSize: '0.75rem',
              color: isNormal ? '#3fb950' : isTremor ? '#f85149' : 'var(--text-muted)',
            }}>
              {isNormal && 'No pathological tremor detected'}
              {isTremor && 'Tremor pattern detected'}
              {result === '—' && 'Waiting for data…'}
            </div>
          </div>

          {/* Frequency metrics */}
          <div className="card">
            <h4 style={{ marginBottom: 10 }}>Frequency Analysis</h4>
            <div className="metrics-grid">
              <div className="metric-item">
                <div className="metric-label">Dominant Freq</div>
                <div className="metric-value">{domFreq.toFixed(1)} Hz</div>
              </div>
              <div className="metric-item">
                <div className="metric-label">Parkinson Band</div>
                <div className="metric-value" style={{ color: pPark > 40 ? 'var(--red)' : 'inherit' }}>
                  {pPark.toFixed(1)}%
                </div>
              </div>
              <div className="metric-item">
                <div className="metric-label">Essential Band</div>
                <div className="metric-value" style={{ color: pEss > 40 ? 'var(--yellow)' : 'inherit' }}>
                  {pEss.toFixed(1)}%
                </div>
              </div>
              <div className="metric-item">
                <div className="metric-label">Model</div>
                <div className="metric-value" style={{ fontSize: '0.75rem' }}>RF Binary</div>
              </div>
            </div>
          </div>

          {/* Model info */}
          <div className="card">
            <h4 style={{ marginBottom: 8 }}>Model Info</h4>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.9 }}>
              <div>Training: real glove data</div>
              <div>Normal: ~1,755 s recorded</div>
              <div>Tremor: Parkinson's dataset</div>
              <div>Test accuracy: 99%</div>
              <div>Tremor recall: 83%</div>
            </div>
          </div>

        </aside>

        <div className="screen-main">
          <LiveChart
            data={chart}
            color="#ffd600"
            title="Dominant Accel Axis (g)"
            yLabel="g"
            height="100%"
          />
        </div>
      </div>

      <div className="screen-footer">
        <StatusDot mode={connected ? 'live' : 'offline'} label={connected ? 'Connected' : 'Reconnecting…'} />
        <span>Model: tremor_binary_classifier.pkl (RandomForest)</span>
        <span>Binary: normal vs tremor</span>
        <span>15 FFT features · 2 s window</span>
      </div>
    </div>
  );
}
