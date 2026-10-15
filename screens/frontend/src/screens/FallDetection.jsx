import Navbar from '../components/Navbar';
import LiveChart from '../components/LiveChart';
import ConfidenceBar from '../components/ConfidenceBar';
import ModelEvalPanel from '../components/ModelEvalPanel';
import StatusDot from '../components/StatusDot';
import { useWebSocket } from '../hooks/useWebSocket';

export default function FallDetection() {
  const { data, connected } = useWebSocket('/ws/fall');

  const label      = data?.prediction ?? 'NORMAL';
  const confidence = data?.confidence ?? 0;
  const chart      = data?.chart ?? [];
  const bufPct     = data?.buffer_pct ?? 0;
  const isFall     = label === 'FALL';
  const color      = isFall ? 'var(--red)' : 'var(--green)';

  return (
    <div className="screen fade-in">
      <Navbar title="Fall Detection" color="var(--red)" />

      <div className="screen-body">
        {/* Sidebar */}
        <aside className="screen-sidebar">
          <div className="card" style={{ textAlign: 'center' }}>
            <h4 style={{ marginBottom: 12 }}>Prediction</h4>
            <div className="pred-badge" style={{ color, borderColor: color, margin: '0 auto 12px' }}>
              {isFall ? '⚠ FALL' : 'NORMAL'}
            </div>
            <ConfidenceBar value={confidence} color={color} label="Fall probability" />
            <div className="buf-progress" style={{ marginTop: 12 }}>
              <div className="buf-label"><span>Buffer</span><span>{bufPct}%</span></div>
              <div className="buf-track"><div className="buf-fill" style={{ width: `${bufPct}%`, background: 'var(--blue)' }} /></div>
            </div>
          </div>

          <div className="metrics-grid">
            <div className="metric-item">
              <div className="metric-label">Status</div>
              <div className="metric-value" style={{ color, fontSize: '0.9rem' }}>{label}</div>
            </div>
            <div className="metric-item">
              <div className="metric-label">Prob</div>
              <div className="metric-value" style={{ color: 'var(--blue)' }}>{(data?.prob ?? 0).toFixed(3)}</div>
            </div>
            <div className="metric-item">
              <div className="metric-label">Threshold</div>
              <div className="metric-value">{(data?.threshold ?? 0.65).toFixed(2)}</div>
            </div>
            <div className="metric-item">
              <div className="metric-label">Samples</div>
              <div className="metric-value" style={{ fontSize: '0.85rem' }}>{chart.length}</div>
            </div>
          </div>

          <ModelEvalPanel feature="fall" />
        </aside>

        {/* Main chart */}
        <div className="screen-main">
          <LiveChart
            data={chart}
            color={isFall ? 'var(--red)' : 'var(--blue)'}
            title="Accel Magnitude (z-scored)"
            yLabel="σ"
            height="100%"
          />
        </div>
      </div>

      <div className="screen-footer">
        <StatusDot mode={connected ? 'live' : 'offline'} label={connected ? 'Connected' : 'Reconnecting…'} />
        <span>Model: fall_detection_cnn.h5</span>
        <span>Window: 512 samples (~10.24 s)</span>
        <span>Threshold: {(data?.threshold ?? 0.65).toFixed(2)}</span>
        <span>Missing fall ≻ false alarm</span>
      </div>
    </div>
  );
}
