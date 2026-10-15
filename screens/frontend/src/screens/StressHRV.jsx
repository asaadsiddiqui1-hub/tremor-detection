import { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import LiveChart from '../components/LiveChart';
import ConfidenceBar from '../components/ConfidenceBar';
import ModelEvalPanel from '../components/ModelEvalPanel';
import StatusDot from '../components/StatusDot';
import { useWebSocket } from '../hooks/useWebSocket';

export default function StressHRV() {
  const { data, connected, send } = useWebSocket('/ws/stress');
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (data && data.label && data.label !== '—') setResult(data);
  }, [data]);

  const ppg    = data?.ppgChart ?? [];
  const bufPct = data?.bufferPct ?? 0;
  const mode   = data?.mode ?? 'sim';

  const label      = result?.label ?? '—';
  const confidence = result?.confidence ?? 0;
  const color      = result?.color ?? 'var(--text-muted)';
  const hrBpm      = result?.hrBpm ?? 0;
  const rmssd      = result?.rmssd ?? 0;
  const lfHf       = result?.lfHf ?? 0;
  const sdnn       = result?.sdnn ?? 0;

  const isDemo = mode === 'sim';
  const ready  = true;  // accel-based — no 60s buffer needed

  const runInfer = () => send({ cmd: 'infer' });

  return (
    <div className="screen fade-in">
      <Navbar title="Stress / HRV" color="var(--blue)" />

      <div className="screen-body">
        <aside className="screen-sidebar">
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <h4 style={{ margin: 0 }}>Prediction</h4>
              {isDemo
                ? <span style={{ fontSize: '0.68rem', padding: '2px 8px', borderRadius: 4, border: '1px solid #ff9800', color: '#ff9800', background: 'rgba(255,152,0,0.1)' }}>DEMO</span>
                : <span style={{ fontSize: '0.68rem', padding: '2px 8px', borderRadius: 4, border: '1px solid var(--green)', color: 'var(--green)', background: 'rgba(63,185,80,0.1)' }}>LIVE</span>
              }
            </div>

            {ready ? (
              <>
                <div className="pred-badge" style={{ color, borderColor: color, margin: '0 auto 12px' }}>
                  {label.toUpperCase()}
                </div>
                <ConfidenceBar value={confidence} color={color} />
              </>
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '8px 0' }}>
                Accumulating 60 s of PPG…
              </div>
            )}

            <button
              className="btn primary"
              style={{ width: '100%', marginTop: 12 }}
              onClick={runInfer}
              disabled={!connected}
            >
              ▶ Run HRV Analysis
            </button>

            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 8 }}>
              Prediction based on MPU6050 hand motion · PPG chart is simulated
            </div>
          </div>

          <div className="card">
            <h4 style={{ marginBottom: 10 }}>HRV Metrics</h4>
            <div className="metrics-grid">
              <div className="metric-item">
                <div className="metric-label">Heart Rate</div>
                <div className="metric-value" style={{ color: 'var(--red)' }}>{hrBpm.toFixed(0)} bpm</div>
              </div>
              <div className="metric-item">
                <div className="metric-label">RMSSD</div>
                <div className="metric-value" style={{ color: 'var(--green)' }}>{rmssd.toFixed(1)} ms</div>
              </div>
              <div className="metric-item">
                <div className="metric-label">LF / HF</div>
                <div className="metric-value" style={{ color: 'var(--yellow)' }}>{lfHf.toFixed(2)}</div>
              </div>
              <div className="metric-item">
                <div className="metric-label">SDNN</div>
                <div className="metric-value">{sdnn.toFixed(1)} ms</div>
              </div>
            </div>
          </div>

          <ModelEvalPanel feature="stress" />
        </aside>

        <div className="screen-main">
          <LiveChart
            data={ppg}
            color="var(--orange)"
            title="PPG Signal (last 200 samples)"
            yLabel="raw"
            height="100%"
          />
        </div>
      </div>

      <div className="screen-footer">
        <StatusDot mode={connected ? 'live' : 'offline'} label={connected ? 'Connected' : 'Reconnecting…'} />
        <span>Model: stress_classifier.pkl (RandomForest)</span>
        <span>Window: 60 s at 50 Hz</span>
        <span>8 HRV features</span>
      </div>
    </div>
  );
}
