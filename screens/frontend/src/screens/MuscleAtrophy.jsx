import { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import LiveChart from '../components/LiveChart';
import ModelEvalPanel from '../components/ModelEvalPanel';
import StatusDot from '../components/StatusDot';
import { useWebSocket } from '../hooks/useWebSocket';

export default function MuscleAtrophy() {
  const { data, connected, send } = useWebSocket('/ws/atrophy');
  const [age,  setAge]  = useState(35);
  const [imm,  setImm]  = useState(0);
  const [result, setResult] = useState(null);

  const chart = data?.type === 'chart' ? (data.accel ?? []) : [];

  useEffect(() => {
    if (data?.type === 'result') setResult(data);
  }, [data]);

  const res = result;

  const runInference = () => {
    send({ cmd: 'infer', age, immobilisedDays: imm });
  };

  const atrophyColor = res
    ? (res.atrophyPct < 5 ? 'var(--green)' : res.atrophyPct < 15 ? 'var(--yellow)' : 'var(--red)')
    : 'var(--text-muted)';

  return (
    <div className="screen fade-in">
      <Navbar title="Muscle Atrophy Estimate" color="var(--green)" />

      <div className="screen-body">

        {/* Sidebar — result only */}
        <aside className="screen-sidebar">
          {res ? (
            <div className="card" style={{ textAlign: 'center' }}>
              <h4 style={{ marginBottom: 10 }}>Estimated Atrophy</h4>
              <div style={{ fontSize: '2.6rem', fontWeight: 900, color: atrophyColor, fontFamily: 'var(--font-mono)' }}>
                {res.atrophyPct}%
              </div>
              <div className="metrics-grid" style={{ marginTop: 12 }}>
                <div className="metric-item">
                  <div className="metric-label">Spatial Var</div>
                  <div className="metric-value" style={{ fontSize: '0.88rem' }}>{res.spatialVar}</div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">Resting HR</div>
                  <div className="metric-value" style={{ fontSize: '0.88rem', color: 'var(--red)' }}>{res.restingHr} bpm</div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">HRV</div>
                  <div className="metric-value" style={{ fontSize: '0.88rem' }}>{res.hrVariability} ms</div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">Activity</div>
                  <div className="metric-value" style={{ fontSize: '0.88rem' }}>{res.activityLevel}</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="card" style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem', padding: '20px 14px' }}>
              Set inputs above and press Run Estimate
            </div>
          )}

          <div className="note-warning" style={{ fontSize: '0.72rem' }}>
            ⚠ Trained on real-time sensor data. Running in PPG simulation mode — MAX30102 sensor short-circuited
          </div>

          <ModelEvalPanel feature="atrophy" />
        </aside>

        {/* Main — input bar + chart */}
        <div className="screen-main">
          {/* Horizontal input bar */}
          <div className="card" style={{
            display: 'flex', alignItems: 'center', gap: 16,
            padding: '10px 16px', flexShrink: 0,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <label style={{ fontSize: '0.82rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>Age (yrs)</label>
              <input
                type="number" className="input" min={10} max={99} value={age}
                onChange={e => setAge(Number(e.target.value))}
                style={{ width: 72 }}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <label style={{ fontSize: '0.82rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>Days Immobilised</label>
              <input
                type="number" className="input" min={0} max={365} step={0.5} value={imm}
                onChange={e => setImm(Number(e.target.value))}
                style={{ width: 80 }}
              />
            </div>
            <button className="btn primary" onClick={runInference} style={{ whiteSpace: 'nowrap' }}>
              ▶ Run Estimate
            </button>
          </div>

          <LiveChart
            data={chart}
            color="var(--green)"
            title="Accel Magnitude (live)"
            yLabel="m/s²"
            height="100%"
          />
        </div>
      </div>

      <div className="screen-footer">
        <StatusDot mode={connected ? 'live' : 'offline'} label={connected ? 'Connected' : 'Reconnecting…'} />
        <span>Model: atrophy_regressor.pkl (GradBoost)</span>
        <span>MAE=0.44%  R²=0.96</span>
        <span>Inference: button-triggered</span>
      </div>
    </div>
  );
}
