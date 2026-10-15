// Home dashboard — 7 feature cards + connection controls
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useConnection } from '../context/ConnectionContext';

const CARDS = [
  { route: '/fall',    icon: '🫸', name: 'Fall Detection',     desc: '1D-CNN · 10.2 s window',      color: '#ff2d78', ready: true,  demo: false },
  { route: '/tremor',  icon: '🤝', name: 'Tremor Severity',   desc: 'RandomForest · 2 s window',    color: '#ffd600', ready: true,  demo: false },
  { route: '/stress',  icon: '💓', name: 'Stress / HRV',      desc: 'Simulated PPG · demo mode',    color: '#00d4ff', ready: true,  demo: true  },
  { route: '/atrophy', icon: '💪', name: 'Muscle Atrophy',    desc: 'Simulated HR · demo mode',     color: '#00e676', ready: true,  demo: true  },
  { route: '/ptform',  icon: '🏃', name: 'PT Form',           desc: 'LSTM · 2 s window',            color: '#7b2ff7', ready: true,  demo: false },
  { route: '/gesture', icon: '✋', name: 'Gesture Translator', desc: 'Live stream · model pending',  color: '#ff6d00', ready: true,  demo: false },
  { route: '/sensor',        icon: '📡', name: 'Live Sensor',         desc: 'Raw 7-channel stream',          color: '#00bcd4', ready: true,  demo: false },
  { route: '/tremor-binary', icon: '🧪', name: 'Tremor Binary (v2)',  desc: 'Real data · normal vs tremor',  color: '#ffd600', ready: true,  demo: false },
  { route: '/gesture-v2',    icon: '🖐', name: 'Gesture v2',          desc: '42 features · TTS · idle class', color: '#ff6d00', ready: true,  demo: false },
  { route: '/fall-binary',   icon: '🫸', name: 'Fall Binary (v2)',    desc: 'IsoForest · anomaly detection · 7 features', color: '#f85149', ready: true,  demo: false },
];

const SCENARIOS = ['normal', 'fall', 'tremor', 'stress'];

export default function Home() {
  const navigate = useNavigate();
  const { status, ports, fetchStatus, fetchPorts, connect, setSampleRate } = useConnection();
  const [selectedPort, setSelectedPort] = useState('COM3');
  const [selectedScenario, setSelectedScenario] = useState('normal');
  const [sampleHz, setSampleHz] = useState(50);
  const [connecting, setConnecting] = useState(false);

  useEffect(() => {
    fetchStatus();
    fetchPorts();
    const t = setInterval(fetchStatus, 3000);
    return () => clearInterval(t);
  }, [fetchStatus, fetchPorts]);

  const handleConnect = async (mode) => {
    setConnecting(true);
    await connect(mode, selectedPort, selectedScenario);
    setConnecting(false);
  };

  const handleSampleRate = async (e) => {
    const hz = Number(e.target.value);
    setSampleHz(hz);
    await setSampleRate(hz);
  };

  return (
    <div className="home-page fade-in">

      {/* Hero header */}
      <div className="home-hero">
        <h1 className="holo">NeuroPath AI</h1>
        <p className="sub">Wearable Diagnostic Glove · ESP32 + MPU6050</p>
      </div>

      {/* Connection bar */}
      <div className="conn-bar">
        {/* ESP32 connection indicator */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '6px 14px', borderRadius: 8,
          border: `1px solid ${status.mode === 'live' ? '#238636' : '#3d1f1f'}`,
          background: status.mode === 'live' ? 'rgba(35,134,54,0.12)' : 'rgba(248,81,73,0.10)',
          fontFamily: 'var(--font-mono)', fontSize: '0.78rem', whiteSpace: 'nowrap',
        }}>
          <span style={{
            width: 9, height: 9, borderRadius: '50%', flexShrink: 0,
            background: status.mode === 'live' ? '#2ea043' : '#f85149',
            boxShadow: status.mode === 'live' ? '0 0 6px #2ea043' : '0 0 6px #f85149',
            animation: status.mode === 'live' ? 'pulse 2s infinite' : 'none',
          }} />
          <span style={{ color: status.mode === 'live' ? '#3fb950' : '#f85149', fontWeight: 600 }}>
            {status.mode === 'live' ? 'ESP32 CONNECTED' : 'ESP32 NOT CONNECTED'}
          </span>
          {status.mode === 'live' && (
            <span style={{ color: 'var(--text-muted)' }}>
              {status.port} · {Math.round(status.actual_rate)} Hz
            </span>
          )}
        </div>

        <div className="input-group" style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
          <label>COM Port:</label>
          <select className="input" value={selectedPort} onChange={e => setSelectedPort(e.target.value)} style={{ width: 110 }}>
            {ports.length ? ports.map(p => <option key={p}>{p}</option>) : <option>COM3</option>}
          </select>
          <button className="btn primary" onClick={() => handleConnect('live')} disabled={connecting}>
            {connecting ? '…' : '🔌 Connect ESP32'}
          </button>
        </div>

        <div className="input-group" style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
          <label>Scenario:</label>
          <select className="input" value={selectedScenario} onChange={e => setSelectedScenario(e.target.value)} style={{ width: 100 }}>
            {SCENARIOS.map(s => <option key={s}>{s}</option>)}
          </select>
          <button className="btn" onClick={() => handleConnect('sim')} disabled={connecting}>
            🎭 Simulate
          </button>
        </div>

        <div className="spacer" />

        <div className="input-group" style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
          <label>Sample Rate:</label>
          <input
            type="number" className="input" min={1} max={500} value={sampleHz}
            onChange={handleSampleRate} style={{ width: 70 }}
          />
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Hz</span>
        </div>

        <div style={{ fontSize: '0.72rem', color: 'var(--text-faint)', textAlign: 'right', lineHeight: 1.6, fontFamily: 'var(--font-mono)' }}>
          <div>BUF <span style={{ color: 'var(--text-muted)' }}>{status.buffer_fill}%</span></div>
          <div>FRM <span style={{ color: 'var(--text-muted)' }}>{status.lines_read?.toLocaleString()}</span></div>
        </div>
      </div>

      {/* Feature cards grid */}
      <div className="cards-grid">
        {CARDS.map(card => (
          <div
            key={card.route}
            className={`feature-card ${!card.ready ? 'disabled' : ''}`}
            data-feature={card.route.slice(1)}
            onClick={() => card.ready && navigate(card.route)}
          >
            {/* Tinted ambient overlay */}
            <div style={{
              position: 'absolute', inset: 0, borderRadius: 'inherit',
              background: `linear-gradient(160deg, ${card.color}0d, transparent 55%)`,
              pointerEvents: 'none',
            }} />

            {/* Neural icon ring */}
            <div className="icon-ring" style={{
              borderColor: `${card.color}4a`,
              background: `radial-gradient(circle at 36% 28%, ${card.color}1c, rgba(2,4,16,0.97) 70%)`,
            }}>
              {card.icon}
            </div>

            <div className="name" style={{ color: card.color }}>{card.name}</div>
            <div className="desc">{card.desc}</div>

            {card.demo
              ? <span className="badge-status" style={{ color: '#ff9800', borderColor: '#ff9800', background: 'rgba(255,152,0,0.10)' }}>DEMO</span>
              : card.ready
                ? <span className="badge-status" style={status.mode === 'live' ? {} : { color: '#8b949e', borderColor: '#8b949e', background: 'rgba(139,148,158,0.10)' }}>
                    {status.mode === 'live' ? 'LIVE' : 'SIM'}
                  </span>
                : <span className="badge" style={{ color: 'var(--text-muted)', borderColor: 'var(--border)' }}>SOON</span>
            }
          </div>
        ))}
      </div>
    </div>
  );
}
