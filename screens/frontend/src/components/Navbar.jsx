// Top navigation bar used on feature screens
import { useNavigate } from 'react-router-dom';
import { useConnection } from '../context/ConnectionContext';
import StatusDot from './StatusDot';

export default function Navbar({ title, color }) {
  const navigate = useNavigate();
  const { status } = useConnection();

  return (
    <nav className="navbar">
      <button className="back-btn" onClick={() => navigate('/')}>
        ‹ HOME
      </button>

      <span className="screen-title" style={color ? { color, textShadow: `0 0 12px ${color}66` } : undefined}>
        {title}
      </span>

      <div style={{ flex: 1 }} />

      <div className="navbar-tac">
        <div className="tac-item">
          <span>SYS</span>
          <span className="tac-val">ONLINE</span>
        </div>
        <div className="tac-item">
          <span>RATE</span>
          <span className="tac-val">{status.sample_rate ?? 50} HZ</span>
        </div>
        <div className="tac-item">
          <span>MODE</span>
          <span className="tac-val" style={{
            color: status.mode === 'live' ? 'var(--green)' : 'var(--cyan)',
          }}>
            {status.mode?.toUpperCase() ?? 'SIM'}
          </span>
        </div>
      </div>

      <StatusDot
        mode={status.mode}
        label={status.mode === 'live' ? status.port : `SIM · ${status.sample_rate} Hz`}
      />
    </nav>
  );
}
