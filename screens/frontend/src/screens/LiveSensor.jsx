import Navbar from '../components/Navbar';
import StatusDot from '../components/StatusDot';
import { useWebSocket } from '../hooks/useWebSocket';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';
import { useMemo } from 'react';

const CHANNELS = [
  { key: 'accel_x', label: 'Accel X',  color: '#58a6ff', unit: 'm/s²' },
  { key: 'accel_y', label: 'Accel Y',  color: '#3fb950', unit: 'm/s²' },
  { key: 'accel_z', label: 'Accel Z',  color: '#d29922', unit: 'm/s²' },
  { key: 'gyro_x',  label: 'Gyro X',   color: '#bc8cff', unit: 'rad/s' },
  { key: 'gyro_y',  label: 'Gyro Y',   color: '#e07b39', unit: 'rad/s' },
  { key: 'gyro_z',  label: 'Gyro Z',   color: '#f85149', unit: 'rad/s' },
  { key: 'ppg_raw', label: 'PPG Raw',  color: '#ff7b93', unit: 'count' },
];

function MiniChart({ values = [], color, label, unit }) {
  const chartData = useMemo(() =>
    values.map((v, i) => ({ i, v: parseFloat((v || 0).toFixed(3)) })), [values]
  );
  const latest = values.at(-1);
  return (
    <div className="sensor-mini">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
        <h4 style={{ color }}>{label}</h4>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color }}>
          {latest != null ? latest.toFixed(3) : '—'} <span style={{ color: 'var(--text-faint)', fontSize: '0.7rem' }}>{unit}</span>
        </span>
      </div>
      <ResponsiveContainer width="100%" height={90}>
        <LineChart data={chartData} margin={{ top: 0, right: 2, left: -30, bottom: 0 }}>
          <CartesianGrid strokeDasharray="2 2" stroke="var(--border)" />
          <XAxis dataKey="i" hide />
          <YAxis domain={['auto', 'auto']} tick={{ fill: 'var(--text-muted)', fontSize: 10 }} tickLine={false} axisLine={false} />
          <Line type="monotoneX" dataKey="v" stroke={color} strokeWidth={1.5} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function LiveSensor() {
  const { data, connected } = useWebSocket('/ws/sensor');
  const channels = data?.channels ?? {};
  const latest   = data?.latest   ?? {};
  const sampleRate = data?.sample_rate ?? '—';
  const linesRead  = data?.lines_read ?? 0;

  return (
    <div className="screen fade-in">
      <Navbar title="Live Sensor Stream" color="var(--text-muted)" />

      <div className="screen-body" style={{ padding: 12 }}>
        <div className="sensor-grid" style={{ flex: 1 }}>
          {CHANNELS.map(ch => (
            <MiniChart
              key={ch.key}
              values={channels[ch.key] ?? []}
              color={ch.color}
              label={ch.label}
              unit={ch.unit}
            />
          ))}
        </div>
      </div>

      <div className="screen-footer">
        <StatusDot mode={connected ? 'live' : 'offline'} label={connected ? 'Connected' : 'Reconnecting…'} />
        <span>7 channels · 100 ms refresh</span>
        <span>Actual rate: {Math.round(sampleRate)} Hz</span>
        <span>Frames: {linesRead?.toLocaleString()}</span>
        {Object.entries(latest).map(([k, v]) => (
          <span key={k}>{k}: <strong style={{ color: 'var(--text)' }}>{v}</strong></span>
        ))}
      </div>
    </div>
  );
}
