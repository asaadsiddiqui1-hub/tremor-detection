export default function StatusDot({ mode = 'sim', label }) {
  const dotClass = mode === 'live' ? 'live' : mode === 'offline' ? 'offline' : 'sim';
  return (
    <div className="conn-status">
      <span className={`status-dot ${dotClass}`} />
      <span>{label ?? (mode === 'live' ? 'Live' : mode === 'offline' ? 'Offline' : 'Simulated')}</span>
    </div>
  );
}
