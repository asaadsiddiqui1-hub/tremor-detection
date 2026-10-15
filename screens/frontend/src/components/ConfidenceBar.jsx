export default function ConfidenceBar({ value = 0, color = 'var(--blue)', label = 'Confidence' }) {
  return (
    <div className="conf-bar-wrap">
      <div className="conf-bar-label">
        <span>{label}</span>
        <span style={{ color }}>{value}%</span>
      </div>
      <div className="conf-bar-track">
        <div
          className="conf-bar-fill"
          style={{ width: `${value}%`, background: color }}
        />
      </div>
    </div>
  );
}
