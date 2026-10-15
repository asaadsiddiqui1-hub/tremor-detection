import Navbar from '../components/Navbar';
import LiveChart from '../components/LiveChart';
import StatusDot from '../components/StatusDot';
import { useWebSocket } from '../hooks/useWebSocket';

const STATE_META = {
  MONITORING:    { label: 'MONITORING',    color: '#3fb950' },
  FALL_DETECTED: { label: 'FALL DETECTED', color: '#f85149' },
};

export default function FallBinary() {
  const { data, connected } = useWebSocket('/ws/fall-binary');

  const modelReady   = data?.model_ready   ?? false;
  const prediction   = data?.prediction    ?? 'NORMAL';
  const confidence   = data?.confidence    ?? 0;
  const anomalyScore = data?.anomaly_score ?? null;
  const threshold    = data?.threshold     ?? null;
  const fallVotes    = data?.fall_votes    ?? 0;
  const state        = data?.state         ?? 'MONITORING';
  const chartMag     = data?.chart_mag     ?? [];

  const isFall    = prediction === 'FALL';
  const stateMeta = STATE_META[state] ?? STATE_META.MONITORING;
  const predColor  = isFall ? '#f85149' : '#3fb950';
  const cardBorder = isFall ? '1px solid #f8514966' : '1px solid #3fb95033';
  const cardBg     = isFall
    ? 'linear-gradient(160deg, #f8514918, transparent 60%)'
    : 'linear-gradient(160deg, #3fb95010, transparent 60%)';

  // Anomaly score bar: score relative to threshold
  // Score above threshold = more normal (green), below = anomaly (red)
  const hasScore = anomalyScore !== null && threshold !== null;
  const scoreBarPct = hasScore
    ? Math.max(0, Math.min(100, ((anomalyScore - threshold) / Math.max(Math.abs(threshold), 0.01)) * 50 + 50))
    : 50;
  const scoreColor = hasScore && anomalyScore < threshold ? '#f85149' : '#3fb950';

  return (
    <div className="screen fade-in">
      <Navbar title="Fall Binary (v2)" color="#f85149" />

      <div className="screen-body">
        <aside className="screen-sidebar">

          {/* Main prediction card */}
          <div className="card" style={{
            textAlign: 'center', border: cardBorder,
            background: cardBg, position: 'relative', overflow: 'hidden',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <h4 style={{ margin: 0 }}>Prediction</h4>
              <span style={{
                fontSize: '0.65rem', padding: '2px 8px', borderRadius: 4,
                border: '1px solid #f85149', color: '#f85149',
                background: 'rgba(248,81,73,0.08)',
              }}>
                v2 · IsoForest
              </span>
            </div>

            {modelReady ? (
              <>
                <div style={{
                  fontSize: '2.4rem', fontWeight: 900, letterSpacing: 3,
                  color: predColor, fontFamily: 'var(--font-mono)',
                  padding: '16px 0 8px',
                  textShadow: `0 0 24px ${predColor}55`,
                }}>
                  {prediction}
                </div>

                <div style={{
                  display: 'inline-block', marginBottom: 14,
                  padding: '3px 14px', borderRadius: 20,
                  border: `1px solid ${stateMeta.color}`,
                  color: stateMeta.color,
                  background: `${stateMeta.color}18`,
                  fontSize: '0.7rem', letterSpacing: 1,
                  animation: isFall ? 'pulse 0.8s infinite' : 'none',
                }}>
                  {stateMeta.label}
                </div>

                {/* Anomaly score bar */}
                {hasScore && (
                  <div style={{ marginTop: 14, textAlign: 'left' }}>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 4 }}>
                      Anomaly Score
                    </div>
                    <div style={{
                      height: 8, borderRadius: 4, background: 'var(--border)', overflow: 'hidden',
                    }}>
                      <div style={{
                        width: `${Math.round(scoreBarPct)}%`,
                        height: '100%',
                        background: scoreColor,
                        transition: 'width 0.3s ease, background 0.3s',
                        borderRadius: 4,
                      }} />
                    </div>
                    <div style={{
                      display: 'flex', justifyContent: 'space-between',
                      fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 3,
                    }}>
                      <span style={{ color: scoreColor }}>
                        {anomalyScore.toFixed(4)}
                      </span>
                      <span>threshold {threshold.toFixed(4)}</span>
                    </div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-faint)', marginTop: 2 }}>
                      score &lt; threshold → FALL
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '16px 0' }}>
                Loading model…
              </div>
            )}
          </div>

          {/* Model info */}
          <div className="card">
            <h4 style={{ marginBottom: 10 }}>Model Info</h4>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 2 }}>
              <div>Type: Isolation Forest</div>
              <div>Features: 7 (accel magnitude)</div>
              <div>Window: 100 samples (2 s)</div>
              <div>Normal class: glove normal data only</div>
              <div>Threshold: {threshold !== null ? threshold.toFixed(4) : '—'}</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                Debounce votes:
                <span style={{ fontFamily: 'var(--font-mono)', color: isFall ? '#f85149' : '#3fb950' }}>
                  {fallVotes}/5
                </span>
              </div>
            </div>
          </div>

          {/* Class legend */}
          <div className="card">
            <h4 style={{ marginBottom: 10 }}>Classes</h4>
            {[
              { label: 'Normal',  color: '#3fb950', active: !isFall },
              { label: 'Fall',    color: '#f85149', active: isFall  },
            ].map(cls => (
              <div key={cls.label} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '5px 0', borderBottom: '1px solid var(--border)',
              }}>
                <div style={{
                  width: 10, height: 10, borderRadius: '50%',
                  background: cls.color,
                  boxShadow: cls.active ? `0 0 8px ${cls.color}` : 'none',
                  flexShrink: 0,
                }} />
                <span style={{
                  fontSize: '0.8rem',
                  color: cls.active ? cls.color : 'var(--text-muted)',
                  fontWeight: cls.active ? 700 : 400,
                }}>
                  {cls.label}
                </span>
                {cls.active && (
                  <span style={{ marginLeft: 'auto', fontSize: '0.68rem', color: cls.color }}>●</span>
                )}
              </div>
            ))}
          </div>

        </aside>

        <div className="screen-main">
          <LiveChart
            data={chartMag}
            color={isFall ? '#f85149' : '#3fb950'}
            title="Accel Magnitude"
            yLabel="m/s²"
          />
        </div>
      </div>

      <div className="screen-footer">
        <StatusDot mode={connected ? 'live' : 'offline'} label={connected ? 'Connected' : 'Reconnecting…'} />
        <span>Model: fall_binary_classifier.pkl</span>
        <span>IsolationForest · 7 features · anomaly detection</span>
        <span>normal | fall</span>
      </div>
    </div>
  );
}
