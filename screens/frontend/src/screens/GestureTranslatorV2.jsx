import Navbar from '../components/Navbar';
import LiveChart from '../components/LiveChart';
import ConfidenceBar from '../components/ConfidenceBar';
import StatusDot from '../components/StatusDot';
import { useWebSocket } from '../hooks/useWebSocket';

const GESTURE_COLORS = {
  goodbye: '#58a6ff',
  thanks:  '#3fb950',
  welcome: '#ff9800',
};

const STATE_META = {
  LISTENING: { label: 'LISTENING', color: '#6e7681', pulse: true  },
  DETECTED:  { label: 'DETECTED',  color: '#3fb950', pulse: false },
  COOLDOWN:  { label: 'COOLDOWN',  color: '#d29922', pulse: false },
  UNCLEAR:   { label: 'UNCLEAR',   color: '#8b949e', pulse: false },
};

export default function GestureTranslatorV2() {
  const { data, connected } = useWebSocket('/ws/gesture-v2');

  const modelReady  = data?.model_ready ?? false;
  const gesture     = data?.gesture    ?? '—';
  const confidence  = data?.confidence ?? 0;
  const state       = data?.state      ?? 'LISTENING';
  const featCount   = data?.features   ?? '…';
  const modelType   = data?.model_type ?? '…';

  const accelX = data?.accel_x ?? [];
  const accelY = data?.accel_y ?? [];
  const accelZ = data?.accel_z ?? [];
  const gyroX  = data?.gyro_x  ?? [];

  const gestureColor = GESTURE_COLORS[gesture] ?? 'var(--text-muted)';
  const stateMeta    = STATE_META[state] ?? STATE_META.LISTENING;
  const isDetected   = state === 'DETECTED';

  return (
    <div className="screen fade-in">
      <Navbar title="Gesture Translator v2" color="var(--orange)" />

      <div className="screen-body">
        <aside className="screen-sidebar">

          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <h4 style={{ margin: 0 }}>Gesture</h4>
              <span style={{ fontSize: '0.65rem', padding: '2px 8px', borderRadius: 4,
                border: '1px solid var(--orange)', color: 'var(--orange)',
                background: 'rgba(255,109,0,0.08)' }}>
                v2 · {featCount} features
              </span>
            </div>

            {modelReady ? (
              <>
                {/* Gesture name + voice indicator */}
                <div style={{
                  fontSize: '2rem', fontWeight: 900, letterSpacing: 2,
                  color: gestureColor, fontFamily: 'var(--font-mono)',
                  padding: '12px 0 4px',
                  textShadow: gesture !== '—' ? `0 0 20px ${gestureColor}66` : 'none',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                }}>
                  {gesture !== '—' ? gesture.toUpperCase() : '—'}
                  {isDetected && gesture !== '—' && (
                    <span style={{ fontSize: '1.2rem' }} title="Speaking aloud">🔊</span>
                  )}
                </div>

                {/* State badge */}
                <div style={{
                  display: 'inline-block', marginBottom: 12, marginTop: 4,
                  padding: '3px 12px', borderRadius: 20,
                  border: `1px solid ${stateMeta.color}`,
                  color: stateMeta.color,
                  background: `${stateMeta.color}18`,
                  fontSize: '0.7rem', letterSpacing: 1,
                  animation: stateMeta.pulse ? 'pulse 2s infinite' : 'none',
                }}>
                  {stateMeta.label}
                </div>

                {state !== 'LISTENING' && (
                  <ConfidenceBar value={confidence} color={gestureColor} />
                )}

                {isDetected && (
                  <div style={{
                    marginTop: 8, fontSize: '0.72rem',
                    color: 'var(--text-muted)', fontStyle: 'italic',
                  }}>
                    Speaking "{gesture}" aloud…
                  </div>
                )}
              </>
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '16px 0' }}>
                Loading model…
              </div>
            )}
          </div>

          {/* Gesture list */}
          <div className="card">
            <h4 style={{ marginBottom: 10 }}>Gestures</h4>
            {['goodbye', 'thanks', 'welcome'].map(g => (
              <div key={g} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '5px 0', borderBottom: '1px solid var(--border)',
              }}>
                <div style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: GESTURE_COLORS[g],
                  boxShadow: gesture === g ? `0 0 8px ${GESTURE_COLORS[g]}` : 'none',
                  flexShrink: 0,
                }} />
                <span style={{
                  fontSize: '0.8rem', textTransform: 'capitalize',
                  color: gesture === g ? GESTURE_COLORS[g] : 'var(--text-muted)',
                  fontWeight: gesture === g ? 700 : 400,
                }}>
                  {g}
                </span>
                {gesture === g && isDetected && (
                  <span style={{ marginLeft: 'auto', fontSize: '0.7rem', color: GESTURE_COLORS[g] }}>
                    🔊
                  </span>
                )}
              </div>
            ))}
          </div>

          {/* Detection info — live from backend */}
          <div className="card">
            <h4 style={{ marginBottom: 8 }}>Model v2</h4>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.9 }}>
              <div>Algorithm: <span style={{ color: 'var(--cyan)' }}>{modelType}</span></div>
              <div>Features: <span style={{ color: 'var(--cyan)' }}>{featCount} per window</span></div>
              <div>Classes: 3 · idle via motion gate</div>
              <div>Motion gate: 0.8–1.0 m/s²</div>
              <div>Confidence: ≥55%</div>
              <div>Cooldown: 2 s</div>
              <div>Vote: 2 consecutive windows</div>
            </div>
          </div>

        </aside>

        <div className="screen-main">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, flex: 1, minHeight: 0 }}>
            <LiveChart data={accelX} color="#58a6ff" title="Accel X" yLabel="m/s²" />
            <LiveChart data={accelY} color="#3fb950" title="Accel Y" yLabel="m/s²" />
            <LiveChart data={accelZ} color="#d29922" title="Accel Z" yLabel="m/s²" />
            <LiveChart data={gyroX}  color="#bc8cff" title="Gyro X"  yLabel="rad/s" />
          </div>
        </div>
      </div>

      <div className="screen-footer">
        <StatusDot mode={connected ? 'live' : 'offline'} label={connected ? 'Connected' : 'Reconnecting…'} />
        <span>Model: gesture_brain_v2.pkl ({modelType})</span>
        <span>4 classes · {featCount} features · TTS enabled</span>
        <span>goodbye | thanks | welcome | idle</span>
      </div>
    </div>
  );
}
