// Shared connection context — COM port, sample rate, status
import { createContext, useContext, useState, useCallback } from 'react';

const API = 'http://localhost:8000';

const ConnectionContext = createContext(null);

export function ConnectionProvider({ children }) {
  const [status, setStatus] = useState({
    mode: 'sim', port: 'SIM', sample_rate: 50,
    buffer_fill: 0, lines_read: 0, actual_rate: 50,
  });
  const [ports, setPorts] = useState([]);

  const fetchStatus = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/status`);
      setStatus(await r.json());
    } catch { /* backend not ready yet */ }
  }, []);

  const fetchPorts = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/ports`);
      const d = await r.json();
      setPorts(d.ports || []);
    } catch { setPorts([]); }
  }, []);

  const connect = useCallback(async (mode, port, scenario) => {
    try {
      const r = await fetch(`${API}/api/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, port, scenario }),
      });
      const d = await r.json();
      await fetchStatus();
      return d;
    } catch (e) { return { ok: false, error: String(e) }; }
  }, [fetchStatus]);

  const setSampleRate = useCallback(async (hz) => {
    try {
      await fetch(`${API}/api/sample-rate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hz: Number(hz) }),
      });
      await fetchStatus();
    } catch { /* ignore */ }
  }, [fetchStatus]);

  return (
    <ConnectionContext.Provider value={{
      status, ports, fetchStatus, fetchPorts, connect, setSampleRate, API,
    }}>
      {children}
    </ConnectionContext.Provider>
  );
}

export const useConnection = () => useContext(ConnectionContext);
