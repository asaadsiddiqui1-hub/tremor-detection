// Custom hook — manages a WebSocket connection to the backend
// Automatically connects on mount, disconnects on unmount.
import { useEffect, useRef, useState, useCallback } from 'react';

const WS_BASE = 'ws://localhost:8000';

export function useWebSocket(path) {
  const [data, setData] = useState(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const mountedRef = useRef(true);

  const send = useCallback((payload) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    const connect = () => {
      const ws = new WebSocket(`${WS_BASE}${path}`);
      wsRef.current = ws;

      ws.onopen  = () => { if (mountedRef.current) setConnected(true); };
      ws.onclose = () => { if (mountedRef.current) setConnected(false); };
      ws.onerror = () => { ws.close(); };

      ws.onmessage = (evt) => {
        if (!mountedRef.current) return;
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type !== 'meta') setData(msg);
        } catch { /* ignore */ }
      };
    };

    connect();

    return () => {
      mountedRef.current = false;
      wsRef.current?.close();
    };
  }, [path]);

  return { data, connected, send };
}
