import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { useMemo } from 'react';

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'var(--surface)',
      borderRadius: 10,
      padding: '6px 12px',
      fontSize: '0.8rem',
      color: 'var(--text)',
      fontFamily: 'var(--font)',
      fontWeight: 500,
      boxShadow: 'var(--shadow-deep)',
      border: '0.5px solid rgba(0,0,0,0.08)',
    }}>
      {typeof payload[0].value === 'number'
        ? payload[0].value.toFixed(4)
        : payload[0].value}
    </div>
  );
}

export default function LiveChart({
  data = [],
  color = '#007AFF',
  yLabel = '',
  title = '',
  height = '100%',
  referenceY = null,
  refColor = '#FF9500',
  showDots = false,
  strokeWidth = 2,
  yDomain = ['auto', 'auto'],
}) {
  const chartData = useMemo(() =>
    data.map((v, i) => ({ i, v: typeof v === 'number' ? parseFloat(v.toFixed(4)) : 0 })),
    [data]
  );

  return (
    <div className="chart-card fade-in" style={{ height }}>
      {title && <h4>{title}</h4>}
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
          <CartesianGrid strokeDasharray="0" stroke="rgba(0,0,0,0.05)" vertical={false} />
          <XAxis dataKey="i" hide />
          <YAxis
            domain={yDomain}
            tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font)' }}
            tickLine={false}
            axisLine={false}
            label={yLabel ? {
              value: yLabel, angle: -90, position: 'insideLeft',
              fill: 'var(--text-muted)', fontSize: 11, offset: 14,
            } : undefined}
            width={yLabel ? 50 : 34}
          />
          <Tooltip content={<ChartTooltip />} />
          {referenceY !== null && (
            <ReferenceLine
              y={referenceY}
              stroke={refColor}
              strokeDasharray="4 4"
              strokeWidth={1.5}
            />
          )}
          <Line
            type="monotoneX"
            dataKey="v"
            stroke={color}
            strokeWidth={strokeWidth}
            dot={showDots}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
