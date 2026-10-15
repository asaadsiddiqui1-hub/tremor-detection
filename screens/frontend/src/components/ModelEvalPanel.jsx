// Expandable model evaluation image panel
import { useConnection } from '../context/ConnectionContext';

export default function ModelEvalPanel({ feature }) {
  const { API } = useConnection();
  const url = `${API}/api/eval-image/${feature}`;

  return (
    <details className="eval-panel card">
      <summary>📊 Model Evaluation Metrics</summary>
      <img src={url} alt={`${feature} evaluation`} loading="lazy" />
    </details>
  );
}
