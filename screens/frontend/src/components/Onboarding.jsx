import { useState } from 'react';

const STEPS = [
  {
    icon: '🧠',
    iconBg: 'rgba(0,122,255,0.10)',
    accentColor: '#007AFF',
    title: 'Welcome to NeuroAI',
    subtitle: 'AI-Powered Neurological Health Monitoring',
    body: 'NeuroAI is a cutting-edge diagnostic platform that combines wearable sensor technology with advanced machine learning to monitor and analyze neurological health indicators in real time.',
  },
  {
    icon: '🧤',
    iconBg: 'rgba(52,199,89,0.10)',
    accentColor: '#34C759',
    title: 'The Smart Glove',
    subtitle: 'Precision Sensing at Your Fingertips',
    body: 'The NeuroAI glove integrates an MPU6050 IMU sensor and MAX30102 heart rate sensor. Together they capture 7 channels of movement, orientation, and cardiovascular data with high precision.',
  },
  {
    icon: '📡',
    iconBg: 'rgba(88,86,214,0.10)',
    accentColor: '#5856D6',
    title: 'Connecting Your Device',
    subtitle: 'Seamless ESP32 Integration',
    body: 'The glove connects via an ESP32 microcontroller over USB serial. Select your COM port from the dashboard and tap "Connect ESP32" to begin streaming live sensor data instantly.',
  },
  {
    icon: '📊',
    iconBg: 'rgba(255,149,0,0.10)',
    accentColor: '#FF9500',
    title: 'Real-Time Data Collection',
    subtitle: 'Continuous Streaming at Up to 500 Hz',
    body: 'Once connected, the system continuously reads and buffers all 7 sensor channels. No manual triggering required — data flows in real time and is instantly ready for AI analysis.',
  },
  {
    icon: '🤖',
    iconBg: 'rgba(175,82,222,0.10)',
    accentColor: '#AF52DE',
    title: 'Seven AI Models',
    subtitle: 'Each Specialized for a Different Task',
    body: 'NeuroAI runs 7 independent models in parallel: Fall Detection (1D-CNN), Tremor Severity (RandomForest), Stress & HRV (RandomForest), Muscle Atrophy (GradBoost), PT Form (LSTM), Gesture Translation, and Raw Sensor streaming.',
  },
  {
    icon: '🔒',
    iconBg: 'rgba(255,45,85,0.10)',
    accentColor: '#FF2D55',
    title: 'Your Privacy, Protected',
    subtitle: 'All Processing Happens Locally',
    body: 'All sensor data and AI inference runs entirely on your local device. No health data is ever sent to external servers. Your biometric information stays private by design, always.',
  },
  {
    icon: '▶️',
    iconBg: 'rgba(0,199,190,0.10)',
    accentColor: '#00C7BE',
    title: 'Starting a Session',
    subtitle: 'Three Steps to Get Going',
    body: 'Select your COM port, choose a sample rate (50 Hz recommended for most models), then tap "Connect ESP32". No hardware? Use Simulate mode to explore all 7 features with realistic synthetic data.',
  },
  {
    icon: '📈',
    iconBg: 'rgba(50,173,230,0.10)',
    accentColor: '#32ADE6',
    title: 'Reading Your Results',
    subtitle: 'Understanding the Analysis',
    body: 'Each diagnostic screen shows a real-time prediction with a confidence score and live chart. Green indicates normal/healthy readings, yellow indicates caution, and red signals a detected event requiring attention.',
  },
];

export default function Onboarding({ onDismiss }) {
  const [step, setStep] = useState(0);
  const [animKey, setAnimKey] = useState(0);

  const goNext = () => {
    if (step === STEPS.length - 1) { onDismiss(); return; }
    setAnimKey(k => k + 1);
    setStep(s => s + 1);
  };

  const goBack = () => {
    if (step === 0) return;
    setAnimKey(k => k + 1);
    setStep(s => s - 1);
  };

  const current = STEPS[step];
  const isLast = step === STEPS.length - 1;

  return (
    <div className="onboarding-overlay" role="dialog" aria-modal="true" aria-label="Welcome to NeuroAI">
      <div className="onboarding-modal">

        {/* Progress dots */}
        <div className="onboarding-dots" role="list" aria-label="Progress">
          {STEPS.map((_, i) => (
            <div
              key={i}
              role="listitem"
              className={`onboarding-dot ${i === step ? 'active' : i < step ? 'done' : ''}`}
              aria-label={`Step ${i + 1}${i === step ? ' (current)' : ''}`}
            />
          ))}
        </div>

        {/* Icon */}
        <div className="onboarding-icon-wrap">
          <div
            className="onboarding-icon"
            style={{ background: current.iconBg }}
            aria-hidden="true"
          >
            {current.icon}
          </div>
        </div>

        {/* Step content */}
        <div className="onboarding-step" key={animKey}>
          <h2 className="onboarding-title">{current.title}</h2>
          <p className="onboarding-subtitle" style={{ color: current.accentColor }}>
            {current.subtitle}
          </p>
          <p className="onboarding-body">{current.body}</p>
        </div>

        {/* Navigation */}
        <div className="onboarding-actions">
          {step > 0 ? (
            <button className="onboarding-back" onClick={goBack} aria-label="Go back">
              ‹ Back
            </button>
          ) : (
            <div style={{ width: 80 }} />
          )}

          <button
            className="onboarding-continue"
            onClick={goNext}
            style={{ background: current.accentColor }}
            aria-label={isLast ? 'Get started' : 'Continue to next step'}
          >
            {isLast ? 'Get Started' : 'Continue'}
          </button>

          <button className="onboarding-skip" onClick={onDismiss} aria-label="Skip onboarding">
            Skip
          </button>
        </div>

      </div>
    </div>
  );
}
