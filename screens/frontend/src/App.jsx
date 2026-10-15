import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ConnectionProvider } from './context/ConnectionContext';

import Home              from './screens/Home';
import FallDetection     from './screens/FallDetection';
import TremorSeverity    from './screens/TremorSeverity';
import StressHRV         from './screens/StressHRV';
import MuscleAtrophy     from './screens/MuscleAtrophy';
import PTForm            from './screens/PTForm';
import GestureTranslator   from './screens/GestureTranslator';
import GestureTranslatorV2 from './screens/GestureTranslatorV2';
import LiveSensor          from './screens/LiveSensor';
import TremorBinary        from './screens/TremorBinary';
import FallBinary          from './screens/FallBinary';

export default function App() {
  return (
    <ConnectionProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/"        element={<Home />} />
          <Route path="/fall"    element={<FallDetection />} />
          <Route path="/tremor"  element={<TremorSeverity />} />
          <Route path="/stress"  element={<StressHRV />} />
          <Route path="/atrophy" element={<MuscleAtrophy />} />
          <Route path="/ptform"  element={<PTForm />} />
          <Route path="/gesture"       element={<GestureTranslator />} />
          <Route path="/sensor"        element={<LiveSensor />} />
          <Route path="/tremor-binary" element={<TremorBinary />} />
          <Route path="/gesture-v2"    element={<GestureTranslatorV2 />} />
          <Route path="/fall-binary"   element={<FallBinary />} />
        </Routes>
      </BrowserRouter>
    </ConnectionProvider>
  );
}
