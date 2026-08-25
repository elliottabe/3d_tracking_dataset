import { Canvas } from './canvas/Canvas';
import { EditorProvider } from './state/editorStore';

function App() {
  return (
    <EditorProvider>
      <Canvas />
    </EditorProvider>
  );
}

export default App;
