import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import HomePage from './pages/HomePage';
import CameraPage from './pages/CameraPage';
import InventoryPage from './pages/InventoryPage';
import { useState } from 'react'; // ADD this!

function App() {
  const [photos, setPhotos] = useState([]); // ADD this!

  return (
    <Router className='dashboard'>
      <div className="p-4">
        <nav className="mb-4">
          <Link to="/" className="link">Home</Link>
          <Link to="/camera" className="link">Camera</Link>
          <Link to="/inventory" className="link">Inventory</Link>
        </nav>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/camera" element={<CameraPage photos={photos} setPhotos={setPhotos} />} />
          <Route path="/inventory" element={<InventoryPage photos={photos} />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
