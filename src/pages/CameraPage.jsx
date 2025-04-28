import { useRef } from 'react';
import Webcam from 'react-webcam';
import { Link } from 'react-router-dom';

function CameraPage({ photos, setPhotos }) {
  const webcamRef = useRef(null);

  const capturePhoto = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      setPhotos(prev => [...prev, imageSrc]);
    }
  };

  return (
    <div className="camera-box">
      <Webcam
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        className="camera"
        videoConstraints={{
          facingMode: "environment" // Try to use back camera if mobile
        }}
      />
      <div className='buttons'>
        <button 
          onClick={capturePhoto}
          className="mt-4 px-6 py-2 bg-green-500 text-white rounded hover:bg-green-600"
        >
          Capture Photo
        </button>
        <Link to="/inventory" className="to-inventory">
          View Inventory
        </Link>
      </div>
    </div>
  );
}

export default CameraPage;
