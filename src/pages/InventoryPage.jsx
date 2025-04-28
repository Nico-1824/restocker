import { useState } from 'react';
import { Link } from 'react-router-dom';

function InventoryPage({ photos }) {
  const [predictions, setPredictions] = useState({});
  const [loading, setLoading] = useState({});

  const identifyItem = async (photoUrl, index) => {
    if (predictions[index]) return; // Skip if already identified
    
    setLoading(prev => ({ ...prev, [index]: true }));
    
    try {
      // This URL would point to your backend API that runs the PyTorch model
      const response = await fetch('http://localhost:5001/identify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: photoUrl }),
      });
      
      const data = await response.json();
      setPredictions(prev => ({ ...prev, [index]: data.prediction }));
    } catch (error) {
      console.error('Error identifying item:', error);
      setPredictions(prev => ({ ...prev, [index]: 'Error identifying item' }));
    } finally {
      setLoading(prev => ({ ...prev, [index]: false }));
    }
  };

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold text-center mb-8 text-purple-600">Inventory</h1>
      {photos.length === 0 ? (
        <p className="text-center text-gray-600">No items captured yet.</p>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
          {photos.map((photo, index) => (
            <div key={index} className="flex flex-col">
              <img
                src={photo}
                alt={`Captured Item ${index}`}
                className="rounded-lg shadow-md object-cover w-full h-48"
              />
              <div className="mt-2 flex justify-between items-center">
                {predictions[index] ? (
                  <p className="text-sm font-medium">{predictions[index]}</p>
                ) : (
                  <p className="text-sm text-gray-500">Item {index + 1}</p>
                )}
                <button
                  onClick={() => identifyItem(photo, index)}
                  disabled={loading[index]}
                  className="px-3 py-1 bg-purple-600 text-white text-sm rounded hover:bg-purple-700 disabled:bg-purple-300"
                >
                  {loading[index] ? 'Identifying...' : 'Identify Item'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="flex justify-center mt-8">
        <Link to="/camera" className="text-blue-500 underline">
          Back to Camera
        </Link>
      </div>
    </div>
  );
}

export default InventoryPage;