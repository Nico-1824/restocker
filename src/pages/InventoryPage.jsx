import { Link } from 'react-router-dom';

function InventoryPage({ photos }) {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold text-center mb-8 text-purple-600">Inventory</h1>
      {photos.length === 0 ? (
        <p className="text-center text-gray-600">No items captured yet.</p>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
          {photos.map((photo, index) => (
            <img
              key={index}
              src={photo}
              alt={`Captured Item ${index}`}
              className="rounded-lg shadow-md object-cover w-full h-48"
            />
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
