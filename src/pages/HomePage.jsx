// src/pages/HomePage.jsx
import React from "react";
import { Link } from "react-router-dom";

const HomePage = () => {
  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-br from-indigo-100 via-purple-100 to-pink-100 text-center px-4 animate-fadeIn">
      {/* Logo or icon (optional) */}
      <img
        src="/vite.svg" // replace with your logo or icon path if desired
        alt="RestockerAI Logo"
        className="w-20 mb-6"
      />

      <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
        Welcome to RestockerAI
      </h1>

      <p className="text-lg md:text-xl text-gray-600 max-w-xl mb-8">
        Seamlessly track your pantry with just a photo. Capture items, check your inventory, and never run out again.
      </p>

      <div className="flex space-x-4">
        <Link to="/camera">
          <button className="px-6 py-2 bg-blue-600 text-white rounded-md shadow hover:bg-blue-700 transition">
            Scan Item
          </button>
        </Link>
        <Link to="/inventory">
          <button className="px-6 py-2 bg-gray-300 text-gray-800 rounded-md shadow hover:bg-gray-400 transition">
            View Inventory
          </button>
        </Link>
      </div>
    </div>
  );
};

export default HomePage;
import React from "react";
import { Link } from "react-router-dom";

const HomePage = () => {
  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-br from-indigo-100 via-purple-100 to-pink-100 text-center px-4 animate-fadeIn">
      {/* Logo or icon (optional) */}
      <img
        src="/vite.svg" // replace with your logo or icon path if desired
        alt="RestockerAI Logo"
        className="w-20 mb-6"
      />

      <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
        Welcome to RestockerAI
      </h1>

      <p className="text-lg md:text-xl text-gray-600 max-w-xl mb-8">
        Seamlessly track your pantry with just a photo. Capture items, check your inventory, and never run out again.
      </p>

      <div className="flex space-x-4">
        <Link to="/camera">
          <button className="px-6 py-2 bg-blue-600 text-white rounded-md shadow hover:bg-blue-700 transition">
            Scan Item
          </button>
        </Link>
        <Link to="/inventory">
          <button className="px-6 py-2 bg-gray-300 text-gray-800 rounded-md shadow hover:bg-gray-400 transition">
            View Inventory
          </button>
        </Link>
      </div>
    </div>
  );
};

export default HomePage;

