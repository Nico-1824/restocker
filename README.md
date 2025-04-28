# Restocker

Restocker is a food inventory management application that uses computer vision to automatically identify fruits and vegetables. Take photos, get instant identification, and manage your inventory efficiently.

## Features

- Camera capture for adding items to inventory
- Automatic food item recognition using a PyTorch deep learning model
- Clean, mobile-friendly user interface
- Inventory management

## Setup and Installation

### Prerequisites

- Node.js (v14 or newer)
- Python 3.8+
- pip (Python package manager)
- Git

### Installation Steps

1. Clone the repository
   ```bash
   git clone https://github.com/Nico-1824/restocker
   cd restocker
   ```

2. Install frontend dependencies
   ```bash
   npm install
   ```

3. Install Python backend dependencies
   ```bash
   pip install flask flask-cors torch torchvision pillow
   ```

## Running the Application

1. Start the backend server
   ```bash
   python server.py
   ```
   The server will start on port 5001.

2. In a new terminal, start the React frontend
   ```bash
   npm start
   ```
   The application will open in your browser at http://localhost:3000.

## Using the Application

- **Home Page**: Navigate between features and see overview information
- **Camera Page**: Click the camera icon to take photos of your food items
- **Inventory Page**: View your inventory and click "Identify Item" to use the AI model to recognize each item

## Project Structure

- **src/**: React frontend code
  - **/pages**: Main application pages
  - **/components**: Reusable React components
- **model/**: PyTorch model and related files
  - **model.pth**: Trained model weights
  - **Restocker.py**: Model architecture definition
- **server.py**: Flask backend for serving the PyTorch model

## Troubleshooting

- If you encounter CORS issues, make sure both the frontend and backend are running
- If the model isn't loading, check that the path to model.pth is correct
- If identification doesn't work, ensure images are correctly formatted (PNG/JPEG) and are clear photos of food items

## License

MIT

## Acknowledgements

- The model was trained on a dataset of common fruits and vegetables
- Built with React, Flask, and PyTorch