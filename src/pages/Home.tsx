import React, { useState } from 'react';
import { Upload } from 'lucide-react';

const API_URL = 'http://localhost:5000';

interface AnalysisResults {
  visualization: string;
  georeferenced: string;
  binary_mask: string;
  upload_id: string;
}

const Home = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<AnalysisResults | null>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length !== 4) {
      setError('Please select exactly 4 files (S1, S2, elevation, and slope)');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
      console.log('Added file:', files[i].name);
    }

    try {
      console.log('Sending files to backend...');
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
        mode: 'cors',
        credentials: 'include',
        headers: {
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));
        throw new Error(errorData.error || `Server returned ${response.status}`);
      }

      const data = await response.json();
      console.log('Response data:', data);

      if (!data.visualization || !data.georeferenced || !data.binary_mask || !data.upload_id) {
        throw new Error('Invalid response format from server');
      }

      setResults({
        visualization: `${API_URL}${data.visualization}`,
        georeferenced: `${API_URL}${data.georeferenced}`,
        binary_mask: `${API_URL}${data.binary_mask}`,
        upload_id: data.upload_id
      });
    } catch (err) {
      console.error('Error:', err);
      if (err instanceof TypeError && err.message === 'Failed to fetch') {
        setError('Could not connect to the server. Please ensure the Flask server is running at http://localhost:5000');
      } else {
        setError(err instanceof Error ? err.message : 'Failed to connect to server. Please ensure the backend is running.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold text-gray-800">Flood Mapping</h1>
        </div>

        {/* Upload Section */}
        <div className="mb-8">
          <label className="block mb-4">
            <div className="border-2 border-dashed border-blue-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors cursor-pointer">
              <Upload className="mx-auto h-12 w-12 text-blue-500 mb-4" />
              <p className="text-gray-600">Upload Files</p>
              <p className="text-sm text-gray-500 mt-2">Select all 4 TIF files: S1, S2, Elevation, Slope</p>
              <input
                type="file"
                className="hidden"
                accept=".tif,.tiff"
                multiple
                onChange={handleFileUpload}
                disabled={loading}
              />
            </div>
          </label>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center my-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 text-red-600 p-4 rounded-lg mb-8">
            {error}
          </div>
        )}

        {/* Results Section */}
        {results && (
          <div className="space-y-6">
            {/* First Row - Visualization */}
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <h3 className="text-lg font-semibold p-4">Flood Analysis</h3>
              <img
                src={results.visualization}
                alt="Visualization"
                className="w-full h-auto"
                onError={(e) => {
                  console.error('Error loading visualization:', e);
                  setError('Failed to load visualization');
                }}
              />
            </div>

            {/* Second Row - Binary Mask and Georeferenced Results */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg shadow-md overflow-hidden">
                <h3 className="text-lg font-semibold p-4">Binary Flood Mask</h3>
                <img
                  src={results.binary_mask}
                  alt="Binary Flood Mask"
                  className="w-full h-auto"
                  onError={(e) => {
                    console.error('Error loading binary mask:', e);
                    setError('Failed to load binary mask');
                  }}
                />
              </div>
              <div className="bg-white rounded-lg shadow-md overflow-hidden">
                <h3 className="text-lg font-semibold p-4">Flood Polygons</h3>
                <img
                  src={results.georeferenced}
                  alt="Flood Polygons"
                  className="w-full h-auto"
                  onError={(e) => {
                    console.error('Error loading flood polygons:', e);
                    setError('Failed to load flood polygons');
                  }}
                />
              </div>
            </div>

            {/* Third Row - Map Button */}
            <div className="bg-white rounded-lg shadow-md p-4">
              <button
                onClick={() => window.open(`${API_URL}/map/${results.upload_id}`, '_blank')}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-md shadow transition duration-200 flex items-center justify-center gap-2"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
                </svg>
                View Map Visualization
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Home;