import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import EvacuationPaths from './pages/EvacuationPaths';

const App = () => {
  console.log('🚀 Starting application...');

  return (
    <div className="min-h-screen bg-gray-100">
      <Navbar />
      <main className="container mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/evacuation/:uploadId" element={<EvacuationPaths />} />
        </Routes>
      </main>
    </div>
  );
};

export default App;