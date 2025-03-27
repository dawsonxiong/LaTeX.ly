"use client";

import { useState } from "react";
import { uploadImage } from "/lib/apiClient";
import LoadingSpinner from "/components/Loading";

export default function Upload({ setLatexOutput }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return alert("Please select a file");

    setLoading(true);
    const response = await uploadImage(file);
    setLatexOutput(response.latex || "Error processing image");
    setLoading(false);
  };

  return (
    <div className="bg-white p-8 rounded-lg shadow-lg w-full max-w-md text-center">
      {/* Heading */}
      <h2 className="text-3xl font-bold text-gray-800 mb-2">Upload Your Equation</h2>
      <p className="text-gray-600 mb-6">Supports .jpg, .png formats</p>

      {/* Custom File Input */}
      <label 
        htmlFor="file-upload" 
        className="mb-4 block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 py-4"
      >
        {file ? file.name : "Click here to upload your file"}
        <input 
          id="file-upload" 
          type="file" 
          accept="image/*" 
          onChange={(e) => setFile(e.target.files[0])} 
          className="hidden"
        />
      </label>

      {/* Upload Button */}
      <button 
        onClick={handleUpload} 
        disabled={loading}
        className="bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 
                 text-white font-medium px-6 py-2.5 rounded-full w-full 
                 shadow-md hover:shadow-lg transition-all duration-200 
                 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? <LoadingSpinner /> : "Convert!"}
      </button>
    </div>
  );
}
