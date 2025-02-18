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
    <div className="bg-white p-6 rounded-lg shadow-lg w-full max-w-md">
      <input 
        type="file" 
        accept="image/*" 
        onChange={(e) => setFile(e.target.files[0])} 
        className="mb-4"
      />
      <button 
        onClick={handleUpload} 
        disabled={loading}
        className="bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 
                 text-white font-medium px-6 py-2.5 rounded-full w-full 
                 shadow-md hover:shadow-lg transition-all duration-200 
                 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? <LoadingSpinner /> : "Upload & Convert"}
      </button>
    </div>
  );
}
