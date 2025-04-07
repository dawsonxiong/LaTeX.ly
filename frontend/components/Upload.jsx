"use client";

import { useState } from "react";
import { uploadImage } from "/lib/apiClient";
import LoadingSpinner from "/components/Loading";

export default function Upload({ setLatexOutput }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      // if (file.name === "ex8.png") {
      //   await new Promise(resolve => setTimeout(resolve, 2000));
      //   setLatexOutput("\\forall x \\in R , \\exists 2 \\sum 3 \\neq 4 v");
      //   setLoading(false);
      //   return;
      // }

      const response = await uploadImage(file);
      
      if (response.error) {
        throw new Error(response.error);
      }
      
      setLatexOutput(response.latex_output);
    } catch (err) {
      setError(err.message || "Error processing image");
      console.error("Upload error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg w-full max-w-lg text-center">
      <h2 className="text-3xl font-bold text-gray-800 mb-1 font-figtree">
        Upload Your Equation
      </h2>
      <p className="text-gray-600 mb-4 font-figtree">
        Supports .jpg, .png formats
      </p>

      <label 
        htmlFor="file-upload" 
        className={`mb-3 block w-full text-sm border border-gray-300 
                   rounded-lg cursor-pointer bg-gray-50 
                   hover:bg-gray-100 focus:outline-none 
                   focus:ring-2 focus:ring-blue-500 
                   focus:border-blue-500 py-3 px-4 font-figtree
                   ${error ? 'border-red-500' : ''}`}
      >
        <div>
          {file ? file.name : "Click here to upload your file"}
        </div>
        <input 
          id="file-upload" 
          type="file" 
          accept="image/*" 
          onChange={(e) => {
            setFile(e.target.files[0]);
            setError(null);
          }} 
          className="hidden"
        />
      </label>

      {error && (
        <p className="text-red-500 text-sm mb-4 font-figtree">{error}</p>
      )}

      <div className="flex gap-3">
        <button 
          onClick={handleUpload} 
          disabled={loading}
          className="bg-gradient-to-r from-blue-500 to-blue-600 
                   hover:from-blue-600 hover:to-blue-700 
                   text-white font-medium px-6 py-2.5 rounded-full 
                   flex-1 shadow-md hover:shadow-lg 
                   transition-all duration-200
                   disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? <LoadingSpinner /> : "Convert!"}
        </button>

        <button 
          className="bg-gradient-to-r from-green-600 to-green-600 
                   hover:from-green-600 hover:to-green-700 
                   text-white font-medium px-6 py-2.5 rounded-full 
                   flex-1 shadow-md hover:shadow-lg 
                   transition-all duration-200"
        >
          Preview Output
        </button>
      </div>
    </div>
  );
}
