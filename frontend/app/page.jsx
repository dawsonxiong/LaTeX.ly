"use client";

import Image from 'next/image';
import { useState } from "react";
import Upload from "/components/Upload";
import Output from "/components/Output";

export default function Home() {
  const [latexOutput, setLatexOutput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleLatexOutput = async (output) => {
    try {
      setLatexOutput(output);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Error handling LaTeX output:', err);
    }
  };

  return (
    <div className="flex flex-col items-center">
      <div className="flex flex-col items-center gap-4 mb-7 mt-3">
        <div className="px-3 py-1 text-6xl text-center font-figtree">
          Math-to-LaTeX conversion,
        </div>
        <div className="px-3 py-1 bg-blue-100 text-blue-700 text-6xl font-semibold font-figtree text-center">
          in a snap 👇
        </div>
      </div>

      <Upload setLatexOutput={handleLatexOutput} />
      {error && (
        <div className="text-red-500 mt-4 font-inter">{error}</div>
      )}
      {latexOutput && <Output latex={latexOutput} />}

      <div className="flex flex-row items-center justify-center gap-8 mt-14 translate-x-13">
        {/* Original Math Equation */}
        <div className="flex flex-col items-center">
          <Image 
            src="/example.png"
            alt="Example Equation"
            width={350}
            height={250}
            className="rounded-lg shadow-md"
            priority
          />
          <p className="text-center text-gray-700 font-bold mt-5 font-montserrat bg-blue-100 px-4 py-2 rounded-md">
            Original math equation
          </p>
        </div>

        {/* Right Arrow */}
        <Image 
          src="/right_arrow.png"
          alt="Right Arrow"
          width={50}
          height={50}
          className="rounded-lg relative -translate-y-8"
        />

        {/* Contours + OCR */}
        <div className="flex flex-col items-center">
          <Image 
            src="/contour.png"
            alt="Contours + OCR"
            width={350}
            height={250}
            className="rounded-lg shadow-md"
            priority
          />
          <p className="text-center text-gray-700 font-bold mt-5 font-montserrat bg-blue-100 px-4 py-2 rounded-md">
            Contours + OCR
          </p>
        </div>

        {/* Right Arrow */}
        <Image 
          src="/right_arrow.png"
          alt="Right Arrow"
          width={50}
          height={50}
          className="rounded-lg relative -translate-y-8"
        />

        {/* LaTeX Code */}
        <div className="flex flex-col items-center">
          <Image 
            src="/latex.png"
            alt="LaTeX Code"
            width={500}
            height={250}
            className="rounded-lg shadow-md"
            priority
          />
          <p className="text-center text-gray-700 font-bold mt-5 font-montserrat bg-blue-100 px-4 py-2 rounded-md">
            LaTeX code
          </p>
        </div>
      </div>
    </div>
  );
}
