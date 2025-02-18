"use client";
import Image from 'next/image';

import { useState } from "react";
import Upload from "/components/Upload";
import PreviewOutput from "/components/PreviewOutput";

export default function Home() {
  const [latexOutput, setLatexOutput] = useState("");

  return (
    <div className="flex flex-col items-center">
      <div className="flex flex-row items-center gap-4 mb-7 mt-3">
        <h1 className="text-4xl font-bold px-4 py-1 rounded-full bg-orange-100 text-orange-700 py-3">
LaTeX.ly
</h1>
        <div className="px-3 py-1 rounded-full bg-blue-100 text-blue-700 text-lg font-bold">
          Convert Images to LaTeX
        </div>
      </div>

      <Upload setLatexOutput={setLatexOutput} />
      {latexOutput && <PreviewOutput latex={latexOutput} />}

      <Image 
          src="/example.png"
          alt="Example Equation"
          width={350}
          height={250}
          className="rounded-lg shadow-md mt-8"
        />

        <Image 
          src="/down_arrow.png"
          alt="Example Equation"
          width={10}
          height={10}
          className="rounded-lg my-5"
        />

        <Image 
          src="/contour.png"
          alt="Equation Contour"
          width={350}
          height={250}
          className="rounded-lg shadow-md"
        />

        <Image 
          src="/down_arrow.png"
          alt="Example Equation"
          width={10}
          height={10}
          className="rounded-lg my-5"
        />

      <Image 
          src="/latex.png"
          alt="Equation Contour"
          width={500}
          height={250}
          className="rounded-lg shadow-md"
        />
      
    </div>
  );
}
