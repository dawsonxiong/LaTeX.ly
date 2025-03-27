export default function PreviewOutput({ latex }) {
    return (
      <div className="bg-white p-6 mt-4 rounded-lg shadow-md w-full max-w-md">
        <h2 className="text-2xl font-bold text-center mb-4">Generated LaTeX:</h2>
        <pre className="bg-gray-100 p-4 mt-2 rounded-md overflow-x-auto">{latex}</pre>
      </div>
    );
  }
  