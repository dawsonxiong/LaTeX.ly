import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="bg-white shadow-md p-4">
      <div className="container mx-auto flex justify-between items-center">
        <Link href="/" className="text-xl font-bold">
          LaTeX.ly
        </Link>
        <div className="space-x-4">
          <Link href="/convert" className="hover:underline">Convert</Link>
          <Link href="/history" className="hover:underline">History</Link>
        </div>
      </div>
    </nav>
  );
}
