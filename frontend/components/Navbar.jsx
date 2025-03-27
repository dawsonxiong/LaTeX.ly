import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="bg-white shadow-md p-4">
      <div className="mx-6 container flex justify-start items-center ">
        <Link href="/" className="text-3xl font-semibold text-blue-700 bg-blue-100 py-3 px-4 rounded-full hover:underline font-inter">
          LaTeX.ly
        </Link>
      </div>
    </nav>
  );
}
