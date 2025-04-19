import "../styles/globals.css";
import Navbar from "/components/Navbar";

export const metadata = {
  title: 'LaTeX.ly | Math to LaTeX Converter',
  description: 'Convert mathematical equations to LaTeX code instantly!',
};

export default function Layout({ children }) {
  return (
    <html lang="en">
      <body className="bg-blue-50 text-gray-900">
        <Navbar />
        <main className="container mx-auto p-6">{children}</main>
      </body>
    </html>
  );
}
