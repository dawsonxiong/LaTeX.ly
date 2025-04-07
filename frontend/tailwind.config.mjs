/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        montserrat: ["Montserrat", "sans-serif"],
        inter: ["Inter", "sans-serif"],
        figtree: ["Figtree", "sans-serif"],
      },
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
      },
      translate: {
        '13': '4.45rem',
      },
    },
  },
  plugins: [],
};
