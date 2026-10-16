import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          bg: "#090d16",
          surface: "#0f1523",
          card: "#121929",
          border: "#1c2638",
          subtle: "#161f30",
        },
        gov: {
          blue: "#1d4ed8",
          hover: "#2563eb",
          muted: "#94a3b8",
        }
      },
    },
  },
  plugins: [],
};
export default config;
