import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VĀHAN-LATIS | National Intelligent Traffic Management System (ITMS)",
  description: "Government of India • Ministry of Road Transport & Highways • AI Computer Vision Intelligence Portal",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-gov-dark text-slate-100 antialiased min-h-screen flex flex-col">
        {children}
      </body>
    </html>
  );
}
