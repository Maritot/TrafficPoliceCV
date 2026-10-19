"use client";

import React, { useState } from "react";
import Navbar from "@/components/Navbar";
import LiveMonitoring from "@/components/LiveMonitoring";
import VideoAnalysis from "@/components/VideoAnalysis";
import StatsDashboard from "@/components/StatsDashboard";
import { ShieldCheck, Info, ExternalLink, Activity, Server, FileText } from "lucide-react";

export default function Home() {
  const [activeModule, setActiveModule] = useState<"cctv" | "video_analysis" | "analytics">("cctv");

  return (
    <div className="flex flex-col min-h-screen bg-gov-dark">
      {/* Government Navigation & Identity Header */}
      <Navbar 
        activeModule={activeModule} 
        setActiveModule={setActiveModule} 
      />

      {/* Main Body Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-5">
        {activeModule === "cctv" ? (
          <LiveMonitoring />
        ) : activeModule === "video_analysis" ? (
          <VideoAnalysis />
        ) : (
          <StatsDashboard />
        )}
      </main>

      {/* Official Government Footer */}
      <footer className="border-t border-gov-border bg-gov-navy/90 text-slate-400 text-xs py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-4">
          
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 pb-4 border-b border-gov-border/60">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded bg-amber-400/10 border border-amber-400/20 flex items-center justify-center text-amber-400">
                <ShieldCheck className="w-4 h-4" />
              </div>
              <div>
                <div className="font-bold text-white tracking-wide">
                  PROJECT LATIS • INTELLIGENT TRAFFIC SURVEILLANCE SYSTEM
                </div>
                <div className="text-[11px] text-slate-400">
                  Developed in conformance with Indian Road Congress (IRC:SP:110-2015) Guidelines for ITS.
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4 text-[11px]">
              <span className="flex items-center gap-1.5 text-police-emerald">
                <span className="w-2 h-2 rounded-full bg-police-emerald animate-pulse" />
                <span>Central AI Cluster: <strong>OPERATIONAL</strong></span>
              </span>
              <span className="text-slate-500">|</span>
              <span className="text-slate-400">NIC Node: <strong>IN-NDC-01</strong></span>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row items-center justify-between gap-2 text-[11px] text-slate-500">
            <div>
              © 2026 Project LATIS Team. Government of India Traffic Intelligence Research Initiative.
            </div>
            <div className="flex items-center gap-4">
              <span className="hover:text-slate-300 transition-colors">Privacy Policy</span>
              <span>•</span>
              <span className="hover:text-slate-300 transition-colors">Terms of Use</span>
              <span>•</span>
              <span className="hover:text-slate-300 transition-colors">Audit & Compliance Log</span>
            </div>
          </div>

        </div>
      </footer>
    </div>
  );
}
