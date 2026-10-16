"use client";

import React, { useState, useEffect } from "react";
import { Shield, Radio, FileText, Clock } from "lucide-react";

interface NavbarProps {
  activeModule: "cctv" | "video_analysis";
  setActiveModule: (module: "cctv" | "video_analysis") => void;
}

export default function Navbar({ activeModule, setActiveModule }: NavbarProps) {
  const [timeStr, setTimeStr] = useState<string>("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(
        now.toLocaleTimeString("en-IN", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false
        }) + " IST"
      );
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <header className="border-b border-dark-border bg-dark-bg/90 backdrop-blur-sm sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-2.5 flex items-center justify-between">
        
        {/* Brand / Identity */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-dark-card border border-dark-border flex items-center justify-center text-slate-300">
            <Shield className="w-4 h-4 text-gov-blue" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-sm tracking-tight text-white">VĀHAN-LATIS</span>
              <span className="text-[10px] font-mono bg-dark-surface text-slate-400 px-1.5 py-0.5 rounded border border-dark-border">
                ITMS v2.4
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              Government of India • Ministry of Road Transport & Highways
            </p>
          </div>
        </div>

        {/* Central Module Selector Tabs */}
        <nav className="flex items-center bg-dark-surface p-1 rounded-lg border border-dark-border">
          <button
            onClick={() => setActiveModule("cctv")}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeModule === "cctv"
                ? "bg-gov-blue text-white shadow-sm"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <Radio className={`w-3.5 h-3.5 ${activeModule === "cctv" ? "text-emerald-400" : ""}`} />
            <span>Live CCTV Surveillance</span>
          </button>

          <button
            onClick={() => setActiveModule("video_analysis")}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeModule === "video_analysis"
                ? "bg-gov-blue text-white shadow-sm"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Video Analysis & Audit</span>
          </button>
        </nav>

        {/* Live Status Indicator */}
        <div className="hidden md:flex items-center gap-3 text-xs text-slate-400">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span className="font-mono text-[11px] text-slate-300">API CONNECTED</span>
          </div>
          <span className="text-slate-600">|</span>
          <div className="flex items-center gap-1 font-mono text-[11px] text-slate-400">
            <Clock className="w-3 h-3 text-slate-500" />
            <span>{timeStr || "--:--:--"}</span>
          </div>
        </div>

      </div>
    </header>
  );
}
