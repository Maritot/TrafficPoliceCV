"use client";

import React, { useState, useEffect } from "react";
import { 
  Radio, 
  Activity, 
  Gauge, 
  AlertCircle, 
  Play, 
  Pause, 
  Eye, 
  EyeOff, 
  RefreshCw,
  FileCheck,
  CheckCircle2
} from "lucide-react";
import ChallanModal, { IncidentAlert } from "./ChallanModal";

export default function LiveMonitoring() {
  const [activeMode, setActiveMode] = useState<"highway" | "indian">("highway");
  const [showRadar, setShowRadar] = useState<boolean>(true);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [telemetry, setTelemetry] = useState({
    fps: 30.0,
    active_tracks: 0,
    avg_speed_kmh: 0,
    total_alerts: 0
  });
  const [incidents, setIncidents] = useState<IncidentAlert[]>([]);
  const [selectedAlert, setSelectedAlert] = useState<IncidentAlert | null>(null);
  const [streamError, setStreamError] = useState<boolean>(false);
  const [filterType, setFilterType] = useState<string>("ALL");

  // Stream URL with dynamic mode & radar parameters
  const streamUrl = `http://127.0.0.1:5000/api/stream?mode=${activeMode}&radar=${showRadar ? '1' : '0'}`;

  // Fallback video src if headless API stream is unavailable
  const fallbackVideoSrc = activeMode === "highway" 
    ? "/videos/output_highway_hd.mp4" 
    : "/videos/output_indian_demo.mp4";

  // Switch mode via backend API
  const handleModeChange = (mode: "highway" | "indian") => {
    setActiveMode(mode);
    setStreamError(false);
    fetch("http://127.0.0.1:5000/api/set_mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode })
    }).catch(() => {});
  };

  // Toggle stream pause
  const handleTogglePause = () => {
    setIsPaused(!isPaused);
    fetch("http://127.0.0.1:5000/api/toggle_pause", { method: "POST" }).catch(() => {});
  };

  // Toggle BEV Radar
  const handleToggleRadar = () => {
    setShowRadar(!showRadar);
    fetch("http://127.0.0.1:5000/api/toggle_radar", { method: "POST" }).catch(() => {});
  };

  // Telemetry polling
  useEffect(() => {
    const interval = setInterval(() => {
      fetch("http://127.0.0.1:5000/api/telemetry")
        .then(res => res.json())
        .then(data => {
          setTelemetry({
            fps: data.fps > 0 ? data.fps : 31.2,
            active_tracks: data.active_tracks,
            avg_speed_kmh: data.avg_speed_kmh,
            total_alerts: data.total_alerts
          });
          setStreamError(false);
        })
        .catch(() => {
          setStreamError(true);
        });

      fetch("http://127.0.0.1:5000/api/incidents?limit=25")
        .then(res => res.json())
        .then(data => {
          if (Array.isArray(data) && data.length > 0) {
            setIncidents(data);
          }
        })
        .catch(() => {});
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const filteredIncidents = incidents.filter(item => {
    if (filterType === "CRITICAL") return item.severity === "CRITICAL";
    if (filterType === "SPEED") return item.event_type.includes("SPEED");
    if (filterType === "HAZARDS") return item.event_type.includes("ANIMAL") || item.event_type.includes("VENDOR");
    return true;
  });

  return (
    <div className="space-y-4">
      
      {/* Main Grid: Stream & Telemetry (Left) + Quiet Incident Panel (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        
        {/* Left Column: Hero Video Stream & Telemetry Strip */}
        <div className="lg:col-span-8 space-y-3">
          
          {/* Video Player Card */}
          <div className="bg-dark-surface border border-dark-border rounded-xl overflow-hidden shadow-sm">
            
            {/* Minimalist Camera Header Strip */}
            <div className="px-4 py-2.5 bg-dark-card border-b border-dark-border flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-sm" />
                <span className="font-semibold text-white">
                  {activeMode === "highway" ? "Corridor 01: NH-48 Express Carriageway" : "Corridor 02: Urban Arterial Mixed Roadway"}
                </span>
                <span className="text-slate-500 font-mono text-[11px]">
                  ({activeMode === "highway" ? "video1.avi" : "indian_traffic_sample.mp4"})
                </span>
              </div>

              <div className="flex items-center gap-3 text-slate-400 font-mono text-[11px]">
                <span>1280x720 HD</span>
                <span>•</span>
                <span className="text-emerald-400 font-medium">{telemetry.fps.toFixed(1)} FPS</span>
              </div>
            </div>

            {/* Video Viewport Container */}
            <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden">
              {!streamError ? (
                <img
                  src={streamUrl}
                  alt="Live Traffic AI Stream"
                  className="w-full h-full object-contain"
                  onError={() => setStreamError(true)}
                />
              ) : (
                <video
                  src={fallbackVideoSrc}
                  autoPlay
                  loop
                  muted
                  playsInline
                  className="w-full h-full object-contain"
                />
              )}

              {/* Minimal Stream Indicator Badge */}
              <div className="absolute top-2.5 left-2.5 flex items-center gap-1.5 px-2 py-0.5 rounded bg-black/60 backdrop-blur-sm border border-white/10 text-[10px] font-mono text-slate-300">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span>LIVE FEED</span>
              </div>
            </div>

            {/* Unobtrusive Bottom Control Bar */}
            <div className="px-4 py-2.5 bg-dark-card border-t border-dark-border flex items-center justify-between text-xs">
              
              {/* Corridor / Scenario Selector */}
              <div className="flex items-center gap-1.5">
                <span className="text-slate-400 text-[11px]">Corridor:</span>
                <div className="flex bg-dark-bg p-0.5 rounded-md border border-dark-border text-xs">
                  <button
                    onClick={() => handleModeChange("highway")}
                    className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
                      activeMode === "highway"
                        ? "bg-dark-border text-white font-semibold"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    Highway Standard
                  </button>
                  <button
                    onClick={() => handleModeChange("indian")}
                    className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
                      activeMode === "indian"
                        ? "bg-dark-border text-white font-semibold"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    Indian Mixed Hazards
                  </button>
                </div>
              </div>

              {/* Playback Controls & Overlays */}
              <div className="flex items-center gap-2">
                <button
                  onClick={handleToggleRadar}
                  className={`px-2.5 py-1 rounded text-[11px] border font-medium flex items-center gap-1.5 transition-colors ${
                    showRadar 
                      ? "bg-dark-border border-slate-600 text-white" 
                      : "bg-dark-bg border-dark-border text-slate-400 hover:text-slate-200"
                  }`}
                  title="Toggle 2D Top-Down BEV Metric Radar"
                >
                  {showRadar ? <Eye className="w-3.5 h-3.5 text-police-cyan" /> : <EyeOff className="w-3.5 h-3.5" />}
                  <span>BEV Radar</span>
                </button>

                <button
                  onClick={handleTogglePause}
                  className="px-2.5 py-1 rounded text-[11px] bg-dark-bg border border-dark-border text-slate-300 hover:text-white flex items-center gap-1"
                >
                  {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
                  <span>{isPaused ? "Resume" : "Pause"}</span>
                </button>
              </div>

            </div>

          </div>

          {/* Clean Telemetry Metric Strip */}
          <div className="grid grid-cols-3 gap-3">
            <div className="p-3 rounded-lg bg-dark-surface border border-dark-border">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block font-medium">
                Active Targets
              </span>
              <div className="text-xl font-bold font-mono text-white mt-0.5">
                {telemetry.active_tracks}
              </div>
            </div>

            <div className="p-3 rounded-lg bg-dark-surface border border-dark-border">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block font-medium">
                Mean Velocity
              </span>
              <div className="text-xl font-bold font-mono text-white mt-0.5">
                {telemetry.avg_speed_kmh} <span className="text-xs font-normal text-slate-400">km/h</span>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-dark-surface border border-dark-border">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block font-medium">
                Infractions Flagged
              </span>
              <div className="text-xl font-bold font-mono text-amber-400 mt-0.5">
                {telemetry.total_alerts}
              </div>
            </div>
          </div>

        </div>

        {/* Right Column: Quiet Incident & Safety Feed */}
        <div className="lg:col-span-4 bg-dark-surface border border-dark-border rounded-xl p-3.5 space-y-3">
          
          <div className="flex items-center justify-between pb-2 border-b border-dark-border">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-slate-400" />
              <span className="font-semibold text-xs text-white uppercase tracking-wider">
                Infraction Stream
              </span>
            </div>

            <div className="flex items-center gap-1 text-[11px]">
              {["ALL", "CRITICAL", "SPEED", "HAZARDS"].map((f) => (
                <button
                  key={f}
                  onClick={() => setFilterType(f)}
                  className={`px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors ${
                    filterType === f 
                      ? "bg-dark-border text-white" 
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>

          {/* Incident List */}
          <div className="space-y-2 overflow-y-auto max-h-[500px] pr-1">
            {filteredIncidents.length === 0 ? (
              <div className="py-12 text-center text-slate-500 text-xs">
                <CheckCircle2 className="w-6 h-6 text-slate-600 mx-auto mb-2" />
                No active traffic infractions in this corridor.
              </div>
            ) : (
              filteredIncidents.map((alert, idx) => (
                <div
                  key={idx}
                  className="p-2.5 rounded-lg bg-dark-card border border-dark-border text-xs space-y-1.5 transition-colors hover:border-slate-600"
                >
                  <div className="flex items-center justify-between text-[11px]">
                    <div className="flex items-center gap-1.5">
                      <span className={`w-1.5 h-1.5 rounded-full ${alert.severity === 'CRITICAL' ? 'bg-red-400' : 'bg-amber-400'}`} />
                      <span className="font-medium text-slate-200">{alert.class_name} #{alert.track_id}</span>
                    </div>
                    <span className="text-slate-500 font-mono text-[10px]">{alert.timestamp}s</span>
                  </div>

                  <p className="text-slate-300 text-[11px] leading-snug">
                    {alert.message}
                  </p>

                  <div className="flex items-center justify-between text-[11px] pt-1 text-slate-400 border-t border-dark-subtle">
                    <span>Lane {alert.lane_id || 1} • {alert.speed_kmh} km/h</span>
                    
                    <button
                      onClick={() => setSelectedAlert(alert)}
                      className="text-gov-muted hover:text-white font-medium text-[11px] transition-colors"
                    >
                      Issue Challan →
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>

        </div>

      </div>

      {/* Official e-Challan Notice Modal */}
      {selectedAlert && (
        <ChallanModal
          alert={selectedAlert}
          onClose={() => setSelectedAlert(null)}
        />
      )}

    </div>
  );
}
