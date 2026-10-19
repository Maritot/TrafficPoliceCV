"use client";

import React, { useState, useEffect, useRef } from "react";
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
  CheckCircle2,
  Maximize2,
  Minimize2,
  LayoutGrid,
  MapPin,
  Shield,
  Layers,
  Video,
  Camera,
  Server
} from "lucide-react";
import ChallanModal, { IncidentAlert } from "./ChallanModal";

export interface CameraNode {
  id: string;
  name: string;
  area: string;
  area_code: string;
  location: string;
  resolution: string;
  corridor_type: string;
  status: string;
  is_active?: boolean;
  active_tracks?: number;
  avg_speed_kmh?: number;
}

const DEFAULT_CAMERAS: CameraNode[] = [
  {
    id: "CAM-01",
    name: "NH-48 Km 32 Toll Plaza Express",
    area: "Zone 1 - North Highway Corridor",
    area_code: "ZONE_NORTH",
    location: "NH-48 Toll Approach (Full HD 50FPS)",
    resolution: "1920x1080 Full HD",
    corridor_type: "dual_carriageway",
    status: "ONLINE",
    is_active: true,
    active_tracks: 4,
    avg_speed_kmh: 74
  },
  {
    id: "CAM-02",
    name: "NH-48 Sector 21 High-Speed Median",
    area: "Zone 1 - North Highway Corridor",
    area_code: "ZONE_NORTH",
    location: "NH-48 Sec 21 Express Flyover",
    resolution: "1280x720 HD",
    corridor_type: "highway",
    status: "ONLINE",
    is_active: false,
    active_tracks: 6,
    avg_speed_kmh: 68
  },
  {
    id: "CAM-03",
    name: "Western Peripheral Flyover Split",
    area: "Zone 1 - North Highway Corridor",
    area_code: "ZONE_NORTH",
    location: "WPE Km 14 Interchange Split",
    resolution: "1280x720 HD",
    corridor_type: "highway",
    status: "ONLINE",
    is_active: false,
    active_tracks: 4,
    avg_speed_kmh: 65
  },
  {
    id: "CAM-04",
    name: "Central Avenue Commercial Arterial",
    area: "Zone 2 - Central Urban District",
    area_code: "ZONE_CENTRAL",
    location: "MG Road / Commercial Hub",
    resolution: "1920x1080 Full HD",
    corridor_type: "dual_carriageway",
    status: "ONLINE",
    is_active: false,
    active_tracks: 5,
    avg_speed_kmh: 48
  },
  {
    id: "CAM-05",
    name: "Inner Ring Road Transit Corridor",
    area: "Zone 2 - Central Urban District",
    area_code: "ZONE_CENTRAL",
    location: "Ring Road BRT Lane",
    resolution: "1280x720 HD",
    corridor_type: "highway",
    status: "ONLINE",
    is_active: false,
    active_tracks: 6,
    avg_speed_kmh: 52
  },
  {
    id: "CAM-06",
    name: "Suburban Feeder - Farmland Perimeter",
    area: "Zone 3 - Suburban & Rural Belt",
    area_code: "ZONE_RURAL",
    location: "SH-12 Agricultural Feeder Road",
    resolution: "1920x1080 Full HD",
    corridor_type: "dual_carriageway",
    status: "ONLINE",
    is_active: false,
    active_tracks: 3,
    avg_speed_kmh: 55
  }
];

const ZONE_DEFINITIONS = [
  { code: "ALL", label: "All Jurisdictions", count: 6 },
  { code: "ZONE_NORTH", label: "Zone 1 - North Highway Corridor", count: 3 },
  { code: "ZONE_CENTRAL", label: "Zone 2 - Central Urban District", count: 2 },
  { code: "ZONE_RURAL", label: "Zone 3 - Suburban & Rural Belt", count: 1 }
];

export default function LiveMonitoring() {
  const [cameras, setCameras] = useState<CameraNode[]>(DEFAULT_CAMERAS);
  const [activeZone, setActiveZone] = useState<string>("ALL");
  const [selectedCameraId, setSelectedCameraId] = useState<string>("CAM-01");
  const [viewLayout, setViewLayout] = useState<"hero" | "grid">("hero");
  const [showRadar, setShowRadar] = useState<boolean>(true);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [streamError, setStreamError] = useState<boolean>(false);
  const [streamKey, setStreamKey] = useState<number>(Date.now());
  const [liveClock, setLiveClock] = useState<string>("");
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const videoContainerRef = useRef<HTMLDivElement>(null);

  const [telemetry, setTelemetry] = useState({
    fps: 25.0,
    active_tracks: 4,
    avg_speed_kmh: 74,
    total_alerts: 0,
    camera_name: "NH-48 Km 32 Toll Plaza Express",
    area: "Zone 1 - North Highway Corridor",
    resolution: "1920x1080 Full HD"
  });

  const [incidents, setIncidents] = useState<IncidentAlert[]>([]);
  const [selectedAlert, setSelectedAlert] = useState<IncidentAlert | null>(null);
  const [filterType, setFilterType] = useState<string>("ALL");

  // Fullscreen toggle handler
  const toggleFullScreen = () => {
    const el = videoContainerRef.current;
    if (!el) return;
    if (!document.fullscreenElement) {
      if (el.requestFullscreen) {
        el.requestFullscreen().catch(() => {});
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen().catch(() => {});
      }
    }
  };

  // Fullscreen event listener and 'F' key shortcut
  useEffect(() => {
    const handleFsChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "f" || e.key === "F") {
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
        toggleFullScreen();
      }
    };
    document.addEventListener("fullscreenchange", handleFsChange);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("fullscreenchange", handleFsChange);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  // Real-time IST Timestamp Clock
  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      setLiveClock(now.toLocaleTimeString("en-IN", { hour12: false }) + " IST");
    };
    updateClock();
    const interval = setInterval(updateClock, 1000);
    return () => clearInterval(interval);
  }, []);

  // Fetch camera directory from backend API
  const fetchCameras = () => {
    fetch("http://127.0.0.1:5000/api/cameras")
      .then((res) => res.json())
      .then((data: CameraNode[]) => {
        if (Array.isArray(data) && data.length > 0) {
          setCameras(data);
        }
      })
      .catch(() => {});
  };

  useEffect(() => {
    fetchCameras();
    const interval = setInterval(fetchCameras, 6000);
    return () => clearInterval(interval);
  }, []);

  // Poll real-time telemetry & incidents from backend
  useEffect(() => {
    const interval = setInterval(() => {
      fetch("http://127.0.0.1:5000/api/telemetry")
        .then((res) => res.json())
        .then((data) => {
          setTelemetry((prev) => ({
            fps: data.fps > 0 ? data.fps : prev.fps,
            active_tracks: data.active_tracks,
            avg_speed_kmh: data.avg_speed_kmh,
            total_alerts: data.total_alerts,
            camera_name: data.camera_name || prev.camera_name,
            area: data.area || prev.area,
            resolution: data.resolution || prev.resolution
          }));
          if (streamError) {
            setStreamError(false);
          }
        })
        .catch(() => {});

      fetch("http://127.0.0.1:5000/api/incidents?limit=25")
        .then((res) => res.json())
        .then((data) => {
          if (Array.isArray(data) && data.length > 0) {
            setIncidents(data);
          }
        })
        .catch(() => {});
    }, 1000);

    return () => clearInterval(interval);
  }, [streamError]);

  // Select camera node across geographic areas
  const handleSelectCamera = (camId: string) => {
    setSelectedCameraId(camId);
    setStreamError(false);
    setStreamKey(Date.now());
    fetch("http://127.0.0.1:5000/api/select_camera", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ camera_id: camId })
    })
      .then(() => fetchCameras())
      .catch(() => {});
  };

  // Toggle BEV Radar
  const handleToggleRadar = () => {
    setShowRadar(!showRadar);
    fetch("http://127.0.0.1:5000/api/toggle_radar", { method: "POST" })
      .then(() => setStreamKey(Date.now()))
      .catch(() => {});
  };

  // Toggle Stream Pause
  const handleTogglePause = () => {
    setIsPaused(!isPaused);
    fetch("http://127.0.0.1:5000/api/toggle_pause", { method: "POST" }).catch(() => {});
  };

  // Manual reconnect handler
  const handleReconnect = () => {
    setStreamError(false);
    setStreamKey(Date.now());
    fetch("http://127.0.0.1:5000/api/select_camera", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ camera_id: selectedCameraId })
    }).catch(() => {});
  };

  // Filter cameras by geographic zone
  const visibleCameras = cameras.filter((cam) => {
    if (activeZone === "ALL") return true;
    return cam.area_code === activeZone;
  });

  const selectedCam = cameras.find((c) => c.id === selectedCameraId) || cameras[0] || DEFAULT_CAMERAS[0];

  const streamUrl = `http://127.0.0.1:5000/api/stream?camera_id=${selectedCameraId}&radar=${showRadar ? '1' : '0'}&t=${streamKey}`;

  const filteredIncidents = incidents.filter((item) => {
    if (filterType === "CRITICAL") return item.severity === "CRITICAL";
    if (filterType === "SPEED") return item.event_type.includes("SPEED");
    if (filterType === "HAZARDS") return item.event_type.includes("ANIMAL") || item.event_type.includes("VENDOR") || item.event_type.includes("WRONG_WAY");
    return true;
  });

  // Calculate cumulative zone metrics
  const zoneActiveTargets = visibleCameras.reduce((acc, c) => acc + (c.id === selectedCameraId ? telemetry.active_tracks : (c.active_tracks || 3)), 0);

  return (
    <div className="space-y-4">
      
      {/* Area / Zone Navigation & View Mode Header */}
      <div className="bg-dark-surface border border-dark-border rounded-xl p-3 shadow-sm flex flex-col md:flex-row items-center justify-between gap-3">
        
        {/* Left: Geographic Zone Selector Chips */}
        <div className="flex items-center gap-1.5 flex-wrap w-full md:w-auto">
          <div className="flex items-center gap-1 text-slate-400 text-xs font-semibold uppercase tracking-wider mr-1.5">
            <Layers className="w-3.5 h-3.5 text-blue-400" />
            <span>Jurisdiction:</span>
          </div>

          {ZONE_DEFINITIONS.map((zone) => {
            const isSelected = activeZone === zone.code;
            return (
              <button
                key={zone.code}
                onClick={() => setActiveZone(zone.code)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 border ${
                  isSelected
                    ? "bg-blue-600/20 border-blue-500/50 text-white shadow-sm"
                    : "bg-dark-card border-dark-border text-slate-400 hover:text-slate-200 hover:border-slate-700"
                }`}
              >
                <span>{zone.label}</span>
                <span className={`px-1.5 py-0.2 rounded text-[10px] font-mono ${
                  isSelected ? "bg-blue-500 text-white font-bold" : "bg-dark-border text-slate-400"
                }`}>
                  {zone.count}
                </span>
              </button>
            );
          })}
        </div>

        {/* Right: Layout Switcher (Hero Focus vs Multi-Camera Matrix) */}
        <div className="flex items-center gap-2 self-end md:self-auto">
          <div className="flex bg-dark-card p-1 rounded-lg border border-dark-border text-xs">
            <button
              onClick={() => setViewLayout("hero")}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors flex items-center gap-1.5 ${
                viewLayout === "hero"
                  ? "bg-dark-border text-white font-semibold shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
              title="Hero Focus View: Primary camera with Metric BEV Radar"
            >
              <Maximize2 className="w-3.5 h-3.5 text-blue-400" />
              <span>Hero Focus</span>
            </button>
            <button
              onClick={() => setViewLayout("grid")}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors flex items-center gap-1.5 ${
                viewLayout === "grid"
                  ? "bg-dark-border text-white font-semibold shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
              title="Multi-Camera Matrix Grid: Simultaneous monitoring of all area feeds"
            >
              <LayoutGrid className="w-3.5 h-3.5 text-emerald-400" />
              <span>Multi-Cam Matrix</span>
            </button>
          </div>
        </div>

      </div>

      {/* Main Monitoring Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        
        {/* Left Column (8 Cols): Video Streams & Telemetry */}
        <div className="lg:col-span-8 space-y-4">
          
          {/* VIEW MODE 1: HERO FOCUS VIEW */}
          {viewLayout === "hero" && (
            <div className="space-y-3">
              
              {/* Primary Video Player Card */}
              <div 
                ref={videoContainerRef} 
                className={`bg-dark-surface border border-dark-border rounded-xl overflow-hidden shadow-sm transition-all ${
                  isFullscreen ? "fixed inset-0 z-50 bg-black flex flex-col justify-between rounded-none" : ""
                }`}
              >
                
                {/* Official CCTV Header Strip */}
                <div className="px-4 py-2.5 bg-dark-card border-b border-dark-border flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-sm animate-pulse" />
                    <div>
                      <div className="font-semibold text-white flex items-center gap-2">
                        <span>[{selectedCam.id}] {selectedCam.name}</span>
                        <span className="px-2 py-0.5 rounded text-[10px] bg-blue-500/10 border border-blue-500/20 text-blue-400 font-mono">
                          {selectedCam.area}
                        </span>
                      </div>
                      <div className="text-[11px] text-slate-400 flex items-center gap-2">
                        <MapPin className="w-3 h-3 text-slate-500 inline" />
                        <span>{selectedCam.location}</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 text-slate-400 font-mono text-[11px]">
                    <span className="text-slate-300 font-medium">{selectedCam.resolution}</span>
                    <span>•</span>
                    <span className="text-emerald-400 font-semibold">{telemetry.fps.toFixed(1)} FPS</span>
                    <span>•</span>
                    <span className="text-slate-400">{liveClock}</span>
                  </div>
                </div>

                {/* Video Viewport Container (Zero Dummy Fallback Video) */}
                <div 
                  onDoubleClick={toggleFullScreen}
                  className={`relative bg-black flex items-center justify-center overflow-hidden ${
                    isFullscreen ? "flex-1 w-full h-[calc(100vh-100px)]" : "aspect-video"
                  }`}
                  title="Double-click to toggle Fullscreen"
                >
                  {!streamError ? (
                    <img
                      key={streamKey}
                      src={streamUrl}
                      alt={`Live Stream - ${selectedCam.name}`}
                      className="w-full h-full object-contain select-none"
                      onError={() => setStreamError(true)}
                    />
                  ) : (
                    /* High-Tech Tactical Surveillance HUD Connecting Screen (NO DUMMY VIDEO) */
                    <div className="w-full h-full bg-[#070b12] flex flex-col items-center justify-center text-center p-6 space-y-4">
                      
                      {/* Animated Radar Scanning HUD Reticle */}
                      <div className="relative w-28 h-28 rounded-full border-2 border-emerald-500/30 flex items-center justify-center">
                        <div className="absolute inset-0 rounded-full border border-dashed border-emerald-500/40 animate-spin" style={{ animationDuration: '6s' }} />
                        <div className="w-20 h-20 rounded-full border border-emerald-500/20 flex items-center justify-center">
                          <Radio className="w-8 h-8 text-emerald-400 animate-pulse" />
                        </div>
                        <div className="absolute w-full h-[1px] bg-emerald-500/50" />
                        <div className="absolute h-full w-[1px] bg-emerald-500/50" />
                      </div>

                      <div className="space-y-1.5 max-w-md">
                        <div className="text-xs font-mono font-bold text-emerald-400 tracking-wider uppercase">
                          [CONNECTING TO LIVE RTSP/MJPEG NODE]
                        </div>
                        <div className="text-sm font-semibold text-white">
                          CCTV Node {selectedCam.id} • {selectedCam.name}
                        </div>
                        <div className="text-[11px] font-mono text-slate-400">
                          Deploying High-Precision WAV-DET Perception Tensor (YOLOv8s • 1.0x Real-Time)
                        </div>
                      </div>

                      <button
                        onClick={handleReconnect}
                        className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium text-xs flex items-center gap-2 shadow-sm transition-colors"
                      >
                        <RefreshCw className="w-3.5 h-3.5" />
                        <span>Reconnect Live Signal</span>
                      </button>

                    </div>
                  )}

                  {/* Top-Left Live Watermark Overlay */}
                  <div className="absolute top-3 left-3 flex items-center gap-2 px-2.5 py-1 rounded bg-black/75 backdrop-blur-md border border-white/10 text-[10px] font-mono text-slate-200 z-10 pointer-events-none">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="font-bold text-white tracking-wider">LIVE SURVEILLANCE</span>
                    <span className="text-slate-400">|</span>
                    <span className="text-emerald-300 font-mono">{selectedCam.id}</span>
                  </div>

                  {/* Top-Right Quick Fullscreen Icon Button */}
                  <button
                    onClick={toggleFullScreen}
                    className="absolute top-3 right-3 p-2 rounded-lg bg-black/70 hover:bg-black/90 backdrop-blur-md border border-white/15 text-slate-300 hover:text-white transition-all z-20 shadow-md group"
                    title={isFullscreen ? "Exit Fullscreen (Esc / F)" : "Fullscreen View (F)"}
                  >
                    {isFullscreen ? (
                      <Minimize2 className="w-4 h-4 text-blue-400 group-hover:scale-110 transition-transform" />
                    ) : (
                      <Maximize2 className="w-4 h-4 text-slate-300 group-hover:text-white group-hover:scale-110 transition-transform" />
                    )}
                  </button>

                  {/* Bottom-Right Regulatory Overlay */}
                  <div className="absolute bottom-3 right-3 px-2 py-0.5 rounded bg-black/60 backdrop-blur-sm border border-white/10 text-[9px] font-mono text-slate-400 pointer-events-none">
                    GOVT OF INDIA • ITMS REGULATORY CONFORMANCE (IRC:SP:110)
                  </div>
                </div>

                {/* Bottom Control Strip */}
                <div className="px-4 py-2.5 bg-dark-card border-t border-dark-border flex items-center justify-between text-xs">
                  
                  {/* Left: Quick Camera Node Switcher */}
                  <div className="flex items-center gap-2">
                    <span className="text-slate-400 text-[11px] font-medium">Node:</span>
                    <select
                      value={selectedCameraId}
                      onChange={(e) => handleSelectCamera(e.target.value)}
                      className="bg-dark-bg border border-dark-border rounded px-2.5 py-1 text-xs text-white focus:outline-none focus:border-blue-500"
                    >
                      {cameras.map((c) => (
                        <option key={c.id} value={c.id}>
                          {c.id}: {c.name} ({c.area.split(" - ")[0]})
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Right: Radar Toggle, Pause, Fullscreen & Refresh Controls */}
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleToggleRadar}
                      className={`px-2.5 py-1 rounded text-[11px] border font-medium flex items-center gap-1.5 transition-colors ${
                        showRadar 
                          ? "bg-dark-border border-slate-600 text-white" 
                          : "bg-dark-bg border-dark-border text-slate-400 hover:text-slate-200"
                      }`}
                      title="Toggle 2D Top-Down Metric BEV Radar"
                    >
                      {showRadar ? <Eye className="w-3.5 h-3.5 text-blue-400" /> : <EyeOff className="w-3.5 h-3.5" />}
                      <span>BEV Radar</span>
                    </button>

                    <button
                      onClick={toggleFullScreen}
                      className={`px-2.5 py-1 rounded text-[11px] border font-medium flex items-center gap-1.5 transition-colors ${
                        isFullscreen
                          ? "bg-blue-600/30 border-blue-500 text-white"
                          : "bg-dark-bg border-dark-border text-slate-300 hover:text-white hover:border-slate-600"
                      }`}
                      title="Expand to Fullscreen (F)"
                    >
                      {isFullscreen ? <Minimize2 className="w-3.5 h-3.5 text-blue-400" /> : <Maximize2 className="w-3.5 h-3.5 text-slate-400" />}
                      <span>{isFullscreen ? "Exit Fullscreen" : "Fullscreen"}</span>
                    </button>

                    <button
                      onClick={handleTogglePause}
                      className="px-2.5 py-1 rounded text-[11px] bg-dark-bg border border-dark-border text-slate-300 hover:text-white flex items-center gap-1 transition-colors"
                      title={isPaused ? "Resume live feed" : "Pause live feed"}
                    >
                      {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
                      <span>{isPaused ? "Resume" : "Pause"}</span>
                    </button>

                    <button
                      onClick={handleReconnect}
                      className="p-1 rounded bg-dark-bg border border-dark-border text-slate-400 hover:text-white transition-colors"
                      title="Refresh & reconnect RTSP signal"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                    </button>
                  </div>

                </div>

              </div>

              {/* Real-Time Telemetry Metric Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="p-3 rounded-lg bg-dark-surface border border-dark-border">
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-semibold">
                    Monitored Targets
                  </span>
                  <div className="text-xl font-bold font-mono text-white mt-0.5">
                    {telemetry.active_tracks} <span className="text-xs font-normal text-slate-400">vehicles</span>
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-dark-surface border border-dark-border">
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-semibold">
                    Corridor Velocity
                  </span>
                  <div className="text-xl font-bold font-mono text-white mt-0.5">
                    {telemetry.avg_speed_kmh} <span className="text-xs font-normal text-slate-400">km/h</span>
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-dark-surface border border-dark-border">
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-semibold">
                    Infractions Flagged
                  </span>
                  <div className="text-xl font-bold font-mono text-amber-400 mt-0.5">
                    {telemetry.total_alerts}
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-dark-surface border border-dark-border">
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-semibold">
                    Perception Pipeline
                  </span>
                  <div className="text-xl font-bold font-mono text-emerald-400 mt-0.5">
                    {telemetry.fps.toFixed(1)} <span className="text-xs font-normal text-slate-400">FPS</span>
                  </div>
                </div>
              </div>

              {/* Area Camera Directory Strip (Below Hero Feed) */}
              <div className="bg-dark-surface border border-dark-border rounded-xl p-3 space-y-2">
                <div className="flex items-center justify-between text-xs pb-1 border-b border-dark-border">
                  <div className="flex items-center gap-1.5 font-semibold text-slate-200">
                    <Camera className="w-3.5 h-3.5 text-blue-400" />
                    <span>Area Camera Directory ({visibleCameras.length} Nodes in {activeZone === 'ALL' ? 'All Jurisdictions' : activeZone})</span>
                  </div>
                  <span className="text-[11px] text-slate-500 font-mono">Click card to switch primary monitor</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5 pt-1">
                  {visibleCameras.map((cam) => {
                    const isFocused = cam.id === selectedCameraId;
                    return (
                      <div
                        key={cam.id}
                        onClick={() => handleSelectCamera(cam.id)}
                        className={`cursor-pointer rounded-lg border p-2.5 transition-all text-left flex flex-col justify-between ${
                          isFocused
                            ? "bg-blue-950/20 border-blue-500/80 ring-1 ring-blue-500/40 shadow-sm"
                            : "bg-dark-card border-dark-border hover:border-slate-600 hover:bg-dark-subtle"
                        }`}
                      >
                        <div className="flex items-center justify-between text-[11px] mb-1">
                          <span className={`font-mono font-bold ${isFocused ? 'text-blue-400' : 'text-slate-300'}`}>
                            [{cam.id}]
                          </span>
                          <span className="flex items-center gap-1 text-[10px] font-medium text-emerald-400">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                            <span>ONLINE</span>
                          </span>
                        </div>

                        <div className="font-medium text-xs text-white truncate" title={cam.name}>
                          {cam.name}
                        </div>

                        <div className="text-[11px] text-slate-400 truncate mt-0.5">
                          {cam.location}
                        </div>

                        <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono mt-2 pt-1 border-t border-dark-border/60">
                          <span>{cam.resolution}</span>
                          <span className="text-slate-400">
                            {cam.id === selectedCameraId ? `${telemetry.active_tracks} targets` : `${cam.active_tracks || 4} targets`}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

            </div>
          )}

          {/* VIEW MODE 2: MULTI-CAMERA MATRIX VIEW (GRID MODE) */}
          {viewLayout === "grid" && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs px-1">
                <span className="font-semibold text-slate-200 uppercase tracking-wider">
                  Live Multi-Camera Matrix ({visibleCameras.length} Nodes Active)
                </span>
                <span className="text-slate-400 text-xs">
                  Click &apos;Focus on Monitor&apos; on any camera to expand with BEV Radar
                </span>
              </div>

              {/* Multi-CCTV Monitor Matrix Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                {visibleCameras.map((cam) => {
                  const isPrimary = cam.id === selectedCameraId;
                  return (
                    <div
                      key={cam.id}
                      className={`bg-dark-surface border rounded-xl overflow-hidden flex flex-col justify-between transition-all ${
                        isPrimary ? "border-blue-500/80 ring-1 ring-blue-500/40" : "border-dark-border hover:border-slate-600"
                      }`}
                    >
                      {/* Cell Monitor Header */}
                      <div className="px-3 py-2 bg-dark-card border-b border-dark-border flex items-center justify-between text-xs">
                        <div className="flex items-center gap-2 truncate">
                          <span className="w-2 h-2 rounded-full bg-emerald-500" />
                          <span className="font-bold font-mono text-white">[{cam.id}]</span>
                          <span className="text-slate-300 font-medium truncate">{cam.name}</span>
                        </div>
                        <span className="text-[10px] font-mono text-slate-400 shrink-0">{cam.resolution}</span>
                      </div>

                      {/* Cell Snapshot / Stream Viewport */}
                      <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden">
                        <img
                          src={`http://127.0.0.1:5000/api/camera_snapshot?camera_id=${cam.id}&frame=35&t=${streamKey}`}
                          alt={cam.name}
                          className="w-full h-full object-cover"
                        />

                        {/* Top-Right Status Badge */}
                        <div className="absolute top-2 right-2 flex items-center gap-1 px-1.5 py-0.5 rounded bg-black/70 backdrop-blur-sm border border-white/10 text-[9px] font-mono text-emerald-400">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                          <span>LIVE</span>
                        </div>

                        {/* Bottom-Left Location Watermark */}
                        <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/60 backdrop-blur-sm border border-white/10 text-[10px] text-slate-300 truncate max-w-[85%]">
                          {cam.location}
                        </div>
                      </div>

                      {/* Cell Footer Bar */}
                      <div className="px-3 py-2 bg-dark-card border-t border-dark-border flex items-center justify-between text-xs">
                        <div className="flex items-center gap-3 text-[11px] text-slate-400 font-mono">
                          <span>Targets: <strong className="text-white">{cam.id === selectedCameraId ? telemetry.active_tracks : (cam.active_tracks || 4)}</strong></span>
                          <span>•</span>
                          <span>Speed: <strong className="text-white">{cam.id === selectedCameraId ? telemetry.avg_speed_kmh : (cam.avg_speed_kmh || 70)} km/h</strong></span>
                        </div>

                        <button
                          onClick={() => {
                            handleSelectCamera(cam.id);
                            setViewLayout("hero");
                          }}
                          className="px-2.5 py-1 rounded text-[11px] bg-blue-600/20 border border-blue-500/40 text-blue-300 hover:bg-blue-600 hover:text-white font-medium transition-colors"
                        >
                          Focus on Monitor →
                        </button>
                      </div>

                    </div>
                  );
                })}
              </div>

            </div>
          )}

        </div>

        {/* Right Column (4 Cols): Quiet Infraction Feed & Challan Issuance */}
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
                      ? "bg-dark-border text-white font-semibold" 
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>

          {/* Incident List */}
          <div className="space-y-2 overflow-y-auto max-h-[560px] pr-1">
            {filteredIncidents.length === 0 ? (
              <div className="py-16 text-center text-slate-500 text-xs">
                <CheckCircle2 className="w-7 h-7 text-emerald-500/60 mx-auto mb-2" />
                <div className="font-medium text-slate-400">All Corridors Clear</div>
                <div className="text-[11px] text-slate-500 mt-0.5">No active traffic safety violations detected.</div>
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
                      className="text-blue-400 hover:text-blue-300 font-medium text-[11px] transition-colors"
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
