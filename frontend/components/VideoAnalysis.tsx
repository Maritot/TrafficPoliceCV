"use client";

import React, { useState, useRef, useEffect } from "react";
import { 
  UploadCloud, 
  FileVideo, 
  CheckCircle2, 
  Download, 
  Sliders,
  Sparkles,
  FileCheck,
  X,
  Play,
  Activity,
  AlertTriangle,
  Gauge,
  Car,
  ShieldCheck,
  RefreshCw,
  Info
} from "lucide-react";
import ChallanModal, { IncidentAlert } from "./ChallanModal";

interface CaseStudy {
  id: string;
  name: string;
  filename: string;
  mode: "highway" | "indian";
  rawSrc: string;
  annotatedSrc: string;
  description: string;
}

interface ViolationItem {
  ref: string;
  track_id: number;
  entity: string;
  infraction: string;
  speed: string;
  lane: string;
  timestamp?: string;
  mva: string;
  penalty: string;
  severity?: "CRITICAL" | "WARNING";
}

interface AnalysisResult {
  status: string;
  clip_name: string;
  processed_video_url: string;
  raw_video_url: string;
  total_frames: number;
  duration_s: number;
  active_targets_count: number;
  avg_speed_kmh: number;
  max_speed_kmh: number;
  total_violations: number;
  violations: ViolationItem[];
  processing_time_s: number;
}

const PRESET_CASES: CaseStudy[] = [
  {
    id: "case_indian",
    name: "Indian Arterial Mixed Hazards",
    filename: "indian_traffic_sample.mp4",
    mode: "indian",
    rawSrc: "http://127.0.0.1:5000/api/media/indian_traffic_sample.mp4",
    annotatedSrc: "http://127.0.0.1:5000/api/media/output_indian_demo.mp4",
    description: "Cattle on live carriageway, mobile pushcart vendors on shoulder, and evasive swerves."
  },
  {
    id: "case_highway",
    name: "NH-48 National Highway Corridor",
    filename: "video1.avi",
    mode: "highway",
    rawSrc: "http://127.0.0.1:5000/api/media/video1.avi",
    annotatedSrc: "http://127.0.0.1:5000/api/media/output_highway_hd.mp4",
    description: "Multi-lane high-speed traffic, solid lane boundary compliance, and metric Frenet kinematics."
  }
];

export default function VideoAnalysis() {
  const [ingestMode, setIngestMode] = useState<"upload" | "preset">("upload");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<CaseStudy>(PRESET_CASES[0]);
  const [modelMode, setModelMode] = useState<"highway" | "indian">("highway");
  
  // Real Processing States
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [processStatusText, setProcessStatusText] = useState<string>("");
  const [progress, setProgress] = useState<number>(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [hasProcessed, setHasProcessed] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // UI View States
  const [viewMode, setViewMode] = useState<"annotated" | "raw" | "side_by_side">("annotated");
  const [selectedAlert, setSelectedAlert] = useState<IncidentAlert | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoPlayerRef = useRef<HTMLVideoElement>(null);

  // File Upload Handlers
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setUploadedFile(file);
      if (file.name.toLowerCase().includes("video1") || file.name.toLowerCase().includes("highway")) {
        setModelMode("highway");
      } else if (file.name.toLowerCase().includes("indian")) {
        setModelMode("indian");
      }
      setAnalysisResult(null);
      setHasProcessed(false);
      setErrorMessage(null);
      setIngestMode("upload");
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setUploadedFile(file);
      if (file.name.toLowerCase().includes("video1") || file.name.toLowerCase().includes("highway")) {
        setModelMode("highway");
      } else if (file.name.toLowerCase().includes("indian")) {
        setModelMode("indian");
      }
      setAnalysisResult(null);
      setHasProcessed(false);
      setErrorMessage(null);
      setIngestMode("upload");
    }
  };

  const handleClearUpload = () => {
    setUploadedFile(null);
    setAnalysisResult(null);
    setHasProcessed(false);
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Preset Selection Handler
  const handleSelectPreset = (preset: CaseStudy) => {
    setSelectedPreset(preset);
    setModelMode(preset.mode);
    setAnalysisResult(null);
    setHasProcessed(false);
    setErrorMessage(null);
    setIngestMode("preset");
  };

  // Real Computer Vision Processing Trigger
  const handleProcess = async () => {
    setIsProcessing(true);
    setProgress(15);
    setProcessStatusText("Uploading video footage to LATIS AI Perception Engine...");
    setErrorMessage(null);

    // Smooth progress progression
    const timer = setInterval(() => {
      setProgress((prev) => {
        if (prev < 40) return prev + 6;
        if (prev < 70) return prev + 3;
        if (prev < 90) return prev + 1;
        return prev;
      });
    }, 350);

    try {
      const formData = new FormData();
      if (ingestMode === "upload" && uploadedFile) {
        formData.append("file", uploadedFile);
        formData.append("model", modelMode);
        formData.append("max_frames", "350");
      } else {
        formData.append("preset", selectedPreset.id);
        formData.append("model", modelMode);
        formData.append("max_frames", "350");
      }

      setProcessStatusText("Executing YOLOv8 + ByteTrack 1-to-1 Metric Tracker & Curvilinear Frenet Homography...");

      const res = await fetch("http://127.0.0.1:5000/api/analyze-video", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.error || `Server responded with HTTP ${res.status}`);
      }

      const data: AnalysisResult = await res.json();
      clearInterval(timer);
      setProgress(100);
      setProcessStatusText("Analysis Complete! Generating evidence video and statutory incident register...");

      setTimeout(() => {
        setAnalysisResult(data);
        setHasProcessed(true);
        setIsProcessing(false);
      }, 400);

    } catch (err: any) {
      clearInterval(timer);
      setIsProcessing(false);
      setErrorMessage(err.message || "Failed to communicate with LATIS Computer Vision API (Port 5000).");
      console.error("Forensic video analysis error:", err);
    }
  };

  // Determine current active video names and source URLs
  const currentVideoName = ingestMode === "upload" && uploadedFile
    ? uploadedFile.name
    : selectedPreset.name;

  // Annotated Video URL (pointing to backend media or preset)
  let annotatedVideoSource = "";
  let rawVideoSource = "";

  if (analysisResult) {
    annotatedVideoSource = `http://127.0.0.1:5000${analysisResult.processed_video_url}`;
    rawVideoSource = `http://127.0.0.1:5000${analysisResult.raw_video_url}`;
  } else if (ingestMode === "preset") {
    annotatedVideoSource = selectedPreset.annotatedSrc;
    rawVideoSource = selectedPreset.rawSrc;
  }

  // Active list of violations
  const violations: ViolationItem[] = analysisResult
    ? analysisResult.violations
    : (modelMode === "indian" ? [
        {
          ref: "VIO-01",
          track_id: 10,
          entity: "Street Vendor Pushcart",
          infraction: "Mobile Vendor Encroaching Active Lane",
          speed: "2 km/h",
          lane: "Lane 1",
          mva: "Sec 283 IPC / Sec 201 MVA",
          penalty: "₹ 2,000",
          severity: "WARNING"
        },
        {
          ref: "VIO-02",
          track_id: 38,
          entity: "Cow/Cattle",
          infraction: "Live Cattle Hazard on Fast Lane",
          speed: "4 km/h",
          lane: "Lane 2",
          mva: "Section 289, IPC",
          penalty: "₹ 1,000",
          severity: "CRITICAL"
        },
        {
          ref: "VIO-03",
          track_id: 30,
          entity: "Street Vendor Pushcart",
          infraction: "Unauthorized Roadway Obstruction",
          speed: "3 km/h",
          lane: "Lane 1",
          mva: "Sec 283 IPC / Sec 201 MVA",
          penalty: "₹ 2,000",
          severity: "WARNING"
        }
      ] : [
        {
          ref: "VIO-01",
          track_id: 102,
          entity: "Car",
          infraction: "Exceeded Speed Limit (89 km/h in 80 zone)",
          speed: "89 km/h",
          lane: "Lane 3",
          mva: "Section 183, MVA 2019",
          penalty: "₹ 2,000",
          severity: "WARNING"
        },
        {
          ref: "VIO-02",
          track_id: 73,
          entity: "Car",
          infraction: "Crossed Solid Lane Boundary",
          speed: "62 km/h",
          lane: "Lane 1",
          mva: "Section 177A, MVA 2019",
          penalty: "₹ 1,000",
          severity: "WARNING"
        }
      ]);

  return (
    <div className="space-y-5">
      
      {/* 1. Ingestion & Setup Card */}
      <div className="bg-dark-surface border border-dark-border rounded-xl p-4 space-y-4">
        
        {/* Header & Source Mode Switcher */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-dark-border">
          <div>
            <h2 className="text-sm font-semibold text-white">Video Ingestion & Forensic Processing</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Upload custom traffic video clips or select verified government case studies for computer vision analysis.
            </p>
          </div>

          {/* Ingestion Source Switcher */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Ingest Source:</span>
            <div className="flex bg-dark-bg p-0.5 rounded-md border border-dark-border text-xs">
              <button
                onClick={() => setIngestMode("upload")}
                className={`px-3 py-1 rounded text-xs transition-colors ${
                  ingestMode === "upload" 
                    ? "bg-dark-border text-white font-semibold" 
                    : "text-slate-400 hover:text-white"
                }`}
              >
                Upload File
              </button>
              <button
                onClick={() => setIngestMode("preset")}
                className={`px-3 py-1 rounded text-xs transition-colors ${
                  ingestMode === "preset" 
                    ? "bg-dark-border text-white font-semibold" 
                    : "text-slate-400 hover:text-white"
                }`}
              >
                Preset Case Studies
              </button>
            </div>
          </div>
        </div>

        {/* Error Alert Banner */}
        {errorMessage && (
          <div className="p-3 bg-red-950/40 border border-red-500/40 rounded-lg flex items-center justify-between gap-3 text-red-200 text-xs">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button 
              onClick={() => setErrorMessage(null)}
              className="text-slate-400 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Ingestion Area: Drag & Drop Upload Zone OR Preset Case Study Cards */}
        {ingestMode === "upload" ? (
          <div>
            {/* Hidden native file input */}
            <input 
              type="file" 
              ref={fileInputRef} 
              accept="video/mp4,video/avi,video/quicktime,video/mkv,video/webm" 
              onChange={handleFileSelect}
              className="hidden" 
            />

            {!uploadedFile ? (
              /* Drag & Drop Box */
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all flex flex-col items-center justify-center gap-2 ${
                  isDragging 
                    ? "border-gov-blue bg-gov-blue/10" 
                    : "border-dark-border bg-dark-card/40 hover:bg-dark-card hover:border-slate-500"
                }`}
              >
                <div className="w-12 h-12 rounded-full bg-dark-bg border border-dark-border flex items-center justify-center text-slate-300">
                  <UploadCloud className="w-6 h-6 text-gov-blue" />
                </div>
                <div>
                  <span className="text-xs font-semibold text-white">
                    Click to browse or drag & drop video clip here
                  </span>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Supports CCTV video exports: .avi, .mp4, .mov, .mkv (Up to 500 MB)
                  </p>
                </div>
              </div>
            ) : (
              /* Uploaded File Selected Card */
              <div className="p-3.5 rounded-lg bg-dark-card border border-dark-border flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-lg bg-gov-blue/10 border border-gov-blue/30 flex items-center justify-center text-gov-blue">
                    <FileVideo className="w-5 h-5 text-gov-blue" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-xs text-white">{uploadedFile.name}</span>
                      <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${
                        hasProcessed 
                          ? "bg-emerald-950/60 text-emerald-400 border-emerald-500/30" 
                          : "bg-blue-950/60 text-blue-400 border-blue-500/30"
                      }`}>
                        {hasProcessed ? "Analyzed with AI" : "Staged for Ingestion"}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5">
                      Size: {(uploadedFile.size / (1024 * 1024)).toFixed(1)} MB • Type: {uploadedFile.type || "Surveillance Video"}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="px-2.5 py-1 rounded bg-dark-bg border border-dark-border hover:bg-dark-surface text-slate-300 text-xs font-medium transition-colors"
                  >
                    Change File
                  </button>
                  <button
                    onClick={handleClearUpload}
                    className="p-1 rounded bg-dark-bg border border-dark-border hover:text-red-400 text-slate-400 transition-colors"
                    title="Remove file"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Preset Case Studies Grid */
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {PRESET_CASES.map((cs) => {
              const isSelected = selectedPreset.id === cs.id;
              return (
                <button
                  key={cs.id}
                  onClick={() => handleSelectPreset(cs)}
                  className={`p-3 rounded-lg border text-left text-xs transition-all flex flex-col gap-1 ${
                    isSelected
                      ? "bg-dark-card border-gov-blue text-white shadow-sm"
                      : "bg-dark-card/50 border-dark-border text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white text-[13px]">{cs.name}</span>
                    <span className="font-mono text-[10px] text-slate-400">{cs.filename}</span>
                  </div>
                  <p className="text-[11px] text-slate-400 line-clamp-2">{cs.description}</p>
                </button>
              );
            })}
          </div>
        )}

        {/* Model Pipeline Selector & Action Button */}
        <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-t border-dark-border">
          <div className="flex items-center gap-2 text-xs">
            <span className="text-slate-400">Target Perception Model:</span>
            <div className="flex bg-dark-bg p-0.5 rounded-md border border-dark-border text-xs">
              <button
                onClick={() => setModelMode("highway")}
                className={`px-2.5 py-1 rounded text-xs transition-colors ${
                  modelMode === "highway" ? "bg-dark-border text-white font-semibold" : "text-slate-400 hover:text-white"
                }`}
              >
                Highway Standard
              </button>
              <button
                onClick={() => setModelMode("indian")}
                className={`px-2.5 py-1 rounded text-xs transition-colors ${
                  modelMode === "indian" ? "bg-dark-border text-white font-semibold" : "text-slate-400 hover:text-white"
                }`}
              >
                Indian Hazards (Cattle/Vendors)
              </button>
            </div>
          </div>

          <button
            onClick={handleProcess}
            disabled={isProcessing || (ingestMode === "upload" && !uploadedFile)}
            className={`px-5 py-2 rounded-lg text-xs font-semibold transition-all flex items-center justify-center gap-2 ${
              isProcessing || (ingestMode === "upload" && !uploadedFile)
                ? "bg-slate-700/60 text-slate-400 cursor-not-allowed border border-slate-700"
                : "bg-gov-blue hover:bg-gov-hover text-white shadow shadow-gov-blue/20"
            }`}
          >
            {isProcessing ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Processing Video Pipeline...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Run Forensic Analysis on {currentVideoName}</span>
              </>
            )}
          </button>
        </div>

        {/* Linear Progress Bar with Dynamic Step Text */}
        {isProcessing && (
          <div className="space-y-1.5 pt-1">
            <div className="flex justify-between text-xs text-slate-300">
              <span className="font-medium flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-gov-blue animate-pulse" />
                {processStatusText}
              </span>
              <span className="font-mono text-gov-blue font-bold">{progress}%</span>
            </div>
            <div className="w-full h-1.5 bg-dark-bg rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-gov-blue to-emerald-500 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

      </div>

      {/* Real Ingestion KPI Summary Bar (when analysis has completed) */}
      {analysisResult && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          <div className="bg-dark-surface border border-dark-border rounded-lg p-3">
            <span className="text-[11px] text-slate-400 flex items-center gap-1">
              <FileCheck className="w-3.5 h-3.5 text-gov-blue" />
              Frames Analyzed
            </span>
            <div className="text-base font-semibold text-white mt-1">
              {analysisResult.total_frames} <span className="text-xs text-slate-400">({analysisResult.duration_s}s)</span>
            </div>
          </div>

          <div className="bg-dark-surface border border-dark-border rounded-lg p-3">
            <span className="text-[11px] text-slate-400 flex items-center gap-1">
              <Car className="w-3.5 h-3.5 text-blue-400" />
              Tracked Targets
            </span>
            <div className="text-base font-semibold text-white mt-1">
              {analysisResult.active_targets_count} <span className="text-xs text-slate-400">vehicles</span>
            </div>
          </div>

          <div className="bg-dark-surface border border-dark-border rounded-lg p-3">
            <span className="text-[11px] text-slate-400 flex items-center gap-1">
              <Gauge className="w-3.5 h-3.5 text-emerald-400" />
              Average Speed
            </span>
            <div className="text-base font-semibold text-white mt-1">
              {analysisResult.avg_speed_kmh} <span className="text-xs text-slate-400">km/h</span>
            </div>
          </div>

          <div className="bg-dark-surface border border-dark-border rounded-lg p-3">
            <span className="text-[11px] text-slate-400 flex items-center gap-1">
              <Gauge className="w-3.5 h-3.5 text-amber-400" />
              Peak Velocity
            </span>
            <div className="text-base font-semibold text-white mt-1">
              {analysisResult.max_speed_kmh} <span className="text-xs text-slate-400">km/h</span>
            </div>
          </div>

          <div className="bg-dark-surface border border-dark-border rounded-lg p-3">
            <span className="text-[11px] text-slate-400 flex items-center gap-1">
              <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
              Safety Violations
            </span>
            <div className="text-base font-semibold text-red-400 mt-1">
              {analysisResult.total_violations} <span className="text-xs text-slate-400">infractions</span>
            </div>
          </div>
        </div>
      )}

      {/* 2. Forensic Evidence Video Player */}
      <div className="bg-dark-surface border border-dark-border rounded-xl p-4 space-y-3">
        
        {/* View Mode Toggle Strip */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-dark-border">
          <div>
            <h3 className="text-sm font-semibold text-white">Forensic Evidence Player</h3>
            <span className="text-xs text-slate-400">
              Active Video: <strong className="text-slate-200">{currentVideoName}</strong>
            </span>
          </div>

          <div className="flex bg-dark-bg p-0.5 rounded-md border border-dark-border text-xs">
            <button
              onClick={() => setViewMode("annotated")}
              className={`px-3 py-1 rounded text-xs transition-colors ${
                viewMode === "annotated" ? "bg-dark-border text-white font-semibold" : "text-slate-400 hover:text-white"
              }`}
            >
              AI Annotated Output
            </button>
            <button
              onClick={() => setViewMode("raw")}
              className={`px-3 py-1 rounded text-xs transition-colors ${
                viewMode === "raw" ? "bg-dark-border text-white font-semibold" : "text-slate-400 hover:text-white"
              }`}
            >
              Raw Input
            </button>
            <button
              onClick={() => setViewMode("side_by_side")}
              className={`px-3 py-1 rounded text-xs transition-colors ${
                viewMode === "side_by_side" ? "bg-dark-border text-white font-semibold" : "text-slate-400 hover:text-white"
              }`}
            >
              Side-by-Side Dual
            </button>
          </div>
        </div>

        {/* Video Canvases or Ingestion Prompt */}
        {ingestMode === "upload" && !hasProcessed && !analysisResult ? (
          /* Staged Upload Waiting for Execution Card */
          <div className="aspect-video bg-dark-card/60 rounded-lg border border-dashed border-dark-border flex flex-col items-center justify-center p-6 text-center gap-3">
            <div className="w-12 h-12 rounded-full bg-dark-bg border border-dark-border flex items-center justify-center text-gov-blue">
              <FileVideo className="w-6 h-6 text-gov-blue" />
            </div>
            <div className="max-w-md">
              <h4 className="text-sm font-semibold text-white">
                {uploadedFile ? `"${uploadedFile.name}" Ready for AI Analysis` : "No Video Clip Selected"}
              </h4>
              <p className="text-xs text-slate-400 mt-1">
                {uploadedFile
                  ? "Click the 'Run Forensic Analysis' button above to transmit this clip to the Python CV engine for frame-by-frame homography projection and behavior detection."
                  : "Upload a surveillance video clip or select a Preset Case Study from above to inspect forensic evidence."}
              </p>
            </div>
            {uploadedFile && (
              <button
                onClick={handleProcess}
                disabled={isProcessing}
                className="mt-1 px-4 py-2 rounded-lg bg-gov-blue hover:bg-gov-hover text-white text-xs font-semibold flex items-center gap-2 shadow"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Run Forensic Analysis Now</span>
              </button>
            )}
          </div>
        ) : (
          /* Active Playing Video Canvases */
          <div className={`grid gap-3 ${viewMode === "side_by_side" ? "grid-cols-1 md:grid-cols-2" : "grid-cols-1"}`}>
            
            {(viewMode === "raw" || viewMode === "side_by_side") && (
              <div className="relative aspect-video bg-black rounded-lg overflow-hidden border border-dark-border flex flex-col">
                <span className="absolute top-2 left-2 z-10 text-[10px] font-mono px-2 py-0.5 rounded bg-black/70 text-slate-300 border border-slate-700/50">
                  RAW UNPROCESSED INPUT
                </span>
                <video
                  key={rawVideoSource}
                  src={rawVideoSource}
                  controls
                  autoPlay
                  loop
                  muted
                  playsInline
                  className="w-full h-full object-contain"
                />
              </div>
            )}

            {(viewMode === "annotated" || viewMode === "side_by_side") && (
              <div className="relative aspect-video bg-black rounded-lg overflow-hidden border border-dark-border flex flex-col">
                <span className="absolute top-2 left-2 z-10 text-[10px] font-mono px-2 py-0.5 rounded bg-black/70 text-emerald-400 font-semibold border border-emerald-500/30">
                  AI ANNOTATED EVIDENCE + BEV RADAR
                </span>
                <video
                  ref={videoPlayerRef}
                  key={annotatedVideoSource}
                  src={annotatedVideoSource}
                  controls
                  autoPlay
                  loop
                  muted
                  playsInline
                  className="w-full h-full object-contain"
                />
              </div>
            )}

          </div>
        )}

      </div>

      {/* 3. Forensic Infraction Audit Register */}
      <div className="bg-dark-surface border border-dark-border rounded-xl p-4 space-y-3">
        
        <div className="flex items-center justify-between pb-2 border-b border-dark-border">
          <div>
            <h3 className="text-sm font-semibold text-white">Incident & Infraction Audit Register</h3>
            <span className="text-xs text-slate-400">
              Deterministic violations mapped to the Motor Vehicles (Amendment) Act 2019
            </span>
          </div>

          <div className="flex items-center gap-2">
            {annotatedVideoSource && (
              <a
                href={annotatedVideoSource}
                download={`LATIS_Forensic_${currentVideoName}.mp4`}
                className="px-2.5 py-1 rounded bg-dark-bg border border-dark-border hover:bg-dark-card text-xs text-slate-300 flex items-center gap-1.5 transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download Evidence Video</span>
              </a>
            )}
          </div>
        </div>

        {/* Minimal Clean Table or Zero Violations Banner */}
        {hasProcessed && violations.length === 0 ? (
          <div className="p-4 rounded-lg bg-emerald-950/30 border border-emerald-500/30 flex items-center gap-3 text-emerald-300 text-xs">
            <ShieldCheck className="w-5 h-5 text-emerald-400 flex-shrink-0" />
            <div>
              <span className="font-semibold text-white">100% Traffic Safety Compliance</span>
              <p className="text-[11px] text-emerald-400/80 mt-0.5">
                No statutory speed violations, solid lane crossing, or roadway obstructions detected in the analyzed video clip.
              </p>
            </div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-[11px] text-slate-400 uppercase border-b border-dark-border bg-dark-card">
                <tr>
                  <th className="py-2 px-3">Ref</th>
                  <th className="py-2 px-3">Target ID</th>
                  <th className="py-2 px-3">Entity Class</th>
                  <th className="py-2 px-3">Observed Infraction</th>
                  <th className="py-2 px-3">Velocity</th>
                  <th className="py-2 px-3">Lane</th>
                  <th className="py-2 px-3">Legal Provision</th>
                  <th className="py-2 px-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-border/40 text-slate-300">
                {violations.map((v, i) => (
                  <tr key={i} className="hover:bg-dark-card/40 transition-colors">
                    <td className="py-2.5 px-3 font-mono text-slate-400">{v.ref}</td>
                    <td className="py-2.5 px-3 font-mono font-semibold text-white">#{v.track_id}</td>
                    <td className="py-2.5 px-3">{v.entity}</td>
                    <td className="py-2.5 px-3 text-white font-medium">{v.infraction}</td>
                    <td className="py-2.5 px-3 font-mono">{v.speed}</td>
                    <td className="py-2.5 px-3 text-slate-400">{v.lane}</td>
                    <td className="py-2.5 px-3 text-amber-400 font-mono text-[11px]">{v.mva}</td>
                    <td className="py-2.5 px-3 text-right">
                      <button
                        onClick={() => setSelectedAlert({
                          track_id: v.track_id,
                          class_name: v.entity,
                          event_type: v.infraction,
                          severity: v.severity || "WARNING",
                          message: v.infraction,
                          timestamp: 10.0,
                          speed_kmh: parseFloat(v.speed) || 0,
                          lane_id: v.lane
                        })}
                        className="text-gov-blue hover:text-gov-hover font-medium text-[11px]"
                      >
                        Issue Challan →
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

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
