"use client";

import React, { useState, useRef } from "react";
import { 
  UploadCloud, 
  FileVideo, 
  CheckCircle2, 
  Download, 
  Sliders,
  Sparkles,
  FileCheck,
  X,
  Play
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

const PRESET_CASES: CaseStudy[] = [
  {
    id: "case_indian",
    name: "Indian Arterial Mixed Hazards",
    filename: "indian_traffic_sample.mp4",
    mode: "indian",
    rawSrc: "/videos/indian_traffic_sample.mp4",
    annotatedSrc: "/videos/output_indian_demo.mp4",
    description: "Cattle on live carriageway, mobile pushcart vendors on shoulder, and evasive swerves."
  },
  {
    id: "case_highway",
    name: "NH-48 National Highway Corridor",
    filename: "video1.avi",
    mode: "highway",
    rawSrc: "/videos/video1.avi",
    annotatedSrc: "/videos/output_highway_hd.mp4",
    description: "Multi-lane high-speed traffic, solid lane boundary compliance, and metric Frenet kinematics."
  }
];

export default function VideoAnalysis() {
  const [ingestMode, setIngestMode] = useState<"upload" | "preset">("upload");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<CaseStudy>(PRESET_CASES[0]);
  const [modelMode, setModelMode] = useState<"highway" | "indian">("indian");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(100);
  const [viewMode, setViewMode] = useState<"annotated" | "raw" | "side_by_side">("annotated");
  const [selectedAlert, setSelectedAlert] = useState<IncidentAlert | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // File Upload Handlers
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setUploadedFile(file);
      setUploadedVideoUrl(URL.createObjectURL(file));
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
      setUploadedVideoUrl(URL.createObjectURL(file));
      setIngestMode("upload");
    }
  };

  const handleClearUpload = () => {
    setUploadedFile(null);
    setUploadedVideoUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Preset Selection Handler
  const handleSelectPreset = (preset: CaseStudy) => {
    setSelectedPreset(preset);
    setModelMode(preset.mode);
    setIngestMode("preset");
  };

  // Execute Pipeline Simulation
  const handleProcess = () => {
    setIsProcessing(true);
    setProgress(0);

    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsProcessing(false);
          return 100;
        }
        return prev + 25;
      });
    }, 350);
  };

  // Determine current active video sources
  const currentVideoName = ingestMode === "upload" && uploadedFile
    ? uploadedFile.name
    : selectedPreset.name;

  const rawVideoSource = ingestMode === "upload" && uploadedVideoUrl
    ? uploadedVideoUrl
    : selectedPreset.rawSrc;

  const annotatedVideoSource = modelMode === "highway"
    ? "/videos/output_highway_hd.mp4"
    : "/videos/output_indian_demo.mp4";

  // Structured Incident Register
  const violations = modelMode === "indian" ? [
    {
      ref: "VIO-01",
      track_id: 10,
      entity: "Street Vendor Pushcart",
      infraction: "Mobile Vendor Encroaching Active Lane",
      speed: "2 km/h",
      lane: "Lane 1",
      mva: "Sec 283 IPC / Sec 201 MVA",
      penalty: "₹ 2,000"
    },
    {
      ref: "VIO-02",
      track_id: 38,
      entity: "Cow/Cattle",
      infraction: "Live Cattle Hazard on Fast Lane",
      speed: "4 km/h",
      lane: "Lane 2",
      mva: "Section 289, IPC",
      penalty: "₹ 1,000"
    },
    {
      ref: "VIO-03",
      track_id: 30,
      entity: "Street Vendor Pushcart",
      infraction: "Unauthorized Roadway Obstruction",
      speed: "3 km/h",
      lane: "Lane 1",
      mva: "Sec 283 IPC / Sec 201 MVA",
      penalty: "₹ 2,000"
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
      penalty: "₹ 2,000"
    },
    {
      ref: "VIO-02",
      track_id: 73,
      entity: "Car",
      infraction: "Crossed Solid Lane Boundary",
      speed: "62 km/h",
      lane: "Lane 1",
      mva: "Section 177A, MVA 2019",
      penalty: "₹ 1,000"
    }
  ];

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
                    Supports CCTV video exports: .mp4, .avi, .mov, .mkv (Up to 500 MB)
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
                      <span className="text-[10px] font-mono bg-emerald-950/60 text-emerald-400 px-1.5 py-0.2 rounded border border-emerald-500/30">
                        Ready to Ingest
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5">
                      Size: {(uploadedFile.size / (1024 * 1024)).toFixed(1)} MB • Type: {uploadedFile.type || "Video Stream"}
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
            disabled={isProcessing}
            className={`px-5 py-2 rounded-lg text-xs font-semibold transition-all flex items-center justify-center gap-2 ${
              isProcessing
                ? "bg-slate-700 text-slate-400 cursor-not-allowed"
                : "bg-gov-blue hover:bg-gov-hover text-white shadow"
            }`}
          >
            <span>{isProcessing ? "Processing Video Pipeline..." : `Run Forensic Analysis on ${currentVideoName}`}</span>
          </button>
        </div>

        {/* Linear Progress Bar */}
        {isProcessing && (
          <div className="space-y-1.5 pt-1">
            <div className="flex justify-between text-xs text-slate-400">
              <span>Executing Ground Homography & ByteTrack Detection...</span>
              <span className="font-mono">{progress}%</span>
            </div>
            <div className="w-full h-1.5 bg-dark-bg rounded-full overflow-hidden">
              <div 
                className="h-full bg-gov-blue transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

      </div>

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

        {/* Video Canvases */}
        <div className={`grid gap-3 ${viewMode === "side_by_side" ? "grid-cols-1 md:grid-cols-2" : "grid-cols-1"}`}>
          
          {(viewMode === "raw" || viewMode === "side_by_side") && (
            <div className="relative aspect-video bg-black rounded-lg overflow-hidden border border-dark-border flex flex-col">
              <span className="absolute top-2 left-2 z-10 text-[10px] font-mono px-2 py-0.5 rounded bg-black/60 text-slate-300">
                RAW UNPROCESSED INPUT
              </span>
              <video
                key={rawVideoSource}
                src={rawVideoSource}
                controls
                loop
                muted
                playsInline
                className="w-full h-full object-contain"
              />
            </div>
          )}

          {(viewMode === "annotated" || viewMode === "side_by_side") && (
            <div className="relative aspect-video bg-black rounded-lg overflow-hidden border border-dark-border flex flex-col">
              <span className="absolute top-2 left-2 z-10 text-[10px] font-mono px-2 py-0.5 rounded bg-black/60 text-emerald-400 font-semibold">
                AI ANNOTATED EVIDENCE + BEV RADAR
              </span>
              <video
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
            <a
              href={annotatedVideoSource}
              download
              className="px-2.5 py-1 rounded bg-dark-bg border border-dark-border hover:bg-dark-card text-xs text-slate-300 flex items-center gap-1.5 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download Evidence Video</span>
            </a>
          </div>
        </div>

        {/* Minimal Clean Table */}
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
                        severity: "WARNING",
                        message: v.infraction,
                        timestamp: 10.0,
                        speed_kmh: parseFloat(v.speed),
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
