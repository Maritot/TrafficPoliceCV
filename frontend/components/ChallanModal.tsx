"use client";

import React, { useState } from "react";
import { 
  X, 
  ShieldAlert, 
  CheckCircle2, 
  Printer, 
  Send
} from "lucide-react";

export interface IncidentAlert {
  id?: string;
  track_id: number;
  class_name: string;
  event_type: string;
  severity: "CRITICAL" | "WARNING" | string;
  message: string;
  timestamp: number;
  speed_kmh: number;
  lane_id?: number | string;
  location?: string;
}

interface ChallanModalProps {
  alert: IncidentAlert | null;
  onClose: () => void;
}

export default function ChallanModal({ alert, onClose }: ChallanModalProps) {
  const [transmitted, setTransmitted] = useState<boolean>(false);

  if (!alert) return null;

  let mvaSection = "Section 177, MVA 1988 (General Traffic Regulation)";
  let fineAmount = "₹ 500";
  let offenseTitle = "General Lane / Traffic Infraction";

  if (alert.event_type.includes("WEAVING") || alert.event_type.includes("OSCILLATION")) {
    mvaSection = "Section 184, MVA (Dangerous / Reckless Driving)";
    fineAmount = "₹ 5,000";
    offenseTitle = "Dangerous & Erratic Lane Weaving";
  } else if (alert.event_type.includes("WRONG_WAY")) {
    mvaSection = "Section 184(c), MVA (Driving against traffic flow)";
    fineAmount = "₹ 5,000";
    offenseTitle = "Wrong-Way Travel on Live Carriageway";
  } else if (alert.event_type.includes("STOPPED")) {
    mvaSection = "Section 122 r/w 177, MVA (Dangerous Vehicle Stoppage)";
    fineAmount = "₹ 1,500";
    offenseTitle = "Hazardous Stoppage in Travel Lane";
  } else if (alert.event_type.includes("LANE_DEPARTURE")) {
    mvaSection = "Section 177A, MVA (Lane Discipline Violation)";
    fineAmount = "₹ 1,000";
    offenseTitle = "Crossed Solid Lane Boundary Marking";
  } else if (alert.event_type.includes("ENCROACHMENT") || alert.event_type.includes("VENDOR")) {
    mvaSection = "Section 283, IPC & Section 201, MVA (Road Obstruction)";
    fineAmount = "₹ 2,000";
    offenseTitle = "Unauthorized Mobile Cart / Vendor Obstruction";
  } else if (alert.event_type.includes("ANIMAL") || alert.event_type.includes("CATTLE")) {
    mvaSection = "Section 289, IPC (Negligent conduct with respect to animal)";
    fineAmount = "₹ 1,000";
    offenseTitle = "Stray Cattle / Livestock Roadway Hazard";
  }

  const challanNumber = `CH-DL-2026-${Math.floor(100000 + Math.random() * 900000)}`;

  const handleTransmit = () => {
    setTransmitted(true);
    setTimeout(() => {
      setTimeout(onClose, 1400);
    }, 600);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm">
      <div className="relative w-full max-w-lg bg-dark-card border border-dark-border rounded-xl shadow-2xl overflow-hidden text-slate-100">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between px-5 py-3.5 bg-dark-surface border-b border-dark-border">
          <div>
            <span className="text-[10px] font-semibold text-slate-400 tracking-wider uppercase">
              MINISTRY OF ROAD TRANSPORT & HIGHWAYS (MoRTH)
            </span>
            <h3 className="text-sm font-bold text-white">
              Official Electronic Violation Notice (e-Challan)
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-dark-bg transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-5 space-y-3.5 text-xs">
          
          <div className="flex items-center justify-between p-2.5 rounded-lg bg-dark-bg border border-dark-border text-xs">
            <div>
              <span className="text-slate-400">Notice Number:</span>{" "}
              <strong className="text-white font-mono">{challanNumber}</strong>
            </div>
            <div>
              <span className="text-slate-400">Timestamp:</span>{" "}
              <strong className="text-white font-mono">{alert.timestamp.toFixed(2)}s</strong>
            </div>
          </div>

          <div className="p-3.5 rounded-lg bg-dark-surface border border-dark-border space-y-1">
            <div className="flex items-start justify-between">
              <div>
                <span className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider">
                  Violation Category
                </span>
                <h4 className="text-base font-bold text-white">
                  {offenseTitle}
                </h4>
                <p className="text-xs text-slate-300 mt-0.5">
                  {alert.message}
                </p>
              </div>
              <div className="text-right">
                <span className="text-[10px] text-slate-400 uppercase">Penalty</span>
                <div className="text-lg font-bold text-amber-400">
                  {fineAmount}
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2.5">
            <div className="p-2 rounded-lg bg-dark-bg border border-dark-border">
              <span className="text-slate-400 block text-[10px]">Target ID</span>
              <strong className="text-white font-mono text-xs">#{alert.track_id}</strong>
            </div>
            <div className="p-2 rounded-lg bg-dark-bg border border-dark-border">
              <span className="text-slate-400 block text-[10px]">Entity Class</span>
              <strong className="text-white text-xs">{alert.class_name}</strong>
            </div>
            <div className="p-2 rounded-lg bg-dark-bg border border-dark-border">
              <span className="text-slate-400 block text-[10px]">Measured Speed</span>
              <strong className="text-emerald-400 font-mono text-xs">{alert.speed_kmh.toFixed(0)} km/h</strong>
            </div>
          </div>

          <div className="p-2.5 rounded-lg bg-dark-bg border border-dark-border">
            <span className="text-slate-400 block text-[10px]">Statutory Legal Section:</span>
            <span className="text-slate-200 font-medium">{mvaSection}</span>
          </div>

          {transmitted && (
            <div className="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-500/40 text-emerald-300 text-xs flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>
                Transmitted to <strong>NIC Vahan & Sarathi Central Database</strong>.
              </span>
            </div>
          )}

        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-between px-5 py-3 bg-dark-surface border-t border-dark-border">
          <button
            onClick={() => window.print()}
            className="px-3 py-1.5 rounded-md bg-dark-bg border border-dark-border text-slate-300 hover:text-white text-xs font-medium transition-colors"
          >
            Print
          </button>

          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-3 py-1.5 rounded-md border border-dark-border text-slate-400 hover:text-white text-xs font-medium"
            >
              Cancel
            </button>
            <button
              onClick={handleTransmit}
              disabled={transmitted}
              className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-all ${
                transmitted 
                  ? "bg-emerald-600 text-white cursor-default"
                  : "bg-gov-blue hover:bg-gov-hover text-white"
              }`}
            >
              {transmitted ? "Transmitted ✓" : "Issue Notice"}
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
