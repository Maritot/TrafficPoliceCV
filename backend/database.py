"""
Project WAV-DET - Persistent Incident & Active Learning SQLite Database.
Stores detected incidents, evidence frames, severity ratings, calibrated justifications,
and active human feedback (Confirmed True Positive vs. False Alarm) for model retraining.
"""

import os
import sqlite3
import time
from typing import List, Dict, Optional, Any

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "events.db")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite schema for traffic events and user feedback."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            reason TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp_s REAL NOT NULL,
            speed_kmh REAL NOT NULL,
            lane_id INTEGER DEFAULT 1,
            evidence_frame_url TEXT,
            user_feedback TEXT DEFAULT 'unreviewed',
            notes TEXT DEFAULT '',
            created_at REAL NOT NULL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_video ON events(video_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_feedback ON events(user_feedback)")
    conn.commit()
    conn.close()


def insert_event(event_data: Dict[str, Any]) -> str:
    """Inserts a new incident event or updates if exists."""
    conn = get_db_connection()
    cursor = conn.cursor()
    event_id = event_data.get("id") or f"INC-{int(time.time()*1000)}"
    cursor.execute("""
        INSERT OR REPLACE INTO events (
            id, video_id, track_id, class_name, event_type, severity,
            reason, confidence, timestamp_s, speed_kmh, lane_id,
            evidence_frame_url, user_feedback, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event_id,
        event_data.get("video_id", "live_stream"),
        event_data.get("track_id", 0),
        event_data.get("class_name", "Vehicle"),
        event_data.get("event_type", "WAVERING_VEHICLE"),
        event_data.get("severity", "MODERATE"),
        event_data.get("reason", "Trajectory anomaly"),
        float(event_data.get("confidence", 0.90)),
        float(event_data.get("timestamp_s", 0.0)),
        float(event_data.get("speed_kmh", 0.0)),
        int(event_data.get("lane_id", 1)),
        event_data.get("evidence_frame_url", ""),
        event_data.get("user_feedback", "unreviewed"),
        event_data.get("notes", ""),
        time.time()
    ))
    conn.commit()
    conn.close()
    return event_id


def list_events(severity: Optional[str] = None, 
                feedback: Optional[str] = None, 
                video_id: Optional[str] = None,
                limit: int = 50, 
                offset: int = 0) -> List[Dict[str, Any]]:
    """Retrieves paginated events with optional filters."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM events WHERE 1=1"
    params = []
    
    if severity and severity.upper() != "ALL":
        query += " AND severity = ?"
        params.append(severity.upper())
        
    if feedback and feedback.lower() != "all":
        query += " AND user_feedback = ?"
        params.append(feedback.lower())
        
    if video_id:
        query += " AND video_id = ?"
        params.append(video_id)
        
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results


def update_event_feedback(event_id: str, feedback: str, notes: str = "") -> bool:
    """Updates user feedback ('confirmed_tp', 'false_alarm', 'unreviewed')."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE events 
        SET user_feedback = ?, notes = ?
        WHERE id = ?
    """, (feedback, notes, event_id))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def get_event_stats() -> Dict[str, Any]:
    """Computes comprehensive incident and active learning statistics."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM events")
    total_events = cursor.fetchone()[0]
    
    cursor.execute("SELECT severity, COUNT(*) FROM events GROUP BY severity")
    sev_counts = dict(cursor.fetchall())
    
    cursor.execute("SELECT user_feedback, COUNT(*) FROM events GROUP BY user_feedback")
    fb_counts = dict(cursor.fetchall())
    
    conn.close()
    
    confirmed = fb_counts.get("confirmed_tp", 0)
    false_alarms = fb_counts.get("false_alarm", 0)
    reviewed_total = confirmed + false_alarms
    far_pct = round((false_alarms / reviewed_total * 100.0), 1) if reviewed_total > 0 else 0.0
    
    return {
        "total_events": total_events,
        "severity_breakdown": {
            "SEVERE": sev_counts.get("SEVERE", 0) + sev_counts.get("CRITICAL", 0),
            "MODERATE": sev_counts.get("MODERATE", 0) + sev_counts.get("WARNING", 0),
            "INFO": sev_counts.get("INFO", 0)
        },
        "feedback_breakdown": {
            "unreviewed": fb_counts.get("unreviewed", 0),
            "confirmed_tp": confirmed,
            "false_alarm": false_alarms
        },
        "false_alarm_rate_pct": far_pct,
        "active_learning_samples": false_alarms
    }


# Auto-initialize table on import
init_db()
