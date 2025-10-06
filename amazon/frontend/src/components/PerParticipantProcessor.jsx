// src/components/PerParticipantProcessor.jsx
import { useEffect, useRef } from "react";
import { RtProcessor } from "../lib/rtProcessor";

export default function PerParticipantProcessor({ participantId, videoEl, onText, drawOverlay = false }) {
  const procRef = useRef(null);
  const holdTimerRef = useRef(null);
  const HOLD_MS = 1500; // boş sonuçta sabit tutma süresi

  useEffect(() => {
    if (!videoEl) return;

    const wsUrl = import.meta.env.VITE_RT_WS_URL || "ws://127.0.0.1:8008/ws";
    const p = new RtProcessor(wsUrl, {
      drawOverlay,
      emitText: (txt) => {
        if (txt) {
          if (holdTimerRef.current) {
            clearTimeout(holdTimerRef.current);
            holdTimerRef.current = null;
          }
          onText?.(participantId, txt);
        } else {
          if (!holdTimerRef.current) {
            holdTimerRef.current = setTimeout(() => {
              onText?.(participantId, "");
              holdTimerRef.current = null;
            }, HOLD_MS);
          }
        }
      },
    });
    procRef.current = p;

    p.attach(videoEl);

    return () => {
      try { if (holdTimerRef.current) clearTimeout(holdTimerRef.current); } catch {}
      holdTimerRef.current = null;
      try { p.dispose(); } catch {}
      procRef.current = null;
    };
  }, [videoEl, participantId, onText, drawOverlay]);

  return null;
}
