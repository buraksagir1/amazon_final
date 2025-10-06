import { useCallback, useMemo, useState } from "react";
import {
  ParticipantView,
  useParticipantViewContext,
} from "@stream-io/video-react-sdk";
import PerParticipantProcessor from "./PerParticipantProcessor";

// Tile üstü overlay UI
function OverlayUI({ text }) {
  return (
    <div
      style={{
        position: "absolute",
        top: 12,
        left: 0,
        right: 0,
        textAlign: "center",
        pointerEvents: "none",
      }}
      className="w-full flex justify-center"
    >
      <span className="bg-black bg-opacity-70 text-white px-3 py-1 rounded text-sm md:text-base shadow">
        {text || "Detecting..."}
      </span>
    </div>
  );
}

export default function ParticipantTile({ participant, text, onText }) {
  // Bu tile'a ait video elementini tutuyoruz
  const [videoEl, setVideoEl] = useState(null);

  // ref callback STABİL: dependency boş; içinde identity check var
  const handleSetVideoEl = useCallback((el) => {
    setVideoEl((prev) => (prev === el ? prev : el));
  }, []);

  // refs objesini de STABİL tut
  const refs = useMemo(
    () => ({ setVideoElement: handleSetVideoEl }),
    [handleSetVideoEl]
  );

  return (
    <div className="relative">
      <ParticipantView
        participant={participant}
        className="w-full h-full"
        ParticipantViewUI={<OverlayUI text={text} />}
        refs={refs}
      />
      {/* gizli işlemci: videoEl değişince attach/detach */}
      <PerParticipantProcessor
        participantId={participant.sessionId}
        videoEl={videoEl}
        onText={onText}
        drawOverlay={false}
      />
    </div>
  );
}
