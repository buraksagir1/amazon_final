// src/components/CustomLayout.jsx
import { useState } from "react";
import { useCallStateHooks } from "@stream-io/video-react-sdk";
import ParticipantTile from "./ParticipantTile";

export default function CustomLayout() {
  const { useParticipants } = useCallStateHooks();
  const participants = useParticipants(); // local + remote

  const [textMap, setTextMap] = useState({});

  const setText = (pid, txt) =>
    setTextMap((m) => (m[pid] === txt ? m : { ...m, [pid]: txt }));

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-2 w-full h-full p-2">
      {participants.map((p) => (
        <ParticipantTile
          key={p.sessionId}
          participant={p}
          text={textMap[p.sessionId] || ""}
          onText={setText}
        />
      ))}
    </div>
  );
}
