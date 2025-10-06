import { useEffect, useRef } from "react";

type Props = {
  wsUrl: string;           // ör: ws://localhost:8001/ws/rt
  onText?: (s: string) => void;
  sendEveryMs?: number;     // 250ms öneri
};

export default function RTFromRemote({ wsUrl, onText, sendEveryMs = 250 }: Props) {
  const wsRef = useRef<WebSocket | null>(null);
  const loopRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      const canvas = document.createElement("canvas");
      canvasRef.current = canvas;
      const ctx = canvas.getContext("2d");

      const pickRemoteVideo = (): HTMLVideoElement | null => {
        const vids = Array.from(document.querySelectorAll("video")) as HTMLVideoElement[];

        // görünür videolar
        const visible = vids.filter((v) => {
          const r = v.getBoundingClientRect();
          return r.width > 0 && r.height > 0 && !v.hidden && getComputedStyle(v).display !== "none";
        });

        // local genelde muted olur → remote adaylar
        const remoteCandidates = visible.filter((v) => !v.muted);

        // en büyük videoyu seç (genelde aktif konuşan/odakta)
        const pick = (arr: HTMLVideoElement[]) =>
          arr.sort((a, b) => (b.videoWidth * b.videoHeight) - (a.videoWidth * a.videoHeight))[0] || null;

        return pick(remoteCandidates) || pick(visible) || null;
      };

      const sendFrame = () => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        const v = pickRemoteVideo();
        if (!v || v.readyState < 2 || !ctx) return;

        const W = 320;
        const H = Math.max(1, Math.round((v.videoHeight / Math.max(1, v.videoWidth)) * W));
        canvas.width = W; canvas.height = H;
        ctx.drawImage(v, 0, 0, W, H);
        canvas.toBlob((blob) => {
          if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
            blob.arrayBuffer().then((buf) => wsRef.current?.send(new Uint8Array(buf)));
          }
        }, "image/jpeg", 0.7);
      };

      const tick = () => {
        sendFrame();
        loopRef.current = window.setTimeout(tick, sendEveryMs) as any;
      };
      tick();
    };

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (Array.isArray(data?.topk) && data.topk.length > 0) {
          const [label, p] = data.topk[0];
          onText?.(`${label} ${(p * 100).toFixed(1)}%`);
        } else {
          // düşük güven → istersen temizle
          // onText?.("");
        }
      } catch {}
    };

    const cleanup = () => {
      if (loopRef.current) { clearTimeout(loopRef.current); loopRef.current = null; }
      try { ws.close(); } catch {}
      wsRef.current = null; canvasRef.current = null;
    };

    ws.onerror = cleanup;
    ws.onclose = cleanup;
    return cleanup;
  }, [wsUrl, onText, sendEveryMs]);

  return null; // görünmeyen yardımcı
}
