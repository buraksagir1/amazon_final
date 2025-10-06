import { useEffect, useState, useRef } from "react";
import { useNavigate, useParams } from "react-router";
import useAuthUser from "../hooks/useAuthUser";
import { useQuery } from "@tanstack/react-query";
import { getStreamToken } from "../lib/api";
import CustomLayout from "../components/CustomLayout";
import {
  StreamVideo,
  StreamVideoClient,
  StreamCall,
  CallControls,
  SpeakerLayout,
  StreamTheme,
  CallingState,
  useCallStateHooks,
  useCall,
} from "@stream-io/video-react-sdk";

import "@stream-io/video-react-sdk/dist/css/styles.css";
import toast from "react-hot-toast";
import PageLoader from "../components/PageLoader";

const STREAM_API_KEY = import.meta.env.VITE_STREAM_API_KEY;

const SubtitleOverlay = ({ subtitle }) => (
  <div
    style={{
      position: "absolute",
      bottom: 32,
      left: 0,
      right: 0,
      textAlign: "center",
      zIndex: 20,
      pointerEvents: "none",
    }}
    className="w-full flex justify-center"
  >
    <span
      className="bg-black bg-opacity-70 text-white px-4 py-2 rounded text-lg shadow-lg max-w-2xl inline-block"
      style={{ maxWidth: "80vw", overflowWrap: "break-word" }}
    >
      {subtitle}
    </span>
  </div>
);

const ToggleTranscriptionButton = ({
  onToggle,
  isEnabled,
  speechLanguage,
  onLanguageChange,
}) => {
  const call = useCall();
  const { useCallSettings, useIsCallTranscribingInProgress } =
    useCallStateHooks();
  const { transcription } = useCallSettings() || {};
  if (transcription?.mode === "disabled") return null;
  const isTranscribing = useIsCallTranscribingInProgress();

  return (
    <div className="absolute top-4 right-4 z-30 flex gap-2">
      <select
        value={speechLanguage}
        onChange={(e) => onLanguageChange(e.target.value)}
        className="px-3 py-2 rounded bg-white text-gray-900 border shadow text-sm"
        disabled={isTranscribing || isEnabled}
      >
        <option value="en-US">English</option>
        <option value="tr-TR">Turkish</option>
      </select>
      <button
        onClick={() => {
          if (isTranscribing || isEnabled) {
            call
              ?.stopTranscription()
              .catch((err) => console.log("Failed to stop transcriptions", err));
            onToggle(false);
          } else {
            call
              ?.startTranscription()
              .catch((err) => console.error("Failed to start transcription", err));
            onToggle(true);
          }
        }}
        className={`px-4 py-2 rounded font-semibold shadow transition-colors ${isTranscribing || isEnabled ? "bg-red-600 text-white" : "bg-green-600 text-white"
          }`}
        style={{ minWidth: 160 }}
      >
        {isTranscribing || isEnabled ? "Disable Subtitles" : "Enable Subtitles"}
      </button>
    </div>
  );
};

const RTCapture = ({ onText }) => {
  const { useCameraState } = useCallStateHooks();
  const { mediaStream } = useCameraState();

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const loopRef = useRef(null);

  const clearTimerRef = useRef(null);
  const LAST_HOLD_MS = 1500;
  useEffect(() => {
    if (!mediaStream) return;

    const v = videoRef.current;
    if (v && v.srcObject !== mediaStream) {
      v.srcObject = mediaStream;
      v.play().catch(() => { });
    }

    const wsUrl = import.meta.env.VITE_RT_WS_URL || "ws://127.0.0.1:8008/ws";
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");

      const SEND_W = 320;
      const SEND_H = 180;

      const sendFrame = () => {
        if (!v || v.readyState < 2 || ws.readyState !== 1) return;
        canvas.width = SEND_W;
        canvas.height = SEND_H;
        ctx.drawImage(v, 0, 0, SEND_W, SEND_H);
        canvas.toBlob(
          (blob) => {
            if (blob && ws.readyState === 1) {
              blob.arrayBuffer().then((ab) => ws.send(new Uint8Array(ab)));
            }
          },
          "image/jpeg",
          0.65
        );
      };

      loopRef.current = setInterval(sendFrame, 65);
    };

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data?.ok) {
          if (Array.isArray(data.topk) && data.topk.length > 0) {
            const first = data.topk[0];
            const name = Array.isArray(first) ? first[0] : first?.name ?? "";
            const prob = Array.isArray(first) ? first[1] : first?.prob ?? 0;
            const text = name ? `${name}  ${(prob * 100).toFixed(1)}%` : "";
            if (text) {
              if (clearTimerRef.current) {
                clearTimeout(clearTimerRef.current);
                clearTimerRef.current = null;
              }
              onText?.(text);
            } else {
              if (clearTimerRef.current) clearTimerRef.current = clearTimerRef.current; // no-op
              else {
                clearTimerRef.current = setTimeout(() => {
                  onText?.("");
                  clearTimerRef.current = null;
                }, LAST_HOLD_MS);
              }
            }
          } else {

            if (!clearTimerRef.current) {
              clearTimerRef.current = setTimeout(() => {
                onText?.("");
                clearTimerRef.current = null;
              }, LAST_HOLD_MS);
            }
          }
        }
      } catch {
        /* ignore parse errors */
      }
    };

    const cleanup = () => {
      if (loopRef.current) {
        clearInterval(loopRef.current);
        loopRef.current = null;
      }
      if (clearTimerRef.current) {
        clearTimeout(clearTimerRef.current);
        clearTimerRef.current = null;
      }
      try {
        ws.close();
      } catch { }
      wsRef.current = null;
    };

    ws.onerror = cleanup;
    ws.onclose = cleanup;
    return cleanup;
  }, [mediaStream, onText]);

  return (
    <>
      {/* gizli yakalama elemanları */}
      <video ref={videoRef} muted playsInline style={{ display: "none" }} />
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </>
  );
};

/* ---------------- Ana sayfa ---------------- */

const CallPage = () => {
  const { id: callId } = useParams();
  const [client, setClient] = useState(null);
  const [call, setCall] = useState(null);
  const [isConnecting, setIsConnecting] = useState(true);
  const [transcriptionText, setTranscriptionText] = useState("");
  const [webSpeechRecognition, setWebSpeechRecognition] = useState(null);
  const [subtitlesEnabled, setSubtitlesEnabled] = useState(true);
  const webSpeechInitialized = useRef(false);
  const isRecognitionActive = useRef(false);
  const [callReady, setCallReady] = useState(false);
  const restartTimerRef = useRef(null);
  const lastSpeechErrorRef = useRef(null);
  const [micReady, setMicReady] = useState(false);
  const [hasMicDevice, setHasMicDevice] = useState(true);
  const [speechLanguage, setSpeechLanguage] = useState("en-US");

  const { authUser, isLoading } = useAuthUser();

  const { data: tokenData } = useQuery({
    queryKey: ["streamToken"],
    queryFn: getStreamToken,
    enabled: !!authUser,
  });

  // mic permission helper
  const ensureMicPermission = async () => {
    try {
      if (!navigator.mediaDevices?.getUserMedia) return false;
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((t) => t.stop());
      setMicReady(true);
      return true;
    } catch {
      setMicReady(false);
      toast.error("Please allow microphone access in your browser");
      return false;
    }
  };

  // Stream video init / join
  useEffect(() => {
    const initCall = async () => {
      if (!tokenData?.token || !authUser || !callId) return;
      try {
        const user = {
          id: authUser._id,
          name: authUser.fullName,
          image: authUser.profilePic,
        };

        const videoClient = new StreamVideoClient({
          apiKey: STREAM_API_KEY,
          user,
          token: tokenData.token,
          options: {
            enable_insights: true,
            enable_transcription: true,
            transcription: { mode: "available", language: "en" },
          },
        });

        const callInstance = videoClient.call("default", callId);

        // mic izni non-blocking
        try {
          if (navigator.mediaDevices?.getUserMedia) {
            const p = navigator.mediaDevices
              .getUserMedia({ audio: true })
              .then((s) => {
                try {
                  s.getTracks().forEach((t) => t.stop());
                } catch { }
                setMicReady(true);
                return true;
              })
              .catch(() => false);
            await Promise.race([p, new Promise((r) => setTimeout(() => r(false), 1500))]);
          }
        } catch { }

        // join (varsa katıl, yoksa oluştur)
        try {
          await callInstance.join({
            create: false,
            data: {
              transcription: {
                mode: "available",
                enabled: true,
                language: "en",
                auto_start: true,
              },
            },
          });
        } catch {
          await callInstance.join({
            create: true,
            data: {
              transcription: {
                mode: "available",
                enabled: true,
                language: "en",
                auto_start: true,
              },
            },
          });
        }

        // transcription’ı başlatmayı dene
        try {
          await callInstance.startTranscription();
        } catch { }

        setClient(videoClient);
        setCall(callInstance);
        setCallReady(true);

        // transcription event’leri
        callInstance.on("transcription.updated", (ev) => {
          if (ev?.text) setTranscriptionText(ev.text);
        });
        callInstance.on("transcription.started", () => {
          setTranscriptionText("🎤 Listening... Speak now!");
        });
        callInstance.on("transcription.stopped", () => setTranscriptionText(""));
      } catch (err) {
        console.error("Error joining call:", err);
        toast.error("Could not join the call. Please try again.");
      } finally {
        setIsConnecting(false);
      }
    };

    initCall();

    return () => {
      if (webSpeechRecognition) {
        try {
          webSpeechRecognition.stop();
        } catch { }
      }
    };
  }, [tokenData, authUser, callId]);

  // device check
  useEffect(() => {
    const checkDevices = async () => {
      try {
        const devices = await navigator.mediaDevices?.enumerateDevices?.();
        const hasAudioInput =
          Array.isArray(devices) && devices.some((d) => d.kind === "audioinput");
        setHasMicDevice(hasAudioInput);
      } catch { }
    };
    checkDevices();
    const onDeviceChange = () => checkDevices();
    if (navigator.mediaDevices?.addEventListener) {
      navigator.mediaDevices.addEventListener("devicechange", onDeviceChange);
      return () =>
        navigator.mediaDevices.removeEventListener("devicechange", onDeviceChange);
    }
  }, []);

  // Web Speech init
  useEffect(() => {
    if (
      callReady &&
      !webSpeechInitialized.current &&
      !webSpeechRecognition &&
      ("webkitSpeechRecognition" in window || "SpeechRecognition" in window)
    ) {
      webSpeechInitialized.current = true;
      try {
        const SpeechRecognition =
          window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();

        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = speechLanguage;

        recognition.onstart = () => {
          setTranscriptionText("🎤 Web Speech API listening...");
          isRecognitionActive.current = true;
        };

        recognition.onresult = (event) => {
          let finalTranscript = "";
          let interimTranscript = "";
          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) finalTranscript += transcript;
            else interimTranscript += transcript;
          }
          if (finalTranscript) setTranscriptionText(finalTranscript);
          else if (interimTranscript) setTranscriptionText(interimTranscript);
        };

        recognition.onerror = (event) => {
          lastSpeechErrorRef.current = event.error;
          if (
            (event.error === "no-speech" || event.error === "audio-capture") &&
            subtitlesEnabled &&
            callReady &&
            document.visibilityState === "visible"
          ) {
            try {
              recognition.stop();
            } catch { }
            clearTimeout(restartTimerRef.current);
            restartTimerRef.current = setTimeout(() => {
              try {
                if (!isRecognitionActive.current) recognition.start();
              } catch { }
            }, 1500);
          }
        };

        recognition.onend = () => {
          isRecognitionActive.current = false;
          if (
            subtitlesEnabled &&
            callReady &&
            document.visibilityState === "visible" &&
            lastSpeechErrorRef.current !== "aborted"
          ) {
            clearTimeout(restartTimerRef.current);
            restartTimerRef.current = setTimeout(() => {
              try {
                if (!isRecognitionActive.current) recognition.start();
              } catch { }
            }, 1200);
          }
        };

        setWebSpeechRecognition(recognition);
      } catch (e) {
        console.log("Error setting up Web Speech API:", e);
      }
    }
  }, [webSpeechRecognition, subtitlesEnabled, callReady, speechLanguage]);

  // speech dil değişimi
  useEffect(() => {
    if (webSpeechRecognition && speechLanguage) {
      webSpeechRecognition.lang = speechLanguage;
      if (isRecognitionActive.current && subtitlesEnabled && callReady) {
        try {
          webSpeechRecognition.stop();
          setTimeout(() => {
            try {
              if (!isRecognitionActive.current) webSpeechRecognition.start();
            } catch { }
          }, 500);
        } catch { }
      }
    }
  }, [speechLanguage, webSpeechRecognition, subtitlesEnabled, callReady]);

  // speech başlat/durdur
  useEffect(() => {
    if (!webSpeechRecognition) return;
    if (subtitlesEnabled && callReady && micReady) {
      if (!hasMicDevice) {
        toast.error("No microphone found. Connect a mic to use subtitles.");
        return;
      }
      ensureMicPermission().then((ok) => {
        if (!ok) return;
        try {
          if (!isRecognitionActive.current) webSpeechRecognition.start();
        } catch (e) { }
      });
    } else {
      try {
        if (isRecognitionActive.current) webSpeechRecognition.stop();
      } catch (e) { }
    }
  }, [subtitlesEnabled, webSpeechRecognition, callReady, hasMicDevice, micReady]);

  // tab görünürlüğü
  useEffect(() => {
    const onVisibility = () => {
      if (!webSpeechRecognition) return;
      if (document.visibilityState === "hidden") {
        try {
          if (isRecognitionActive.current) webSpeechRecognition.stop();
        } catch { }
      } else if (document.visibilityState === "visible") {
        if (subtitlesEnabled && callReady && micReady) {
          try {
            if (!isRecognitionActive.current) webSpeechRecognition.start();
          } catch { }
        }
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, [webSpeechRecognition, subtitlesEnabled, callReady, micReady]);

  // altyazı kapandığında temizle
  useEffect(() => {
    if (!subtitlesEnabled) setTranscriptionText("");
  }, [subtitlesEnabled]);

  if (isLoading || isConnecting) return <PageLoader />;

  return (
    <div className="h-screen flex flex-col items-center justify-center">
      <div className="relative">
        {client && call ? (
          <StreamVideo client={client}>
            <StreamCall call={call}>
              <CallContent
                transcriptionText={transcriptionText}
                subtitlesEnabled={subtitlesEnabled}
                onToggleSubtitles={setSubtitlesEnabled}
                speechLanguage={speechLanguage}
                onLanguageChange={setSpeechLanguage}
              />
            </StreamCall>
          </StreamVideo>
        ) : (
          <div className="flex items-center justify-center h-full">
            <p>Could not initialize call. Please refresh or try again later.</p>
          </div>
        )}
      </div>
    </div>
  );
};

/* --------- StreamCall içi UI + RT overlay --------- */
const CallContent = ({
  transcriptionText,
  subtitlesEnabled,
  onToggleSubtitles,
  speechLanguage,
  onLanguageChange,
}) => {
  const { useCallCallingState, useCallSettings } = useCallStateHooks();
  const callingState = useCallCallingState();
  const callSettings = useCallSettings();
  const navigate = useNavigate();

  console.log("Call settings:", callSettings);
  console.log("Transcription text:", transcriptionText);
  console.log("Subtitles enabled:", subtitlesEnabled);

  if (callingState === CallingState.LEFT) return navigate("/");

  return (
    <StreamTheme>
      <div className="relative w-full h-full">





        {/* RT model çıktısı overlay */}

        <div
          style={{
            position: "absolute",
            top: 16,
            left: 0,
            right: 0,
            zIndex: 30,
            textAlign: "center",
            pointerEvents: "none",
          }}
          className="w-full flex justify-center"
        >
          <span
            className="bg-black bg-opacity-70 text-white px-4 py-2 rounded text-base md:text-lg shadow-lg inline-block"
            style={{ maxWidth: "80vw", overflowWrap: "break-word" }}
          >
          </span>
        </div>
        <CustomLayout />
        <CallControls />
        {/* Senin mevcut altyazı overlay'in */}
        <ToggleTranscriptionButton
          onToggle={onToggleSubtitles}
          isEnabled={subtitlesEnabled}
          speechLanguage={speechLanguage}
          onLanguageChange={onLanguageChange}
        />
        {subtitlesEnabled && (
          <SubtitleOverlay
            subtitle={
              transcriptionText ||
              "🎤 Web Speech API ready - speak to see live subtitles..."
            }
          />
        )}
      </div>
    </StreamTheme>
  );
};

export default CallPage;
