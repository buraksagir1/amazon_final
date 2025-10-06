# server.py
from PIL import Image
import io
import numpy as np
import cv2

import os, io, base64
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

import numpy as np

from rt_core import RTInferEngine

CKPT_PATH = os.getenv("RT_CKPT", "checkpoints/model.pt")
LABEL_MAP = os.getenv("RT_LABEL_MAP", "Data/ProcessedDatasetLSA/label_map.json")
DEVICE = os.getenv("RT_DEVICE", "cpu")
STRIDE = int(os.getenv("RT_STRIDE", "8"))
EMA = float(os.getenv("RT_EMA", "0.6"))
THRESH = float(os.getenv("RT_THRESH", "0.20"))
TOPK = int(os.getenv("RT_TOPK", "3"))
MAXLEN = int(os.getenv("RT_MAXLEN", "90"))
DEBUG = os.getenv("RT_DEBUG", "0") == "1"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

engine = None

@app.on_event("startup")
def startup():
    global engine
    engine = RTInferEngine(
        ckpt=CKPT_PATH,
        label_map_path=LABEL_MAP,
        device=DEVICE,
        max_len=MAXLEN,
        stride=STRIDE,
        ema=EMA,
        prob_thresh=THRESH,
        topk=TOPK,
        debug=DEBUG,
    )
    print("[RT] Engine ready")

@app.get("/health")
def health():
    return {"ok": True}
# server.py (yalnızca ws_endpoint'i değiştir)
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive()  # hem text hem bytes gelebilir
            t = msg.get("type")

            if t == "websocket.receive":
                # 1) BYTES: kamera frame'i (JPEG)
                if msg.get("bytes") is not None:
                    bytes_data = msg["bytes"]  # << BURASI ÖNEMLİ

                    # PIL -> numpy -> BGR (cv2.imdecode kullanmıyoruz)
                    try:
                        img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
                        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print("[WS ERROR] decode failed:", repr(e))
                        await ws.send_json({"ok": False, "error": "decode-failed"})
                        continue

                    # inference
                    resp = engine.push_frame_and_predict(frame)
                    await ws.send_json(resp)

                # 2) TEXT: ping/hello v.b.
                elif msg.get("text") is not None:
                    await ws.send_json({"ok": True, "topk": [], "msg": "text-ignored"})

            elif t in ("websocket.disconnect", "websocket.close"):
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback
        print("[WS ERROR]", e)
        print(traceback.format_exc())
        try:
            await ws.close()
        except:
            pass