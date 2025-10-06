# rt_core.py
import json, time, collections, traceback
import numpy as np
import torch
import cv2

from models import load_from_checkpoint
import os, sys, importlib.util

try:
    # PreProcess/preprocess_videos.py içine eklediğin frame_to_v4
    from PreProcess.preprocess_videos import frame_to_v4  # (frame_bgr, normalise=True, V_target=None) -> (V,4)
    HAS_FRAME_TO_V4 = True
except Exception:
    HAS_FRAME_TO_V4 = False






HOLD_SEC_DEFAULT = 0.35         # 0.7 → 0.35: overlay daha çabuk güncellensin
EMA_ALPHA_DEFAULT = 0.5         # 0.6 → 0.5: biraz daha hızlı tepki
PROB_THRESH_DEFAULT = 0.25      # 0.20 → 0.25: yanlış pozitifleri azalt
STRIDE_FRAMES_DEFAULT = 4       # 8 → 4: daha sık tahmin
MIN_INFER_FRAMES_DEFAULT = 28   # YENİ: pencere dolmadan bu kadar kare varsa tahmin
RESIZE_SHORT_SIDE = 320  

def infer_F_expected(model):
    conv1d = None
    linear = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d) and conv1d is None:
            conv1d = m.in_channels
        if isinstance(m, torch.nn.Linear) and linear is None:
            linear = m.in_features
    if conv1d is not None:
        return ("conv1d", conv1d)
    if linear is not None:
        return ("linear", linear)
    return (None, None)

def dummy_frame_to_v4(frame, normalise=True, V=None):
    if V is None: V = 21
    h, w = frame.shape[:2]
    xs = np.linspace(0.2, 0.8, V) * w
    ys = np.linspace(0.2, 0.8, V) * h
    v4 = np.stack([xs, ys, np.zeros(V), np.ones(V)], axis=1).astype(np.float32)
    noise = np.random.normal(0, 0.5, size=v4.shape).astype(np.float32)
    v4[:, :2] += noise[:, :2]
    if normalise:
        v4[:, 0] = np.clip(v4[:, 0] / max(1, w), 0, 1)
        v4[:, 1] = np.clip(v4[:, 1] / max(1, h), 0, 1)
    return v4

def to_features(arr_tv4: np.ndarray) -> np.ndarray:
    T, V, C = arr_tv4.shape
    assert C == 4, f"expected 4 channels, got {C}"
    return arr_tv4.reshape(T, V * C).astype(np.float32)

def _maybe_take_logits(y):
    if isinstance(y, dict):
        for k in ("logits", "preds", "y", "out"):
            if k in y and torch.is_tensor(y[k]): return y[k]
    if isinstance(y, (list, tuple)) and len(y) > 0 and torch.is_tensor(y[0]): return y[0]
    return y

def _build_attn_2d_from_pad(mask_bool):
    b, T = mask_bool.shape
    valid = (~mask_bool).to(torch.float32)
    valid_any = (valid.sum(dim=0) > 0).to(torch.float32)
    m2d = valid_any[None, :] * valid_any[:, None]
    attn2d = torch.where(m2d > 0, torch.zeros_like(m2d), torch.full_like(m2d, float("-inf")))
    return attn2d

def smart_forward(model, x_btf, pad_mask_bt, prefer_conv=False, debug=False):
    candidates = []
    x_tbf = x_btf.transpose(0, 1).contiguous()  # (T,B,F)
    x_bft = x_btf.transpose(1, 2).contiguous()  # (B,F,T)
    if prefer_conv:
        candidates = [("x_bft", x_bft), ("x_btf", x_btf), ("x_tbf", x_tbf)]
    else:
        candidates = [("x_btf", x_btf), ("x_tbf", x_tbf), ("x_bft", x_bft)]

    tried = []
    for name, x in candidates:
        try:
            y = model(x); y = _maybe_take_logits(y)
            if torch.is_tensor(y): return y
        except Exception as e:
            tried.append((f"forward({name})", repr(e)))

    for name, x in candidates:
        for kw in ("key_padding_mask", "src_key_padding_mask", "padding_mask", "mask"):
            try:
                y = model(x, **{kw: pad_mask_bt}); y = _maybe_take_logits(y)
                if torch.is_tensor(y): return y
            except Exception as e:
                tried.append((f"forward({name}, {kw}=B×T)", repr(e)))

    attn_valid = (~pad_mask_bt).to(torch.long)
    for name, x in candidates:
        try:
            y = model(x, attention_mask=attn_valid); y = _maybe_take_logits(y)
            if torch.is_tensor(y): return y
        except Exception as e:
            tried.append((f"forward({name}, attention_mask=valid)", repr(e)))

    attn2d = _build_attn_2d_from_pad(pad_mask_bt)
    for name, x in candidates:
        try:
            y = model(x, attn_mask=attn2d); y = _maybe_take_logits(y)
            if torch.is_tensor(y): return y
        except Exception as e:
            tried.append((f"forward({name}, attn_mask=T×T)", repr(e)))

    seqlen = int((~pad_mask_bt[0]).sum().item())
    lengths = torch.tensor([seqlen], dtype=torch.long, device=x_btf.device)
    for name, x in candidates:
        try:
            y = model(x, lengths=lengths); y = _maybe_take_logits(y)
            if torch.is_tensor(y): return y
        except Exception as e:
            tried.append((f"forward({name}, lengths)", repr(e)))

    msg = ["Model forward kombinasyonlarının hiçbiri çalışmadı. Denenenler:"]
    for desc, err in tried[:10]:
        msg.append(f"  - {desc}: {err}")
    raise RuntimeError("\n".join(msg))


class RTInferEngine:
    def __init__(
        self,
        ckpt: str,
        label_map_path: str,
        device: str = "cpu",
        max_len: int = 90,
        stride: int = STRIDE_FRAMES_DEFAULT,
        ema: float = EMA_ALPHA_DEFAULT,
        prob_thresh: float = PROB_THRESH_DEFAULT,
        topk: int = 3,
        debug: bool = False,
        min_infer_frames: int = MIN_INFER_FRAMES_DEFAULT,   # YENİ
        resize_short: int = RESIZE_SHORT_SIDE,   
    ):
        self.device = device
        self.model = load_from_checkpoint(ckpt, map_location=device).eval()

        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        # İki formatı da destekle
        if all(isinstance(v, int) for v in label_map.values()):
            self.inv_map = {v: k for k, v in label_map.items()}
        else:
            self.inv_map = {int(k): v for k, v in label_map.items()}

        proj_type, F_expected = infer_F_expected(self.model)
        self.prefer_conv = (proj_type == "conv1d")
        self.V_expected = (F_expected // 4) if (F_expected and F_expected % 4 == 0) else None
        self.debug = debug

        self.max_len = max_len
        self.stride = max(1, stride)
        self.ema_alpha = float(ema)
        self.prob_thresh = float(prob_thresh)
        self.topk = int(topk)
        self.min_infer_frames = int(min_infer_frames)
        self.resize_short = int(resize_short) if resize_short else 0

        self.buf = collections.deque(maxlen=self.max_len)
        self.frame_idx = 0
        self.ema_probs = None
        self.normalise = True

        self.last_lines = ["Starting..."]
        self.hold_until = time.time() + HOLD_SEC_DEFAULT

    def _frame_to_v4(self, frame_bgr):
        if self.resize_short and self.resize_short > 0:
            h, w = frame_bgr.shape[:2]
            short = min(h, w)
            if short > self.resize_short:
                scale = self.resize_short / float(short)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame_bgr = cv2.resize(
                    frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        v4 = None
        print(f"[RT] HAS_FRAME_TO_V4={HAS_FRAME_TO_V4}, prefer_conv={self.prefer_conv}, V_expected={self.V_expected}")

        if HAS_FRAME_TO_V4:
            try:
                v4 = frame_to_v4(frame_bgr, normalise=self.normalise, V_target=self.V_expected)
            except Exception:
                if self.debug:
                    print("[ERROR] frame_to_v4 exception:\n" + traceback.format_exc())
                v4 = None
        if v4 is None:
            v4 = dummy_frame_to_v4(frame_bgr, normalise=self.normalise, V=self.V_expected)
        if v4.dtype != np.float32:
            v4 = v4.astype(np.float32)
        return v4  # (V,4)

    def push_frame_and_predict(self, frame_bgr) -> dict:
        """
        WS’ten gelen JPEG'i BGR frame olarak ver; JSON döner:
        { ok: true, topk: [[label, prob], ...], msg: "..." }
        """
        self.frame_idx += 1
        v4 = self._frame_to_v4(frame_bgr)
        self.buf.append(v4)

        T = len(self.buf)
        have_min = (T >= self.min_infer_frames)

        new_lines = None
        if have_min and (self.frame_idx % self.stride == 0):
            # (T,V,4) -> (T,F)
            arr = np.stack(list(self.buf), axis=0)
            feat = to_features(arr)         # (T,F)
            F = feat.shape[1]

            # T < max_len ise pad + mask
            if T < self.max_len:
                pad = np.zeros((self.max_len - T, F), dtype=np.float32)
                feat_full = np.vstack([feat, pad])   # (max_len, F)
                mask = np.zeros(self.max_len, dtype=bool)
                mask[T:] = True                      # pad bölgeleri True = PAD
            else:
                # T >= max_len ise son max_len’i kullan
                feat_full = feat[-self.max_len:]
                mask = np.zeros(self.max_len, dtype=bool)

            x = torch.from_numpy(feat_full).unsqueeze(0)  # (1, max_len, F)
            m = torch.from_numpy(mask).unsqueeze(0)       # (1, max_len)

            with torch.inference_mode():
                logits = smart_forward(self.model, x, m, prefer_conv=self.prefer_conv, debug=self.debug)
                if logits.ndim == 2 and logits.shape[0] == 1:
                    logits = logits[0]
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

            # EMA
            if self.ema_probs is None or self.ema_probs.shape != probs.shape:
                self.ema_probs = probs
            else:
                a = float(self.ema_alpha)
                self.ema_probs = a * probs + (1.0 - a) * self.ema_probs

            use_probs = self.ema_probs
            k = min(self.topk, use_probs.shape[-1])
            idx = np.argsort(-use_probs)[:k]
            topk = [(self.inv_map[i], float(use_probs[i])) for i in idx]

            if topk and topk[0][1] >= self.prob_thresh:
                new_lines = [f"{name}: {p:.2f}" for name, p in topk]
            else:
                new_lines = ["Nothing detected (low conf)"]

            self.last_topk = [[self.inv_map[i], float(use_probs[i])] for i in idx]
        else:
            self.last_topk = None

        # hold / mesaj
        now = time.time()
        if new_lines is not None:
            self.last_lines = new_lines
            self.hold_until = now + HOLD_SEC_DEFAULT
        if (new_lines is None) and (now >= self.hold_until):
            if not have_min:
                self.last_lines = [f"Waiting frames: {T}/{self.min_infer_frames}"]
            else:
                self.last_lines = ["Nothing detected"]
            self.hold_until = now + HOLD_SEC_DEFAULT
            
        return {"ok": True, "topk": (self.last_topk or []), "msg": self.last_lines[0]}

