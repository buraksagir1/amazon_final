export class RtProcessor {
  constructor(wsUrl, opts = {}) {
    this.wsUrl = wsUrl;
    this.emitText = typeof opts.emitText === "function" ? opts.emitText : null;
    this.drawOverlay = opts.drawOverlay !== false; // default: true
    this.srcVideo = null;
    this.canvas = null;
    this.ctx = null;
    this.stream = null;

    this.rafId = null;
    this.sendTimer = null;

    this.connected = false;
    this.lastTopk = null;
    this.lastStatus = "Initializing…";

    this.fps = 30;       // overlay output fps
    this.sendFps = 8;    // WS'e gönderim fps
    this.jpegQuality = 0.75;

    this._frameCounter = 0;  // canlılık göstergesi
  }

  getProcessedTrack() {
    if (!this.stream) return null;
    const [t] = this.stream.getVideoTracks();
    return t || null;
  }

  async attach(videoEl) {
    this.srcVideo = videoEl;

    // 1) Video hazır olana kadar bekle
    await this._waitForVideoReady(videoEl);

    // 2) Canvas kur
    this.canvas = document.createElement("canvas");
    this.canvas.width  = Math.max(1, videoEl.videoWidth  || 640);
    this.canvas.height = Math.max(1, videoEl.videoHeight || 480);
    this.ctx = this.canvas.getContext("2d");

    // 3) Çıkış stream'i
    this.stream = this.canvas.captureStream(this.fps);

    // 4) WS bağlan
    await this._openWs();

    // 5) Çizim döngüsü
    const draw = () => {
      try {
        const v = this.srcVideo;
        const c = this.canvas;
        const ctx = this.ctx;
        if (!v || !c || !ctx) {
          // video/ctx/canvas hazır değilse bu kareyi atla+          this.rafId = requestAnimationFrame(draw);
          return;
        }
        // video element var ama henüz hazır değilse (metadata/ölçüler yok)
        if (v.readyState < 2 || !v.videoWidth || !v.videoHeight) {
          this.rafId = requestAnimationFrame(draw);
          return;
        }
        const vw = v.videoWidth;
        const vh = v.videoHeight;
        if ((vw !== c.width) || (vh !== c.height)) {
          c.width = vw;
          c.height = vh;
        }

        // Ham kare
        ctx.drawImage(v, 0, 0, c.width, c.height);

        // Overlay
        this._drawOverlay(ctx, c.width, c.height);
        this._frameCounter++;
      } catch (e) {
        console.warn("[RT] draw error:", e?.message || e);
      }
      this.rafId = requestAnimationFrame(draw);
    };
    this.rafId = requestAnimationFrame(draw);

    // 6) Gönderim döngüsü (jpeg -> ws)
    const sendInterval = Math.max(1, Math.floor(1000 / this.sendFps));
    this.sendTimer = setInterval(() => {
      this._sendFrame().catch(() => {});
    }, sendInterval);
  }

  async _waitForVideoReady(videoEl) {
    // Eğer stream bağlanmadıysa bağla (bazı durumlarda lazımdır)
    try {
      if (videoEl.srcObject && videoEl.paused) await videoEl.play().catch(() => {});
    } catch {}

    // Zaten hazır mı?
    if (videoEl.readyState >= 2 && videoEl.videoWidth > 0 && videoEl.videoHeight > 0) return;

    // loadedmetadata + playing bekle
    await new Promise((resolve) => {
      let metaOK = false, playOK = false;
      const maybe = () => (metaOK && playOK) && resolve();

      const onMeta = () => { metaOK = true; maybe(); };
      const onPlay = () => { playOK = true; maybe(); };

      videoEl.addEventListener("loadedmetadata", onMeta, { once: true });
      videoEl.addEventListener("playing", onPlay, { once: true });

      // Güvenlik ağı: 1.5s sonra yine çöz
      setTimeout(() => resolve(), 1500);
    });
  }

  async _openWs() {
    return new Promise((resolve, reject) => {
      try {
        const url =
       this.wsUrl ||
       (import.meta.env && import.meta.env.VITE_RT_WS_URL) ||
       "ws://127.0.0.1:8008/ws";   
        console.log("[RT] WS connecting:", url);
        this.ws = new WebSocket(url);
        this.ws.binaryType = "arraybuffer";

        this.ws.onopen = () => {
          this.connected = true;
          this.lastStatus = "Connected";
          console.log("[RT] WS connected");
          resolve();
        };

        this.ws.onmessage = (ev) => {
          try {
            const raw = typeof ev.data === "string"
              ? ev.data
              : new TextDecoder().decode(ev.data);
            const data = JSON.parse(raw);
            // { ok: true, topk: [], msg?: "..." }
            if (data?.ok) {
              if (Array.isArray(data.topk) && data.topk.length) {
                this.lastTopk = data.topk;
                this.lastStatus = null;
                if (this.emitText) {
                    const [label, prob] = this.lastTopk[0];
                    const txt = `${label}  ${(Number(prob) * 100).toFixed(1)}%`;
                    try { this.emitText(txt); } catch {}
               }
              } else {
                this.lastTopk = [];
                this.lastStatus = data.msg || "No results";
                if (this.emitText) { try { this.emitText(""); } catch {} }
              }
            } else {
              this.lastTopk = [];
              this.lastStatus = data?.error || "No data";
              if (this.emitText) { try { this.emitText(""); } catch {} }
            }
          } catch (e) {
            console.warn("[RT] WS parse error:", e);
            this.lastStatus = "Parse error";
            if (this.emitText) { try { this.emitText(""); } catch {} }
          }
        };

        this.ws.onerror = (e) => {
          console.error("[RT] WS error:", e);
          this.lastStatus = "WS error";
        };

        this.ws.onclose = () => {
          console.warn("[RT] WS closed");
          this.connected = false;
          this.lastStatus = "Disconnected";
        };
      } catch (e) {
        reject(e);
      }
    });
  }

  async _sendFrame() {
    if (!this.connected || !this.ws || this.ws.readyState !== 1) return;
    if (!this.srcVideo || this.srcVideo.readyState < 2) return;

    return new Promise((resolve) => {
      this.canvas.toBlob(
        (blob) => {
          if (!blob) return resolve();
          blob.arrayBuffer().then((buf) => {
            try { this.ws.send(buf); } catch {}
            resolve();
          }).catch(() => resolve());
        },
        "image/jpeg",
        this.jpegQuality
      );
    });
  }

  _drawOverlay(ctx, W, H) {
    if (!this.drawOverlay) return; // UI’yi React tarafında göstereceğiz
    // Fontu ÖNCE ayarla (measureText için)
    ctx.font = "16px sans-serif";
    ctx.textBaseline = "top";

    // Çizilecek metinler
    let lines = [];
    if (this.lastTopk && this.lastTopk.length > 0) {
      for (let i = 0; i < Math.min(3, this.lastTopk.length); i++) {
        const [label, prob] = this.lastTopk[i];
        lines.push(`${label}: ${Number(prob).toFixed(2)}`);
      }
    } else {
      lines.push(this.lastStatus || "Waiting…");
    }

    // Canlılık göstergesi: sağ-alt'a küçük sayaç (her kare artar)
    const tick = `frames: ${this._frameCounter}`;

    // Kutu ölçüleri
    const pad = 10;
    const lineH = 22;
    const textWidths = lines.map(t => ctx.measureText(t).width);
    const maxTextW = textWidths.length ? Math.max(...textWidths) : 80;
    const boxW = Math.min(420, Math.max(160, maxTextW + 2 * pad));
    const boxH = lineH * lines.length + 2 * pad;

    // Arka kutu
    ctx.save();
    ctx.globalAlpha = 0.45;
    const alert = (lines[0] && /nothing/i.test(lines[0]));
    ctx.fillStyle = alert ? "red" : "black";
    ctx.fillRect(10, 10, boxW, boxH);
    ctx.restore();

    // Yazı
    ctx.fillStyle = "white";
    let y = 10 + pad;
    for (const t of lines) {
      ctx.fillText(t, 10 + pad, y);
      y += lineH;
    }

    // Sağ-alt köşe: kare sayacı (canlı mı görmek için)
    ctx.save();
    ctx.globalAlpha = 0.8;
    const tw = ctx.measureText(tick).width;
    ctx.fillStyle = "black";
    ctx.fillRect(W - tw - 20, H - 28, tw + 12, 22);
    ctx.fillStyle = "white";
    ctx.fillText(tick, W - tw - 14, H - 26);
    ctx.restore();
  }

  dispose() {
    try { if (this.rafId) cancelAnimationFrame(this.rafId); } catch {}
    this.rafId = null;

    try { if (this.sendTimer) clearInterval(this.sendTimer); } catch {}
    this.sendTimer = null;

    try { this.ws && this.ws.close(); } catch {}
    this.ws = null;
    this.connected = false;

    try { if (this.stream) this.stream.getTracks().forEach(t => t.stop()); } catch {}
    this.stream = null;

    this.canvas = null;
    this.ctx = null;
    this.srcVideo = null;
  }
}
