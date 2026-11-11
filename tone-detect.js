// Tone-lock detector: audio samples in; marks, spaces and unknown stretches (ms) out.
// Lock: N tone bursts inside T seconds agree in frequency and level; that gives f0 and on-level A.
// Marks: amplitude at f0 against A; a mark is kept only if its level, frequency and line share match.
// Browser: defines window.ToneDetect. Node: module.exports.
(function (root) {
  'use strict';

  const P = {
    N: 6, T: 5.0,          // lock: bursts needed, inside this many seconds
    df: 4, dl: 3,          // lock: bursts agree within this many Hz and dB
    weakDb: 6,             // while marks are arriving, a lock this much weaker than A is ignored
    lvlLo: -5, lvlHi: 5,   // mark level relative to A, dB
    fTol: 8, fTolShort: 8, // mark frequency relative to f0, Hz; short: under 20 ms measured
    share: 0.75,           // a frame is clean if the line holds this share of the energy near it
    shareFrac: 0.9,        // a mark needs this fraction of clean frames
    guardHz: 350,          // clean-frame test: energy this far either side of f0 counts as nearby
    guard2Hz: 150,         // the same test, narrower, for marks judged by their edges; 0 = no test
    win2: 1,               // window for the amplitude at f0, in units of 20 ms
    minGapMs: 20,          // a mark with other sound at half its level closer than this is unknown; 0 = no test
    fitTol: 0.06,          // a mark that fails the 350 Hz test is kept if each edge's level curve is within this
                           // rms distance (fraction of mark level) of this sender's clean edges; 0 = never
    minMs: 15,             // shorter marks are unknown
    offLev: 0.35,          // the level must fall below this fraction of the mark's level beside each edge
    track: 0.2,            // weight of each kept mark in the running A, f0 and edge curves
  };
  const F_MIN = 200, F_MAX = 3000; // tone search range, Hz
  const LOBE_HZ = 75;       // the line: this far either side of its peak
  const GUARD_HZ = 350;     // "near": this far either side, or 0.8 x frequency if less
  const RUN_SHARE = 0.8;    // burst frames: line share
  const RUN_AMP = Math.pow(10, -55 / 20); // burst frames: amplitude, full scale = 1
  const RUN_TOL_HZ = 30;    // a burst's peak stays within this of its first frame
  const PAD = 15;           // frames examined either side of a burst
  const EDGE = 12;          // frames beside an edge searched for the fall below offLev
  const NFFT = 1024;
  const FN = 4096, FH = FN - 64; // frame ring size and usable history
  const XN = 65536;         // sample ring size

  // In-place radix-2 complex FFT.
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let j = 0; j < half; j++) {
          const a = i + j, b = a + half;
          const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
          const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
        }
      }
    }
  }

  // Quantile with linear interpolation; a is sorted ascending.
  function quantile(a, q) {
    const p = q * (a.length - 1), i = Math.floor(p);
    return i + 1 < a.length ? a[i] + (p - i) * (a[i + 1] - a[i]) : a[i];
  }
  function median(a) {
    const s = Array.from(a).sort((x, y) => x - y);
    return quantile(s, 0.5);
  }

  class ToneTimer {
    constructor(rate) {
      this.rate = rate;
      // Work near 8 kHz: low-pass, then keep every M-th sample.
      this.M = Math.max(1, Math.round(rate / 8000));
      this.R = rate / this.M;
      const L = this.M === 1 ? 1 : 32 * this.M + 1;
      this.h = new Float64Array(L);
      if (L === 1) this.h[0] = 1;
      else {
        let sum = 0;
        for (let i = 0; i < L; i++) {
          const t = i - (L - 1) / 2, x = Math.PI * t / this.M;
          this.h[i] = (t === 0 ? 1 : Math.sin(x) / x) * (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (L - 1)));
          sum += this.h[i];
        }
        for (let i = 0; i < L; i++) this.h[i] /= sum;
      }
      this.carry = new Float32Array(0); // input not yet consumed by the low-pass
      this.total = 0;                   // input samples pushed
      this.x = new Float32Array(XN);    // decimated samples, ring
      this.nx = 0;
      this.W = Math.round(0.02 * this.R);   // spectrum window, 20 ms
      this.HOP = Math.round(0.002 * this.R); // frame spacing, 2 ms
      this.DF = this.R / NFFT;
      this.win = new Float64Array(this.W);
      let ws = 0;
      for (let i = 0; i < this.W; i++) { this.win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / this.W); ws += this.win[i]; }
      this.scale = 2 / ws; // a full-scale steady sine reads amplitude 1
      this.cw = new Float64Array(this.W + 1); // cumulative window weight, 0..1
      for (let i = 0; i < this.W; i++) this.cw[i + 1] = this.cw[i] + this.win[i] / ws;
      this.W2 = P.win2 * this.W; // window for the amplitude at the locked frequency
      this.win2 = new Float64Array(this.W2);
      let ws2 = 0;
      for (let i = 0; i < this.W2; i++) { this.win2[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / this.W2); ws2 += this.win2[i]; }
      this.scale2 = 2 / ws2;
      this.cw2 = new Float64Array(this.W2 + 1);
      for (let i = 0; i < this.W2; i++) this.cw2[i + 1] = this.cw2[i] + this.win2[i] / ws2;
      this.lag = Math.ceil((this.W2 - this.W) / 2 / this.HOP); // frames the wider window needs beyond the newest
      this.nz = 0;                                 // frames measured at the locked frequency
      this.re = new Float64Array(NFFT); this.im = new Float64Array(NFFT);
      this.kLo = Math.ceil(F_MIN / this.DF); this.kHi = Math.floor(F_MAX / this.DF);
      this.NB = Math.min(NFFT / 2, Math.ceil((F_MAX + GUARD_HZ) / this.DF) + 2); // bins kept per frame
      this.nl = Math.round(LOBE_HZ / this.DF);
      this.nf = 0;                                 // frames made
      this.pw = new Float32Array(FN * this.NB);    // power spectrum per frame
      this.kTop = new Int16Array(FN);              // strongest bin per frame
      this.ok = new Uint8Array(FN);                // frame qualifies for a burst
      this.rp = 0; this.run = null; this.ended = []; // burst finder state
      this.runs = [];                              // fitted bursts: {f, Adb, on, off, len}
      // Lock state.
      this.toneHz = null; this.levelDb = null;
      this.f0 = 0; this.A = 0; this.d = 0;
      this.zr = new Float32Array(FN); this.zi = new Float32Array(FN); this.za = new Float32Array(FN); this.zs = new Float32Array(FN); this.zs2 = new Float32Array(FN);
      this.from = 0; this.sp = 0; this.segI = -1; this.segJ = -1;
      this.prev = null; this.prevEnd = 0; this.doneMs = -1; this.seenMs = -1;
      this.aliveT = -1e9; // end of the last kept mark, s
      this.out = [];
    }

    nowMs() { return this.total * 1000 / this.rate; }
    // Time in ms of decimated sample position p (may be fractional).
    posMs(p) { return (p * this.M + (this.h.length - 1) / 2) * 1000 / this.rate; }
    // Centre time in ms of frame index e (may be fractional).
    frameMs(e) { return this.posMs(e * this.HOP + this.W / 2); }

    push(samples) {
      this.total += samples.length;
      const L = this.h.length, M = this.M;
      let buf = samples;
      if (this.carry.length) {
        buf = new Float32Array(this.carry.length + samples.length);
        buf.set(this.carry); buf.set(samples, this.carry.length);
      }
      let s = 0;
      for (; s + L <= buf.length; s += M) {
        let v = 0;
        for (let t = 0; t < L; t++) v += this.h[t] * buf[s + t];
        this.x[this.nx++ & (XN - 1)] = v;
        while (this.nx >= this.nf * this.HOP + this.W) this.frame();
      }
      this.carry = buf.slice(s);
      const o = this.out; this.out = [];
      return o;
    }

    // One new frame: spectrum, strongest line, bursts, lock, marks.
    frame() {
      const i = this.nf, s = i * this.HOP, W = this.W, re = this.re, im = this.im;
      for (let n = 0; n < W; n++) { re[n] = this.x[(s + n) & (XN - 1)] * this.win[n]; im[n] = 0; }
      re.fill(0, W); im.fill(0, W);
      fft(re, im);
      const row = (i % FN) * this.NB, sc = this.scale * this.scale;
      let k = this.kLo, best = -1;
      for (let b = 0; b < this.NB; b++) {
        const p = (re[b] * re[b] + im[b] * im[b]) * sc;
        this.pw[row + b] = p;
        if (b >= this.kLo && b <= this.kHi && p > best) { best = p; k = b; }
      }
      this.kTop[i % FN] = k;
      this.ok[i % FN] = this.shareOf(i, k) >= RUN_SHARE && Math.sqrt(best) >= RUN_AMP ? 1 : 0;
      this.nf++;
      if (this.toneHz !== null && i - this.lag >= this.nz) { this.lockFrame(i - this.lag); this.nz = i - this.lag + 1; }
      this.findRuns();
      while (this.ended.length && this.ended[0].i1 + PAD < this.nf) this.addRun(this.ended.shift());
      if (this.toneHz !== null) this.scan();
    }

    // Share of frame i's energy near bin k that lies in the line at k.
    shareOf(i, k, guardHz = GUARD_HZ) {
      const row = (i % FN) * this.NB, top = this.NB - 1;
      const ng = Math.round(Math.min(guardHz, 0.8 * k * this.DF) / this.DF);
      let el = 0, eg = 0;
      for (let b = Math.max(0, k - ng); b <= Math.min(top, k + ng); b++) {
        const p = this.pw[row + b];
        eg += p;
        if (b >= k - this.nl && b <= k + this.nl) el += p;
      }
      return eg > 1e-20 ? el / eg : 0;
    }

    // Complex amplitude of frame i at f Hz; phase is referred to sample 0, so a steady tone at f gives a constant.
    zat(f, i) {
      const s = i * this.HOP, w = -2 * Math.PI * f / this.R;
      const cw = Math.cos(w), sw = Math.sin(w), ph = w * s;
      let cr = Math.cos(ph), ci = Math.sin(ph), zr = 0, zi = 0;
      for (let n = 0; n < this.W; n++) {
        const v = this.x[(s + n) & (XN - 1)] * this.win[n];
        zr += v * cr; zi += v * ci;
        const t = cr * cw - ci * sw; ci = cr * sw + ci * cw; cr = t;
      }
      return [zr * this.scale, zi * this.scale];
    }

    // As zat, with the wider window centred on the same instant.
    zat2(f, i) {
      const s = i * this.HOP + (this.W - this.W2) / 2, w = -2 * Math.PI * f / this.R;
      if (s < 0) return [0, 0];
      const cw = Math.cos(w), sw = Math.sin(w), ph = w * s;
      let cr = Math.cos(ph), ci = Math.sin(ph), zr = 0, zi = 0;
      for (let n = 0; n < this.W2; n++) {
        const v = this.x[(s + n) & (XN - 1)] * this.win2[n];
        zr += v * cr; zi += v * ci;
        const t = cr * cw - ci * sw; ci = cr * sw + ci * cw; cr = t;
      }
      return [zr * this.scale2, zi * this.scale2];
    }

    // Bursts: consecutive frames whose strongest line stays put and holds most of the nearby energy.
    findRuns() {
      while (this.rp < this.nf) {
        const t = this.rp, ok = this.ok[t % FN], k = this.kTop[t % FN], r = this.run;
        if (!r) { if (ok) this.run = { i0: t, last: t, k0: k }; this.rp++; continue; }
        if (t - r.i0 > 3000) { this.run = null; continue; } // a steady tone, not keying
        if (ok && Math.abs(k - r.k0) * this.DF <= RUN_TOL_HZ) { r.last = t; this.rp++; continue; }
        if (t - r.last >= 2) {
          if (r.last - r.i0 + 1 >= 5) this.ended.push({ i0: r.i0, i1: r.last });
          this.rp = r.last + 1; this.run = null; continue;
        }
        this.rp++;
      }
    }

    // Measures an ended burst and tests for a lock.
    addRun(r) {
      const lo = this.nf - FH;
      if (r.i0 - PAD - 1 < Math.max(lo, 0)) return;
      const ks = [];
      for (let i = r.i0; i <= r.i1; i++) ks.push(this.kTop[i % FN]);
      const f = Math.round(median(ks)) * this.DF;
      // Frequency: mean phase advance per frame, over frames near the top level.
      const z = [];
      for (let i = r.i0 - 1; i <= r.i1; i++) z.push(this.zat(f, i));
      const a = z.slice(1).map((c) => Math.hypot(c[0], c[1]));
      const peak = quantile(a.slice().sort((x, y) => x - y), 0.8);
      let pr = 0, pi = 0, first = true;
      for (let j = 0; j < a.length; j++) {
        if (a[j] < 0.7 * peak) continue;
        if (first) { first = false; continue; }
        const c = z[j + 1], p = z[j];
        pr += c[0] * p[0] + c[1] * p[1]; pi += c[1] * p[0] - c[0] * p[1];
      }
      const g = this.fit(f + Math.atan2(pi, pr) * this.R / (2 * Math.PI * this.HOP), r.i0, r.i1);
      if (!g || g.len < 20) return;
      this.runs.push(g);
      this.tryLock(g.off);
    }

    // Fits one tone switched on once and off once to frames i0..i1: frequency, level in dB, edges in s.
    fit(f, i0, i1) {
      const b = i0 - PAD, n = i1 + PAD - b + 1, z = [], a = [];
      for (let j = 0; j < n; j++) { const c = this.zat(f, b + j); z.push(c); a.push(Math.hypot(c[0], c[1])); }
      const lo = i0 - b, hi = i1 - b;
      const top = quantile(a.slice(lo, hi + 1).sort((x, y) => x - y), 0.8);
      const inner = [];
      for (let j = lo; j <= hi; j++) if (a[j] >= 0.7 * top) inner.push(j);
      const half = 0.5 * median(inner.map((j) => a[j]));
      let j = inner[0];
      while (j > 0 && a[j - 1] >= half) j--;
      if (j === 0) return null;
      const eOn = j - 1 + (half - a[j - 1]) / (a[j] - a[j - 1]);
      j = inner[inner.length - 1];
      while (j < n - 1 && a[j + 1] >= half) j++;
      if (j === n - 1) return null;
      const eOff = j + (a[j] - half) / (a[j] - a[j + 1]);
      const c0 = Math.ceil(eOn) + 2, c1 = Math.floor(eOff) - 2;
      let df = 0;
      if (c1 - c0 >= 4) {
        let pr = 0, pi = 0;
        for (let q = c0 + 1; q <= c1; q++) { pr += z[q][0] * z[q - 1][0] + z[q][1] * z[q - 1][1]; pi += z[q][1] * z[q - 1][0] - z[q][0] * z[q - 1][1]; }
        df = Math.atan2(pi, pr) * this.R / (2 * Math.PI * this.HOP);
      }
      // Least-squares amplitude against the window's response to an ideal burst between the edges.
      const nOn = (b + eOn) * this.HOP + this.W / 2, nOff = (b + eOff) * this.HOP + this.W / 2;
      let nr = 0, ni = 0, den = 0;
      for (let q = 0; q < n; q++) {
        const s0 = (b + q) * this.HOP;
        const sh = this.cwAt(nOff - s0) - this.cwAt(nOn - s0);
        if (!(sh > 0)) continue;
        const ph = -2 * Math.PI * df * (s0 + this.W / 2) / this.R, cr = Math.cos(ph), ci = Math.sin(ph);
        nr += (z[q][0] * cr - z[q][1] * ci) * sh; ni += (z[q][0] * ci + z[q][1] * cr) * sh; den += sh * sh;
      }
      if (!(den > 0)) return null;
      const A = Math.hypot(nr, ni) / den;
      return { f: f + df, Adb: 20 * Math.log10(A + 1e-12), on: nOn / this.R, off: nOff / this.R, len: (nOff - nOn) * 1000 / this.R };
    }

    // Cumulative window weight up to offset p samples into the window.
    cwAt(p) {
      if (p <= 0) return 0;
      if (p >= this.W) return 1;
      const i = Math.floor(p);
      return this.cw[i] + (p - i) * (this.cw[i + 1] - this.cw[i]);
    }

    cw2At(p) {
      if (p <= 0) return 0;
      if (p >= this.W2) return 1;
      const i = Math.floor(p);
      return this.cw2[i] + (p - i) * (this.cw2[i + 1] - this.cw2[i]);
    }

    // Lock when N bursts in the last T seconds agree in frequency and level.
    tryLock(now) {
      this.runs = this.runs.filter((q) => q.off > now - P.T);
      const groups = [];
      for (const c of this.runs) {
        const grp = this.runs.filter((q) => Math.abs(q.f - c.f) <= P.df && Math.abs(q.Adb - c.Adb) <= P.dl);
        if (grp.length >= P.N) groups.push({ f: median(grp.map((q) => q.f)), Adb: median(grp.map((q) => q.Adb)), since: Math.min(...grp.map((q) => q.on)) });
      }
      // Loudest first: echoes and tails of the tone are weaker than the tone.
      groups.sort((x, y) => y.Adb - x.Adb);
      const alive = this.toneHz !== null && this.aliveT > now - P.T; // a mark was kept in the last T seconds
      for (const g of groups) {
        if (this.toneHz === null) { this.setLock(g.f, g.Adb, g.since); return; }
        const up = g.Adb - 20 * Math.log10(this.A), same = Math.abs(g.f - (this.f0 + this.d)) <= 10;
        if (same && Math.abs(up) <= 2) return;        // the current lock, still right
        if (alive && up < (same ? 0 : -P.weakDb)) continue;
        this.setLock(g.f, g.Adb, g.since); return;
      }
    }

    // Starts measuring at f0, re-running stored frames from 0.1 s before the first agreeing burst.
    setLock(f, Adb, since) {
      this.f0 = f; this.A = Math.pow(10, Adb / 20); this.d = 0;
      this.toneHz = f; this.levelDb = Adb;
      this.from = Math.max(0, this.nf - FH, Math.floor((since - 0.1) * this.R / this.HOP));
      this.nz = Math.max(this.from, this.nf - this.lag);
      for (let i = this.from; i < this.nz; i++) this.lockFrame(i);
      this.sp = this.from + 1; this.segI = -1;
      // Edge curves start as those of a clean switch; clean marks then replace them with this sender's.
      const H = this.W2 / (2 * this.HOP);
      this.oA = Math.ceil(H / 2); this.oB = Math.ceil(H) + 2;
      const np = this.oA + this.oB + 1;
      this.tOn = new Float64Array(np); this.tOff = new Float64Array(np);
      for (let k = 0; k < np; k++) {
        this.tOn[k] = this.cw2At((k - this.oA) * this.HOP + this.W2 / 2);
        this.tOff[k] = 1 - this.cw2At((k - this.oB) * this.HOP + this.W2 / 2);
      }
    }

    // Amplitude and line share of frame i at the locked frequency.
    lockFrame(i) {
      const c = this.zat2(this.f0, i), m = i % FN;
      this.zr[m] = c[0]; this.zi[m] = c[1]; this.za[m] = Math.hypot(c[0], c[1]);
      this.zs[m] = this.shareOf(i, Math.round(this.f0 / this.DF), P.guardHz);
      this.zs2[m] = P.guard2Hz ? this.shareOf(i, Math.round(this.f0 / this.DF), P.guard2Hz) : 1;
    }

    // Finds stretches at or above half of A, and finishes each once enough later frames exist.
    scan() {
      const n = this.nz, a = this.za;
      for (;;) {
        const thr = 0.5 * this.A;
        if (this.segI < 0) {
          while (this.sp < n && !(a[this.sp % FN] >= thr && a[(this.sp - 1) % FN] < thr)) this.sp++;
          if (this.sp >= n) return;
          this.segI = this.segJ = this.sp;
        }
        while (this.segJ + 1 < n && a[(this.segJ + 1) % FN] >= thr) this.segJ++;
        if (this.segJ + 1 >= n) return;
        const next = this.finish(this.segI, this.segJ, n);
        if (next < 0) return;
        this.sp = next; this.segI = -1;
      }
    }

    // Decides one stretch i..j: mark or unknown. Returns the next frame to scan, or -1 to wait for more frames.
    finish(i, j, n) {
      const a = (q) => this.za[q % FN], low = Math.max(this.from, n - FH);
      if (i <= low || j - i > 2500) { this.emit(null, this.frameMs(i), this.frameMs(j), 'long'); return j + 1; }
      const seg = [];
      for (let q = i; q <= j; q++) seg.push(a(q));
      const top = 0.7 * quantile(seg.slice().sort((x, y) => x - y), 0.8);
      const plat = median(seg.filter((v) => v >= top)), half = 0.5 * plat;
      let u = i, v = j;
      while (a(u) < top) u++;
      while (a(v) < top) v--;
      // Each edge is placed at half of the level just inside it.
      const H = this.W2 / (2 * this.HOP); // frames from an edge to full level
      const near = (q0, q1) => { const w = []; for (let q = Math.max(q0, u); q <= Math.min(q1, v); q++) w.push(a(q)); return w.length >= 3 ? median(w) : plat; };
      const platOn = near(u + H - 2, u + H + 8), platOff = near(v - H - 8, v - H + 2), halfOn = 0.5 * platOn, halfOff = 0.5 * platOff;
      while (u - 1 > low && a(u - 1) >= halfOn) u--;
      while (v + 1 < n && a(v + 1) >= halfOff) v++;
      // Wait for the frames after the mark that the tests below read.
      const gf = Math.round(P.minGapMs * this.R / 1000 / this.HOP);
      if (v + Math.max(gf, this.oA + 2) >= n) return -1;
      // The level must fall well below the mark's level just after it, and just before it.
      let offOk = false, onOk = false;
      for (let q = v + 1; q <= v + EDGE && q < n; q++) if (a(q) <= P.offLev * plat) { offOk = true; break; }
      if (!offOk && v + EDGE >= n) return -1;
      for (let q = u - 1; q >= u - EDGE && q >= low; q--) if (a(q) <= P.offLev * plat) { onOk = true; break; }
      // Nothing else at half the mark's level within minGapMs of either edge.
      let alone = true;
      for (let q = v + 2; q <= v + gf; q++) if (a(q) >= halfOff) alone = false;
      for (let q = u - 2; q >= u - gf && q >= low; q--) if (a(q) >= halfOn) alone = false;
      const next = Math.max(j, v) + 1;
      if (a(u - 1) >= halfOn) { this.emit(null, this.frameMs(u), this.frameMs(v), 'edge'); return next; }
      const eOn = u - 1 + (halfOn - a(u - 1)) / (a(u) - a(u - 1));
      const eOff = v + (a(v) - halfOff) / (a(v) - a(v + 1));
      // Edge curves: level around each edge, as a fraction of the level just inside it, on a 2 ms grid.
      // Points whose window reaches the mark's other edge are left out.
      const lvl = (e) => { const q = Math.floor(e); return q <= low || q + 1 >= n ? NaN : a(q) + (e - q) * (a(q + 1) - a(q)); };
      const np = this.oA + this.oB + 1, cOn = new Float64Array(np), cOff = new Float64Array(np), lenF = eOff - eOn;
      let sOn = 0, nOn = 0, sOff = 0, nOff = 0;
      for (let k = 0; k < np; k++) {
        const o1 = k - this.oA, o2 = k - this.oB;
        cOn[k] = o1 <= lenF - H ? lvl(eOn + o1) / platOn : NaN;
        cOff[k] = -o2 <= lenF - H ? lvl(eOff + o2) / platOff : NaN;
        if (cOn[k] === cOn[k]) { sOn += (cOn[k] - this.tOn[k]) ** 2; nOn++; }
        if (cOff[k] === cOff[k]) { sOff += (cOff[k] - this.tOff[k]) ** 2; nOff++; }
      }
      const fitOn = nOn >= 4 ? Math.sqrt(sOn / nOn) : 1, fitOff = nOff >= 4 ? Math.sqrt(sOff / nOff) : 1;
      const on = this.frameMs(eOn), off = this.frameMs(eOff), len = off - on;
      // Frequency offset from f0: mean phase advance per frame, away from the edges.
      let c0 = Math.ceil(eOn) + 5, c1 = Math.floor(eOff) - 5, tol = P.fTol;
      if (c1 - c0 < 10) { c0 = Math.ceil(eOn) + 1; c1 = Math.floor(eOff) - 1; tol = P.fTolShort; }
      let pr = 0, pi = 0;
      for (let q = c0 + 1; q <= c1; q++) {
        const m = q % FN, p = (q - 1) % FN;
        pr += this.zr[m] * this.zr[p] + this.zi[m] * this.zi[p]; pi += this.zi[m] * this.zr[p] - this.zr[m] * this.zi[p];
      }
      const fm = c1 > c0 ? Math.atan2(pi, pr) * this.R / (2 * Math.PI * this.HOP) : 0;
      const ldb = 20 * Math.log10(plat / this.A);
      let clean = 0, clean2 = 0;
      for (let q = u; q <= v; q++) { if (this.zs[q % FN] >= P.share) clean++; if (this.zs2[q % FN] >= P.share) clean2++; }
      const need = P.shareFrac * (v - u + 1), spectrumClean = clean >= need;
      let why = null;
      if (len < P.minMs) why = 'short';
      else if (ldb < P.lvlLo || ldb > P.lvlHi) why = 'level';
      else if (Math.abs(fm - this.d) > tol) why = 'freq';
      else if (!alone) why = 'gap';
      else if (!onOk || !offOk) why = 'edge';
      else if (!spectrumClean) {
        if (!P.fitTol || clean2 < need) why = 'share';
        else if (Math.max(fitOn, fitOff) > P.fitTol) why = 'fit';
      }
      if (!why && spectrumClean) {
        for (let k = 0; k < np; k++) {
          if (cOn[k] === cOn[k]) this.tOn[k] += P.track * (cOn[k] - this.tOn[k]);
          if (cOff[k] === cOff[k]) this.tOff[k] += P.track * (cOff[k] - this.tOff[k]);
        }
      }
      this.lastFit = [fitOn, fitOff];
      this.emit(why ? null : { toneHz: this.f0 + fm, levelDb: 20 * Math.log10(plat) }, on, off, why);
      if (!why) {
        this.aliveT = (eOff * this.HOP + this.W / 2) / this.R;
        this.A *= Math.pow(10, P.track * ldb / 20);
        if (tol === P.fTol) this.d += P.track * (fm - this.d);
        this.toneHz = this.f0 + this.d; this.levelDb = 20 * Math.log10(this.A);
      }
      return next;
    }

    // Queues a mark (with the space before it, if nothing unknown lies between) or an unknown stretch.
    emit(mark, on, off, why) {
      if (mark) {
        if (on < this.doneMs) return;
        if (this.prev === 'mark') this.out.push({ type: 'space', start: this.prevEnd, len: on - this.prevEnd });
        this.out.push({ type: 'mark', start: on, len: off - on, toneHz: mark.toneHz, levelDb: mark.levelDb, fit: this.lastFit });
        this.prev = 'mark'; this.prevEnd = off; this.doneMs = off;
      } else {
        if (on < this.doneMs) return;
        this.prev = 'unknown';
        if (on >= this.seenMs) this.out.push({ type: 'unknown', start: on, len: off - on, why, fit: this.lastFit });
      }
      this.seenMs = Math.max(this.seenMs, off);
    }

    // Start in ms of a stretch at the locked frequency that is in progress or not yet decided, or null.
    liveStart() {
      return this.toneHz !== null && this.segI >= 0 ? this.frameMs(this.segI) : null;
    }
  }

  // Offline: all events in a sample array.
  function measure(samples, rate) {
    const t = new ToneTimer(rate), out = [];
    for (let i = 0; i < samples.length; i += 4096) for (const e of t.push(samples.subarray(i, i + 4096))) out.push(e);
    return out;
  }

  const api = { ToneTimer, measure, P };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ToneDetect = api;
})(this);
