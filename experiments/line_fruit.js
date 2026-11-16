/* Drawn avocados. One renderer, used both to make the training pictures and to draw the page. */
const FRUIT = (function () {
  const CONDS = ['good', 'bruised', 'rotten', 'underripe', 'cut'];
  function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function outline(ctx, k, wob, r) {            // pear shape in unit coordinates, scaled by k; wob > 0 gives a shrivelled edge
    const j = () => 1 + (wob ? (r() - .5) * wob : 0);
    ctx.beginPath(); ctx.moveTo(0, -1.25 * k);
    ctx.bezierCurveTo(.35 * k * j(), -1.25 * k, .42 * k * j(), -.6 * k, .72 * k * j(), -.05 * k);
    ctx.bezierCurveTo(1.0 * k * j(), .5 * k, .6 * k * j(), 1.05 * k * j(), 0, 1.05 * k);
    ctx.bezierCurveTo(-.6 * k * j(), 1.05 * k * j(), -1.0 * k * j(), .5 * k, -.72 * k * j(), -.05 * k);
    ctx.bezierCurveTo(-.42 * k * j(), -.6 * k, -.35 * k * j(), -1.25 * k, 0, -1.25 * k); ctx.closePath();
  }
  // s = { seed, cond, size }.  size 1.0 is a normal fruit, about 100 mm long.
  function draw(ctx, cx, cy, R, s) {
    const r = rng(s.seed * 7919 + 13), k = R * .8 * s.size, rot = (r() - .5) * 1.3 + (r() < .5 ? 0 : Math.PI);
    const under = s.cond === 'underripe', rotten = s.cond === 'rotten';
    const hue = under ? 86 + r() * 10 : rotten ? 28 + r() * 14 : 96 + r() * 22, sat = under ? 38 + r() * 26 : rotten ? 18 + r() * 12 : 22 + r() * 14, lit = under ? 27 + r() * 19 : rotten ? 11 + r() * 6 : 15 + r() * 8;
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(rot);
    ctx.save(); ctx.translate(k * .09, k * .12); outline(ctx, k, 0, r); ctx.fillStyle = 'rgba(0,0,0,.32)'; ctx.fill(); ctx.restore();          // shadow on the belt
    outline(ctx, k, rotten ? .16 : 0, r); ctx.fillStyle = 'hsl(' + hue + ',' + sat + '%,' + lit + '%)'; ctx.fill(); ctx.save(); ctx.clip();
    for (let i = 0; i < 70; i++) { const x = (r() * 2 - 1) * k, y = (r() * 2.4 - 1.3) * k, d = r(); ctx.fillStyle = 'hsla(' + hue + ',' + sat + '%,' + (lit + (d < .5 ? 9 : -6)) + '%,.55)'; ctx.beginPath(); ctx.arc(x, y, k * (.015 + r() * .03), 0, 7); ctx.fill(); }   // pebbled skin
    if (r() < .35) { ctx.fillStyle = 'hsla(35,30%,' + (lit + 12) + '%,.5)'; ctx.beginPath(); ctx.ellipse((r() - .5) * k, (r() - .4) * k, k * .05, k * .03, r() * 3, 0, 7); ctx.fill(); }   // a harmless scuff, on any fruit
    const blot = (x, y, a, b, col, soft) => { const g = ctx.createRadialGradient(x, y, 0, x, y, a); g.addColorStop(0, col); g.addColorStop(soft, col); g.addColorStop(1, 'rgba(0,0,0,0)'); ctx.save(); ctx.translate(x, y); ctx.rotate(r() * 3); ctx.scale(1, b / a); ctx.translate(-x, -y); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, a, 0, 7); ctx.fill(); ctx.restore(); };
    if (s.cond === 'bruised') { const n = 1 + (r() < .4 ? 1 : 0); for (let i = 0; i < n; i++) { const a = k * (.05 + r() * .3); blot((r() - .5) * k * .9, (r() * 1.3 - .45) * k, a, a * (.6 + r() * .4), 'hsla(' + (22 + r() * 10) + ',48%,' + (27 + r() * 8) + '%,.92)', .55); } }
    if (rotten) { for (let i = 0; i < 4 + r() * 3; i++) { const a = k * (.22 + r() * .25); blot((r() - .5) * k * 1.2, (r() * 1.8 - .8) * k, a, a * (.5 + r() * .5), 'rgba(8,6,5,.9)', .6); }
      for (let c = 0; c < 2 + r() * 2; c++) { const mx = (r() - .5) * k, my = (r() * 1.4 - .5) * k; for (let i = 0; i < 26; i++) { ctx.fillStyle = 'rgba(' + (205 + r() * 40 | 0) + ',' + (210 + r() * 35 | 0) + ',' + (200 + r() * 40 | 0) + ',' + (.5 + r() * .4) + ')'; ctx.beginPath(); ctx.arc(mx + (r() - .5) * k * .34, my + (r() - .5) * k * .34, k * (.012 + r() * .03), 0, 7); ctx.fill(); } } }
    if (s.cond === 'cut') { const x = (r() - .5) * k * .7, y = (r() * 1.2 - .4) * k, len = k * (.07 + r() * .5), wid = k * (.02 + r() * .06); ctx.save(); ctx.translate(x, y); ctx.rotate(r() * 3.14);
      ctx.fillStyle = 'hsl(30,35%,16%)'; ctx.beginPath(); ctx.ellipse(0, 0, len * 1.08, wid * 1.5, 0, 0, 7); ctx.fill(); ctx.fillStyle = 'hsl(' + (62 + r() * 10) + ',62%,' + (66 + r() * 10) + '%)'; ctx.beginPath(); ctx.ellipse(0, 0, len, wid, 0, 0, 7); ctx.fill(); ctx.restore(); }
    const hl = ctx.createRadialGradient(-k * .3, -k * .25, 0, -k * .3, -k * .25, k * 1.1); hl.addColorStop(0, 'rgba(255,255,255,' + (under ? .3 : .2) + ')'); hl.addColorStop(.5, 'rgba(255,255,255,0)'); ctx.fillStyle = hl; ctx.fillRect(-k * 1.2, -k * 1.4, k * 2.4, k * 2.6);
    ctx.restore();
    ctx.fillStyle = 'hsl(28,35%,24%)'; ctx.beginPath(); ctx.arc(0, -1.24 * k, k * .07, 0, 7); ctx.fill();      // stem end
    ctx.restore();
  }
  function belt(ctx, w, h, off) { ctx.fillStyle = '#3b4148'; ctx.fillRect(0, 0, w, h); ctx.strokeStyle = 'rgba(255,255,255,.055)'; ctx.lineWidth = 1; for (let x = -(off % 12); x < w; x += 12) { ctx.beginPath(); ctx.moveTo(x + .5, 0); ctx.lineTo(x + .5, h); ctx.stroke(); } }
  // What the small network sees: a 32 x 32 colour crop.  o = { blur (pixels at 96), dx, dy, bright, nseed }
  let work = null;
  function crop(s, o) { o = o || {}; if (!work) { work = document.createElement('canvas'); work.width = work.height = 96; }
    const ctx = work.getContext('2d', { willReadFrequently: true }); belt(ctx, 96, 96, (s.seed * 5) % 12); draw(ctx, 48 + (o.dx || 0), 48 + (o.dy || 0), 30, s);
    let d = ctx.getImageData(0, 0, 96, 96).data; const b = Math.round(o.blur || 0);
    if (b >= 1) { const e = new Uint8ClampedArray(d.length), half = b >> 1; for (let y = 0; y < 96; y++) for (let x = 0; x < 96; x++) { let R = 0, G = 0, B = 0, n = 0; for (let u = x - half; u <= x + b - half; u++) { if (u < 0 || u > 95) continue; const i = (y * 96 + u) * 4; R += d[i]; G += d[i + 1]; B += d[i + 2]; n++; } const i = (y * 96 + x) * 4; e[i] = R / n; e[i + 1] = G / n; e[i + 2] = B / n; e[i + 3] = 255; } d = e; }
    const out = new Float32Array(32 * 32 * 3), r = rng((o.nseed || 1) * 31 + s.seed), br = o.bright || 1;
    for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) for (let c = 0; c < 3; c++) { let v = 0; for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) v += d[((y * 3 + j) * 96 + x * 3 + i) * 4 + c]; out[(y * 32 + x) * 3 + c] = Math.max(0, Math.min(1, v / 9 / 255 * br + (r() - .5) * .04)); }
    return out; }
  return { CONDS, rng, draw, belt, crop };
})();
