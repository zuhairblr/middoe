/* MIDDoE documentation: interactive figures.
 *
 * A figure is a <div class="mdw" data-widget="NAME"></div> in the .rst source (raw-HTML block).
 * Plain SVG + JavaScript, no libraries. Elements are created once and updated in place; drags are
 * continuous and snap to the grid on release; every handle can also be moved with the arrow keys
 * (Shift = larger steps); animations respect "prefers-reduced-motion".
 *
 *   loop         the study round
 *   sampling     where to sample: sensitivities, Fisher information, D/A/E, 95 % ellipse
 *   discrim      two rival models, Hunter-Reiner divergence, a design that separates them
 *   noise        absolute vs relative error, sigma_floor, weights 1/sigma^2
 *   designspace  the design space a `system` dictionary defines, with MIDDoE's constraints
 *   flow         which function reads which dictionary and writes which file
 *   quickstart   the getting-started results, round by round
 *   datafile     the layout of data.xlsx, column by column
 *   residuals    how to read weighted residuals and the fit statistic
 */
(function () {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';
  var REDUCED = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var CHI2_95 = [0, 3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307,
                 19.675, 21.026, 22.362, 23.685, 24.996, 26.296, 27.587, 28.869, 30.144, 31.410];

  // ================================================================ DOM helpers
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    if (attrs) set(e, attrs);
    if (parent) parent.appendChild(e);
    return e;
  }
  function set(e, attrs) { for (var k in attrs) e.setAttribute(k, attrs[k]); return e; }
  function txt(e, s) { if (e.textContent !== s) e.textContent = s; return e; }
  function h(tag, attrs, parent, html) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (html !== undefined) e.innerHTML = html;
    if (parent) parent.appendChild(e);
    return e;
  }
  function clear(g) { while (g.firstChild) g.removeChild(g.firstChild); }
  function scale(d0, d1, r0, r1) {
    var s = function (v) { return s.r0 + (v - s.d0) / (s.d1 - s.d0) * (s.r1 - s.r0); };
    s.d0 = d0; s.d1 = d1; s.r0 = r0; s.r1 = r1;
    s.inv = function (p) { return s.d0 + (p - s.r0) / (s.r1 - s.r0) * (s.d1 - s.d0); };
    return s;
  }
  function pathD(pts) {
    var out = '';
    for (var i = 0; i < pts.length; i++) out += (i ? 'L' : 'M') + pts[i][0].toFixed(1) + ',' + pts[i][1].toFixed(1);
    return out;
  }
  function fmt(v, d) {
    if (v === Infinity) return '∞';
    if (!isFinite(v)) return '–';
    var a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(d === undefined ? 2 : d).replace('e+', 'e');
    return v.toFixed(d === undefined ? 2 : d);
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function niceStep(span, n) {
    var raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
    return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  }
  function ticks(lo, hi, n) {
    var st = niceStep(hi - lo, n || 4), out = [];
    for (var v = Math.ceil(lo / st - 1e-9) * st; v <= hi + 1e-9 * st; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : v);
    return out;
  }
  function niceUp(v) {
    var p = Math.pow(10, Math.floor(Math.log10(v))), m = v / p;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
  }
  function rng(seed) {                       // mulberry32 + Box-Muller: reproducible noise
    var a = seed >>> 0;
    function u() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
    return { u: u, n: function () { return Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u()); } };
  }
  function sortedUniform(n, r) {             // n sorted values, uniform on the ordered simplex
    var c = new Array(n), nxt = 1;
    for (var k = n; k >= 1; k--) { nxt *= Math.pow(r.u(), 1 / k); c[k - 1] = nxt; }
    return c;
  }

  // one animation channel per owner: a new request cancels the running one
  function channel() {
    var id = null;
    var run = function (from, to, ms, step, done) {
      if (id !== null) cancelAnimationFrame(id);
      id = null;
      if (REDUCED || !ms || document.hidden) { step(to); if (done) done(); return; }
      var t0 = null;
      var f = function (ts) {
        if (t0 === null) t0 = ts;
        var a = Math.min(1, (ts - t0) / ms);
        var e = a < 0.5 ? 4 * a * a * a : 1 - Math.pow(-2 * a + 2, 3) / 2;
        step(from.map(function (v, i) { return v + (to[i] - v) * e; }));
        if (a < 1) id = requestAnimationFrame(f); else { id = null; if (done) done(); }
      };
      id = requestAnimationFrame(f);
    };
    run.cancel = function () { if (id !== null) cancelAnimationFrame(id); id = null; };
    return run;
  }

  // ================================================================ chart pieces
  function svgBox(parent, w, hgt, label) {
    return el('svg', { viewBox: '0 0 ' + w + ' ' + hgt, 'class': 'mdw-svg', role: 'img', 'aria-label': label }, parent);
  }
  function drawAxes(g, sx, sy, o) {
    clear(g);
    var xt = o.xt || ticks(sx.d0, sx.d1, o.nx || 5), yt = o.yt || ticks(sy.d0, sy.d1, o.ny || 4);
    yt.forEach(function (v) {
      el('line', { x1: sx.r0, x2: sx.r1, y1: sy(v), y2: sy(v), 'class': v === 0 && o.zero ? 'mdw-zero' : 'mdw-grid' }, g);
      txt(el('text', { x: sx.r0 - 6, y: sy(v) + 3.5, 'text-anchor': 'end', 'class': 'mdw-tick' }, g), o.fy ? o.fy(v) : String(+v.toPrecision(6)));
    });
    xt.forEach(function (v) {
      if (o.zero && v === 0) el('line', { x1: sx(0), x2: sx(0), y1: sy.r0, y2: sy.r1, 'class': 'mdw-zero' }, g);
      txt(el('text', { x: sx(v), y: sy.r0 + 15, 'text-anchor': 'middle', 'class': 'mdw-tick' }, g), o.fx ? o.fx(v) : String(+v.toPrecision(6)));
    });
    el('line', { x1: sx.r0, x2: sx.r1, y1: sy.r0, y2: sy.r0, 'class': 'mdw-axis' }, g);
    if (o.xl) txt(el('text', { x: sx.r1, y: sy.r0 + 30, 'text-anchor': 'end', 'class': 'mdw-label' }, g), o.xl);
    if (o.yl) txt(el('text', { x: sx.r0 - 6, y: sy.r1 - 9, 'class': 'mdw-label' }, g), o.yl);
  }
  function bubble(parent) {
    var g = el('g', { 'class': 'mdw-bubble', opacity: 0 }, parent);
    var r = el('rect', { rx: 5, ry: 5, height: 20 }, g), t = el('text', { y: 14, x: 7 }, g);
    var svg = parent.ownerSVGElement || parent;
    return {
      show: function (x, y, s, below) {
        txt(t, s);
        var w = 14 + (t.getComputedTextLength ? t.getComputedTextLength() : s.length * 6.4);
        if (!w || w < 30) w = s.length * 6.4 + 14;
        r.setAttribute('width', w.toFixed(1));
        var vb = svg.viewBox && svg.viewBox.baseVal, W = vb && vb.width ? vb.width : 640;
        var bx = clamp(x - w / 2, 2, W - w - 2), by = below ? y + 12 : y - 32;
        set(g, { transform: 'translate(' + bx.toFixed(1) + ',' + by.toFixed(1) + ')', opacity: 1 });
      },
      hide: function () { g.setAttribute('opacity', 0); }
    };
  }
  function svgPoint(svg, evt) {
    var p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY;
    var m = svg.getScreenCTM();
    return m ? p.matrixTransform(m.inverse()) : { x: 0, y: 0 };
  }
  // hover line + readout; the hit area sits under everything else
  function crosshair(svg, layer, area, sx, read) {
    var g = el('g', { 'class': 'mdw-cross', opacity: 0 }, layer);
    var line = el('line', { y1: area.y0, y2: area.y1 }, g);
    var b = bubble(layer);
    var hit = el('rect', { x: area.x0, y: area.y1, width: area.x1 - area.x0, height: area.y0 - area.y1, 'class': 'mdw-overlay' });
    svg.insertBefore(hit, svg.firstChild);
    hit.addEventListener('pointermove', function (e) {
      if (svg.classList.contains('mdw-dragging')) return;
      var v = clamp(sx.inv(svgPoint(svg, e).x), sx.d0, sx.d1);
      set(line, { x1: sx(v), x2: sx(v) }); g.setAttribute('opacity', 1);
      b.show(sx(v), area.y1 - 6, read(v), true);
    });
    hit.addEventListener('pointerleave', function () { g.setAttribute('opacity', 0); b.hide(); });
  }
  // drag + keyboard for every element carrying data-h inside svg
  function interact(svg, o) {
    var active = null;
    svg.addEventListener('pointerdown', function (e) {
      var t = e.target.closest ? e.target.closest('[data-h]') : null;
      if (!t || !svg.contains(t)) return;
      e.preventDefault();
      active = t.getAttribute('data-h');
      try { t.focus({ preventScroll: true }); } catch (err) { /* ignore */ }
      try { svg.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
      svg.classList.add('mdw-dragging');
      if (o.start) o.start(active);
      o.drag(active, svgPoint(svg, e));
    });
    svg.addEventListener('pointermove', function (e) { if (active !== null) o.drag(active, svgPoint(svg, e)); });
    var end = function () {
      if (active === null) return;
      var a = active; active = null; svg.classList.remove('mdw-dragging');
      if (o.drop) o.drop(a);
    };
    svg.addEventListener('pointerup', end);
    svg.addEventListener('pointercancel', end);
    svg.addEventListener('lostpointercapture', end);
    svg.addEventListener('keydown', function (e) {
      var t = e.target.closest ? e.target.closest('[data-h]') : null;
      if (!t || !o.key) return;
      var d = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
      if (!d) return;
      e.preventDefault();
      o.key(t.getAttribute('data-h'), d, e.shiftKey);
    });
    svg.addEventListener('focusin', function (e) {
      var t = e.target.closest ? e.target.closest('[data-h]') : null;
      if (t && o.focus && active === null) o.focus(t.getAttribute('data-h'));
    });
    svg.addEventListener('focusout', function () { if (o.blur && active === null) o.blur(); });
  }
  function handle(parent, id, label, shape) {
    var g = el('g', { 'class': 'mdw-h', 'data-h': id, tabindex: 0, role: 'slider', 'aria-label': label }, parent);
    el('circle', { r: 15, 'class': 'mdw-hit' }, g);
    if (shape === 'diamond') el('path', { d: 'M0,-7 L7,0 L0,7 L-7,0 Z', 'class': 'mdw-pt' }, g);
    else if (shape === 'bar') el('rect', { x: -10, y: -4, width: 20, height: 8, rx: 3, 'class': 'mdw-pt' }, g);
    else el('circle', { r: 6.5, 'class': 'mdw-pt' }, g);
    g.at = function (x, y) { g.setAttribute('transform', 'translate(' + x.toFixed(1) + ',' + y.toFixed(1) + ')'); return g; };
    return g;
  }

  // ================================================================ controls
  function bar(parent, cls) { return h('div', { 'class': 'mdw-bar' + (cls ? ' ' + cls : '') }, parent); }
  function slider(parent, label, min, max, step, val, onInput, show) {
    var w = h('label', { 'class': 'mdw-ctl' }, parent);
    h('span', { 'class': 'mdw-ctl-l' }, w, label);
    var i = h('input', { type: 'range', min: min, max: max, step: step, value: val }, w);
    var o = h('output', null, w);
    var paint = function () { o.textContent = show ? show(+i.value) : i.value; };
    i.addEventListener('input', function () { paint(); onInput(+i.value); });
    paint();
    return { set: function (v) { i.value = v; paint(); }, get: function () { return +i.value; },
             range: function (mn, mx, st) { i.min = mn; i.max = mx; i.step = st; paint(); } };
  }
  function button(parent, label, onClick, cls, title) {
    var b = h('button', { type: 'button', 'class': 'mdw-btn' + (cls ? ' ' + cls : '') }, parent, label);
    if (title) { b.title = title; b.setAttribute('aria-label', title); }
    b.addEventListener('click', onClick);
    return b;
  }
  function seg(parent, options, value, onChange, label) {
    var w = h('div', { 'class': 'mdw-seg', role: 'radiogroup' }, parent);
    if (label) { w.setAttribute('aria-label', label); h('span', { 'class': 'mdw-seg-l' }, w, label); }
    var bs = options.map(function (o) {
      var b = h('button', { type: 'button', role: 'radio' }, w, o[1]);
      b.dataset.v = o[0];
      b.addEventListener('click', function () { api.set(o[0]); onChange(o[0]); });
      return b;
    });
    var api = { set: function (v) { bs.forEach(function (b) { var on = b.dataset.v === String(v); b.classList.toggle('md-on', on); b.setAttribute('aria-checked', on); }); } };
    api.set(value);
    return api;
  }
  function toggle(parent, label, value, onChange) {
    var w = h('label', { 'class': 'mdw-switch' }, parent);
    var i = h('input', { type: 'checkbox' }, w); i.checked = value;
    h('span', { 'class': 'mdw-switch-ui', 'aria-hidden': 'true' }, w);
    h('span', null, w, label);
    i.addEventListener('change', function () { onChange(i.checked); });
    return { set: function (v) { i.checked = v; } };
  }
  function stats(parent, rows) {
    var g = h('div', { 'class': 'mdw-stats' }, parent), cells = {};
    rows.forEach(function (r) {
      var d = h('div', { 'class': 'mdw-stat' + (r.wide ? ' mdw-wide' : '') }, g);
      h('span', { 'class': 'mdw-stat-l' }, d, r.label);
      cells[r.key] = { v: h('span', { 'class': 'mdw-stat-v' }, d), box: d };
      if (r.note) h('span', { 'class': 'mdw-stat-n' }, d, r.note);
    });
    return { set: function (k, html, hl) { var c = cells[k]; if (c.v.innerHTML !== html) c.v.innerHTML = html; c.box.classList.toggle('mdw-hl', !!hl); } };
  }
  function meter(parent, label) {
    var w = h('div', { 'class': 'mdw-meter' }, parent);
    h('div', { 'class': 'mdw-meter-l' }, w, label);
    var track = h('div', { 'class': 'mdw-meter-t' }, w);
    var fill = h('div', { 'class': 'mdw-meter-f' }, track), mark = h('div', { 'class': 'mdw-meter-m' }, track);
    var cap = h('div', { 'class': 'mdw-meter-c' }, w);
    return function (val, thr, max, caption, good) {
      fill.style.width = (100 * clamp(val / max, 0, 1)).toFixed(1) + '%';
      mark.style.left = (100 * clamp(thr / max, 0, 1)).toFixed(1) + '%';
      w.classList.toggle('mdw-ok', !!good);
      if (cap.innerHTML !== caption) cap.innerHTML = caption;
    };
  }
  function status(parent) {
    var s = h('div', { 'class': 'mdw-status', 'aria-live': 'polite' }, parent);
    var timer = null;
    return function (html, kind, sticky) {
      if (timer) { clearTimeout(timer); timer = null; }
      if (!html) { s.classList.remove('mdw-on'); return; }
      s.innerHTML = html;
      s.className = 'mdw-status mdw-on' + (kind ? ' mdw-' + kind : '');
      if (!sticky) timer = setTimeout(function () { s.classList.remove('mdw-on'); }, 2600);
    };
  }
  function eig2(a, b, c) {                   // symmetric [[a,b],[b,c]]
    var tr = a + c, det = a * c - b * b, d = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    var l1 = tr / 2 + d, l2 = tr / 2 - d;
    var ang = Math.abs(b) < 1e-300 ? (a >= c ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
    return { l1: l1, l2: l2, ang: ang };
  }

  // ======================================================================= 1. study round
  function loopWidget(root) {
    var steps = [
      { n: 'Design', call: 'mbdoe_pp(…) · mbdoe_md(…)', opt: 'optional',
        what: 'Choose the next experiment: inputs, their profiles and the sampling times.',
        idea: 'max<sub>φ</sub> ψ[<b>M</b>(φ, θ̂)] for precision (D, A, E, ME), or the divergence between rival models (HR, BFF) for discrimination.',
        out: '<code>design/</code>', link: 'ug_setup.html#setup-design' },
      { n: 'Experiment', call: 'lab → data.xlsx · expera(…)', opt: 'required',
        what: 'Run it in the lab, or in silico with known parameters and noise.',
        idea: 'y = f(φ, θ) + ε, &nbsp;ε ~ N(0, σ²), σ from <code>unc</code> and <code>errt</code>.',
        out: '<code>data.xlsx</code>, <code>data_plots/</code>', link: 'ug_setup.html#setup-data' },
      { n: 'Estimate', call: 'parmest(system, models, iden_opt)', opt: 'required',
        what: 'Fit every candidate model to all the data so far.',
        idea: 'θ̂ = arg min Σ [(y − ŷ)/σ]² (or the full likelihood, <code>ob=\'MLE\'</code>).',
        out: 'results in memory', link: 'ug_settings.html' },
      { n: 'Uncertainty', call: 'uncert(resultpr, system, models, iden_opt)', opt: 'required',
        what: 'How precise is each parameter? Confidence intervals, t-values, fit statistic.',
        idea: '<b>V</b> ≈ <b>M</b><sup>−1</sup> = (<b>S</b><sup>T</sup><b>Σ</b><sup>−1</sup><b>S</b>)<sup>−1</sup> (or Hessian, bootstrap); CI<sub>i</sub> = t<sub>0.975</sub>√V<sub>ii</sub>.',
        out: 'results in memory', link: 'ug_settings.html' },
      { n: 'Estimability', call: 'estima(resultun, system, models, iden_opt, k)', opt: 'optional',
        what: 'Rank the parameters and keep only those the data can support.',
        idea: 'Orthogonalised sensitivities rank θ; the rCC criterion picks how many to estimate; the rest are fixed.',
        out: '<code>estimability/</code>', link: 'ug_reference.html#practical-workflow' },
      { n: 'Decide', call: 'save_rounds(k, …)', opt: 'required',
        what: 'Store the round. Precise enough, and one model clearly ahead? Then report; otherwise design the next experiment.',
        idea: 'Stop when every t<sub>i</sub> = θ̂<sub>i</sub>/CI<sub>i</sub> &gt; t<sub>ref</sub> and, with rival models, one P<sub>m</sub> dominates.',
        out: '<code>modelling/</code>, <code>confidence/</code>', link: 'ug_results.html' }
    ];
    var rounds = [{ ci: 42, t: 0.9 }, { ci: 19, t: 1.6 }, { ci: 8.5, t: 3.4 }], tref = 1.98;

    var wrap = h('div', { 'class': 'mdw-loopgrid' }, root);
    var svg = svgBox(wrap, 360, 330, 'The MIDDoE study round');
    svg.classList.add('mdw-loop');
    var panel = h('div', { 'class': 'mdw-panel', 'aria-live': 'polite' }, wrap);
    var cx = 180, cy = 165, R = 120, n = steps.length;
    el('circle', { cx: cx, cy: cy, r: R, 'class': 'mdw-ring' }, svg);
    var ang = function (i) { return -Math.PI / 2 + i * 2 * Math.PI / n; };
    var P = function (a) { return [cx + R * Math.cos(a), cy + R * Math.sin(a)]; };
    for (var i = 0; i < n; i++) {
      var a = ang(i) + Math.PI / n, p = P(a);
      el('path', { d: 'M-5,-4.5 L3,0 L-5,4.5', 'class': 'mdw-arrow',
                   transform: 'translate(' + p[0].toFixed(1) + ',' + p[1].toFixed(1) + ') rotate(' + ((a + Math.PI / 2) * 180 / Math.PI).toFixed(1) + ')' }, svg);
    }
    var trail = el('path', { 'class': 'mdw-trail' }, svg);
    var dot = el('circle', { r: 5.5, 'class': 'mdw-dot' }, svg);
    var cur = 0, timer = null, round = 0;
    var nodes = steps.map(function (s, i) {
      var p = P(ang(i));
      var g = el('g', { 'class': 'mdw-node' + (s.opt === 'optional' ? ' mdw-opt' : ''), tabindex: 0, role: 'button',
                        'aria-label': 'Step ' + (i + 1) + ': ' + s.n, transform: 'translate(' + p[0].toFixed(1) + ',' + p[1].toFixed(1) + ')' }, svg);
      el('circle', { r: 31 }, g);
      txt(el('text', { y: 3.5, 'text-anchor': 'middle' }, g), s.n);
      txt(el('text', { y: -37, 'text-anchor': 'middle', 'class': 'mdw-num' }, g), String(i + 1));
      g.addEventListener('click', function () { stop(); go(i, true); });
      g.addEventListener('keydown', function (e) {
        var k = { Enter: 0, ' ': 0, ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
        if (k === undefined) return;
        e.preventDefault(); stop();
        var to = (i + k + n) % n; go(to, true); if (k) nodes[to].focus();
      });
      return g;
    });
    el('circle', { cx: cx, cy: cy, r: 48, 'class': 'mdw-gauge-bg' }, svg);
    var arc = el('path', { 'class': 'mdw-gauge' }, svg);
    var c1 = el('text', { x: cx, y: cy - 8, 'text-anchor': 'middle', 'class': 'mdw-centre' }, svg);
    var c2 = el('text', { x: cx, y: cy + 10, 'text-anchor': 'middle', 'class': 'mdw-centre-sm' }, svg);
    var c3 = el('text', { x: cx, y: cy + 25, 'text-anchor': 'middle', 'class': 'mdw-centre-sm' }, svg);
    var anim = channel(), ganim = channel(), gaugeFrac = 0;

    function gauge(frac) {
      var a0 = -Math.PI / 2, a1 = a0 + Math.max(1e-4, frac) * 2 * Math.PI * 0.9999, r = 48;
      arc.setAttribute('d', 'M' + (cx + r * Math.cos(a0)).toFixed(2) + ',' + (cy + r * Math.sin(a0)).toFixed(2) +
        ' A' + r + ',' + r + ' 0 ' + (frac > 0.5 ? 1 : 0) + ' 1 ' + (cx + r * Math.cos(a1)).toFixed(2) + ',' + (cy + r * Math.sin(a1)).toFixed(2));
    }
    function setRound(k) {
      var rd = rounds[k];
      txt(c1, 'round ' + (k + 1));
      txt(c2, 'CI ±' + rd.ci + ' %');
      txt(c3, 'min t ' + rd.t + (rd.t > tref ? ' ✓' : ' < t_ref'));
      c3.classList.toggle('mdw-good-t', rd.t > tref);
      ganim([gaugeFrac], [1 - rd.ci / 50], 700, function (v) { gaugeFrac = v[0]; gauge(gaugeFrac); });
    }
    function placeDot(a) { var p = P(a); set(dot, { cx: p[0].toFixed(1), cy: p[1].toFixed(1) }); }
    function show(html) { panel.classList.remove('mdw-in'); panel.innerHTML = html; void panel.offsetWidth; panel.classList.add('mdw-in'); }
    function render(i) {
      var s = steps[i];
      show('<div class="mdw-kicker">Step ' + (i + 1) + ' of ' + n + ' <span class="mdw-tag mdw-' + s.opt + '">' + s.opt + '</span></div>' +
        '<h4>' + s.n + '</h4><p>' + s.what + '</p>' +
        '<div class="mdw-kv"><span>Call</span><code>' + s.call + '</code></div>' +
        '<div class="mdw-kv"><span>Idea</span><div>' + s.idea + '</div></div>' +
        '<div class="mdw-kv"><span>Writes</span><div>' + s.out + '</div></div>' +
        '<a class="mdw-more" href="' + s.link + '">Read more →</a>');
    }
    function go(i, jump, done) {
      nodes.forEach(function (g, j) { g.classList.toggle('mdw-active', j === i); });
      var a0 = ang(cur), a1 = ang(i);
      if (a1 < a0 - 1e-9) a1 += 2 * Math.PI;
      if (jump) a0 = a1 - Math.min(a1 - a0, 2 * Math.PI / n);
      cur = i; render(i);
      anim([a0], [a1], jump ? 380 : 650, function (v) {
        placeDot(v[0]);
        var p0 = P(a0), p1 = P(v[0]);
        trail.setAttribute('d', v[0] - a0 > 1e-3 ? 'M' + p0[0].toFixed(1) + ',' + p0[1].toFixed(1) + ' A' + R + ',' + R + ' 0 0 1 ' + p1[0].toFixed(1) + ',' + p1[1].toFixed(1) : '');
      }, function () { trail.setAttribute('d', ''); if (done) done(); });
    }
    var ctl = bar(root);
    var playB = button(ctl, '▶&nbsp; Play three rounds', play, 'mdw-primary mdw-fixed');
    h('span', { 'class': 'mdw-hint' }, ctl, 'or click a step (arrow keys work too). Dashed steps are optional.');
    function stop() { if (timer !== null) { clearTimeout(timer); timer = null; } playB.innerHTML = '▶&nbsp; Play three rounds'; }
    function tick() {
      var nxt = (cur + 1) % n;
      if (nxt === 0) {
        if (round === rounds.length - 1) {
          stop();
          nodes.forEach(function (g) { g.classList.remove('mdw-active'); });
          show('<div class="mdw-kicker">Round ' + (round + 1) + ' done</div><h4>Report</h4>' +
            '<p>Every t-value is above t<sub>ref</sub> = ' + tref + ' and the intervals are narrow enough: stop designing, ' +
            'save the study with <code>save_to_jac(…)</code> and run <code>run_postprocessing(…)</code>.</p>' +
            '<a class="mdw-more" href="ug_results.html">Read the results →</a>');
          return;
        }
        round += 1; setRound(round);
      }
      go(nxt, false, function () { if (timer !== null) timer = setTimeout(tick, 1500); });
    }
    function play() {
      if (timer !== null) { stop(); return; }
      round = 0; setRound(0); cur = 0; go(0, true);
      playB.innerHTML = '■&nbsp; Stop';
      timer = setTimeout(tick, 1300);
    }
    placeDot(ang(0)); setRound(0); go(0, true);
  }

  // ======================================================================= 2. where to sample
  function samplingWidget(root) {
    // y = A(1 - e^{-kt}); scaled sensitivities S_A = A(1 - e^{-kt}), S_k = k A t e^{-kt}.
    // M = sum s s^T / sigma^2 in relative parameter units (as MIDDoE scales theta), V = M^-1,
    // 95 % joint region: d^T M d <= chi2_0.95(2). Semi-axes a_i = sqrt(chi2 / lambda_i(M)).
    var A = 1, k = 0.5, T1 = 10, TMIN = 0.2, GRID = 0.05, GAP = 0.5, Q = CHI2_95[2];
    var START = [1, 4, 8];
    var st = { t: START.slice(), sigma: 0.03, gap: true, crit: null, zoom: 30, hover: null };
    var W = 640, H = 214, m = { l: 46, r: 16, t: 24, b: 40 }, SH = 70;

    h('p', { 'class': 'mdw-lead' }, root, 'Drag a sampling point (●) along time, or focus it and use the arrow keys. Then let a criterion place them all, and watch how each one reshapes the confidence region.');
    var svg = svgBox(root, W, H + SH + 30, 'Response and sensitivities with draggable sampling times');
    var x = scale(0, T1, m.l, W - m.r), y = scale(0, 1.1, H - m.b, m.t);
    var sy0 = H + SH + 10, sy = scale(0, 1, sy0, H + 30);
    drawAxes(el('g', null, svg), x, y, { yt: [0, 0.5, 1], xl: 'time t', yl: 'response y = A(1 − e^(−kt))' });
    var curve = [], sA = [], sK = [], maxK = A * Math.exp(-1);
    for (var i = 0; i <= 240; i++) {
      var t = T1 * i / 240, e = Math.exp(-k * t);
      curve.push([x(t), y(A * (1 - e))]); sA.push([x(t), sy(1 - e)]); sK.push([x(t), sy(k * A * t * e / maxK)]);
    }
    el('path', { d: pathD(curve), 'class': 'mdw-l1' }, svg);
    txt(el('text', { x: m.l, y: H + 12, 'class': 'mdw-label' }, svg), 'information carried by one sample (scaled sensitivity, max = 1)');
    el('line', { x1: x(0), x2: x(T1), y1: sy0, y2: sy0, 'class': 'mdw-axis' }, svg);
    el('path', { d: pathD(sA), 'class': 'mdw-l2' }, svg);
    el('path', { d: pathD(sK), 'class': 'mdw-l3' }, svg);
    txt(el('text', { x: x(T1) - 2, y: sy(1) - 5, 'text-anchor': 'end', 'class': 'mdw-tick mdw-c2' }, svg), 'about A');
    txt(el('text', { x: x(6), y: sy(0.5 * 6 * Math.exp(-3) / maxK) - 7, 'text-anchor': 'middle', 'class': 'mdw-tick mdw-c3' }, svg), 'about k (peak at t = 1/k)');
    var layer = el('g', null, svg), drops = el('g', null, layer), hs = el('g', null, layer);
    crosshair(svg, layer, { x0: x(0), x1: x(T1), y0: sy0, y1: m.t }, x, function (v) {
      var e = Math.exp(-k * v);
      return 't = ' + v.toFixed(2) + '   y = ' + (A * (1 - e)).toFixed(3) + '   info: A ' + Math.round(100 * (1 - e)) + ' %, k ' + Math.round(100 * k * A * v * e / maxK) + ' %';
    });
    var bub = bubble(layer);

    // ---- confidence region: square plot, fixed scale chosen by the reader, clipped (not clamped)
    var row = h('div', { 'class': 'mdw-row' }, root);
    var ebox = h('div', { 'class': 'mdw-ellbox' }, row);
    var EW = 310, EH = 310, em = { l: 46, r: 16, t: 16, b: 46 };          // plot area 248 x 248
    var esvg = svgBox(ebox, EW, EH, '95 % joint confidence region of A and k');
    var ex = scale(-50, 50, em.l, EW - em.r), ey = scale(-50, 50, EH - em.b, em.t);
    var clipId = 'mdwclip' + Math.random().toString(36).slice(2, 8);
    var defs = el('defs', null, esvg), cp = el('clipPath', { id: clipId }, defs);
    el('rect', { x: em.l, y: em.t, width: EW - em.l - em.r, height: EH - em.t - em.b }, cp);
    var eAx = el('g', null, esvg);
    var gE = el('g', { 'clip-path': 'url(#' + clipId + ')', 'class': 'mdw-egeo' }, esvg);
    var refEll = el('path', { 'class': 'mdw-ellref' }, gE);
    var box = el('path', { 'class': 'mdw-box' }, gE);
    var ell = el('path', { 'class': 'mdw-ell' }, gE);
    var major = el('line', { 'class': 'mdw-major' }, gE), minor = el('line', { 'class': 'mdw-minor' }, gE);
    var centre = el('circle', { r: 3, 'class': 'mdw-true' }, esvg);
    var offView = el('text', { x: (EW + em.l - em.r) / 2, y: em.t + 14, 'text-anchor': 'middle', 'class': 'mdw-tick mdw-offview' }, esvg);
    var sing = el('g', { 'class': 'mdw-sing', opacity: 0 }, esvg);
    el('rect', { x: em.l + 8, y: EH / 2 - 40, width: EW - em.l - em.r - 16, height: 52, rx: 8 }, sing);
    txt(el('text', { x: (EW + em.l - em.r) / 2, y: EH / 2 - 18, 'text-anchor': 'middle' }, sing), 'One sampling time cannot fix');
    txt(el('text', { x: (EW + em.l - em.r) / 2, y: EH / 2, 'text-anchor': 'middle' }, sing), 'two parameters: M is singular.');
    var zbar = h('div', { 'class': 'mdw-bar mdw-tight' }, ebox);
    seg(zbar, [['15', '±15'], ['30', '±30'], ['60', '±60'], ['120', '±120']], String(st.zoom), function (v) {
      var from = [st.zoom]; st.zoom = +v; zanim(from, [st.zoom], 350, function (q) { setZoom(q[0]); drawEllipse(); });
    }, 'scale %');
    h('div', { 'class': 'mdw-legend mdw-small' }, ebox,
      '<span><i class="mdw-sw mdw-c1"></i>now</span><span><i class="mdw-sw mdw-dash1"></i>reference</span>' +
      '<span><i class="mdw-sw mdw-dot1"></i>projections</span>');

    var right = h('div', { 'class': 'mdw-col' }, row);
    var S = stats(right, [
      { key: 'ciA', label: '95 % CI of A' }, { key: 'ciK', label: '95 % CI of k' },
      { key: 'rho', label: 'correlation ρ' }, { key: 'axes', label: 'semi-axes a ≥ b' },
      { key: 'D', label: 'D · det M ↑', note: 'area π·a·b ∝ 1/√det M', wide: true },
      { key: 'A', label: 'A · tr M⁻¹ ↓', note: 'a² + b² = χ²·tr M⁻¹', wide: true },
      { key: 'E', label: 'E · λ<sub>min</sub>(M) ↑', note: 'longest semi-axis a = √(χ²/λ<sub>min</sub>)', wide: true },
      { key: 'ME', label: 'ME · cond M ↓', note: 'elongation a/b = √cond M', wide: true }
    ]);
    var note = h('p', { 'class': 'mdw-note' }, right);

    var ctl = bar(root);
    var bMinus = button(ctl, '−', function () { if (st.t.length > 1) { anim.cancel(); st.t.pop(); st.crit = null; build(); update(); } }, 'mdw-round', 'Remove a sample');
    var nLab = h('span', { 'class': 'mdw-count' }, ctl);
    var bPlus = button(ctl, '+', function () { var c = freeSpot(); if (c === null) return; anim.cancel(); st.t.push(c); st.crit = null; build(); update(); }, 'mdw-round', 'Add a sample');
    slider(ctl, 'σ', 0.005, 0.1, 0.005, st.sigma, function (v) { st.sigma = v; update(); }, function (v) { return v.toFixed(3); });
    toggle(ctl, 'minimum gap ' + GAP + ' (like <code>offt</code>)', st.gap, function (v) { anim.cancel(); st.gap = v; if (v) spread(); st.crit = null; update(); });
    var ctl2 = bar(root);
    h('span', { 'class': 'mdw-hint' }, ctl2, 'Place them by');
    var cb = {};
    ['D', 'A', 'E', 'ME'].forEach(function (c) {
      var b = cb[c] = button(ctl2, c + '-optimal', function () { optimise(c); }, 'mdw-primary');
      b.addEventListener('mouseenter', function () { st.hover = c; paintCrit(); });
      b.addEventListener('mouseleave', function () { st.hover = null; paintCrit(); });
      b.addEventListener('focus', function () { st.hover = c; paintCrit(); });
      b.addEventListener('blur', function () { st.hover = null; paintCrit(); });
    });
    button(ctl2, 'Reset', function () { anim.cancel(); st.t = START.slice(); st.crit = null; build(); refCov = covOf(st.t); update(); });
    button(ctl2, 'Keep as reference', function () { refCov = cov ? cov.slice() : null; update(); }, null, 'Use the current region as the dashed reference');

    var anim = channel(), zanim = channel(), handles = [], dropLines = [], cov = null, refCov = null;
    var CAPTION = {
      D: '<b>D</b> maximises det M: it shrinks the <b>area</b> of the region (its volume with more parameters), whatever its shape.',
      A: '<b>A</b> minimises tr M⁻¹, the sum of the variances: it shrinks the <b>box</b> around the region (the dotted projections), i.e. a² + b².',
      E: '<b>E</b> maximises the smallest eigenvalue of M: it shortens the <b>longest axis</b>, the worst-determined direction.',
      ME: '<b>ME</b> minimises the condition number of M: it makes the region as <b>round</b> as possible (a/b → 1), not necessarily small.'
    };

    function fim(ts) {
      var a = 0, b = 0, c = 0, w = 1 / (st.sigma * st.sigma);
      for (var i = 0; i < ts.length; i++) {
        var e = Math.exp(-k * ts[i]), SA = A * (1 - e), SK = k * A * ts[i] * e;
        a += SA * SA; b += SA * SK; c += SK * SK;
      }
      return [a * w, b * w, c * w];
    }
    function singular(M) { return M[0] * M[2] - M[1] * M[1] <= 1e-9 * (M[0] * M[2] + 1e-300); }
    function covOf(ts) { var M = fim(ts); if (singular(M)) return null; var d = M[0] * M[2] - M[1] * M[1]; return [M[2] / d, -M[1] / d, M[0] / d]; }
    function crit(ts, which) {
      var M = fim(ts), det = M[0] * M[2] - M[1] * M[1];
      if (singular(M)) return -Infinity;
      var ev = eig2(M[0], M[1], M[2]);
      if (which === 'D') return Math.log(det);
      if (which === 'A') return -(M[0] + M[2]) / det;
      if (which === 'E') return ev.l2;
      return -ev.l1 / ev.l2;                                            // ME
    }
    function feasible(t, j, ts) {
      if (t < TMIN - 1e-9 || t > T1 + 1e-9) return false;
      if (!st.gap) return true;
      for (var i = 0; i < ts.length; i++) if (i !== j && Math.abs(ts[i] - t) < GAP - 1e-9) return false;
      return true;
    }
    function nearestFeasible(want, j) {
      want = clamp(want, TMIN, T1);
      if (feasible(want, j, st.t)) return want;
      var cands = [TMIN, T1];
      st.t.forEach(function (u, i) { if (i !== j) cands.push(u - GAP, u + GAP); });
      var best = st.t[j], bd = Infinity;
      cands.forEach(function (c) { if (feasible(c, j, st.t) && Math.abs(c - want) < bd) { bd = Math.abs(c - want); best = c; } });
      return best;
    }
    function freeSpot() {
      for (var q = Math.round(T1 / GRID); q >= Math.round(TMIN / GRID); q--) { var t = q * GRID; if (feasible(t, -1, st.t)) return t; }
      return null;
    }
    function spread() {
      var ts = st.t.slice().sort(function (a, b) { return a - b; });
      for (var i = 1; i < ts.length; i++) if (ts[i] - ts[i - 1] < GAP) ts[i] = ts[i - 1] + GAP;
      var over = ts[ts.length - 1] - T1; if (over > 0) ts = ts.map(function (t) { return t - over; });
      st.t = ts;
    }
    function optimise(which) {
      var grid = []; for (var q = Math.round(TMIN / GRID); q <= Math.round(T1 / GRID); q++) grid.push(q * GRID);
      var r = rng(11), starts = [st.t.slice()];
      for (var s = 0; s < 10; s++) {
        var ts = [];
        for (var j = 0; j < st.t.length; j++)
          for (var tries = 0; tries < 300; tries++) { var c = grid[Math.floor(r.u() * grid.length)]; if (feasible(c, -1, ts)) { ts.push(c); break; } }
        if (ts.length === st.t.length) starts.push(ts);
      }
      var best = null, bv = -Infinity;
      starts.forEach(function (ts0) {
        var ts = ts0.slice(), v = crit(ts, which), it = 0, imp = true;
        while (imp && it++ < 60) {
          imp = false;
          for (var j = 0; j < ts.length; j++) {
            var keep = ts[j];
            for (var q = 0; q < grid.length; q++) {
              if (!feasible(grid[q], j, ts)) continue;
              ts[j] = grid[q]; var w = crit(ts, which);
              if (w > v + 1e-12 * Math.abs(v)) { v = w; keep = grid[q]; imp = true; }
            }
            ts[j] = keep;
          }
        }
        if (v > bv) { bv = v; best = ts.slice(); }
      });
      if (!best) return;
      var from = st.t.slice().sort(function (a, b) { return a - b; }), to = best.sort(function (a, b) { return a - b; });
      st.crit = which;
      anim(from, to, 900, function (v) { st.t = v; update(); }, function () { st.t = to; update(); });
    }
    function build() {
      clear(hs); clear(drops);
      handles = st.t.map(function (t, j) { return handle(hs, j, 'sampling time ' + (j + 1)); });
      dropLines = st.t.map(function () { return el('line', { 'class': 'mdw-drop' }, drops); });
      txt(nLab, st.t.length + (st.t.length === 1 ? ' sample' : ' samples'));
      bMinus.disabled = st.t.length <= 1; bPlus.disabled = st.t.length >= 8;
    }
    function setZoom(R) { ex.d0 = -R; ex.d1 = R; ey.d0 = -R; ey.d1 = R; }
    function geom(c) {                       // semi-axes (in %) and orientation of the 95 % region
      var ev = eig2(c[0], c[1], c[2]);
      return { a: Math.sqrt(Q * ev.l1) * 100, b: Math.sqrt(Q * Math.max(ev.l2, 0)) * 100, ang: ev.ang,
               bx: Math.sqrt(Q * c[0]) * 100, by: Math.sqrt(Q * c[2]) * 100 };
    }
    function ellipsePath(g) {
      var ca = Math.cos(g.ang), sa = Math.sin(g.ang), pts = [];
      for (var i = 0; i <= 120; i++) {
        var th = 2 * Math.PI * i / 120, u = g.a * Math.cos(th), w = g.b * Math.sin(th);
        pts.push([ex(u * ca - w * sa), ey(u * sa + w * ca)]);
      }
      return pathD(pts) + 'Z';
    }
    function drawEllipse() {
      var R = ex.d1;
      drawAxes(eAx, ex, ey, { xt: [-R, -R / 2, 0, R / 2, R], yt: [-R, -R / 2, 0, R / 2, R], zero: true,
                              xl: 'error in A [%]', yl: 'error in k [%]' });
      set(centre, { cx: ex(0), cy: ey(0) });
      refEll.setAttribute('d', refCov ? ellipsePath(geom(refCov)) : '');
      if (!cov) { ell.setAttribute('d', ''); box.setAttribute('d', ''); set(major, { x1: 0, x2: 0, y1: 0, y2: 0 }); set(minor, { x1: 0, x2: 0, y1: 0, y2: 0 }); txt(offView, ''); return; }
      var g = geom(cov), ca = Math.cos(g.ang), sa = Math.sin(g.ang);
      ell.setAttribute('d', ellipsePath(g));
      box.setAttribute('d', 'M' + ex(-g.bx) + ',' + ey(-g.by) + 'H' + ex(g.bx) + 'V' + ey(g.by) + 'H' + ex(-g.bx) + 'Z');
      set(major, { x1: ex(-g.a * ca), y1: ey(-g.a * sa), x2: ex(g.a * ca), y2: ey(g.a * sa) });
      set(minor, { x1: ex(g.b * sa), y1: ey(-g.b * ca), x2: ex(-g.b * sa), y2: ey(g.b * ca) });
      txt(offView, Math.max(g.bx, g.by) > R ? 'region extends beyond this scale: zoom out' : '');
    }
    function paintCrit() { gE.setAttribute('data-crit', st.hover || st.crit || ''); var c = st.hover || st.crit; note.innerHTML = c ? CAPTION[c] : rhoNote; }
    var rhoNote = '';
    function update() {
      st.t.forEach(function (t, j) {
        var yy = y(A * (1 - Math.exp(-k * t)));
        handles[j].at(x(t), yy);
        set(dropLines[j], { x1: x(t).toFixed(1), x2: x(t).toFixed(1), y1: (yy + 7).toFixed(1), y2: sy0 });
      });
      Object.keys(cb).forEach(function (c) { cb[c].classList.toggle('md-on', st.crit === c); });
      var M = fim(st.t);
      cov = covOf(st.t);
      if (!cov) {
        sing.setAttribute('opacity', 1);
        ['ciA', 'ciK', 'rho', 'axes', 'D', 'A', 'E', 'ME'].forEach(function (kk) { S.set(kk, '–'); });
        rhoNote = 'Add a second sampling time at a different moment.';
        drawEllipse(); paintCrit(); return;
      }
      sing.setAttribute('opacity', 0);
      drawEllipse();
      var det = M[0] * M[2] - M[1] * M[1], ev = eig2(M[0], M[1], M[2]), g = geom(cov);
      var rho = cov[1] / Math.sqrt(cov[0] * cov[2]);
      S.set('ciA', '± ' + fmt(1.96 * Math.sqrt(cov[0]) * 100, 1) + ' %');
      S.set('ciK', '± ' + fmt(1.96 * Math.sqrt(cov[2]) * 100, 1) + ' %');
      S.set('rho', fmt(rho, 2), Math.abs(rho) > 0.9);
      S.set('axes', fmt(g.a, 1) + ' %, ' + fmt(g.b, 1) + ' %');
      S.set('D', fmt(det, 3) + ' <span class="mdw-mute">· area ' + fmt(Math.PI * g.a * g.b, 0) + ' %²</span>', st.crit === 'D');
      S.set('A', fmt(cov[0] + cov[2], 4) + ' <span class="mdw-mute">· a² + b² = ' + fmt(g.a * g.a + g.b * g.b, 0) + ' %²</span>', st.crit === 'A');
      S.set('E', fmt(ev.l2, 3) + ' <span class="mdw-mute">· a = ' + fmt(g.a, 1) + ' %</span>', st.crit === 'E');
      S.set('ME', fmt(ev.l1 / ev.l2, 1) + ' <span class="mdw-mute">· a/b = ' + fmt(g.a / g.b, 2) + '</span>', st.crit === 'ME');
      rhoNote = Math.abs(rho) > 0.9
        ? '<span class="mdw-bad">|ρ| &gt; 0.9:</span> the data fix a combination of A and k, not each one (a thin, tilted region).'
        : 'Hover a criterion button to see which feature of the region it acts on.';
      paintCrit();
    }
    var at = function (j) { return [x(st.t[j]), y(A * (1 - Math.exp(-k * st.t[j])))]; };
    interact(svg, {
      start: function () { anim.cancel(); st.crit = null; },
      drag: function (id, p) { var j = +id; st.t[j] = nearestFeasible(x.inv(p.x), j); update(); var q = at(j); bub.show(q[0], q[1], 't = ' + st.t[j].toFixed(2)); },
      drop: function (id) { var j = +id, s = Math.round(st.t[j] / GRID) * GRID; if (feasible(s, j, st.t)) st.t[j] = s; update(); bub.hide(); },
      key: function (id, d, big) {
        var j = +id; anim.cancel(); st.crit = null;
        st.t[j] = nearestFeasible(Math.round((st.t[j] + d * (big ? 0.5 : GRID)) / GRID) * GRID, j); update();
        var q = at(j); bub.show(q[0], q[1], 't = ' + st.t[j].toFixed(2));
      },
      focus: function (id) { var q = at(+id); bub.show(q[0], q[1], 't = ' + st.t[+id].toFixed(2)); },
      blur: function () { bub.hide(); }
    });
    setZoom(st.zoom); build(); refCov = covOf(st.t); update();
  }

  // ======================================================================= 3. rival models
  function discrimWidget(root) {
    var k1 = 0.5, T1 = 10, TMIN = 0.2, GRID = 0.1, GAP = 0.5, prelim = [1, 2, 4, 8];
    var k2 = (function () {
      var lo = 0.05, hi = 5, g = (Math.sqrt(5) - 1) / 2;
      var sse = function (kk) { return prelim.reduce(function (s, t) { var d = (1 - Math.exp(-k1 * t)) - kk * t / (1 + kk * t); return s + d * d; }, 0); };
      for (var i = 0; i < 90; i++) { var a = hi - g * (hi - lo), b = lo + g * (hi - lo); if (sse(a) < sse(b)) hi = b; else lo = a; }
      return (lo + hi) / 2;
    })();
    function x1(t) { return 1 - Math.exp(-k1 * t); }
    function x2(t, c0) { return c0 * k2 * t / (1 + c0 * k2 * t); }
    var st = { c0: 1, t: [1, 3, 6, 9], sigma: 0.03 };

    h('p', { 'class': 'mdw-lead' }, root, 'Both models fit the first experiment (■, c<sub>0</sub> = 1). Change the design until they disagree by more than the noise.');
    var W = 640, H = 250, m = { l: 46, r: 16, t: 24, b: 38 };
    var svg = svgBox(root, W, H + 62, 'Predictions of two rival models and the sampling times');
    var x = scale(0, T1, m.l, W - m.r), y = scale(0, 1.05, H - m.b, m.t);
    drawAxes(el('g', null, svg), x, y, { yt: [0, 0.25, 0.5, 0.75, 1], xl: 'time t [h]', yl: 'conversion x' });
    var band = el('path', { 'class': 'mdw-bandfill' }, svg), ghost = el('path', { 'class': 'mdw-ghost' }, svg);
    var p1 = [];
    for (var i = 0; i <= 200; i++) p1.push([x(T1 * i / 200), y(x1(T1 * i / 200))]);
    el('path', { d: pathD(p1), 'class': 'mdw-l1' }, svg);
    var m2 = el('path', { 'class': 'mdw-l2' }, svg);
    prelim.forEach(function (t) { el('rect', { x: x(t) - 3.5, y: y(x1(t)) - 3.5, width: 7, height: 7, 'class': 'mdw-prelim' }, svg); });
    var layer = el('g', null, svg), gaps = el('g', null, layer), rugY = y(0) + 44;
    el('line', { x1: x(0), x2: x(T1), y1: rugY, y2: rugY, 'class': 'mdw-rug' }, layer);
    txt(el('text', { x: x(0), y: rugY + 20, 'class': 'mdw-tick' }, layer), 'sampling times (drag ● or use the arrow keys)');
    var hs = el('g', null, layer);
    var gapLines = st.t.map(function () { return el('line', { 'class': 'mdw-gapline' }, gaps); });
    var handles = st.t.map(function (t, j) { return handle(hs, j, 'sampling time ' + (j + 1)); });
    crosshair(svg, layer, { x0: x(0), x1: x(T1), y0: y(0), y1: m.t }, x, function (v) {
      var d = Math.abs(x1(v) - x2(v, st.c0));
      return 't = ' + v.toFixed(1) + ' h   x₁ = ' + x1(v).toFixed(3) + '   x₂ = ' + x2(v, st.c0).toFixed(3) + '   |Δ|/σ = ' + (d / st.sigma).toFixed(1);
    });
    var bub = bubble(layer);
    h('div', { 'class': 'mdw-legend' }, root,
      '<span><i class="mdw-sw mdw-c1"></i>model 1, first order: x = 1 − e<sup>−k₁t</sup></span>' +
      '<span><i class="mdw-sw mdw-c2"></i>model 2, second order: x = c₀k₂t / (1 + c₀k₂t)</span>' +
      '<span><i class="mdw-sw mdw-band"></i>±2σ around model 1</span>' +
      '<span><i class="mdw-sw mdw-dash"></i>model 2 at c₀ = 1</span>');
    var S = stats(root, [{ key: 'hr', label: 'HR = Σ(x₁ − x₂)²/σ²' }, { key: 'n', label: 'samples n' }]);
    var Mt = meter(root, 'expected χ² of the wrong model (n + HR) against its 95 % limit (the tick)');
    var ctl = bar(root);
    var cS = slider(ctl, 'initial concentration c₀', 0.2, 3, 0.05, st.c0, function (v) { anim.cancel(); st.c0 = v; update(); }, function (v) { return v.toFixed(2) + ' mol/L'; });
    slider(ctl, 'σ', 0.005, 0.08, 0.005, st.sigma, function (v) { st.sigma = v; update(); }, function (v) { return v.toFixed(3); });
    var ctl2 = bar(root);
    button(ctl2, 'Design to discriminate (max HR)', optimise, 'mdw-primary');
    button(ctl2, 'Back to the first experiment', function () {
      var from = [st.c0].concat(st.t.slice().sort(function (a, b) { return a - b; })), to = [1, 1, 3, 6, 9];
      anim(from, to, 600, function (v) { st.c0 = v[0]; st.t = v.slice(1); cS.set(st.c0); update(); });
    });
    var anim = channel();

    function HR(c0, ts) { var s = 0; ts.forEach(function (t) { var d = x1(t) - x2(t, c0); s += d * d; }); return s / (st.sigma * st.sigma); }
    function ok(t, j) { return t >= TMIN - 1e-9 && t <= T1 + 1e-9 && !st.t.some(function (u, i) { return i !== j && Math.abs(u - t) < GAP - 1e-9; }); }
    function nearest(want, j) {
      want = clamp(want, TMIN, T1); if (ok(want, j)) return want;
      var c = [TMIN, T1]; st.t.forEach(function (u, i) { if (i !== j) c.push(u - GAP, u + GAP); });
      var best = st.t[j], bd = Infinity;
      c.forEach(function (v) { if (ok(v, j) && Math.abs(v - want) < bd) { bd = Math.abs(v - want); best = v; } });
      return best;
    }
    function optimise() {
      var best = null, bv = -1;
      for (var ci = 4; ci <= 60; ci++) {
        var c = ci * 0.05, cand = [];
        for (var q = Math.round(TMIN / GRID); q <= Math.round(T1 / GRID); q++) { var t = q * GRID; cand.push([t, Math.abs(x1(t) - x2(t, c))]); }
        cand.sort(function (a, b) { return b[1] - a[1]; });
        var ts = [];
        for (var i = 0; i < cand.length && ts.length < st.t.length; i++)
          if (ts.every(function (u) { return Math.abs(u - cand[i][0]) >= GAP - 1e-9; })) ts.push(cand[i][0]);
        var v = HR(c, ts); if (v > bv) { bv = v; best = [c, ts.sort(function (a, b) { return a - b; })]; }
      }
      var from = [st.c0].concat(st.t.slice().sort(function (a, b) { return a - b; }));
      anim(from, [best[0]].concat(best[1]), 800, function (v) { st.c0 = v[0]; st.t = v.slice(1); cS.set(st.c0); update(); });
    }
    function update() {
      var pm = [], up = [], dn = [], gh = [];
      for (var i = 0; i <= 200; i++) {
        var t = T1 * i / 200;
        pm.push([x(t), y(x2(t, st.c0))]); gh.push([x(t), y(x2(t, 1))]);
        up.push([x(t), y(Math.min(1.05, x1(t) + 2 * st.sigma))]); dn.push([x(t), y(Math.max(0, x1(t) - 2 * st.sigma))]);
      }
      band.setAttribute('d', pathD(up.concat(dn.reverse())) + 'Z');
      m2.setAttribute('d', pathD(pm));
      set(ghost, { d: pathD(gh), opacity: Math.abs(st.c0 - 1) > 0.02 ? 0.55 : 0 });
      st.t.forEach(function (t, j) {
        handles[j].at(x(t), rugY);
        set(gapLines[j], { x1: x(t).toFixed(1), x2: x(t).toFixed(1), y1: y(x1(t)).toFixed(1), y2: y(x2(t, st.c0)).toFixed(1) });
      });
      var n = st.t.length, hr = HR(st.c0, st.t), lim = CHI2_95[n], val = n + hr;
      S.set('hr', fmt(hr, 1)); S.set('n', String(n));
      Mt(val, lim, Math.max(lim * 3, val * 1.05),
        (val > lim ? '<span class="mdw-good">✓ expected to reject the wrong model</span>' : '<span class="mdw-bad">✗ the models cannot be told apart yet</span>') +
        ' &nbsp;<span class="mdw-mute">n + HR = ' + fmt(val, 1) + ' vs χ²<sub>0.95</sub>(' + n + ') = ' + lim.toFixed(2) + ' · if model 1 is true, refitting ignored</span>', val > lim);
    }
    interact(svg, {
      start: function () { anim.cancel(); },
      drag: function (id, p) { var j = +id; st.t[j] = nearest(x.inv(p.x), j); update(); bub.show(x(st.t[j]), rugY, 't = ' + st.t[j].toFixed(1) + ' h'); },
      drop: function (id) { var j = +id, s = Math.round(st.t[j] / GRID) * GRID; if (ok(s, j)) st.t[j] = s; update(); bub.hide(); },
      key: function (id, d, big) { var j = +id; anim.cancel(); st.t[j] = nearest(Math.round((st.t[j] + d * (big ? 1 : GRID)) / GRID) * GRID, j); update(); bub.show(x(st.t[j]), rugY, 't = ' + st.t[j].toFixed(1) + ' h'); },
      focus: function (id) { var j = +id; bub.show(x(st.t[j]), rugY, 't = ' + st.t[j].toFixed(1) + ' h'); },
      blur: function () { bub.hide(); }
    });
    update();
  }

  // ======================================================================= 4. measurement error
  function noiseWidget(root) {
    var N = 21, T1 = 10;
    var st = { errt: 'rel', unc: 0.05, lfloor: -6, seed: 3 };
    var f = function (t) { return 1 - Math.exp(-0.5 * t); };
    var ts = []; for (var i = 0; i < N; i++) ts.push(T1 * i / (N - 1));
    var z = (function () { var r = rng(st.seed), o = []; for (var i = 0; i < N; i++) o.push(r.n()); return o; })();
    var W = 640, H = 224, m = { l: 46, r: 16, t: 24, b: 36 }, WH = 64;
    var anim = channel();

    var topBar = bar(root);
    seg(topBar, [['abs', "'abs'"], ['rel', "'rel'"]], st.errt, function (v) {
      var from = [st.unc], to = [v === 'abs' ? 0.03 : 0.05];
      st.errt = v;
      uS.range(v === 'abs' ? 0.005 : 0.01, v === 'abs' ? 0.1 : 0.2, 0.005);
      flBox.classList.toggle('mdw-dim', v === 'abs');
      anim(from, to, 350, function (q) { st.unc = q[0]; uS.set(st.unc); update(); });
    }, 'errt');
    var svg = svgBox(root, W, H + WH + 26, 'Measurements with absolute or relative noise and their weights');
    var x = scale(0, T1, m.l, W - m.r), y = scale(-0.15, 1.25, H - m.b, m.t);
    drawAxes(el('g', null, svg), x, y, { yt: [0, 0.5, 1], xl: 'time t', yl: 'y' });
    var band = el('path', { 'class': 'mdw-bandfill' }, svg);
    var cpts = []; for (i = 0; i <= 200; i++) cpts.push([x(T1 * i / 200), y(f(T1 * i / 200))]);
    el('path', { d: pathD(cpts), 'class': 'mdw-l1' }, svg);
    var obs = ts.map(function () { return el('circle', { r: 3.6, 'class': 'mdw-obs' }, svg); });
    var wy0 = H + WH + 6, wy = scale(0, 1, wy0, H + 14);
    txt(el('text', { x: m.l, y: H + 4, 'class': 'mdw-label' }, svg), 'weight of each point in the fit, 1/σ² (largest = full height; hover a bar)');
    el('line', { x1: x(0), x2: x(T1), y1: wy0, y2: wy0, 'class': 'mdw-axis' }, svg);
    var bars = ts.map(function () { var b = el('rect', { width: 12, rx: 2, 'class': 'mdw-wbar' }, svg); b.ttl = el('title', null, b); return b; });
    var Mt = meter(root, 'share of the total weight carried by the point at t = 0, where y = 0 (the tick is an even share)');
    var read = h('p', { 'class': 'mdw-note' }, root);
    var ctl = bar(root);
    var uS = slider(ctl, 'unc', 0.01, 0.2, 0.005, st.unc, function (v) { st.unc = v; update(); },
      function (v) { return st.errt === 'abs' ? v.toFixed(3) : (v * 100).toFixed(1) + ' %'; });
    var flBox = h('span', { 'class': 'mdw-inline' }, ctl);
    slider(flBox, 'sigma_floor', -6, -1, 0.25, st.lfloor, function (v) { st.lfloor = v; update(); }, function (v) { return '1e' + v; });
    button(ctl, 'New noise', function () {
      st.seed += 1; var r = rng(st.seed), to = z.map(function () { return r.n(); });
      anim(z.slice(), to, 450, function (v) { z = v; update(); });
    });
    function sig(v) { return st.errt === 'abs' ? st.unc : Math.max(st.unc * Math.abs(v), Math.pow(10, st.lfloor)); }
    function update() {
      var up = [], dn = [];
      for (var i = 0; i <= 200; i++) {
        var t = T1 * i / 200, v = f(t), s = sig(v);
        up.push([x(t), y(Math.min(1.25, v + 2 * s))]); dn.push([x(t), y(Math.max(-0.15, v - 2 * s))]);
      }
      band.setAttribute('d', pathD(up.concat(dn.reverse())) + 'Z');
      var ws = ts.map(function (t) { var s = sig(f(t)); return 1 / (s * s); });
      var wmax = Math.max.apply(null, ws), tot = ws.reduce(function (a, b) { return a + b; }, 0);
      ts.forEach(function (t, i) {
        var s = sig(f(t)), o = f(t) + s * z[i];
        set(obs[i], { cx: x(t).toFixed(1), cy: y(clamp(o, -0.15, 1.25)).toFixed(1) });
        var hh = Math.max(0.6, wy0 - wy(ws[i] / wmax));
        set(bars[i], { x: (x(t) - 6).toFixed(1), y: (wy0 - hh).toFixed(1), height: hh.toFixed(1) });
        txt(bars[i].ttl, 't = ' + t.toFixed(1) + ': σ = ' + fmt(s, 2) + ', ' + (100 * ws[i] / tot).toFixed(1) + ' % of the weight');
      });
      var share = 100 * ws[0] / tot;
      Mt(share, 100 / N, 100, fmt(share, share < 10 ? 1 : 0) + ' % &nbsp;<span class="mdw-mute">(even share ' + (100 / N).toFixed(1) + ' %)</span>', share < 3 * 100 / N);
      var msg;
      if (st.errt === 'abs') msg = 'Absolute error: σ = <code>unc</code> everywhere, so every point counts the same.';
      else if (share > 50) msg = '<span class="mdw-bad">The point at y = 0 has σ = sigma_floor and dominates the fit.</span> Set the floor to about the smallest σ you can really measure: <code>insilicos[\'sigma_floor\']</code> for generated <code>MES_E</code>, <code>system[\'sigma_floor\']</code> for the design.';
      else msg = 'Relative error: σ = unc·|y|, floored at <code>sigma_floor</code>. Small values are trusted more; the floor limits that near zero.';
      if (read.innerHTML !== msg) read.innerHTML = msg;
    }
    update();
  }

  // ======================================================================= 5. design space of `system`
  function designspaceWidget(root) {
    var TF = 16, TR = 0.1, U = { min: 296.15, max: 306.15 };
    var cfg = { stps: 5, cvp: 'CPF', cst: 'inc', offl: 0.5, offt: 1.0, td: 0.5, sp: 9, sofft: 0.5, forced: true };
    var des = null, r = rng(5), anim = channel();
    var W = 660, m = { l: 52, r: 16 }, H1 = 210, top1 = 30;

    h('p', { 'class': 'mdw-lead' }, root, 'Set the keys and drag the levels (▬), switching times (◆) and samples (●), or focus one and use the arrow keys. Every move obeys the rules MIDDoE\'s design optimiser obeys; when a rule stops you, it is named.');
    var grid = h('div', { 'class': 'mdw-dsgrid' }, root);
    var left = h('div', { 'class': 'mdw-dsmain' }, grid);
    var sTop = top1 + H1 + 6, sy = sTop + 40;
    var svg = svgBox(left, W, sy + 36, 'Design space of the system dictionary');
    var x = scale(0, TF, m.l, W - m.r), y = scale(U.min - 0.6, U.max + 0.6, top1 + H1 - 34, top1 + 14);
    var dz = el('g', null, svg);
    var bandU = el('rect', { 'class': 'mdw-range' }, svg);
    drawAxes(el('g', null, svg), x, y, { yt: [296, 298, 300, 302, 304, 306], xt: [0, 2, 4, 6, 8, 10, 12, 14, 16], xl: 'time [h]', yl: "u1(t) [K] · tvi['u1']" });
    var swLines = el('g', null, svg), gapMarks = el('g', null, svg);
    var prof = el('path', { 'class': 'mdw-l1 mdw-thick' }, svg);
    var layer = el('g', null, svg);
    txt(el('text', { x: m.l - 6, y: sTop + 8, 'class': 'mdw-label' }, svg), "samples of y1 · tvo['y1']");
    el('line', { x1: x(0), x2: x(TF), y1: sy, y2: sy, 'class': 'mdw-rug' }, svg);
    var sGaps = el('g', null, svg), sLayer = el('g', null, svg), bub = bubble(svg);
    var say = status(left);

    var g1 = h('fieldset', { 'class': 'mdw-group' }, root); h('legend', null, g1, "<code>system['tvi']['u1']</code> · the input profile");
    var c1 = bar(g1);
    slider(c1, 'stps', 2, 8, 1, cfg.stps, function (v) { cfg.stps = v; fresh(); }, function (v) { return v + ' segments'; });
    seg(c1, [['CPF', 'CPF'], ['LPF', 'LPF']], cfg.cvp, function (v) { cfg.cvp = v; render(); }, 'cvp');
    seg(c1, [['inc', 'inc'], ['dec', 'dec'], ['rel', 'rel']], cfg.cst, function (v) { cfg.cst = v; fresh(); }, 'const');
    var c1b = bar(g1);
    slider(c1b, 'offt', 0.2, 3, 0.1, cfg.offt, function (v) { cfg.offt = v; fresh(); }, function (v) { return v.toFixed(1) + ' h'; });
    slider(c1b, 'offl', 0, 2, 0.1, cfg.offl, function (v) { cfg.offl = v; fresh(); }, function (v) { return v.toFixed(1) + ' K'; });
    var g2 = h('fieldset', { 'class': 'mdw-group' }, root); h('legend', null, g2, "<code>system['tvo']['y1']</code> · sampling, and <code>t_d</code>");
    var c2 = bar(g2);
    slider(c2, 'sp', 2, 20, 1, cfg.sp, function (v) { cfg.sp = v; fresh(); }, function (v) { return v + ' samples'; });
    slider(c2, 'offt', 0.1, 2, 0.1, cfg.sofft, function (v) { cfg.sofft = v; fresh(); }, function (v) { return v.toFixed(1) + ' h'; });
    var c2b = bar(g2);
    toggle(c2b, '<code>samp_f</code> = [0, 16]', cfg.forced, function (v) { cfg.forced = v; fresh(); });
    slider(c2b, 't_d', 0, 3, 0.1, cfg.td, function (v) { cfg.td = v; fresh(); }, function (v) { return v.toFixed(1) + ' h'; });
    var c3 = bar(root);
    var bRnd = button(c3, '⟳&nbsp; Another feasible design', function () { fresh(true); }, 'mdw-primary');
    h('span', { 'class': 'mdw-hint' }, c3, 'drawn at random from the design space');
    var side = h('div', { 'class': 'mdw-dsside' }, root);
    var chip = h('div', { 'class': 'mdw-chip' }, side);
    var code = h('pre', { 'class': 'mdw-code' }, side);

    var gapS = function () { return Math.max(cfg.sofft, TR); };
    var snapT = function (t) { return Math.round(t / TR) * TR; };
    var lo = function () { return cfg.td; }, hi = function () { return TF - cfg.td; };
    var forcedList = function () { return cfg.forced ? [0, TF] : []; };
    var nFree = function () { return cfg.sp - forcedList().length; };
    function sampleWindow() {
      var a = lo(), b = hi(), g = gapS();
      if (cfg.forced) { a = Math.max(a, g); b = Math.min(b, TF - g); }
      return [a, b];
    }
    function feasibility() {
      var msgs = [], ns = cfg.stps - 1;
      if (ns > 1 && (ns - 1) * cfg.offt > hi() - lo() + 1e-9)
        msgs.push(ns + ' switching times ' + cfg.offt.toFixed(1) + ' h apart do not fit between t_d and t_f − t_d');
      if (cfg.cst !== 'rel' && (cfg.stps - 1) * cfg.offl > U.max - U.min + 1e-9)
        msgs.push(cfg.stps + ' levels changing by at least offl = ' + cfg.offl.toFixed(1) + ' K do not fit in [min, max]');
      var nf = nFree(), g = gapS(), w = sampleWindow();
      if (nf < 0) msgs.push('sp is smaller than the number of forced samples');
      else if (nf > 0 && (nf - 1) * g > w[1] - w[0] + 1e-9) msgs.push(nf + ' free samples ' + g.toFixed(1) + ' h apart do not fit in the sampling window');
      return msgs;
    }
    function randomDesign() {
      var d = { lev: [], sw: [], smp: [] }, ns = cfg.stps - 1, i;
      if (cfg.cst === 'rel') for (i = 0; i < cfg.stps; i++) d.lev.push(U.min + r.u() * (U.max - U.min));
      else {
        var R = U.max - U.min - (cfg.stps - 1) * cfg.offl, c = sortedUniform(cfg.stps, r);
        for (i = 0; i < cfg.stps; i++) d.lev.push(U.min + i * cfg.offl + R * c[i]);
        if (cfg.cst === 'dec') d.lev.reverse();
      }
      var Rs = hi() - lo() - Math.max(0, ns - 1) * cfg.offt, cs = sortedUniform(ns, r);
      for (i = 0; i < ns; i++) d.sw.push(snapT(lo() + i * cfg.offt + Rs * cs[i]));
      var g = gapS(), w = sampleWindow(), nf = nFree(), Rq = w[1] - w[0] - Math.max(0, nf - 1) * g, cq = sortedUniform(Math.max(nf, 0), r);
      for (i = 0; i < nf; i++) d.smp.push(snapT(w[0] + i * g + Rq * cq[i]));
      return d;
    }
    function fresh(animate) {
      var msgs = feasibility();
      bRnd.disabled = msgs.length > 0;
      if (msgs.length) {
        say('<b>No feasible design:</b> ' + msgs.join('; ') + '. MIDDoE would stop with "found no feasible design".', 'bad', true);
        des = null; render(); return;
      }
      say('');
      var nd = randomDesign();
      if (animate && des && des.lev.length === nd.lev.length && des.sw.length === nd.sw.length && des.smp.length === nd.smp.length) {
        var from = des.lev.concat(des.sw, des.smp), to = nd.lev.concat(nd.sw, nd.smp), L = nd.lev.length, Sw = nd.sw.length;
        anim(from, to, 600, function (v) { des = { lev: v.slice(0, L), sw: v.slice(L, L + Sw), smp: v.slice(L + Sw) }; render(); }, function () { des = nd; render(); });
      } else { anim.cancel(); des = nd; render(); }
    }
    // clamp a move and name the rule that held it
    function holdLevel(i, want) {
      var a = U.min, b = U.max, rule = null;
      if (cfg.cst === 'inc') {
        if (i > 0) a = Math.max(a, des.lev[i - 1] + cfg.offl);
        if (i < des.lev.length - 1) b = Math.min(b, des.lev[i + 1] - cfg.offl);
      } else if (cfg.cst === 'dec') {
        if (i > 0) b = Math.min(b, des.lev[i - 1] - cfg.offl);
        if (i < des.lev.length - 1) a = Math.max(a, des.lev[i + 1] + cfg.offl);
      }
      var v = clamp(want, a, b);
      if (Math.abs(v - want) > 1e-6) {
        rule = (v <= U.min + 1e-9 || v >= U.max - 1e-9) ? "the input range <code>min</code>–<code>max</code>"
          : "<code>const = '" + cfg.cst + "'</code>: consecutive levels differ by at least <code>offl</code> = " + cfg.offl.toFixed(1) + ' K';
      }
      return [v, rule];
    }
    function holdSwitch(i, want) {
      var a = lo(), b = hi();
      if (i > 0) a = Math.max(a, des.sw[i - 1] + cfg.offt);
      if (i < des.sw.length - 1) b = Math.min(b, des.sw[i + 1] - cfg.offt);
      var v = clamp(want, a, b), rule = null;
      if (Math.abs(v - want) > 1e-6) rule = (v <= lo() + 1e-9 || v >= hi() - 1e-9) ? 'the dead time <code>t_d</code> = ' + cfg.td.toFixed(1) + ' h'
        : '<code>offt</code> = ' + cfg.offt.toFixed(1) + ' h between switching times';
      return [v, rule];
    }
    function holdSample(i, want) {
      var g = gapS(), others = des.smp.filter(function (u, j) { return j !== i; }).concat(forcedList()), a = lo(), b = hi();
      var okp = function (t) { return t >= a - 1e-9 && t <= b + 1e-9 && others.every(function (u) { return Math.abs(u - t) >= g - 1e-9; }); };
      var c = clamp(want, a, b);
      if (okp(c)) return [c, Math.abs(c - want) > 1e-6 ? 'the dead time <code>t_d</code> = ' + cfg.td.toFixed(1) + ' h' : null];
      var cands = [a, b]; others.forEach(function (u) { cands.push(u - g, u + g); });
      var best = des.smp[i], bd = Infinity;
      cands.forEach(function (t) { if (okp(t) && Math.abs(t - want) < bd) { bd = Math.abs(t - want); best = t; } });
      return [best, 'the sampling gap max(<code>offt</code>, <code>t_r</code>) = ' + g.toFixed(1) + ' h'];
    }
    var lvH = [], swH = [], smH = [], fH = [];
    function ensure(arr, n, mk) {
      while (arr.length > n) { var e = arr.pop(); e.parentNode.removeChild(e); }
      while (arr.length < n) arr.push(mk(arr.length));
    }
    function nodesOf() {
      return { ts: [0].concat(des.sw, [TF]), ls: des.lev.concat([des.lev[des.lev.length - 1]]) };
    }
    function levelX(j, P) { return cfg.cvp === 'CPF' ? (P.ts[j] + P.ts[j + 1]) / 2 : P.ts[j]; }
    function render() {
      clear(dz);
      if (cfg.td > 0) [[0, cfg.td], [TF - cfg.td, TF]].forEach(function (zz) {
        el('rect', { x: x(zz[0]), y: top1 - 10, width: x(zz[1]) - x(zz[0]), height: sy + 16 - top1, 'class': 'mdw-dead' }, dz);
      });
      if (cfg.td > 0.35) txt(el('text', { x: x(cfg.td / 2), y: top1 - 14, 'text-anchor': 'middle', 'class': 'mdw-tick' }, dz), 't_d');
      set(bandU, { x: x(0), width: x(TF) - x(0), y: y(U.max), height: y(U.min) - y(U.max) });
      clear(swLines); clear(gapMarks); clear(sGaps);
      if (!des) {
        prof.setAttribute('d', '');
        ensure(lvH, 0); ensure(swH, 0); ensure(smH, 0); ensure(fH, 0);
        chip.innerHTML = '<b>No feasible design</b> with these settings.';
        writeCode(); return;
      }
      var P = nodesOf(), pts = [];
      if (cfg.cvp === 'CPF') for (var i = 0; i < P.ts.length - 1; i++) { pts.push([x(P.ts[i]), y(P.ls[i])]); pts.push([x(P.ts[i + 1]), y(P.ls[i])]); }
      else for (i = 0; i < P.ts.length; i++) pts.push([x(P.ts[i]), y(P.ls[i])]);
      prof.setAttribute('d', pathD(pts));
      des.sw.forEach(function (t) { el('line', { x1: x(t), x2: x(t), y1: top1, y2: y(U.min - 0.6), 'class': 'mdw-swline' }, swLines); });
      for (i = 1; i < des.sw.length; i++) {
        var a = des.sw[i - 1], b = des.sw[i];
        if (b - a <= cfg.offt * 1.3 + 1e-9) {
          var yy = top1 + 8;
          el('path', { d: 'M' + x(a) + ',' + (yy + 5) + 'V' + yy + 'H' + x(b) + 'V' + (yy + 5), 'class': 'mdw-brk' + (b - a <= cfg.offt + 1e-6 ? ' mdw-tight' : '') }, gapMarks);
        }
      }
      var all = des.smp.concat(forcedList()).sort(function (p, q) { return p - q; }), g = gapS();
      for (i = 1; i < all.length; i++) if (all[i] - all[i - 1] <= g + 1e-6)
        el('path', { d: 'M' + x(all[i - 1]) + ',' + (sy + 12) + 'V' + (sy + 16) + 'H' + x(all[i]) + 'V' + (sy + 12), 'class': 'mdw-brk mdw-tight' }, sGaps);
      ensure(lvH, des.lev.length, function (j) { return handle(layer, 'L' + j, 'level ' + (j + 1), 'bar'); });
      des.lev.forEach(function (l, j) { lvH[j].at(x(levelX(j, P)), y(l)); });
      ensure(swH, des.sw.length, function (j) { return handle(layer, 'S' + j, 'switching time ' + (j + 1), 'diamond'); });
      des.sw.forEach(function (t, j) { swH[j].at(x(t), top1); });
      ensure(smH, des.smp.length, function (j) { return handle(sLayer, 'P' + j, 'sample ' + (j + 1)); });
      des.smp.forEach(function (t, j) { smH[j].at(x(t), sy); });
      ensure(fH, forcedList().length, function () { return el('rect', { width: 11, height: 11, x: -5.5, y: -5.5, rx: 2, 'class': 'mdw-forced' }, sLayer); });
      forcedList().forEach(function (t, j) { fH[j].setAttribute('transform', 'translate(' + x(t).toFixed(1) + ',' + sy + ')'); });
      var nl = des.lev.length, ns = des.sw.length, nsmp = des.smp.length;
      chip.innerHTML = '<span class="mdw-chip-n">' + (1 + nl + ns + nsmp) + '</span> decision variables for the optimiser: ' +
        '1 <code>tii</code> + ' + nl + ' levels + ' + ns + ' switching times + ' + nsmp + ' sampling times' +
        (cfg.forced ? ' (forced samples are fixed)' : '') + (cfg.cvp === 'LPF' ? '. With LPF the last segment is flat.' : '.');
      writeCode();
    }
    function writeCode() {
      var q = function (s) { return '<span class="mdw-s">' + s + '</span>'; }, n = function (v) { return '<span class="mdw-n">' + v + '</span>'; };
      var html =
        'system = {\n' +
        '  ' + q("'t_s'") + ': [' + n(0) + ', ' + n(TF) + '], ' + q("'t_r'") + ': ' + n(TR) + ', ' + q("'t_d'") + ': ' + n(cfg.td.toFixed(1)) + ',\n' +
        '  ' + q("'tvi'") + ': {' + q("'u1'") + ': {\n' +
        '      ' + q("'min'") + ': ' + n(U.min) + ', ' + q("'max'") + ': ' + n(U.max) + ',\n' +
        '      ' + q("'stps'") + ': ' + n(cfg.stps) + ', ' + q("'cvp'") + ': ' + q("'" + cfg.cvp + "'") + ',\n' +
        '      ' + q("'const'") + ': ' + q("'" + cfg.cst + "'") + ',\n' +
        '      ' + q("'offt'") + ': ' + n(cfg.offt.toFixed(1)) + ', ' + q("'offl'") + ': ' + n(cfg.offl.toFixed(1)) + '}},\n' +
        '  ' + q("'tii'") + ': {' + q("'y10'") + ': {' + q("'min'") + ': ' + n(0.3) + ', ' + q("'max'") + ': ' + n(1) + '}},\n' +
        '  ' + q("'tvo'") + ': {' + q("'y1'") + ': {\n' +
        '      ' + q("'meas'") + ': ' + n('True') + ', ' + q("'unc'") + ': ' + n(0.01) + ', ' + q("'init'") + ': ' + n(0) + ',\n' +
        '      ' + q("'sp'") + ': ' + n(cfg.sp) + ', ' + q("'offt'") + ': ' + n(cfg.sofft.toFixed(1)) + ',\n' +
        '      ' + q("'samp_s'") + ': ' + n(1) + ', ' + q("'samp_f'") + ': ' + (cfg.forced ? '[' + n(0) + ', ' + n(TF) + ']' : '[]') + '}},\n' +
        '  ' + q("'tio'") + ': {},\n}';
      if (code.innerHTML !== html) code.innerHTML = html;
    }
    function label(id) {
      var kind = id[0], j = +id.slice(1);
      if (kind === 'L') return 'level ' + (j + 1) + ': ' + des.lev[j].toFixed(2) + ' K';
      if (kind === 'S') return 'switch ' + (j + 1) + ': ' + des.sw[j].toFixed(1) + ' h';
      return 'sample: ' + des.smp[j].toFixed(1) + ' h';
    }
    function where(id) {
      var kind = id[0], j = +id.slice(1);
      if (kind === 'L') return [x(levelX(j, nodesOf())), y(des.lev[j])];
      if (kind === 'S') return [x(des.sw[j]), top1];
      return [x(des.smp[j]), sy];
    }
    function move(id, want) {
      var kind = id[0], j = +id.slice(1), res;
      if (kind === 'L') { res = holdLevel(j, want); des.lev[j] = res[0]; }
      else if (kind === 'S') { res = holdSwitch(j, want); des.sw[j] = res[0]; }
      else { res = holdSample(j, want); des.smp[j] = res[0]; }
      render();
      if (res[1]) say('Held by ' + res[1] + '.', 'warn');
      var p = where(id); bub.show(p[0], p[1], label(id), kind === 'S');
    }
    interact(svg, {
      start: function () { anim.cancel(); },
      drag: function (id, p) { if (des) move(id, id[0] === 'L' ? y.inv(p.y) : x.inv(p.x)); },
      drop: function (id) {
        if (!des) return;
        var kind = id[0], j = +id.slice(1);
        if (kind === 'S') { var s = snapT(des.sw[j]); if (holdSwitch(j, s)[1] === null) des.sw[j] = s; }
        if (kind === 'P') { var t = snapT(des.smp[j]); if (holdSample(j, t)[1] === null) des.smp[j] = t; }
        render(); bub.hide();
      },
      key: function (id, d, big) {
        if (!des) return;
        var kind = id[0], j = +id.slice(1);
        if (kind === 'L') move(id, des.lev[j] + d * (big ? 1 : 0.1));
        else if (kind === 'S') move(id, snapT(des.sw[j] + d * (big ? 1 : TR)));
        else move(id, snapT(des.smp[j] + d * (big ? 1 : TR)));
      },
      focus: function (id) { if (des) { var p = where(id); bub.show(p[0], p[1], label(id), id[0] === 'S'); } },
      blur: function () { bub.hide(); }
    });
    fresh();
  }


  // ======================================================================= 6. package map
  function flowWidget(root) {
    var D = [['system', 'the experiment: inputs, outputs, time, σ'], ['models', 'the candidate models, parameters, bounds'],
             ['insilicos', 'the virtual lab: true model, noise, preliminary runs'], ['iden_opt', 'fitting and uncertainty options'],
             ['des_opt', 'design criterion and optimiser'], ['gsa', 'sensitivity-analysis options']];
    var F = [
      { n: 'sensa', call: 'sensa(gsa, models, system)', what: 'Which parameters influence the outputs at all (Sobol indices), before any data.', r: ['gsa', 'models', 'system'], w: [] },
      { n: 'expera', call: 'expera(system, models, insilicos, design, expr=k)', what: 'Runs experiment k in the virtual lab and adds it to the data file.', r: ['system', 'models', 'insilicos'], w: ['data.xlsx', 'data_plots/'] },
      { n: 'parmest', call: 'resultpr = parmest(system, models, iden_opt)', what: 'Fits every candidate model to all the data so far.', r: ['system', 'models', 'iden_opt'], w: [], i: ['data.xlsx'] },
      { n: 'uncert', call: 'unc = uncert(resultpr, system, models, iden_opt)', what: 'Confidence intervals, t-values, correlations, model probabilities P.', r: ['system', 'models', 'iden_opt'], w: [], i: ['data.xlsx'] },
      { n: 'estima', call: "estima(unc['results'], system, models, iden_opt, k)", what: 'Ranks the parameters and keeps the subset the data can support.', r: ['system', 'models', 'iden_opt'], w: ['estimability/'] },
      { n: 'save_rounds', call: "save_rounds(k, unc['results'], label, round_data, …)", what: 'Stores the round; draws the fit and confidence plots.', r: ['system', 'models', 'iden_opt'], w: ['modelling/ · confidence/'] },
      { n: 'mbdoe_pp · _md', call: 'design = mbdoe_pp(des_opt, system, models, round=k)', what: 'Designs the most informative next experiment (precision or discrimination).', r: ['des_opt', 'system', 'models'], w: ['design/'] },
      { n: 'validation', call: 'validation(system, models, iden_opt, round_data)', what: 'Leave-one-experiment-out cross-validation.', r: ['system', 'models', 'iden_opt'], w: ['validation/'], i: ['data.xlsx'] },
      { n: 'save_to_jac', call: "save_to_jac(round_data, purpose='iden')", what: 'Saves every round in one file; reload with load_from_jac().', r: [], w: ['iden_results.jac'] },
      { n: 'run_postprocessing', call: "run_postprocessing(round_data, ['M'], rounds)", what: 'The final report: plots and Excel across the rounds.', r: [], w: ['post_processing/'] }
    ];
    var FILES = ['data.xlsx', 'data_plots/', 'estimability/', 'modelling/ · confidence/', 'design/', 'validation/', 'iden_results.jac', 'post_processing/'];
    var W = 780, H = 372, top = 22, xD = 18, wD = 116, xF = 290, wF = 168, xO = 600, wO = 178, hN = 26;
    var yOf = function (i, n) { return top + i * (H - top - 20 - hN) / (n - 1); };
    var scroll = h('div', { 'class': 'mdw-scroll' }, root);
    var svg = svgBox(scroll, W, H, 'Which function reads which dictionary and writes which file');
    svg.classList.add('mdw-flow');
    [['Dictionaries', xD], ['Functions, in workflow order', xF], ['Files written', xO]].forEach(function (c) {
      txt(el('text', { x: c[1], y: 10, 'class': 'mdw-colhead' }, svg), c[0]);
    });
    var edges = el('g', null, svg), nodesG = el('g', null, svg), nodes = {}, E = [];
    function node(key, label, x, y, w, kind) {
      var g = el('g', { 'class': 'mdw-fnode mdw-' + kind, tabindex: 0, role: 'button', 'aria-label': label,
                        transform: 'translate(' + x + ',' + y.toFixed(1) + ')' }, nodesG);
      el('rect', { width: w, height: hN, rx: 8 }, g);
      txt(el('text', { x: 10, y: 17 }, g), label);
      nodes[key] = { g: g, x: x, y: y, w: w };
      return g;
    }
    D.forEach(function (d, i) { node('d:' + d[0], d[0], xD, yOf(i, D.length), wD, 'dict'); });
    F.forEach(function (f, i) {
      var g = node('f:' + f.n, f.n, xF, yOf(i, F.length), wF, 'fn');
      txt(el('text', { x: -8, y: 17, 'text-anchor': 'end', 'class': 'mdw-fnum' }, g), String(i + 1));
    });
    FILES.forEach(function (o, i) { node('o:' + o, o, xO, yOf(i, FILES.length), wO, 'file'); });
    function link(a, b, cls) {
      var A = nodes[a], B = nodes[b], x1 = A.x + A.w, y1 = A.y + hN / 2, x2 = B.x, y2 = B.y + hN / 2, m = (x1 + x2) / 2;
      var p = el('path', { d: 'M' + x1 + ',' + y1.toFixed(1) + ' C' + m + ',' + y1.toFixed(1) + ' ' + m + ',' + y2.toFixed(1) + ' ' + x2 + ',' + y2.toFixed(1),
                           'class': 'mdw-edge ' + cls }, edges);
      E.push({ p: p, a: a, b: b });
    }
    F.forEach(function (f) {
      f.r.forEach(function (d) { link('d:' + d, 'f:' + f.n, 'mdw-in'); });
      f.w.forEach(function (o) { link('f:' + f.n, 'o:' + o, 'mdw-out'); });
      (f.i || []).forEach(function (o) { link('f:' + f.n, 'o:' + o, 'mdw-out mdw-rev'); });
    });
    var panel = h('div', { 'class': 'mdw-fpanel', 'aria-live': 'polite' }, root);
    var idle = '<span class="mdw-mute">Point at (or tab to) a dictionary, a function or a file to see its connections, ' +
               'or press ▶ to walk through a study in order.</span>';
    panel.innerHTML = idle;
    function focus(key) {
      var on = {}; on[key] = 1;
      E.forEach(function (e) { var hit = e.a === key || e.b === key; e.p.classList.toggle('mdw-hot', hit); if (hit) { on[e.a] = 1; on[e.b] = 1; } });
      Object.keys(nodes).forEach(function (k) { nodes[k].g.classList.toggle('mdw-dimmed', !on[k]); nodes[k].g.classList.toggle('mdw-sel', k === key); });
      svg.classList.add('mdw-focus');
      var kind = key.slice(0, 1), name = key.slice(2), html;
      var codes = function (arr) { return arr.map(function (x) { return '<code>' + x.n + '</code>'; }).join(', '); };
      if (kind === 'f') {
        var f = F.filter(function (x) { return x.n === name; })[0];
        html = '<code>' + f.call + '</code><div>' + f.what + '</div>';
      } else if (kind === 'd') {
        var d = D.filter(function (x) { return x[0] === name; })[0];
        html = '<b><code>' + name + '</code></b>: ' + d[1] + '.<div>Read by ' + codes(F.filter(function (x) { return x.r.indexOf(name) >= 0; })) + '.</div>';
      } else {
        var rd = F.filter(function (x) { return (x.i || []).indexOf(name) >= 0; });
        html = '<b><code>' + name + '</code></b><div>Written by ' + codes(F.filter(function (x) { return x.w.indexOf(name) >= 0; })) +
               (rd.length ? '; read by ' + codes(rd) : '') + '.</div>';
      }
      panel.innerHTML = html;
    }
    function unfocus() {
      E.forEach(function (e) { e.p.classList.remove('mdw-hot'); });
      Object.keys(nodes).forEach(function (k) { nodes[k].g.classList.remove('mdw-dimmed', 'mdw-sel'); });
      svg.classList.remove('mdw-focus'); panel.innerHTML = idle;
    }
    Object.keys(nodes).forEach(function (k) {
      var g = nodes[k].g;
      g.addEventListener('pointerenter', function () { stop(); focus(k); });
      g.addEventListener('focus', function () { stop(); focus(k); });
      g.addEventListener('click', function () { stop(); focus(k); });
      g.addEventListener('pointerleave', function () { if (timer === null) unfocus(); });
      g.addEventListener('blur', function () { if (timer === null) unfocus(); });
    });
    var ctl = bar(root), timer = null, step = 0;
    var playB = button(ctl, '▶&nbsp; Walk through a study', function () { if (timer !== null) { stop(); unfocus(); } else { step = 0; tick(); } }, 'mdw-primary mdw-fixed');
    h('span', { 'class': 'mdw-hint' }, ctl, 'Blue: reads a dictionary. Orange: writes (or reads) a file.');
    function tick() {
      if (step >= F.length) { stop(); unfocus(); return; }
      playB.innerHTML = '■&nbsp; Stop';
      focus('f:' + F[step].n); step += 1;
      timer = setTimeout(tick, 2000);
    }
    function stop() { if (timer !== null) clearTimeout(timer); timer = null; playB.innerHTML = '▶&nbsp; Walk through a study'; }
  }

  // ======================================================================= 7. quickstart results
  function quickstartWidget(root) {
    var P = [{ n: 'k_ref', est: [0.3099, 0.3074], ci: [0.0146, 0.00652], t: [44.0, 95.8], f: function (v) { return v.toPrecision(3); } },
             { n: 'Ea', est: [57730, 54270], ci: [5800, 2280], t: [20.6, 48.4], f: function (v) { return Math.round(v).toLocaleString('en'); } }];
    var tref = 2.0;
    h('p', { 'class': 'mdw-lead' }, root, 'Each bar is a 95 % confidence interval, as ± % of its estimate. ' +
      'Switch the round, or press ▶, to watch the designed experiment shrink them.');
    var svg = svgBox(root, 640, 150, 'Confidence intervals of k_ref and Ea after each round');
    var x = scale(-12, 12, 130, 470);
    el('line', { x1: x(0), x2: x(0), y1: 14, y2: 132, 'class': 'mdw-zero' }, svg);
    [-10, -5, 5, 10].forEach(function (v) {
      el('line', { x1: x(v), x2: x(v), y1: 14, y2: 128, 'class': 'mdw-grid' }, svg);
      txt(el('text', { x: x(v), y: 146, 'text-anchor': 'middle', 'class': 'mdw-tick' }, svg), (v > 0 ? '+' : '') + v + ' %');
    });
    txt(el('text', { x: 520, y: 10, 'class': 'mdw-colhead' }, svg), 't-value  (line: t_ref ≈ 2)');
    var rows = P.map(function (p, i) {
      var y = 46 + i * 56;
      txt(el('text', { x: 6, y: y + 2, 'class': 'mdw-label mdw-mono' }, svg), p.n);
      var val = el('text', { x: 6, y: y + 18, 'class': 'mdw-tick' }, svg);
      var band = el('rect', { y: y - 9, height: 18, rx: 9, 'class': 'mdw-cibar' }, svg);
      el('circle', { cx: x(0), cy: y, r: 4, 'class': 'mdw-true' }, svg);
      var lab = el('text', { x: x(0), y: y - 14, 'text-anchor': 'middle', 'class': 'mdw-label' }, svg);
      el('rect', { x: 520, y: y - 7, width: 110, height: 14, rx: 7, 'class': 'mdw-tbg' }, svg);
      var tb = el('rect', { x: 520, y: y - 7, height: 14, rx: 7, 'class': 'mdw-tbar' }, svg);
      el('line', { x1: 520 + 110 * tref / 100, x2: 520 + 110 * tref / 100, y1: y - 11, y2: y + 11, 'class': 'mdw-axis' }, svg);
      var tl = el('text', { x: 630, y: y + 22, 'text-anchor': 'end', 'class': 'mdw-tick' }, svg);
      return { val: val, band: band, lab: lab, tb: tb, tl: tl, p: p };
    });
    var target = function (j) { return [P[0].ci[j] / P[0].est[j] * 100, P[1].ci[j] / P[1].est[j] * 100, P[0].t[j], P[1].t[j]]; };
    var ctl = bar(root), anim = channel(), now = target(0);
    var note = h('div', { 'class': 'mdw-note mdw-fixnote' }, root);
    function draw(v) {
      rows.forEach(function (r, i) {
        var hw = v[i], t = v[2 + i];
        set(r.band, { x: x(-hw).toFixed(1), width: (x(hw) - x(-hw)).toFixed(1) });
        txt(r.lab, '± ' + hw.toFixed(1) + ' %');
        set(r.tb, { width: (110 * Math.min(100, t) / 100).toFixed(1) }); txt(r.tl, 't = ' + t.toFixed(1));
      });
    }
    function go(j, ms) {
      sg.set(j);
      rows.forEach(function (r) { txt(r.val, r.p.f(r.p.est[j]) + ' ± ' + r.p.f(r.p.ci[j])); });
      anim(now, target(j), ms, function (v) { now = v; draw(v); });
      note.innerHTML = j ? 'One designed experiment <b>halved both intervals</b> and doubled the t-values.' :
        'Both parameters pass the t-test (t &gt; 2), but the intervals are wide, Ea\'s especially.';
    }
    var sg = seg(ctl, [[0, '2 starting'], [1, '+ 1 designed']], 0, function (v) { go(+v, 900); }, 'experiments');
    button(ctl, '▶&nbsp; Replay', function () { go(0, 0); setTimeout(function () { go(1, 1400); }, 500); }, 'mdw-primary');
    draw(now); go(0, 0);
    if (window.IntersectionObserver) {
      var io = new IntersectionObserver(function (es) {
        if (es[0].isIntersecting) { io.disconnect(); setTimeout(function () { go(1, 1400); }, 700); }
      }, { threshold: 0.6 });
      io.observe(root);
    }
  }

  // ======================================================================= 8. data.xlsx
  function datafileWidget(root) {
    var cols = [
      ['MES_X:y1', 'sampling times of <code>y1</code>', 'm'],
      ['MES_Y:y1', 'measured values of <code>y1</code>; the name after the colon is a key of <code>system[\'tvo\']</code>', 'm'],
      ['MES_E:y1', 'σ of each <code>y1</code> value (absolute); it weights the fit', 'm'],
      ['MES_X:y2', 'sampling times of <code>y2</code>: each output has its own', 'm'],
      ['MES_Y:y2', 'measured values of <code>y2</code>', 'm'],
      ['MES_E:y2', 'σ of each <code>y2</code> value', 'm'],
      ['X:all', 'the time grid, step <code>system[\'t_r\']</code>, on which the inputs are tabulated', 'g'],
      ['y10', 'a time-invariant input (<code>tii</code>): the same value on every row', 'g'],
      ['u1', 'a time-variant input (<code>tvi</code>), its value at each grid time', 'g']];
    var ts = [0.5, 2.3, 4.1, 5.9, 7.7, 9.5], rows = 7, sheet = 1;
    h('p', { 'class': 'mdw-lead' }, root, 'One sheet per experiment. Point at a column header to see what it holds.');
    var tabs = h('div', { 'class': 'mdw-sheets' }, root);
    var wrapT = h('div', { 'class': 'mdw-scroll' }, root);
    var tbl = h('table', { 'class': 'mdw-xl' }, wrapT);
    var cap = h('div', { 'class': 'mdw-fpanel', 'aria-live': 'polite' }, root);
    var capIdle = '<span class="mdw-mute">Measurements (left, blue) have their own number of rows; ' +
                  'the time grid and the inputs (right, green) cover the whole run.</span>';
    function data(s) {
      var y10 = [1.0, 1.5, 2.0][s - 1], k = [0.30, 0.55, 0.42][s - 1], r = rng(11 + s), out = [];
      for (var i = 0; i < rows; i++) {
        var row = [];
        if (i < ts.length) {
          var a = y10 * Math.exp(-k * ts[i]);
          row.push(ts[i].toFixed(1), (a + 0.02 * r.n()).toFixed(3), '0.02', ts[i].toFixed(1), (y10 - a + 0.02 * r.n()).toFixed(3), '0.02');
        } else row.push('', '', '', '', '', '');
        row.push((i * 0.05).toFixed(2), y10.toFixed(1), (s === 3 && i >= 3 ? 343.15 : s === 2 ? 333.15 : 323.15).toFixed(2));
        out.push(row);
      }
      return out;
    }
    function render() {
      tbl.innerHTML = '';
      var hr = h('tr', null, h('thead', null, tbl));
      h('th', { 'class': 'mdw-rn' }, hr, '');
      cols.forEach(function (c, j) {
        var th = h('th', { tabindex: 0, 'class': 'mdw-' + c[2], 'data-c': j }, hr, c[0]);
        var on = function () { hl(j); }, off = function () { hl(-1); };
        th.addEventListener('pointerenter', on); th.addEventListener('focus', on); th.addEventListener('click', on);
        th.addEventListener('pointerleave', off); th.addEventListener('blur', off);
      });
      var tb = h('tbody', null, tbl);
      data(sheet).forEach(function (row, i) {
        var tr = h('tr', null, tb); h('td', { 'class': 'mdw-rn' }, tr, String(i + 2));
        row.forEach(function (v, j) { h('td', { 'data-c': j, 'class': 'mdw-' + cols[j][2] }, tr, v); });
      });
      var tr = h('tr', null, tb); h('td', { 'class': 'mdw-rn' }, tr, '');
      cols.forEach(function (c, j) { h('td', { 'data-c': j, 'class': 'mdw-' + c[2] + ' mdw-mute' }, tr, j < 6 ? '' : '⋮'); });
    }
    function hl(j) {
      tbl.querySelectorAll('[data-c]').forEach(function (c) { c.classList.toggle('mdw-col-on', +c.getAttribute('data-c') === j); });
      cap.innerHTML = j < 0 ? capIdle : '<code>' + cols[j][0] + '</code>: ' + cols[j][1] + '.';
    }
    var tabB = [1, 2, 3].map(function (s) {
      var b = h('button', { type: 'button', 'class': 'mdw-sheet' }, tabs, 'sheet ' + s);
      b.addEventListener('click', function () { sheet = s; tabB.forEach(function (x, i) { x.classList.toggle('md-on', i === s - 1); }); render(); hl(-1); });
      return b;
    });
    h('span', { 'class': 'mdw-hint' }, tabs, 'sheet name = experiment number');
    tabB[0].classList.add('md-on'); render(); hl(-1);
  }

  // ======================================================================= 9. reading the residuals
  function residualsWidget(root) {
    var N = 24, p = 2, r0 = rng(7), base = [], i;
    for (i = 0; i < N; i++) base.push(r0.n());
    var mean = base.reduce(function (a, b) { return a + b; }, 0) / N;
    base = base.map(function (v) { return v - mean; });
    var stat = function (v) { return v.reduce(function (a, b) { return a + b * b; }, 0) / (N - p); };
    var scaled = function (v, target) { var s = stat(v); return v.map(function (z) { return z * Math.sqrt(target / s); }); };
    var trend = base.map(function (v, j) { return 2.4 * Math.cos(Math.PI * j / (N - 1)) - 0.2 + 0.45 * v; });
    var S = {
      good: { r: scaled(base, 1.08), msg: 'Residuals scatter around 0, mostly within ±2, with no pattern, and the fit statistic is inside its range: <b>model and σ agree with the data.</b>' },
      model: { r: scaled(trend, 3.6), msg: 'A <b>trend</b> in time: the model misses part of the behaviour. Fix the model or try a rival one; changing σ will not help.' },
      small: { r: scaled(base, 6.2), msg: 'No pattern, but too spread: the σ you gave (<code>unc</code>, <code>MES_E</code>) is <b>smaller than the real error</b>, so the intervals are too optimistic.' },
      large: { r: scaled(base, 0.16), msg: 'Squeezed near 0: σ is <b>larger than the real error</b>. The fit is fine, but the intervals are wider than they need to be.' }
    };
    var lo = 0.499, hi = 1.672;      // chi2(22) 2.5 % and 97.5 % quantiles, divided by 22
    var svg = svgBox(root, 640, 220, 'Weighted residuals over time');
    var x = scale(0, 10, 50, 620), y = scale(-5, 5, 196, 14);
    drawAxes(el('g', null, svg), x, y, { yt: [-4, -2, 0, 2, 4], zero: true, xl: 'time', yl: '(y − ŷ)/σ' });
    el('rect', { x: x(0), width: x(10) - x(0), y: y(2), height: y(-2) - y(2), 'class': 'mdw-range' }, svg);
    var trendL = el('path', { 'class': 'mdw-l2', opacity: 0 }, svg);
    var dots = [];
    for (i = 0; i < N; i++) dots.push(el('circle', { r: 4.5, cx: x(10 * (i + 0.5) / N), cy: y(0), 'class': 'mdw-obs' }, svg));
    var ctl = bar(root), anim = channel(), cur = base.map(function () { return 0; });
    var m = meter(root, 'Fit statistic Σ[(y − ŷ)/σ]² / (N − p) and its 95 % range (marker = 1)');
    var note = h('div', { 'class': 'mdw-note mdw-fixnote' }, root);
    function draw(v) {
      v.forEach(function (r, j) { dots[j].setAttribute('cy', y(clamp(r, -4.9, 4.9)).toFixed(1)); });
      var s = stat(v), ok = s >= lo && s <= hi;
      m(s, 1, 7, 'value <b>' + s.toFixed(2) + '</b> · range ' + lo.toFixed(2) + ' – ' + hi.toFixed(2) + ' · ' +
        (ok ? '<span class="mdw-good">inside</span>' : '<span class="mdw-bad">' + (s > hi ? 'above' : 'below') + '</span>'), ok);
    }
    function go(k) {
      sg.set(k);
      anim(cur, S[k].r, 800, function (v) { cur = v; draw(v); });
      trendL.setAttribute('opacity', k === 'model' ? 0.8 : 0);
      if (k === 'model') trendL.setAttribute('d', pathD(S.model.r.map(function (r, j) { return [x(10 * (j + 0.5) / N), y(r)]; })));
      note.innerHTML = S[k].msg;
    }
    var sg = seg(ctl, [['good', 'good fit'], ['model', 'model error'], ['small', 'σ too small'], ['large', 'σ too large']], 'good', go, 'case');
    draw(cur); go('good');
  }

  var REG = { flow: flowWidget, quickstart: quickstartWidget, datafile: datafileWidget, residuals: residualsWidget, loop: loopWidget, sampling: samplingWidget, discrim: discrimWidget, noise: noiseWidget, designspace: designspaceWidget };
  function boot() {
    document.querySelectorAll('.mdw[data-widget]').forEach(function (n) {
      if (n.dataset.ready) return;
      var f = REG[n.getAttribute('data-widget')];
      if (!f) return;
      n.dataset.ready = '1';
      try { f(n); } catch (e) {
        n.innerHTML = '<p class="mdw-note">This interactive figure could not be drawn in this browser.</p>';
        if (window.console) console.error('[middoe-widgets]', e);
      }
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
