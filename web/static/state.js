// Shared state: project info, the decoded demo movie, the current parameters and the current analysis result.
import { api, isStatic } from './core.js';

let info = null, movie = null, result = null, params = null;

export async function getInfo() { return info ??= await api('/info'); }

function decode(b64, Type) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer); // little-endian, as written by the server (all supported browsers are LE)
}

// The simulated demo movie: { n, h, w, frame(i) -> Float64Array, truth, sim, camera, ... }
export async function getMovie() {
  if (movie) return movie;
  const m = await api('/demo/movie');
  const [n, h, w] = m.shape, u = decode(m.frames_b64, Uint16Array);
  movie = { ...m, n, h, w, frame: i => Float64Array.from(u.subarray(i * h * w, (i + 1) * h * w)) };
  delete movie.frames_b64;
  return movie;
}

// Mean of all frames (a "widefield" view of the movie).
export function meanImage(mv) {
  const acc = new Float64Array(mv.w * mv.h);
  for (let f = 0; f < mv.n; f++) { const fr = mv.frame(f); for (let i = 0; i < acc.length; i++) acc[i] += fr[i] / mv.n; }
  return acc;
}

export async function getFrame(res, i) {
  if (res.source === 'demo') return (await getMovie()).frame(i);
  const r = await api(`/runs/${res.run_id}/frame/${i}`, { fresh: true });
  return Float64Array.from(decode(r.b64, Float32Array));
}

// Parameters in the /api/info "defaults" shape; the demo uses its own camera (photon counts, no offset).
export async function getParams() {
  if (params) return params;
  const i = await getInfo();
  params = JSON.parse(JSON.stringify({ ...i.defaults, ...i.demo.camera, use_file_pixel_size: true }));
  return params;
}
export function setParams(p) { params = p; }
export function getResult() { return result; }
export function setResult(r) { result = r; }

// Run the analysis: the Python server, or (static site) the JS port on the demo movie via static_api.js.
export async function run(p, file = null) {
  const fd = new FormData();
  fd.append('params', JSON.stringify(p));
  if (file) fd.append('file', file);
  const t0 = performance.now();
  const r = await api('/analyze', { body: fd });
  r.seconds = (performance.now() - t0) / 1000;
  setResult(r);
  return r;
}

// Make sure there is a result to show: the demo with the default settings if nothing has been run yet.
export async function ensureResult() {
  return result ?? run(await getParams());
}

export const engineBadge = r => r.engine === 'python'
  ? '<span class="badge py">Python pipeline</span>' : '<span class="badge js">JS port (in your browser)</span>';
export const sourceBadge = r => r.source === 'demo' ? '<span class="badge sim">simulated demo movie</span>'
  : `<span class="badge">${String(r.source).replace(/[<>&"]/g, '')}</span>`;
export { isStatic };
