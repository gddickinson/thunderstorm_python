// Static-site data layer: answers api() calls from the JSON exported by scripts/build_static.py, and runs the
// analysis of the demo movie in the browser with the JS port of the pipeline (ts_engine.js / ts_fit.js, tested
// against the Python code). Uploaded TIFFs need the local app (Python).
import { analyzeMovie } from './ts_fit.js';
import { evaluateMovie } from './ts_render.js';
import { getMovie } from './state.js';

const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export const NEEDS_SERVER = 'needs the full app running locally (Python server): see the README, "Run the web app"';

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (clean === 'analyze') return analyze(body);
  if (clean.startsWith('runs/')) throw new Error('Uploaded stacks ' + NEEDS_SERVER + '.');
  return file(clean);
}

async function analyze(fd) {
  const up = fd.get('file');
  if (up && up.name) throw new Error('Analysing your own TIFF ' + NEEDS_SERVER + '. The demo movie runs here in your browser.');
  const mv = await getMovie();
  const p = { ...JSON.parse(fd.get('params')), ...mv.camera };
  const r = analyzeMovie(mv, p);
  r.evaluation = evaluateMovie(r.locs, mv.truth, mv.n, mv.tolerance_nm);
  r.source = 'demo';
  r.pixel_size_source = 'simulation';
  return r;
}
