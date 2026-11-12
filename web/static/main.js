// Entry point: lazy page modules, the mode label, a subtle blinking-emitters background, then the router.
import { $, app, isStatic, lazy, route, start } from './core.js';

const PAGES = { 'home.js': ['home'], 'analyze.js': ['analyze'], 'image.js': ['image'], 'table.js': ['table'],
  'validate.js': ['validate'], 'docs.js': ['method', 'about'] };
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: the pipeline runs in your browser (JS port tested against the Python code) on a simulated movie'
  : 'local app: analysis runs in Python on this computer';

// Background: a few faint emitters blinking on and off, like a single-molecule movie. Off under reduced motion.
function background(canvas) {
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const g = canvas.getContext('2d');
  const fit = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
  fit(); addEventListener('resize', fit);
  const dots = Array.from({ length: 40 }, () => ({ x: Math.random(), y: Math.random(), on: Math.random() < 0.2 }));
  let last = 0;
  const tick = t => {
    requestAnimationFrame(tick);
    if (document.hidden || t - last < 140) return;
    last = t;
    g.clearRect(0, 0, canvas.width, canvas.height);
    for (const d of dots) {
      d.on = d.on ? Math.random() > 0.3 : Math.random() < 0.04;
      if (!d.on) continue;
      const x = d.x * canvas.width, y = d.y * canvas.height, r = g.createRadialGradient(x, y, 0, x, y, 14);
      r.addColorStop(0, 'rgba(255,200,90,.8)'); r.addColorStop(1, 'rgba(255,94,58,0)');
      g.fillStyle = r; g.fillRect(x - 14, y - 14, 28, 28);
    }
  };
  requestAnimationFrame(tick);
}
background($('#bg'));
start();
