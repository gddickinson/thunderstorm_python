// MINPACK lmdif (Levenberg-Marquardt, forward-difference Jacobian), as used by scipy.optimize.curve_fit / leastsq,
// ported so the browser lands on the same least-squares solution as the Python fitter (same steps, same stopping
// rule). Routines: enorm, fdjac2, qrfac, qrsolv, lmpar, lmdif (More, Garbow & Hillstrom, Argonne 1980).
// Matrices are arrays of rows: a[i][j] (i = residual, j = parameter).

const EPSMCH = Number.EPSILON, DWARF = 2.2250738585072014e-308;

export function enorm(x, n = x.length, off = 0) {
  const rdwarf = 3.834e-20, rgiant = 1.304e19, agiant = rgiant / n;
  let s1 = 0, s2 = 0, s3 = 0, x1max = 0, x3max = 0;
  for (let i = off; i < off + n; i++) {
    const xabs = Math.abs(x[i]);
    if (xabs > rdwarf && xabs < agiant) s2 += xabs * xabs;
    else if (xabs <= rdwarf) {
      if (xabs > x3max) { s3 = 1 + s3 * (x3max / xabs) ** 2; x3max = xabs; } else if (xabs !== 0) s3 += (xabs / x3max) ** 2;
    } else if (xabs > x1max) { s1 = 1 + s1 * (x1max / xabs) ** 2; x1max = xabs; } else s1 += (xabs / x1max) ** 2;
  }
  if (s1 !== 0) return x1max * Math.sqrt(s1 + (s2 / x1max) / x1max);
  if (s2 !== 0) return s2 >= x3max ? Math.sqrt(s2 * (1 + (x3max / s2) * (x3max * s3))) : Math.sqrt(x3max * ((s2 / x3max) + (x3max * s3)));
  return x3max * Math.sqrt(s3);
}

const col = (a, j, from, m) => { const v = new Float64Array(m - from); for (let i = from; i < m; i++) v[i - from] = a[i][j]; return v; };

function qrfac(m, n, a, ipvt, rdiag, acnorm) {
  const wa = new Float64Array(n);
  for (let j = 0; j < n; j++) { acnorm[j] = enorm(col(a, j, 0, m)); rdiag[j] = acnorm[j]; wa[j] = rdiag[j]; ipvt[j] = j; }
  for (let j = 0; j < Math.min(m, n); j++) {
    let kmax = j;
    for (let k = j; k < n; k++) if (rdiag[k] > rdiag[kmax]) kmax = k;
    if (kmax !== j) {
      for (let i = 0; i < m; i++) { const t = a[i][j]; a[i][j] = a[i][kmax]; a[i][kmax] = t; }
      rdiag[kmax] = rdiag[j]; wa[kmax] = wa[j];
      const t = ipvt[j]; ipvt[j] = ipvt[kmax]; ipvt[kmax] = t;
    }
    let ajnorm = enorm(col(a, j, j, m));
    if (ajnorm !== 0) {
      if (a[j][j] < 0) ajnorm = -ajnorm;
      for (let i = j; i < m; i++) a[i][j] /= ajnorm;
      a[j][j] += 1;
      for (let k = j + 1; k < n; k++) {
        let sum = 0;
        for (let i = j; i < m; i++) sum += a[i][j] * a[i][k];
        const temp = sum / a[j][j];
        for (let i = j; i < m; i++) a[i][k] -= temp * a[i][j];
        if (rdiag[k] !== 0) {
          const t = a[j][k] / rdiag[k];
          rdiag[k] *= Math.sqrt(Math.max(0, 1 - t * t));
          if (0.05 * (rdiag[k] / wa[k]) ** 2 <= EPSMCH) { rdiag[k] = enorm(col(a, k, j + 1, m)); wa[k] = rdiag[k]; }
        }
      }
    }
    rdiag[j] = -ajnorm;
  }
}

function qrsolv(n, r, ipvt, diag, qtb, x, sdiag) {
  const wa = new Float64Array(n);
  for (let j = 0; j < n; j++) { for (let i = j; i < n; i++) r[i][j] = r[j][i]; x[j] = r[j][j]; wa[j] = qtb[j]; }
  for (let j = 0; j < n; j++) {
    const l = ipvt[j];
    if (diag[l] !== 0) {
      for (let k = j; k < n; k++) sdiag[k] = 0;
      sdiag[j] = diag[l];
      let qtbpj = 0;
      for (let k = j; k < n; k++) {
        if (sdiag[k] === 0) continue;
        let sin, cos;
        if (Math.abs(r[k][k]) < Math.abs(sdiag[k])) {
          const cotan = r[k][k] / sdiag[k]; sin = 0.5 / Math.sqrt(0.25 + 0.25 * cotan * cotan); cos = sin * cotan;
        } else {
          const tan = sdiag[k] / r[k][k]; cos = 0.5 / Math.sqrt(0.25 + 0.25 * tan * tan); sin = cos * tan;
        }
        r[k][k] = cos * r[k][k] + sin * sdiag[k];
        const temp = cos * wa[k] + sin * qtbpj;
        qtbpj = -sin * wa[k] + cos * qtbpj;
        wa[k] = temp;
        for (let i = k + 1; i < n; i++) {
          const t = cos * r[i][k] + sin * sdiag[i];
          sdiag[i] = -sin * r[i][k] + cos * sdiag[i];
          r[i][k] = t;
        }
      }
    }
    sdiag[j] = r[j][j]; r[j][j] = x[j];
  }
  let nsing = n;
  for (let j = 0; j < n; j++) { if (sdiag[j] === 0 && nsing === n) nsing = j; if (nsing < n) wa[j] = 0; }
  for (let j = nsing - 1; j >= 0; j--) {
    let sum = 0;
    for (let i = j + 1; i < nsing; i++) sum += r[i][j] * wa[i];
    wa[j] = (wa[j] - sum) / sdiag[j];
  }
  for (let j = 0; j < n; j++) x[ipvt[j]] = wa[j];
}

function lmpar(n, r, ipvt, diag, qtb, delta, par, x, sdiag) {
  const wa1 = new Float64Array(n), wa2 = new Float64Array(n);
  let nsing = n;
  for (let j = 0; j < n; j++) { wa1[j] = qtb[j]; if (r[j][j] === 0 && nsing === n) nsing = j; if (nsing < n) wa1[j] = 0; }
  for (let j = nsing - 1; j >= 0; j--) {
    wa1[j] /= r[j][j]; const temp = wa1[j];
    for (let i = 0; i < j; i++) wa1[i] -= r[i][j] * temp;
  }
  for (let j = 0; j < n; j++) x[ipvt[j]] = wa1[j];
  let iter = 0;
  for (let j = 0; j < n; j++) wa2[j] = diag[j] * x[j];
  let dxnorm = enorm(wa2), fp = dxnorm - delta;
  if (fp <= 0.1 * delta) return 0;
  let parl = 0;
  if (nsing >= n) {
    for (let j = 0; j < n; j++) { const l = ipvt[j]; wa1[j] = diag[l] * (wa2[l] / dxnorm); }
    for (let j = 0; j < n; j++) { let sum = 0; for (let i = 0; i < j; i++) sum += r[i][j] * wa1[i]; wa1[j] = (wa1[j] - sum) / r[j][j]; }
    const temp = enorm(wa1); parl = ((fp / delta) / temp) / temp;
  }
  for (let j = 0; j < n; j++) { let sum = 0; for (let i = 0; i <= j; i++) sum += r[i][j] * qtb[i]; wa1[j] = sum / diag[ipvt[j]]; }
  const gnorm = enorm(wa1);
  let paru = gnorm / delta;
  if (paru === 0) paru = DWARF / Math.min(delta, 0.1);
  par = Math.max(par, parl); par = Math.min(par, paru);
  if (par === 0) par = gnorm / dxnorm;
  for (;;) {
    iter++;
    if (par === 0) par = Math.max(DWARF, 0.001 * paru);
    const s = Math.sqrt(par);
    for (let j = 0; j < n; j++) wa1[j] = s * diag[j];
    qrsolv(n, r, ipvt, wa1, qtb, x, sdiag);
    for (let j = 0; j < n; j++) wa2[j] = diag[j] * x[j];
    dxnorm = enorm(wa2);
    const temp = fp;
    fp = dxnorm - delta;
    if (Math.abs(fp) <= 0.1 * delta || (parl === 0 && fp <= temp && temp < 0) || iter === 10) break;
    for (let j = 0; j < n; j++) { const l = ipvt[j]; wa1[j] = diag[l] * (wa2[l] / dxnorm); }
    for (let j = 0; j < n; j++) { wa1[j] /= sdiag[j]; const t = wa1[j]; for (let i = j + 1; i < n; i++) wa1[i] -= r[i][j] * t; }
    const tn = enorm(wa1), parc = ((fp / delta) / tn) / tn;
    if (fp > 0) parl = Math.max(parl, par);
    if (fp < 0) paru = Math.min(paru, par);
    par = Math.max(parl, par + parc);
  }
  return par;
}

// lmdif with scipy leastsq defaults: epsfcn = machine eps, factor 100, diag scaled internally (mode 1).
// fcn(x) -> Float64Array of m residuals. Returns { x, fvec, info, nfev }; info 1-4 = converged (scipy's success).
export function lmdif(fcn, x0, { ftol = 1.49012e-8, xtol = 1.49012e-8, gtol = 0, maxfev = 1000, factor = 100 } = {}) {
  const n = x0.length, x = Float64Array.from(x0);
  let fvec = fcn(x), nfev = 1;
  const m = fvec.length;
  if (m < n) return { x, fvec, info: 0, nfev };
  const fjac = Array.from({ length: m }, () => new Float64Array(n));
  const diag = new Float64Array(n), qtf = new Float64Array(n), ipvt = new Int32Array(n);
  const wa1 = new Float64Array(n), wa2 = new Float64Array(n), wa3 = new Float64Array(n);
  let fnorm = enorm(fvec), par = 0, iter = 1, delta = 0, xnorm = 0, info = 0;
  const eps = Math.sqrt(EPSMCH);
  outer: for (;;) {
    for (let j = 0; j < n; j++) { // fdjac2
      const temp = x[j]; let h = eps * Math.abs(temp); if (h === 0) h = eps;
      x[j] = temp + h; const wf = fcn(x); x[j] = temp;
      for (let i = 0; i < m; i++) fjac[i][j] = (wf[i] - fvec[i]) / h;
    }
    nfev += n;
    qrfac(m, n, fjac, ipvt, wa1, wa2);
    if (iter === 1) {
      for (let j = 0; j < n; j++) { diag[j] = wa2[j] === 0 ? 1 : wa2[j]; wa3[j] = diag[j] * x[j]; }
      xnorm = enorm(wa3); delta = factor * xnorm; if (delta === 0) delta = factor;
    }
    const wa4 = Float64Array.from(fvec);
    for (let j = 0; j < n; j++) {
      if (fjac[j][j] !== 0) {
        let sum = 0; for (let i = j; i < m; i++) sum += fjac[i][j] * wa4[i];
        const temp = -sum / fjac[j][j]; for (let i = j; i < m; i++) wa4[i] += fjac[i][j] * temp;
      }
      fjac[j][j] = wa1[j]; qtf[j] = wa4[j];
    }
    let gnorm = 0;
    if (fnorm !== 0) for (let j = 0; j < n; j++) {
      const l = ipvt[j];
      if (wa2[l] !== 0) { let sum = 0; for (let i = 0; i <= j; i++) sum += fjac[i][j] * (qtf[i] / fnorm); gnorm = Math.max(gnorm, Math.abs(sum / wa2[l])); }
    }
    if (gnorm <= gtol) { info = 4; break; }
    for (let j = 0; j < n; j++) diag[j] = Math.max(diag[j], wa2[j]);
    for (;;) {
      par = lmpar(n, fjac, ipvt, diag, qtf, delta, par, wa1, wa2);
      for (let j = 0; j < n; j++) { wa1[j] = -wa1[j]; wa2[j] = x[j] + wa1[j]; wa3[j] = diag[j] * wa1[j]; }
      const pnorm = enorm(wa3);
      if (iter === 1) delta = Math.min(delta, pnorm);
      const fv1 = fcn(wa2); nfev++;
      const fnorm1 = enorm(fv1);
      let actred = -1; if (0.1 * fnorm1 < fnorm) actred = 1 - (fnorm1 / fnorm) ** 2;
      for (let j = 0; j < n; j++) { wa3[j] = 0; const temp = wa1[ipvt[j]]; for (let i = 0; i <= j; i++) wa3[i] += fjac[i][j] * temp; }
      const temp1 = enorm(wa3) / fnorm, temp2 = (Math.sqrt(par) * pnorm) / fnorm;
      const prered = temp1 * temp1 + temp2 * temp2 / 0.5, dirder = -(temp1 * temp1 + temp2 * temp2);
      const ratio = prered !== 0 ? actred / prered : 0;
      if (ratio <= 0.25) {
        let temp = actred >= 0 ? 0.5 : 0.5 * dirder / (dirder + 0.5 * actred);
        if (0.1 * fnorm1 >= fnorm || temp < 0.1) temp = 0.1;
        delta = temp * Math.min(delta, pnorm / 0.1); par /= temp;
      } else if (par === 0 || ratio >= 0.75) { delta = pnorm / 0.5; par *= 0.5; }
      if (ratio >= 1e-4) {
        for (let j = 0; j < n; j++) { x[j] = wa2[j]; wa2[j] = diag[j] * x[j]; }
        fvec = fv1; xnorm = enorm(wa2); fnorm = fnorm1; iter++;
      }
      if (Math.abs(actred) <= ftol && prered <= ftol && 0.5 * ratio <= 1) info = 1;
      if (delta <= xtol * xnorm) info = info === 1 ? 3 : 2;
      if (info !== 0) break outer;
      if (nfev >= maxfev) info = 5;
      if (Math.abs(actred) <= EPSMCH && prered <= EPSMCH && 0.5 * ratio <= 1) info = 6;
      if (delta <= EPSMCH * xnorm) info = 7;
      if (gnorm <= EPSMCH) info = 8;
      if (info !== 0) break outer;
      if (ratio >= 1e-4) break;
    }
  }
  return { x, fvec, info, nfev };
}
