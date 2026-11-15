export interface ZoneBounds {
  id: number;
  row: number;
  col: number;
  baseX0: number;
  baseY0: number;
  baseX1: number;
  baseY1: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  pixelCount: number;
}

export interface StarPoint {
  x: number;
  y: number;
  flux: number;
  peak: number;
}

export interface AlignmentResult {
  dx: number;
  dy: number;
  dThetaRad: number;
  dThetaDeg: number;
  matchedStars: number;
  rmsPx: number;
  applied: boolean;
}

export interface DetectionCandidate {
  zoneId: number;
  cx: number;
  cy: number;
  peak: number;
  flux: number;
  fwhmPx: number;
  pixelCount: number;
  snr: number;
  confidence: number;
  inOverlap: boolean;
}

export interface MergedDetection {
  id: number;
  cx: number;
  cy: number;
  peak: number;
  flux: number;
  fwhmPx: number;
  snr: number;
  confidence: number;
  zoneIds: number[];
  isOverlapConfirmed: boolean;
  overlapPairLabel: string;
  classification: 'Moving Body (Dipole)' | 'New Source (+5σ)';
  motionPx: number | null;
  motionPaDeg: number | null;
  e1HoleX: number | null;
  e1HoleY: number | null;
}

export interface FitsImageResult {
  width: number;
  height: number;
  bitpix: number;
  rgba: Uint8ClampedArray;
  objectName?: string;
  crval1?: number;
  crval2?: number;
  plateScaleArcsec?: number;
  mjdObs?: number;
}

export const CANVAS_WIDTH = 1200;
export const CANVAS_HEIGHT = 600;
export const GRID_COLS = 5;
export const GRID_ROWS = 2;
export const OVERLAP_PX = 150;
export const SPHEREX_PLATE_SCALE_ARCSEC = 6.2;

export function buildTenAiZones(
  width = CANVAS_WIDTH,
  height = CANVAS_HEIGHT,
  cols = GRID_COLS,
  rows = GRID_ROWS,
  overlap = OVERLAP_PX
): ZoneBounds[] {
  const halfOverlap = Math.floor(overlap / 2);
  const colW = width / cols;
  const rowH = height / rows;
  const zones: ZoneBounds[] = [];

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const id = r * cols + c + 1;
      const baseX0 = Math.round(c * colW);
      const baseX1 = Math.round((c + 1) * colW);
      const baseY0 = Math.round(r * rowH);
      const baseY1 = Math.round((r + 1) * rowH);

      const x0 = c > 0 ? Math.max(0, baseX0 - halfOverlap) : 0;
      const x1 = c < cols - 1 ? Math.min(width, baseX1 + halfOverlap) : width;
      const y0 = r > 0 ? Math.max(0, baseY0 - halfOverlap) : 0;
      const y1 = r < rows - 1 ? Math.min(height, baseY1 + halfOverlap) : height;

      zones.push({
        id,
        row: r,
        col: c,
        baseX0,
        baseY0,
        baseX1,
        baseY1,
        x0,
        y0,
        x1,
        y1,
        pixelCount: (x1 - x0) * (y1 - y0),
      });
    }
  }
  return zones;
}

function computeLuminanceArray(rgba: Uint8ClampedArray, totalPixels: number): Float32Array {
  const lum = new Float32Array(totalPixels);
  for (let p = 0; p < totalPixels; p++) {
    const i = p * 4;
    lum[p] = rgba[i + 3] === 0 ? 0 : 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
  }
  return lum;
}

function estimateMedianAndMad(values: Float32Array, stride = 8): { median: number; madSigma: number } {
  const sample: number[] = [];
  for (let i = 0; i < values.length; i += stride) {
    sample.push(values[i]);
  }
  if (sample.length === 0) return { median: 0, madSigma: 1 };
  sample.sort((a, b) => a - b);
  const median = sample[Math.floor(sample.length / 2)];
  const absDev = sample.map((v) => Math.abs(v - median)).sort((a, b) => a - b);
  const mad = absDev[Math.floor(absDev.length / 2)];
  return { median, madSigma: Math.max(0.6, 1.4826 * mad) };
}

/**
 * Extracts crisp, well-distributed reference stars from an RGBA buffer using
 * local-sky-subtracted prominence and quadrant spatial balancing for dense crowded fields.
 */
export function extractReferenceStars(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  maxStars = 56
): StarPoint[] {
  const lum = computeLuminanceArray(rgba, width * height);
  const { median, madSigma } = estimateMedianAndMad(lum, 16);
  const minProminence = Math.max(14, 4.5 * madSigma);
  const minAbsVal = median + Math.max(8, 2.5 * madSigma);
  const peaks: Array<{ x: number; y: number; val: number; prom: number; localSky: number; score: number }> = [];
  const margin = 10;
  const ringR = 5;

  for (let y = margin; y < height - margin; y++) {
    const row = y * width;
    for (let x = margin; x < width - margin; x++) {
      const p = row + x;
      if (rgba[p * 4 + 3] === 0) continue;
      const val = lum[p];
      if (val < minAbsVal) continue;
      if (
        val >= lum[row + x - 1] &&
        val >= lum[row + x + 1] &&
        val >= lum[(y - 1) * width + x] &&
        val >= lum[(y + 1) * width + x] &&
        val >= lum[(y - 1) * width + x - 1] &&
        val >= lum[(y - 1) * width + x + 1] &&
        val >= lum[(y + 1) * width + x - 1] &&
        val >= lum[(y + 1) * width + x + 1]
      ) {
        // Estimate local sky using opposite axes at radius 5px to reject extended nebulae
        const sxMean = 0.5 * (lum[row + x - ringR] + lum[row + x + ringR]);
        const syMean = 0.5 * (lum[(y - ringR) * width + x] + lum[(y + ringR) * width + x]);
        const localSky = Math.min(sxMean, syMean);
        const prom = val - localSky;
        if (prom < minProminence) continue;
        // Prefer crisp unsaturated point sources over clipped flat cores
        const score = val >= 253 ? prom * 0.72 : prom;
        peaks.push({ x, y, val, prom, localSky, score });
      }
    }
  }

  peaks.sort((a, b) => b.score - a.score);

  // Two-pass spatial selection: balance across 4x4 regions first, then fill by score
  const stars: StarPoint[] = [];
  const gridCounts = new Uint16Array(16);
  const perCellCap = Math.max(2, Math.ceil(maxStars / 12));

  const tryAccept = (pk: { x: number; y: number; val: number; prom: number; localSky: number }, enforceCellCap: boolean) => {
    const cellX = Math.min(3, Math.floor((pk.x / width) * 4));
    const cellY = Math.min(3, Math.floor((pk.y / height) * 4));
    const cellIdx = cellY * 4 + cellX;
    if (enforceCellCap && gridCounts[cellIdx] >= perCellCap) return false;
    if (stars.some((s) => Math.hypot(s.x - pk.x, s.y - pk.y) < 15)) return false;

    let wSum = 0;
    let wx = 0;
    let wy = 0;
    const base = pk.localSky;
    for (let dy = -3; dy <= 3; dy++) {
      for (let dx = -3; dx <= 3; dx++) {
        const v = Math.max(0, lum[(pk.y + dy) * width + (pk.x + dx)] - base);
        wSum += v;
        wx += (pk.x + dx) * v;
        wy += (pk.y + dy) * v;
      }
    }
    if (wSum <= 0) return false;
    stars.push({ x: wx / wSum, y: wy / wSum, flux: wSum, peak: pk.prom });
    gridCounts[cellIdx]++;
    return true;
  };

  for (const pk of peaks) {
    tryAccept(pk, true);
    if (stars.length >= maxStars) break;
  }
  if (stars.length < maxStars) {
    for (const pk of peaks) {
      tryAccept(pk, false);
      if (stars.length >= maxStars) break;
    }
  }

  return stars;
}

function wrapAngleRad(a: number): number {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

function solveRigidLeastSquares(
  pairs: Array<{ x1: number; y1: number; x2: number; y2: number }>,
  cx0: number,
  cy0: number
): { dx: number; dy: number; dThetaRad: number; rmsPx: number } {
  const n = pairs.length;
  let mx1 = 0;
  let my1 = 0;
  let mx2 = 0;
  let my2 = 0;
  for (const p of pairs) {
    mx1 += p.x1;
    my1 += p.y1;
    mx2 += p.x2;
    my2 += p.y2;
  }
  mx1 /= n;
  my1 /= n;
  mx2 /= n;
  my2 /= n;

  let sCos = 0;
  let sSin = 0;
  for (const p of pairs) {
    const ux = p.x2 - mx2;
    const uy = p.y2 - my2;
    const vx = p.x1 - mx1;
    const vy = p.y1 - my1;
    sCos += ux * vx + uy * vy;
    sSin += ux * vy - uy * vx;
  }
  const theta = Math.atan2(sSin, sCos);
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);

  // p1 = center + R(theta)*(p2 - center) + (dx, dy)
  const rxMean = mx2 - cx0;
  const ryMean = my2 - cy0;
  const dx = mx1 - (cx0 + rxMean * cosT - ryMean * sinT);
  const dy = my1 - (cy0 + rxMean * sinT + ryMean * cosT);

  let errSq = 0;
  for (const p of pairs) {
    const rx = p.x2 - cx0;
    const ry = p.y2 - cy0;
    const predX = cx0 + rx * cosT - ry * sinT + dx;
    const predY = cy0 + rx * sinT + ry * cosT + dy;
    const ex = predX - p.x1;
    const ey = predY - p.y1;
    errSq += ex * ex + ey * ey;
  }

  return { dx, dy, dThetaRad: theta, rmsPx: Math.sqrt(errSq / n) };
}

/**
 * Matches Epoch 2 stars to Epoch 1 stars via geometric pair-length + offset consensus
 * and solves the sub-pixel rigid transform (dx, dy, rotation) mapping Epoch 2 -> Epoch 1.
 */
export function solveStarMatchedTransform(
  starsE1: StarPoint[],
  starsE2: StarPoint[],
  width = CANVAS_WIDTH,
  height = CANVAS_HEIGHT
): AlignmentResult {
  if (starsE1.length < 3 || starsE2.length < 3) {
    return { dx: 0, dy: 0, dThetaRad: 0, dThetaDeg: 0, matchedStars: 0, rmsPx: 0, applied: false };
  }

  const cx0 = width / 2;
  const cy0 = height / 2;

  // Candidate pool 1: Direct pairwise translation voting (fast, high recall for unrotated shifts)
  const transVotes = new Map<string, { count: number; sumDx: number; sumDy: number }>();
  for (const s1 of starsE1) {
    for (const s2 of starsE2) {
      const dx = s1.x - s2.x;
      const dy = s1.y - s2.y;
      if (Math.abs(dx) > width * 0.45 || Math.abs(dy) > height * 0.45) continue;
      const key = `${Math.round(dx / 3)},${Math.round(dy / 3)}`;
      const entry = transVotes.get(key) || { count: 0, sumDx: 0, sumDy: 0 };
      entry.count++;
      entry.sumDx += dx;
      entry.sumDy += dy;
      transVotes.set(key, entry);
    }
  }

  const hypotheses: Array<{ dx: number; dy: number; theta: number }> = [];
  const sortedTrans = [...transVotes.values()].sort((a, b) => b.count - a.count);
  for (let i = 0; i < Math.min(4, sortedTrans.length); i++) {
    const v = sortedTrans[i];
    hypotheses.push({ dx: v.sumDx / v.count, dy: v.sumDy / v.count, theta: 0 });
  }

  // Candidate pool 2: Rotation-invariant pair-length matching (handles rotated telescope frames)
  const top1 = starsE1.slice(0, 36);
  const top2 = starsE2.slice(0, 36);
  interface Segment { i: number; j: number; len: number; ang: number }
  const segs1: Segment[] = [];
  const segs2: Segment[] = [];

  for (let i = 0; i < top1.length; i++) {
    for (let j = i + 1; j < top1.length; j++) {
      const len = Math.hypot(top1[j].x - top1[i].x, top1[j].y - top1[i].y);
      if (len >= 24) {
        segs1.push({ i, j, len, ang: Math.atan2(top1[j].y - top1[i].y, top1[j].x - top1[i].x) });
      }
    }
  }
  for (let i = 0; i < top2.length; i++) {
    for (let j = i + 1; j < top2.length; j++) {
      const len = Math.hypot(top2[j].x - top2[i].x, top2[j].y - top2[i].y);
      if (len >= 24) {
        segs2.push({ i, j, len, ang: Math.atan2(top2[j].y - top2[i].y, top2[j].x - top2[i].x) });
      }
    }
  }

  segs1.sort((a, b) => a.len - b.len);
  segs2.sort((a, b) => a.len - b.len);

  const rotVotes = new Map<string, { count: number; sumDx: number; sumDy: number; sumTh: number }>();
  let ptrStart = 0;

  for (const s2 of segs2) {
    while (ptrStart < segs1.length && segs1[ptrStart].len < s2.len - 2.8) {
      ptrStart++;
    }
    for (let k = ptrStart; k < segs1.length && segs1[k].len <= s2.len + 2.8; k++) {
      const s1 = segs1[k];
      // Test both forward and reversed segment correspondence
      for (const dir of [0, 1]) {
        const a1 = dir === 0 ? top1[s1.i] : top1[s1.j];
        const a2 = top2[s2.i];
        const th = wrapAngleRad(s1.ang + (dir === 1 ? Math.PI : 0) - s2.ang);
        if (Math.abs(th) > (20 * Math.PI) / 180) continue; // telescope roll within +-20 deg
        const cosT = Math.cos(th);
        const sinT = Math.sin(th);
        const rx = a2.x - cx0;
        const ry = a2.y - cy0;
        const dx = a1.x - (cx0 + rx * cosT - ry * sinT);
        const dy = a1.y - (cy0 + rx * sinT + ry * cosT);
        if (Math.abs(dx) > width * 0.45 || Math.abs(dy) > height * 0.45) continue;

        const key = `${Math.round((th * 180) / Math.PI / 0.6)},${Math.round(dx / 5)},${Math.round(dy / 5)}`;
        const entry = rotVotes.get(key) || { count: 0, sumDx: 0, sumDy: 0, sumTh: 0 };
        entry.count++;
        entry.sumDx += dx;
        entry.sumDy += dy;
        entry.sumTh += th;
        rotVotes.set(key, entry);
      }
    }
  }

  const sortedRot = [...rotVotes.values()].sort((a, b) => b.count - a.count);
  for (let i = 0; i < Math.min(6, sortedRot.length); i++) {
    const v = sortedRot[i];
    hypotheses.push({
      dx: v.sumDx / v.count,
      dy: v.sumDy / v.count,
      theta: v.sumTh / v.count,
    });
  }

  // Evaluate hypotheses and refine best match with iterative least-squares
  let bestResult: AlignmentResult = {
    dx: 0,
    dy: 0,
    dThetaRad: 0,
    dThetaDeg: 0,
    matchedStars: 0,
    rmsPx: 0,
    applied: false,
  };

  for (const hyp of hypotheses) {
    let curDx = hyp.dx;
    let curDy = hyp.dy;
    let curTh = hyp.theta;
    let inliers: Array<{ x1: number; y1: number; x2: number; y2: number }> = [];

    for (const tol of [4.8, 2.4, 1.5]) {
      const cosT = Math.cos(curTh);
      const sinT = Math.sin(curTh);
      inliers = [];
      const usedE1 = new Set<number>();

      for (const s2 of starsE2) {
        const rx = s2.x - cx0;
        const ry = s2.y - cy0;
        const px = cx0 + rx * cosT - ry * sinT + curDx;
        const py = cy0 + rx * sinT + ry * cosT + curDy;
        let bestIdx = -1;
        let bestDist = tol;

        for (let idx1 = 0; idx1 < starsE1.length; idx1++) {
          if (usedE1.has(idx1)) continue;
          const d = Math.hypot(starsE1[idx1].x - px, starsE1[idx1].y - py);
          if (d < bestDist) {
            bestDist = d;
            bestIdx = idx1;
          }
        }
        if (bestIdx >= 0) {
          usedE1.add(bestIdx);
          inliers.push({ x1: starsE1[bestIdx].x, y1: starsE1[bestIdx].y, x2: s2.x, y2: s2.y });
        }
      }

      if (inliers.length >= 3) {
        const solved = solveRigidLeastSquares(inliers, cx0, cy0);
        curDx = solved.dx;
        curDy = solved.dy;
        curTh = solved.dThetaRad;
      }
    }

    if (inliers.length >= 3) {
      const finalSolved = solveRigidLeastSquares(inliers, cx0, cy0);
      if (
        inliers.length > bestResult.matchedStars ||
        (inliers.length === bestResult.matchedStars && finalSolved.rmsPx < bestResult.rmsPx)
      ) {
        bestResult = {
          dx: finalSolved.dx,
          dy: finalSolved.dy,
          dThetaRad: finalSolved.dThetaRad,
          dThetaDeg: (finalSolved.dThetaRad * 180) / Math.PI,
          matchedStars: inliers.length,
          rmsPx: finalSolved.rmsPx,
          applied: true,
        };
      }
    }
  }

  return bestResult;
}

function cubicWeights(t: number): [number, number, number, number] {
  const t2 = t * t;
  const t3 = t2 * t;
  return [
    -0.5 * t + t2 - 0.5 * t3,
    1.0 - 2.5 * t2 + 1.5 * t3,
    0.5 * t + 2.0 * t2 - 1.5 * t3,
    -0.5 * t2 + 0.5 * t3,
  ];
}

/**
 * Resamples Epoch 1 RGBA buffer onto Epoch 2's pixel grid via Catmull-Rom bicubic interpolation.
 * Bicubic resampling preserves bright star PSF peaks without bilinear core damping artifacts.
 * Pixels outside the overlapping footprint get alpha=0 so edge regions don't emit false artifacts.
 */
export function resampleAlignedImage(
  rgbaE1: Uint8ClampedArray,
  width: number,
  height: number,
  alignment: AlignmentResult
): Uint8ClampedArray {
  if (
    !alignment.applied ||
    (Math.abs(alignment.dx) < 0.05 &&
      Math.abs(alignment.dy) < 0.05 &&
      Math.abs(alignment.dThetaRad) < 0.0002)
  ) {
    return new Uint8ClampedArray(rgbaE1);
  }

  const out = new Uint8ClampedArray(width * height * 4);
  const cx0 = width / 2;
  const cy0 = height / 2;
  const cosT = Math.cos(alignment.dThetaRad);
  const sinT = Math.sin(alignment.dThetaRad);

  for (let y2 = 0; y2 < height; y2++) {
    const ry = y2 - cy0;
    for (let x2 = 0; x2 < width; x2++) {
      const rx = x2 - cx0;
      const x1 = cx0 + rx * cosT - ry * sinT + alignment.dx;
      const y1 = cy0 + rx * sinT + ry * cosT + alignment.dy;
      const dstIdx = (y2 * width + x2) * 4;

      if (x1 < 2 || x1 >= width - 3 || y1 < 2 || y1 >= height - 3) {
        out[dstIdx] = 4;
        out[dstIdx + 1] = 5;
        out[dstIdx + 2] = 7;
        out[dstIdx + 3] = 0;
        continue;
      }

      const ix = Math.floor(x1);
      const iy = Math.floor(y1);
      const wx = cubicWeights(x1 - ix);
      const wy = cubicWeights(y1 - iy);

      let r = 0;
      let g = 0;
      let b = 0;
      let valid = true;

      for (let m = -1; m <= 2; m++) {
        const wyM = wy[m + 1];
        const rowBase = (iy + m) * width;
        for (let n = -1; n <= 2; n++) {
          const idx = (rowBase + (ix + n)) * 4;
          if (rgbaE1[idx + 3] === 0) {
            valid = false;
            break;
          }
          const w = wx[n + 1] * wyM;
          r += rgbaE1[idx] * w;
          g += rgbaE1[idx + 1] * w;
          b += rgbaE1[idx + 2] * w;
        }
        if (!valid) break;
      }

      if (!valid) {
        out[dstIdx + 3] = 0;
        continue;
      }

      out[dstIdx] = Math.max(0, Math.min(255, Math.round(r)));
      out[dstIdx + 1] = Math.max(0, Math.min(255, Math.round(g)));
      out[dstIdx + 2] = Math.max(0, Math.min(255, Math.round(b)));
      out[dstIdx + 3] = 255;
    }
  }

  return out;
}

/**
 * Computes calibrated difference maps using photometric gain+pedestal matching,
 * warp-border guarding against JPEG seam ringing, and MAD N-sigma thresholding.
 */
export function subtractAlignedFramesSigma(
  alignedE1: Uint8ClampedArray,
  epoch2: Uint8ClampedArray,
  width: number,
  height: number,
  sigmaMultiplier = 5.0,
  mode: 'abs' | 'signed' = 'abs',
  matchBg = true
): {
  out: Uint8ClampedArray;
  posSignalMap: Float32Array;
  negSignalMap: Float32Array;
  alignedLum: Float32Array;
  epoch2Lum: Float32Array;
  skySigma: number;
  thresholdDn: number;
  activeSignalPixels: number;
} {
  const totalPixels = width * height;
  const alignedLum = new Float32Array(totalPixels);
  const epoch2Lum = new Float32Array(totalPixels);
  const rawValid = new Uint8Array(totalPixels);
  const l1Samples: number[] = [];
  const l2Samples: number[] = [];
  let anyInvalid = false;

  for (let p = 0; p < totalPixels; p++) {
    const idx = p * 4;
    if (alignedE1[idx + 3] === 0 || epoch2[idx + 3] === 0) {
      rawValid[p] = 0;
      anyInvalid = true;
      continue;
    }
    rawValid[p] = 1;
    const l1 = 0.299 * alignedE1[idx] + 0.587 * alignedE1[idx + 1] + 0.114 * alignedE1[idx + 2];
    const l2 = 0.299 * epoch2[idx] + 0.587 * epoch2[idx + 1] + 0.114 * epoch2[idx + 2];
    alignedLum[p] = l1;
    epoch2Lum[p] = l2;
    if (p % 13 === 0) {
      l1Samples.push(l1);
      l2Samples.push(l2);
    }
  }

  // Erode validMask by 10 px near invalid warped/border seams to eliminate JPEG edge-block ringing
  const validMask = new Uint8Array(totalPixels);
  const guardR = anyInvalid ? 10 : 4;
  for (let y = 0; y < height; y++) {
    const rowOff = y * width;
    if (y < guardR || y >= height - guardR) continue;
    for (let x = guardR; x < width - guardR; x++) {
      const p = rowOff + x;
      if (!rawValid[p]) continue;
      if (anyInvalid) {
        // Fast axial cross check at radius 3, 6, 10 px catches rotated boundary transitions in O(1)
        if (
          !rawValid[p - 10] || !rawValid[p + 10] ||
          !rawValid[(y - 10) * width + x] || !rawValid[(y + 10) * width + x] ||
          !rawValid[p - 5] || !rawValid[p + 5] ||
          !rawValid[(y - 5) * width + x] || !rawValid[(y + 5) * width + x] ||
          !rawValid[(y - 7) * width + (x - 7)] || !rawValid[(y - 7) * width + (x + 7)] ||
          !rawValid[(y + 7) * width + (x - 7)] || !rawValid[(y + 7) * width + (x + 7)]
        ) {
          continue;
        }
      }
      validMask[p] = 1;
    }
  }

  // Robust photometric scale (gain) matching via upper/lower quantiles of valid samples
  let gain = 1.0;
  if (matchBg && l1Samples.length > 64) {
    const s1 = [...l1Samples].sort((a, b) => a - b);
    const s2 = [...l2Samples].sort((a, b) => a - b);
    const pLow1 = s1[Math.floor(s1.length * 0.2)];
    const pHigh1 = s1[Math.floor(s1.length * 0.88)];
    const pLow2 = s2[Math.floor(s2.length * 0.2)];
    const pHigh2 = s2[Math.floor(s2.length * 0.88)];
    if (pHigh1 - pLow1 > 10 && pHigh2 - pLow2 > 10) {
      const est = (pHigh2 - pLow2) / (pHigh1 - pLow1);
      gain = Math.max(0.65, Math.min(1.55, est));
    }
  }

  const rawSignedDiff = new Float32Array(totalPixels);
  const diffSamples: number[] = [];
  for (let p = 0; p < totalPixels; p++) {
    if (!validMask[p]) continue;
    const d = epoch2Lum[p] - gain * alignedLum[p];
    rawSignedDiff[p] = d;
    if (p % 11 === 0) diffSamples.push(d);
  }

  diffSamples.sort((a, b) => a - b);
  const medianDiff = diffSamples.length > 0 ? diffSamples[Math.floor(diffSamples.length / 2)] : 0;
  const absDevs = diffSamples.map((v) => Math.abs(v - medianDiff)).sort((a, b) => a - b);
  const mad = absDevs.length > 0 ? absDevs[Math.floor(absDevs.length / 2)] : 1;
  const skySigma = Math.max(2.0, 1.4826 * mad);
  const thresholdDn = sigmaMultiplier * skySigma;
  const pedestal = matchBg ? medianDiff : 0;

  if (matchBg && (Math.abs(gain - 1.0) > 1e-4 || Math.abs(pedestal) > 1e-4)) {
    for (let p = 0; p < totalPixels; p++) {
      if (validMask[p]) {
        alignedLum[p] = Math.max(0, Math.min(255, gain * alignedLum[p] + pedestal));
      }
    }
  }

  const out = new Uint8ClampedArray(totalPixels * 4);
  const posSignalMap = new Float32Array(totalPixels);
  const negSignalMap = new Float32Array(totalPixels);
  let activeSignalPixels = 0;

  for (let p = 0; p < totalPixels; p++) {
    const idx = p * 4;
    if (!validMask[p]) {
      out[idx] = 4;
      out[idx + 1] = 5;
      out[idx + 2] = 7;
      out[idx + 3] = 255;
      continue;
    }

    const centered = rawSignedDiff[p] - pedestal;
    if (centered > thresholdDn) {
      const excess = centered - thresholdDn;
      posSignalMap[p] = excess;
      activeSignalPixels++;
      const boost = Math.min(255, Math.round(excess * 2.8));
      out[idx] = Math.min(255, Math.round(boost * 0.24));
      out[idx + 1] = Math.min(255, Math.round(boost * 0.94));
      out[idx + 2] = Math.min(255, Math.round(boost * 0.84));
      out[idx + 3] = 255;
    } else if (-centered > thresholdDn) {
      const excess = -centered - thresholdDn;
      negSignalMap[p] = excess;
      if (mode === 'abs') {
        activeSignalPixels++;
        const boost = Math.min(255, Math.round(excess * 2.4));
        out[idx] = boost;
        out[idx + 1] = Math.round(boost * 0.58);
        out[idx + 2] = Math.round(boost * 0.24);
        out[idx + 3] = 255;
      } else {
        out[idx] = 4;
        out[idx + 1] = 5;
        out[idx + 2] = 7;
        out[idx + 3] = 255;
      }
    } else {
      out[idx] = 4;
      out[idx + 1] = 5;
      out[idx + 2] = 7;
      out[idx + 3] = 255;
    }
  }

  return { out, posSignalMap, negSignalMap, alignedLum, epoch2Lum, skySigma, thresholdDn, activeSignalPixels };
}

function isPointInOverlapEdge(
  x: number,
  y: number,
  zone: ZoneBounds,
  cols = GRID_COLS,
  rows = GRID_ROWS,
  overlap = OVERLAP_PX
): boolean {
  const half = overlap / 2;
  const nearLeftOverlap = zone.col > 0 && x < zone.baseX0 + half;
  const nearRightOverlap = zone.col < cols - 1 && x >= zone.baseX1 - half;
  const nearTopOverlap = zone.row > 0 && y < zone.baseY0 + half;
  const nearBottomOverlap = zone.row < rows - 1 && y >= zone.baseY1 - half;
  return nearLeftOverlap || nearRightOverlap || nearTopOverlap || nearBottomOverlap;
}

export function scanZoneForAllSources(
  signalMap: Float32Array,
  width: number,
  zone: ZoneBounds,
  skySigma = 2.0,
  sigmaMultiplier = 5.0,
  minPeak = 4,
  alignedLum?: Float32Array,
  epoch2Lum?: Float32Array
): DetectionCandidate[] {
  const peaks: Array<{ x: number; y: number; val: number }> = [];
  const effMinPeak = Math.max(minPeak, skySigma * 0.55);

  for (let y = zone.y0 + 1; y < zone.y1 - 1; y++) {
    const row = y * width;
    for (let x = zone.x0 + 1; x < zone.x1 - 1; x++) {
      const val = signalMap[row + x];
      if (val < effMinPeak) continue;
      if (
        val >= signalMap[row + x - 1] &&
        val >= signalMap[row + x + 1] &&
        val >= signalMap[(y - 1) * width + x] &&
        val >= signalMap[(y + 1) * width + x] &&
        val >= signalMap[(y - 1) * width + x - 1] &&
        val >= signalMap[(y - 1) * width + x + 1] &&
        val >= signalMap[(y + 1) * width + x - 1] &&
        val >= signalMap[(y + 1) * width + x + 1]
      ) {
        peaks.push({ x, y, val });
      }
    }
  }

  peaks.sort((a, b) => b.val - a.val);
  const candidates: DetectionCandidate[] = [];
  const used: Array<{ x: number; y: number }> = [];
  const radius = 8;

  for (const pk of peaks) {
    if (used.some((u) => Math.hypot(u.x - pk.x, u.y - pk.y) < 14)) continue;
    // Reject stationary bright star PSF/seeing/interpolation/saturation residuals across 5x5 neighborhood
    if (alignedLum && epoch2Lum) {
      let maxL1 = 0;
      let maxL2 = 0;
      for (let dy = -2; dy <= 2; dy++) {
        const yy = pk.y + dy;
        if (yy < zone.y0 || yy >= zone.y1) continue;
        const rowOff = yy * width;
        for (let dx = -2; dx <= 2; dx++) {
          const xx = pk.x + dx;
          if (xx < zone.x0 || xx >= zone.x1) continue;
          const v1 = alignedLum[rowOff + xx];
          const v2 = epoch2Lum[rowOff + xx];
          if (v1 > maxL1) maxL1 = v1;
          if (v2 > maxL2) maxL2 = v2;
        }
      }
      const sharedStarBase = Math.min(maxL1, maxL2);
      if (sharedStarBase > 46 && pk.val < 0.38 * sharedStarBase) {
        continue;
      }
      if (sharedStarBase >= 245 && pk.val < 0.55 * sharedStarBase) {
        continue;
      }
    }

    let wSum = 0;
    let wx = 0;
    let wy = 0;
    let activePixels = 0;
    const yStart = Math.max(zone.y0, pk.y - radius);
    const yEnd = Math.min(zone.y1 - 1, pk.y + radius);
    const xStart = Math.max(zone.x0, pk.x - radius);
    const xEnd = Math.min(zone.x1 - 1, pk.x + radius);

    for (let y = yStart; y <= yEnd; y++) {
      const rowOffset = y * width;
      for (let x = xStart; x <= xEnd; x++) {
        const val = signalMap[rowOffset + x];
        if (val > 0) {
          wSum += val;
          wx += x * val;
          wy += y * val;
          activePixels++;
        }
      }
    }

    if (wSum <= 0 || activePixels < 4) continue;
    const cx = wx / wSum;
    const cy = wy / wSum;

    let mxx = 0;
    let myy = 0;
    for (let y = yStart; y <= yEnd; y++) {
      const rowOffset = y * width;
      for (let x = xStart; x <= xEnd; x++) {
        const val = signalMap[rowOffset + x];
        if (val > 0) {
          const dx = x - cx;
          const dy = y - cy;
          mxx += val * dx * dx;
          myy += val * dy * dy;
        }
      }
    }

    const sigmaSq = Math.max(0.25, (mxx + myy) / (2 * wSum));
    const fwhmPx = 2.3548 * Math.sqrt(sigmaSq);
    const snr = sigmaMultiplier + pk.val / Math.max(0.5, skySigma);
    const confidence = Math.min(99.9, Math.max(60.0, 58 + Math.min(41.9, snr * 3.0)));

    used.push({ x: cx, y: cy });
    candidates.push({
      zoneId: zone.id,
      cx,
      cy,
      peak: pk.val,
      flux: wSum,
      fwhmPx,
      pixelCount: activePixels,
      snr,
      confidence,
      inOverlap: isPointInOverlapEdge(cx, cy, zone),
    });

    if (candidates.length >= 10) break;
  }

  return candidates;
}

/**
 * Merges multi-zone detections and pairs positive Epoch 2 sources with nearby Epoch 1
 * negative dipole holes to quantify wide-baseline proper motion displacement.
 */
export function mergeDetectionsWithDipolePairing(
  posCandidates: DetectionCandidate[],
  negCandidates: DetectionCandidate[] = [],
  options: { width?: number; height?: number; maxDipoleDistPx?: number } = {}
): MergedDetection[] {
  const width = options.width ?? CANVAS_WIDTH;
  const height = options.height ?? CANVAS_HEIGHT;
  const maxDipoleDistPx =
    options.maxDipoleDistPx ?? Math.max(240, Math.round(0.35 * Math.hypot(width, height)));

  const clusterList = (cands: DetectionCandidate[]) => {
    const sorted = [...cands].sort((a, b) => b.flux - a.flux);
    const clusters: DetectionCandidate[][] = [];
    for (const c of sorted) {
      let hit: DetectionCandidate[] | null = null;
      for (const cl of clusters) {
        if (Math.hypot(c.cx - cl[0].cx, c.cy - cl[0].cy) <= 18) {
          hit = cl;
          break;
        }
      }
      if (hit) hit.push(c);
      else clusters.push([c]);
    }
    return clusters;
  };

  const posClusters = clusterList(posCandidates);
  const negCentroids = clusterList(negCandidates).map((cl) => {
    const tot = cl.reduce((s, c) => s + c.flux, 0);
    return {
      cx: cl.reduce((s, c) => s + c.cx * c.flux, 0) / tot,
      cy: cl.reduce((s, c) => s + c.cy * c.flux, 0) / tot,
      flux: cl[0].flux,
      used: false,
    };
  });

  // Sort deterministically by descending flux then pixel Y/X so results never shuffle
  const merged = posClusters.map((cluster) => {
    const totalFlux = cluster.reduce((acc, c) => acc + c.flux, 0);
    const cx = cluster.reduce((acc, c) => acc + c.cx * c.flux, 0) / totalFlux;
    const cy = cluster.reduce((acc, c) => acc + c.cy * c.flux, 0) / totalFlux;
    const zoneIds = [...new Set(cluster.map((c) => c.zoneId))].sort((a, b) => a - b);
    const primary = cluster[0];
    const isOverlapConfirmed = zoneIds.length >= 2;
    const maxConf = Math.max(...cluster.map((c) => c.confidence));

    // Match Epoch 1 negative dipole hole across wide baselines using flux consistency
    let bestHoleIdx = -1;
    let bestDist = maxDipoleDistPx;
    negCentroids.forEach((h, hIdx) => {
      if (h.used) return;
      const d = Math.hypot(cx - h.cx, cy - h.cy);
      if (d < 4 || d > maxDipoleDistPx) return;
      const ratio = h.flux / Math.max(1, primary.flux);
      if (ratio < 0.25 || ratio > 4.0) return;
      if (d < bestDist) {
        bestDist = d;
        bestHoleIdx = hIdx;
      }
    });

    let classification: 'Moving Body (Dipole)' | 'New Source (+5σ)' = 'New Source (+5σ)';
    let motionPx: number | null = null;
    let motionPaDeg: number | null = null;
    let e1HoleX: number | null = null;
    let e1HoleY: number | null = null;

    if (bestHoleIdx >= 0) {
      const hole = negCentroids[bestHoleIdx];
      hole.used = true;
      classification = 'Moving Body (Dipole)';
      motionPx = bestDist;
      // Astronomical PA: North (-Y) = 0 deg, East (-X) = 90 deg
      const pa = (Math.atan2(-(cx - hole.cx), -(cy - hole.cy)) * 180) / Math.PI;
      motionPaDeg = (pa + 360) % 360;
      e1HoleX = hole.cx;
      e1HoleY = hole.cy;
    }

    return {
      id: 0,
      cx,
      cy,
      peak: primary.peak,
      flux: primary.flux,
      fwhmPx: primary.fwhmPx,
      snr: primary.snr,
      confidence: isOverlapConfirmed ? Math.min(99.9, maxConf + 1.2) : maxConf,
      zoneIds,
      isOverlapConfirmed,
      overlapPairLabel: isOverlapConfirmed
        ? `OL #${zoneIds.join(',#')}`
        : `Zone #${zoneIds[0]}`,
      classification,
      motionPx,
      motionPaDeg,
      e1HoleX,
      e1HoleY,
    };
  });

  merged.sort((a, b) => b.flux - a.flux || a.cy - b.cy || a.cx - b.cx);
  merged.forEach((m, i) => {
    m.id = i + 1;
  });
  return merged;
}

export function pixelToWcsRaDec(
  px: number,
  py: number,
  centerRaDeg: number,
  centerDecDeg: number,
  width = CANVAS_WIDTH,
  height = CANVAS_HEIGHT,
  plateScaleArcsec = SPHEREX_PLATE_SCALE_ARCSEC
): { raDeg: number; decDeg: number; raFormatted: string; decFormatted: string } {
  const degPerPx = plateScaleArcsec / 3600;
  const dxDeg = -(px - width / 2) * degPerPx;
  const dyDeg = -(py - height / 2) * degPerPx;

  const ra0Rad = (centerRaDeg * Math.PI) / 180;
  const dec0Rad = (centerDecDeg * Math.PI) / 180;
  const xi = (dxDeg * Math.PI) / 180;
  const eta = (dyDeg * Math.PI) / 180;

  const denom = Math.cos(dec0Rad) - eta * Math.sin(dec0Rad);
  const raRad = ra0Rad + Math.atan2(xi, denom);
  const decRad = Math.atan2(
    Math.sin(dec0Rad) + eta * Math.cos(dec0Rad),
    Math.hypot(xi, denom)
  );

  let raDeg = ((raRad * 180) / Math.PI) % 360;
  if (raDeg < 0) raDeg += 360;
  const decDeg = (decRad * 180) / Math.PI;

  return {
    raDeg,
    decDeg,
    raFormatted: formatRa(raDeg),
    decFormatted: formatDec(decDeg),
  };
}

export function computeFieldCenterForTarget(
  targetRaDeg: number,
  targetDecDeg: number,
  tx: number,
  ty: number,
  width = CANVAS_WIDTH,
  height = CANVAS_HEIGHT,
  plateScaleArcsec = SPHEREX_PLATE_SCALE_ARCSEC
): { centerRaDeg: number; centerDecDeg: number } {
  let ra0 = targetRaDeg;
  let dec0 = targetDecDeg;

  for (let iter = 0; iter < 12; iter++) {
    const proj = pixelToWcsRaDec(tx, ty, ra0, dec0, width, height, plateScaleArcsec);
    let dRa = targetRaDeg - proj.raDeg;
    if (dRa > 180) dRa -= 360;
    if (dRa < -180) dRa += 360;
    const dDec = targetDecDeg - proj.decDeg;
    ra0 = (ra0 + dRa + 360) % 360;
    dec0 = Math.max(-89.999, Math.min(89.999, dec0 + dDec));
    if (Math.abs(dRa) < 1e-8 && Math.abs(dDec) < 1e-8) break;
  }

  return { centerRaDeg: ra0, centerDecDeg: dec0 };
}

export function formatRa(raDeg: number): string {
  const totalHours = (((raDeg / 15) % 24) + 24) % 24;
  const h = Math.floor(totalHours);
  const mFloat = (totalHours - h) * 60;
  const m = Math.floor(mFloat);
  const s = (mFloat - m) * 60;
  const sRounded = Math.round(s * 10) / 10;
  const finalS = sRounded >= 60 ? 0 : sRounded;
  const finalM = sRounded >= 60 ? (m + 1) % 60 : m;
  const finalH = sRounded >= 60 && m + 1 >= 60 ? (h + 1) % 24 : h;
  return `${String(finalH).padStart(2, '0')}h ${String(finalM).padStart(2, '0')}m ${finalS.toFixed(1).padStart(4, '0')}s`;
}

export function formatDec(decDeg: number): string {
  const sign = decDeg >= 0 ? '+' : '-';
  const abs = Math.abs(decDeg);
  const d = Math.floor(abs);
  const mFloat = (abs - d) * 60;
  const m = Math.floor(mFloat);
  const s = Math.round((mFloat - m) * 60);
  const finalS = s === 60 ? 0 : s;
  const finalM = s === 60 ? (m + 1) % 60 : m;
  const finalD = s === 60 && m + 1 >= 60 ? d + 1 : d;
  return `${sign}${String(finalD).padStart(2, '0')}° ${String(finalM).padStart(2, '0')}' ${String(finalS).padStart(2, '0')}"`;
}

export function parseRaToDeg(raStr: string): number {
  const trimmed = raStr.trim();
  const hms = trimmed.match(/(\d+)h\s*(\d+)m\s*([\d.]+)s/i);
  if (hms) {
    return (Number(hms[1]) + Number(hms[2]) / 60 + Number(hms[3]) / 3600) * 15;
  }
  const colon = trimmed.match(/^(\d+)[:\s]+(\d+)[:\s]+([\d.]+)$/);
  if (colon) {
    return (Number(colon[1]) + Number(colon[2]) / 60 + Number(colon[3]) / 3600) * 15;
  }
  const deg = Number(trimmed);
  return Number.isFinite(deg) ? ((deg % 360) + 360) % 360 : 0;
}

export function parseDecToDeg(decStr: string): number {
  const trimmed = decStr.trim();
  const dms = trimmed.match(/([+-]?)(\d+)[°d\s]+(\d+)['m\s]+([\d.]+)["s]?/i);
  if (dms) {
    const sign = dms[1] === '-' ? -1 : 1;
    return sign * (Number(dms[2]) + Number(dms[3]) / 60 + Number(dms[4]) / 3600);
  }
  const colon = trimmed.match(/^([+-]?)(\d+)[:\s]+(\d+)[:\s]+([\d.]+)$/);
  if (colon) {
    const sign = colon[1] === '-' ? -1 : 1;
    return sign * (Number(colon[2]) + Number(colon[3]) / 60 + Number(colon[4]) / 3600);
  }
  const deg = Number(trimmed);
  return Number.isFinite(deg) ? Math.max(-90, Math.min(90, deg)) : 0;
}
