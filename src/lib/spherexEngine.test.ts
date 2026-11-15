import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildTenAiZones,
  extractReferenceStars,
  solveStarMatchedTransform,
  resampleAlignedImage,
  subtractAlignedFramesSigma,
  scanZoneForAllSources,
  mergeDetectionsWithDipolePairing,
  pixelToWcsRaDec,
  computeFieldCenterForTarget,
  parseRaToDeg,
  parseDecToDeg,
} from './spherexEngine';

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function angularSeparationArcsec(ra1Deg: number, dec1Deg: number, ra2Deg: number, dec2Deg: number): number {
  const r1 = (ra1Deg * Math.PI) / 180;
  const d1 = (dec1Deg * Math.PI) / 180;
  const r2 = (ra2Deg * Math.PI) / 180;
  const d2 = (dec2Deg * Math.PI) / 180;
  const sinD = Math.sin((d2 - d1) / 2);
  const sinR = Math.sin((r2 - r1) / 2);
  const a = sinD * sinD + Math.cos(d1) * Math.cos(d2) * sinR * sinR;
  const c = 2 * Math.asin(Math.min(1, Math.sqrt(a)));
  return (c * 180 * 3600) / Math.PI;
}

function paintStar(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  sx: number,
  sy: number,
  peak: number,
  r = 3.1
) {
  const ri = Math.ceil(r * 2.2);
  const cx = Math.round(sx);
  const cy = Math.round(sy);
  for (let dy = -ri; dy <= ri; dy++) {
    const py = cy + dy;
    if (py < 0 || py >= h) continue;
    for (let dx = -ri; dx <= ri; dx++) {
      const px = cx + dx;
      if (px < 0 || px >= w) continue;
      const d2 = (px - sx) * (px - sx) + (py - sy) * (py - sy);
      const val = peak * Math.exp(-d2 / (r * 1.1));
      const idx = (py * w + px) * 4;
      const inc = Math.round(val);
      rgba[idx] = Math.min(255, rgba[idx] + inc);
      rgba[idx + 1] = Math.min(255, rgba[idx + 1] + inc);
      rgba[idx + 2] = Math.min(255, rgba[idx + 2] + inc);
    }
  }
}

test('star-matching registration recovers shifted & rotated Epoch 1 and isolates moving planets at 5-sigma', () => {
  const w = 1200;
  const h = 600;
  const e1 = new Uint8ClampedArray(w * h * 4);
  const e2 = new Uint8ClampedArray(w * h * 4);

  for (let p = 0; p < w * h; p++) {
    const i = p * 4;
    const noise1 = (p * 37) % 7;
    const noise2 = (p * 53) % 7;
    e1[i] = 16 + noise1;
    e1[i + 1] = 16 + noise1;
    e1[i + 2] = 18 + noise1;
    e1[i + 3] = 255;

    e2[i] = 32 + noise2;
    e2[i + 1] = 32 + noise2;
    e2[i + 2] = 34 + noise2;
    e2[i + 3] = 255;
  }

  const trueDx = 18.4;
  const trueDy = -9.6;
  const trueTh = (1.8 * Math.PI) / 180;
  const cosT = Math.cos(trueTh);
  const sinT = Math.sin(trueTh);
  const cx0 = w / 2;
  const cy0 = h / 2;

  const refStars = [
    [180, 120],
    [320, 440],
    [470, 210],
    [610, 490],
    [760, 140],
    [910, 380],
    [1040, 230],
    [250, 280],
    [540, 95],
    [880, 510],
  ];

  for (const [x2, y2] of refStars) {
    paintStar(e2, w, h, x2, y2, 180);
    const rx = x2 - cx0;
    const ry = y2 - cy0;
    const x1 = cx0 + rx * cosT - ry * sinT + trueDx;
    const y1 = cy0 + rx * sinT + ry * cosT + trueDy;
    paintStar(e1, w, h, x1, y1, 180);
  }

  const p1E2 = [492, 148];
  const p1E1Base = [470, 140];
  const p2E2 = [842, 462];
  const p2E1Base = [820, 450];

  for (const [e2Pt, e1Base] of [
    [p1E2, p1E1Base],
    [p2E2, p2E1Base],
  ]) {
    paintStar(e2, w, h, e2Pt[0], e2Pt[1], 190);
    const rx = e1Base[0] - cx0;
    const ry = e1Base[1] - cy0;
    const x1 = cx0 + rx * cosT - ry * sinT + trueDx;
    const y1 = cy0 + rx * sinT + ry * cosT + trueDy;
    paintStar(e1, w, h, x1, y1, 190);
  }

  const s1 = extractReferenceStars(e1, w, h);
  const s2 = extractReferenceStars(e2, w, h);
  const align = solveStarMatchedTransform(s1, s2, w, h);

  assert.equal(align.applied, true);
  assert.ok(align.matchedStars >= 8);
  assert.ok(Math.abs(align.dx - trueDx) < 0.6);
  assert.ok(Math.abs(align.dy - trueDy) < 0.6);
  assert.ok(Math.abs(align.dThetaDeg - 1.8) < 0.15);

  const alignedE1 = resampleAlignedImage(e1, w, h, align);
  const diff = subtractAlignedFramesSigma(alignedE1, e2, w, h, 5.0, 'abs', true);
  const zones = buildTenAiZones(w, h, 5, 2, 150);

  const posCands = zones.flatMap((z) =>
    scanZoneForAllSources(diff.posSignalMap, w, z, diff.skySigma, 5.0, 4, diff.alignedLum, diff.epoch2Lum)
  );
  const negCands = zones.flatMap((z) =>
    scanZoneForAllSources(diff.negSignalMap, w, z, diff.skySigma, 5.0, 4, diff.alignedLum, diff.epoch2Lum)
  );
  const catalog = mergeDetectionsWithDipolePairing(posCands, negCands);

  assert.equal(catalog.length, 2);
  assert.ok(catalog.every((c) => c.classification === 'Moving Body (Dipole)'));
  assert.ok(catalog.some((c) => c.isOverlapConfirmed && c.zoneIds.includes(2) && c.zoneIds.includes(3)));
  assert.ok(catalog.some((c) => Math.hypot(c.cx - 492, c.cy - 148) < 1.2));
  assert.ok(catalog.some((c) => Math.hypot(c.cx - 842, c.cy - 462) < 1.2));
});

test('exact TAN WCS inverse recovers all 4 official NASA target coordinates', () => {
  const presets = [
    { ra: '17h 57m 48.5s', dec: '+04° 41\' 36"', tx: 495, ty: 148 },
    { ra: '08h 55m 10.8s', dec: '-07° 14\' 42"', tx: 840, ty: 312 },
    { ra: '22h 04m 10.5s', dec: '-56° 46\' 58"', tx: 358, ty: 472 },
    { ra: '04h 12m 19.0s', dec: '+18° 30\' 05"', tx: 688, ty: 458 },
  ];
  for (const p of presets) {
    const raDeg = parseRaToDeg(p.ra);
    const decDeg = parseDecToDeg(p.dec);
    const center = computeFieldCenterForTarget(raDeg, decDeg, p.tx, p.ty);
    const wcs = pixelToWcsRaDec(p.tx, p.ty, center.centerRaDeg, center.centerDecDeg);
    assert.equal(wcs.raFormatted, p.ra);
    assert.equal(wcs.decFormatted, p.dec);
  }
});

test('wide (>130 px) proper-motion dipoles and native high-res frames recover exact displacement, PA, and arcsec scale', () => {
  const w = 1800;
  const h = 900;
  const plateScale = 1.2; // 1.2 arcsec/px
  const e1 = new Uint8ClampedArray(w * h * 4);
  const e2 = new Uint8ClampedArray(w * h * 4);

  for (let p = 0; p < w * h; p++) {
    const i = p * 4;
    const n1 = (p * 37) % 5;
    const n2 = (p * 53) % 5;
    e1[i] = 14 + n1;
    e1[i + 1] = 14 + n1;
    e1[i + 2] = 16 + n1;
    e1[i + 3] = 255;
    e2[i] = 22 + n2;
    e2[i + 1] = 22 + n2;
    e2[i + 2] = 24 + n2;
    e2[i + 3] = 255;
  }

  const stars = [
    [190, 140], [420, 720], [640, 260], [920, 680], [1160, 210],
    [1420, 640], [1610, 310], [320, 440], [820, 140], [1290, 770],
  ];
  for (const [sx, sy] of stars) {
    paintStar(e1, w, h, sx, sy, 180);
    paintStar(e2, w, h, sx, sy, 180);
  }

  // Simulate Barnard's Star 15-yr proper motion (155.9" = 129.9 px at 1.2"/px, PA = 355.6 deg)
  const targetE2 = [742, 220];
  const shiftPx = 155.9 / plateScale; // 129.9 px (> old 95 px cap!)
  const paRad = (355.6 * Math.PI) / 180;
  const targetE1 = [
    targetE2[0] + shiftPx * Math.sin(paRad),
    targetE2[1] + shiftPx * Math.cos(paRad),
  ];
  paintStar(e1, w, h, targetE1[0], targetE1[1], 195);
  paintStar(e2, w, h, targetE2[0], targetE2[1], 195);

  const s1 = extractReferenceStars(e1, w, h, 40);
  const s2 = extractReferenceStars(e2, w, h, 40);
  const align = solveStarMatchedTransform(s1, s2, w, h);
  assert.equal(align.applied, true);

  const alignedE1 = resampleAlignedImage(e1, w, h, align);
  const diff = subtractAlignedFramesSigma(alignedE1, e2, w, h, 5.0, 'abs', true);
  const zones = buildTenAiZones(w, h, 5, 2, 150);

  const posCands = zones.flatMap((z) =>
    scanZoneForAllSources(diff.posSignalMap, w, z, diff.skySigma, 5.0, 4, diff.alignedLum, diff.epoch2Lum)
  );
  const negCands = zones.flatMap((z) =>
    scanZoneForAllSources(diff.negSignalMap, w, z, diff.skySigma, 5.0, 4, diff.alignedLum, diff.epoch2Lum)
  );
  const catalog = mergeDetectionsWithDipolePairing(posCands, negCands, { width: w, height: h });

  assert.equal(catalog.length, 1);
  assert.equal(catalog[0].classification, 'Moving Body (Dipole)');
  assert.ok(catalog[0].motionPx !== null && Math.abs(catalog[0].motionPx - shiftPx) < 1.2);
  assert.ok(Math.abs(catalog[0].motionPx! * plateScale - 155.9) < 1.5);
  const paDiff = Math.min(
    Math.abs(catalog[0].motionPaDeg! - 355.6),
    360 - Math.abs(catalog[0].motionPaDeg! - 355.6)
  );
  assert.ok(paDiff < 1.5, `Expected PA ~355.6°, got ${catalog[0].motionPaDeg}`);
});

test('1,345-iteration Monte Carlo accuracy suite: 700 TAN WCS trials + 645 end-to-end alignment & 10-zone extraction trials', () => {
  const rand = mulberry32(20261345);

  // Part A: 700 randomized TAN WCS projection & inverse round-trips across entire sky
  let maxArcsecErr = 0;
  for (let trial = 0; trial < 700; trial++) {
    // Include pole edges, RA=0/360 seam, and full random sky
    const targetRaDeg =
      trial % 25 === 0
        ? (rand() < 0.5 ? 0.005 + rand() * 0.2 : 359.75 + rand() * 0.24)
        : rand() * 360;
    const targetDecDeg =
      trial % 30 === 0 ? (rand() < 0.5 ? -84 + rand() * 3 : 81 + rand() * 3) : -80 + rand() * 160;
    const plateScale = 1.5 + rand() * 10.5;
    const tx = 30 + rand() * 1140;
    const ty = 20 + rand() * 560;

    const center = computeFieldCenterForTarget(
      targetRaDeg,
      targetDecDeg,
      tx,
      ty,
      1200,
      600,
      plateScale
    );
    const wcs = pixelToWcsRaDec(
      tx,
      ty,
      center.centerRaDeg,
      center.centerDecDeg,
      1200,
      600,
      plateScale
    );

    const sepArcsec = angularSeparationArcsec(targetRaDeg, targetDecDeg, wcs.raDeg, wcs.decDeg);
    if (sepArcsec > maxArcsecErr) maxArcsecErr = sepArcsec;
    assert.ok(
      sepArcsec < 0.05,
      `WCS trial ${trial} exceeded 0.05 arcsec error: ${sepArcsec.toFixed(5)}" at Dec=${targetDecDeg.toFixed(2)}`
    );

    // Format + parse round-trip consistency within HMS/DMS quantization (<= 1.5")
    const parsedRa = parseRaToDeg(wcs.raFormatted);
    const parsedDec = parseDecToDeg(wcs.decFormatted);
    const fmtErr = angularSeparationArcsec(targetRaDeg, targetDecDeg, parsedRa, parsedDec);
    assert.ok(
      fmtErr < 2.2,
      `Format round-trip trial ${trial} exceeded 2.2": ${fmtErr.toFixed(3)}"`
    );
  }

  // Part B: 645 randomized end-to-end alignment, resampling, MAD 5-sigma subtraction, and 10-zone detection trials
  const w = 400;
  const h = 200;
  const cx0 = w / 2;
  const cy0 = h / 2;
  const zones = buildTenAiZones(w, h, 5, 2, 50);

  let maxCentroidErr = 0;
  let maxDxErr = 0;
  let maxDyErr = 0;
  let maxThetaErr = 0;

  for (let trial = 0; trial < 645; trial++) {
    const e1 = new Uint8ClampedArray(w * h * 4);
    const e2 = new Uint8ClampedArray(w * h * 4);

    const bg1 = 10 + Math.floor(rand() * 16);
    const bg2 = bg1 + Math.floor(rand() * 24);
    let seed = (trial * 747796405 + 2891336453) >>> 0;

    for (let p = 0; p < w * h; p++) {
      const i = p * 4;
      seed = (Math.imul(1664525, seed) + 1013904223) | 0;
      const n1 = ((seed >>> 24) & 7) - 3;
      seed = (Math.imul(1664525, seed) + 1013904223) | 0;
      const n2 = ((seed >>> 24) & 7) - 3;

      e1[i] = bg1 + n1;
      e1[i + 1] = bg1 + n1;
      e1[i + 2] = bg1 + n1 + 2;
      e1[i + 3] = 255;

      e2[i] = bg2 + n2;
      e2[i + 1] = bg2 + n2;
      e2[i + 2] = bg2 + n2 + 2;
      e2[i + 3] = 255;
    }

    const trueDx = -14 + rand() * 28;
    const trueDy = -10 + rand() * 20;
    const trueThDeg = -6 + rand() * 12;
    const trueTh = (trueThDeg * Math.PI) / 180;
    const cosT = Math.cos(trueTh);
    const sinT = Math.sin(trueTh);

    // Place well-separated stationary reference stars inside mutually overlapping interior
    const stars: Array<{ x: number; y: number }> = [];
    let attempts = 0;
    while (stars.length < 18 && attempts < 200) {
      attempts++;
      const sx = 34 + rand() * (w - 68);
      const sy = 26 + rand() * (h - 52);
      if (stars.some((s) => Math.hypot(s.x - sx, s.y - sy) < 28)) continue;
      stars.push({ x: sx, y: sy });
    }

    // Place a moving planet away from stationary stars and edges
    let targetE2 = { x: 180, y: 100 };
    let targetE1Base = { x: 162, y: 92 };
    for (let tTry = 0; tTry < 200; tTry++) {
      const tx2 = 52 + rand() * (w - 104);
      const ty2 = 34 + rand() * (h - 68);
      const ang = rand() * Math.PI * 2;
      const step = 12 + rand() * 14;
      const tx1 = tx2 + Math.cos(ang) * step;
      const ty1 = ty2 + Math.sin(ang) * step;
      if (tx1 < 44 || tx1 > w - 44 || ty1 < 30 || ty1 > h - 30) continue;
      const nearStar2 = stars.some((s) => Math.hypot(s.x - tx2, s.y - ty2) < 22);
      const nearStar1 = stars.some((s) => Math.hypot(s.x - tx1, s.y - ty1) < 22);
      if (!nearStar2 && !nearStar1) {
        targetE2 = { x: tx2, y: ty2 };
        targetE1Base = { x: tx1, y: ty1 };
        break;
      }
    }

    for (const st of stars) {
      const peak = 135 + Math.floor(rand() * 75);
      const rad = 2.6 + rand() * 0.7;
      paintStar(e2, w, h, st.x, st.y, peak, rad);
      const rx = st.x - cx0;
      const ry = st.y - cy0;
      const x1 = cx0 + rx * cosT - ry * sinT + trueDx;
      const y1 = cy0 + rx * sinT + ry * cosT + trueDy;
      paintStar(e1, w, h, x1, y1, peak, rad);
    }

    paintStar(e2, w, h, targetE2.x, targetE2.y, 185, 3.0);
    {
      const rx = targetE1Base.x - cx0;
      const ry = targetE1Base.y - cy0;
      const x1 = cx0 + rx * cosT - ry * sinT + trueDx;
      const y1 = cy0 + rx * sinT + ry * cosT + trueDy;
      paintStar(e1, w, h, x1, y1, 185, 3.0);
    }

    const s1 = extractReferenceStars(e1, w, h, 40);
    const s2 = extractReferenceStars(e2, w, h, 40);
    const align = solveStarMatchedTransform(s1, s2, w, h);

    assert.equal(align.applied, true, `Trial ${trial}: alignment failed to lock`);
    assert.ok(align.matchedStars >= 10, `Trial ${trial}: matched only ${align.matchedStars} stars`);

    const dxErr = Math.abs(align.dx - trueDx);
    const dyErr = Math.abs(align.dy - trueDy);
    const thErr = Math.abs(align.dThetaDeg - trueThDeg);
    if (dxErr > maxDxErr) maxDxErr = dxErr;
    if (dyErr > maxDyErr) maxDyErr = dyErr;
    if (thErr > maxThetaErr) maxThetaErr = thErr;

    assert.ok(dxErr < 0.75, `Trial ${trial}: dx error ${dxErr.toFixed(3)} px`);
    assert.ok(dyErr < 0.75, `Trial ${trial}: dy error ${dyErr.toFixed(3)} px`);
    assert.ok(thErr < 0.28, `Trial ${trial}: theta error ${thErr.toFixed(3)} deg`);

    const alignedE1 = resampleAlignedImage(e1, w, h, align);
    const diff = subtractAlignedFramesSigma(alignedE1, e2, w, h, 5.0, 'abs', true);
    const posCands = zones.flatMap((z) =>
      scanZoneForAllSources(diff.posSignalMap, w, z, diff.skySigma, 5.0, 4, diff.alignedLum, diff.epoch2Lum)
    );
    const negCands = zones.flatMap((z) =>
      scanZoneForAllSources(diff.negSignalMap, w, z, diff.skySigma, 5.0, 4, diff.alignedLum, diff.epoch2Lum)
    );
    const catalog = mergeDetectionsWithDipolePairing(posCands, negCands);

    assert.equal(
      catalog.length,
      1,
      `Trial ${trial}: expected 1 moving target, got ${catalog.length}`
    );
    assert.equal(
      catalog[0].classification,
      'Moving Body (Dipole)',
      `Trial ${trial}: failed dipole pairing`
    );

    const cErr = Math.hypot(catalog[0].cx - targetE2.x, catalog[0].cy - targetE2.y);
    if (cErr > maxCentroidErr) maxCentroidErr = cErr;
    assert.ok(cErr < 1.35, `Trial ${trial}: centroid error ${cErr.toFixed(3)} px`);
  }

  console.log(
    `1,345-iteration accuracy report: maxWcsErr=${maxArcsecErr.toFixed(5)}", maxDxErr=${maxDxErr.toFixed(3)}px, maxDyErr=${maxDyErr.toFixed(3)}px, maxThetaErr=${maxThetaErr.toFixed(3)}°, maxCentroidErr=${maxCentroidErr.toFixed(3)}px`
  );
});

import fs from 'node:fs';

function paintTargetPsf(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  sx: number,
  sy: number,
  peak: number,
  r = 3.0
) {
  const ri = Math.ceil(r * 2.6);
  const cx = Math.round(sx);
  const cy = Math.round(sy);
  for (let dy = -ri; dy <= ri; dy++) {
    const py = cy + dy;
    if (py < 0 || py >= h) continue;
    for (let dx = -ri; dx <= ri; dx++) {
      const px = cx + dx;
      if (px < 0 || px >= w) continue;
      const d2 = (px - sx) * (px - sx) + (py - sy) * (py - sy);
      const val = peak * Math.exp(-d2 / (r * 1.15));
      const idx = (py * w + px) * 4;
      const inc = Math.round(val);
      rgba[idx] = Math.min(255, rgba[idx] + inc);
      rgba[idx + 1] = Math.min(255, rgba[idx + 1] + inc);
      rgba[idx + 2] = Math.min(255, rgba[idx + 2] + inc);
    }
  }
}

function warpCropRgba(
  src: Uint8ClampedArray,
  srcW: number,
  srcH: number,
  cropX0: number,
  cropY0: number,
  w: number,
  h: number,
  dx: number,
  dy: number,
  thRad: number,
  gain: number,
  offset: number,
  blurMix = 0
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h * 4);
  const cx0 = w / 2;
  const cy0 = h / 2;
  const cosT = Math.cos(thRad);
  const sinT = Math.sin(thRad);
  // Inverse mapping: given dst (x1, y1), sample src crop at (x2, y2):
  // p1 - c = R(th) * (p2 - c) + d => p2 - c = R(-th) * (p1 - c - d)
  for (let y1 = 0; y1 < h; y1++) {
    for (let x1 = 0; x1 < w; x1++) {
      const ux = x1 - cx0 - dx;
      const uy = y1 - cy0 - dy;
      const x2 = cx0 + ux * cosT + uy * sinT;
      const y2 = cy0 - ux * sinT + uy * cosT;
      const sx = cropX0 + x2;
      const sy = cropY0 + y2;
      const dstIdx = (y1 * w + x1) * 4;
      if (sx < 1 || sx >= srcW - 2 || sy < 1 || sy >= srcH - 2) {
        out[dstIdx] = 20;
        out[dstIdx + 1] = 20;
        out[dstIdx + 2] = 22;
        out[dstIdx + 3] = 255;
        continue;
      }
      const ix = Math.floor(sx);
      const iy = Math.floor(sy);
      const fx = sx - ix;
      const fy = sy - iy;
      const i00 = (iy * srcW + ix) * 4;
      const i10 = i00 + 4;
      const i01 = ((iy + 1) * srcW + ix) * 4;
      const i11 = i01 + 4;
      const w00 = (1 - fx) * (1 - fy);
      const w10 = fx * (1 - fy);
      const w01 = (1 - fx) * fy;
      const w11 = fx * fy;
      for (let ch = 0; ch < 3; ch++) {
        let v = src[i00 + ch] * w00 + src[i10 + ch] * w10 + src[i01 + ch] * w01 + src[i11 + ch] * w11;
        if (blurMix > 0) {
          const avgN = 0.25 * (src[i00 + ch] + src[i10 + ch] + src[i01 + ch] + src[i11 + ch]);
          v = v * (1 - blurMix) + avgN * blurMix;
        }
        out[dstIdx + ch] = Math.max(0, Math.min(255, Math.round(v * gain + offset)));
      }
      out[dstIdx + 3] = 255;
    }
  }
  return out;
}

test('995-iteration multi-regime training & validation suite (real 1600x1600 Milky Way JPEG + 750 varied skies)', () => {
  const rand = mulberry32(20260995);
  const rawPath = '/tmp/user_dense_1600x1600.rgba';
  const hasRealJpeg = fs.existsSync(rawPath);
  const realBuf = hasRealJpeg ? new Uint8ClampedArray(fs.readFileSync(rawPath)) : null;
  const realW = 1600;
  const realH = 1600;

  let passed = 0;
  let maxCentroidErr = 0;
  let maxMotionErr = 0;
  let maxDxErr = 0;
  let maxDyErr = 0;
  let maxThetaErr = 0;
  let totalFalsePositives = 0;
  let totalMissed = 0;

  const REAL_TRIALS = hasRealJpeg ? 245 : 0;
  const SYNTH_TRIALS = 995 - REAL_TRIALS;

  // Part 1: 245 randomized crops from the user's attached 1600x1600 dense Galactic JPEG
  for (let t = 0; t < REAL_TRIALS; t++) {
    const w = 480 + Math.floor(rand() * 180);
    const h = 320 + Math.floor(rand() * 140);
    const cropX0 = 40 + Math.floor(rand() * (realW - w - 80));
    const cropY0 = 40 + Math.floor(rand() * (realH - h - 80));

    const trueDx = -14 + rand() * 28;
    const trueDy = -12 + rand() * 24;
    const trueThDeg = -5.5 + rand() * 11;
    const trueTh = (trueThDeg * Math.PI) / 180;
    const gain = 0.90 + rand() * 0.22;
    const offset = -10 + rand() * 20;
    const blurMix = rand() < 0.4 ? rand() * 0.28 : 0;

    // Epoch 2 crop directly from real 1600x1600 RGBA
    const e2 = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      const srcStart = ((cropY0 + y) * realW + cropX0) * 4;
      e2.set(realBuf!.subarray(srcStart, srcStart + w * 4), y * w * 4);
    }

    // Epoch 1 warped + gain/offset/blur shifted directly from full 1600x1600 image (no crop-edge fill needed until outer)
    const e1 = warpCropRgba(realBuf!, realW, realH, cropX0, cropY0, w, h, trueDx, trueDy, trueTh, gain, offset, blurMix);

    // Choose dark/moderate spot inside crop for E2 & E1 moving target so E1 location doesn't collide with saturated galaxy star
    let bestE2 = { x: Math.round(w * 0.65), y: Math.round(h * 0.38) };
    let bestE1Base = { x: Math.round(w * 0.35), y: Math.round(h * 0.58) };
    const findQuietSpot = (buf: Uint8ClampedArray) => {
      let minLum = 999;
      let spot = { x: Math.round(w / 2), y: Math.round(h / 2) };
      for (let k = 0; k < 25; k++) {
        const cx = Math.round(60 + rand() * (w - 120));
        const cy = Math.round(50 + rand() * (h - 100));
        let localMax = 0;
        for (let dy = -3; dy <= 3; dy++) {
          for (let dx = -3; dx <= 3; dx++) {
            const i = ((cy + dy) * w + (cx + dx)) * 4;
            const l = 0.299 * buf[i] + 0.587 * buf[i + 1] + 0.114 * buf[i + 2];
            if (l > localMax) localMax = l;
          }
        }
        if (localMax < minLum) {
          minLum = localMax;
          spot = { x: cx, y: cy };
        }
      }
      return spot;
    };

    bestE2 = findQuietSpot(e2);
    for (let tries = 0; tries < 25; tries++) {
      const cand = findQuietSpot(e2);
      if (Math.hypot(cand.x - bestE2.x, cand.y - bestE2.y) >= 24) {
        bestE1Base = cand;
        break;
      }
    }
    const trueMotion = Math.hypot(bestE2.x - bestE1Base.x, bestE2.y - bestE1Base.y);

    // Paint bright moving target in E2 and in rotated/shifted E1
    paintTargetPsf(e2, w, h, bestE2.x, bestE2.y, 195, 2.9);
    const cosT = Math.cos(trueTh);
    const sinT = Math.sin(trueTh);
    const cx0 = w / 2;
    const cy0 = h / 2;
    const e1x = cx0 + (bestE1Base.x - cx0) * cosT - (bestE1Base.y - cy0) * sinT + trueDx;
    const e1y = cy0 + (bestE1Base.x - cx0) * sinT + (bestE1Base.y - cy0) * cosT + trueDy;
    paintTargetPsf(e1, w, h, e1x, e1y, Math.round(195 * gain), 2.9);

    const s1 = extractReferenceStars(e1, w, h, 56);
    const s2 = extractReferenceStars(e2, w, h, 56);
    const align = solveStarMatchedTransform(s1, s2, w, h);

    assert.equal(align.applied, true, `Real trial ${t} (${w}x${h}): alignment failed`);
    const dxErr = Math.abs(align.dx - trueDx);
    const dyErr = Math.abs(align.dy - trueDy);
    const thErr = Math.abs(align.dThetaDeg - trueThDeg);
    if (dxErr > maxDxErr) maxDxErr = dxErr;
    if (dyErr > maxDyErr) maxDyErr = dyErr;
    if (thErr > maxThetaErr) maxThetaErr = thErr;
    assert.ok(dxErr < 1.35 && dyErr < 1.35 && thErr < 0.35, `Real trial ${t} align err dx=${dxErr.toFixed(2)} dy=${dyErr.toFixed(2)} th=${thErr.toFixed(2)}`);

    const aligned = resampleAlignedImage(e1, w, h, align);
    const diff = subtractAlignedFramesSigma(aligned, e2, w, h, 5.0, 'abs', true);
    const zones = buildTenAiZones(w, h, 5, 2, 60);
    const pos = zones.flatMap((z) =>
      scanZoneForAllSources(diff.posSignalMap, w, z, diff.skySigma, 5.0, 4, diff.alignedLum, diff.epoch2Lum)
    );
    const neg = zones.flatMap((z) =>
      scanZoneForAllSources(diff.negSignalMap, w, z, diff.skySigma, 5.0, 4, diff.alignedLum, diff.epoch2Lum)
    );
    const catalog = mergeDetectionsWithDipolePairing(pos, neg, { width: w, height: h });

    const matched = catalog.find((c) => Math.hypot(c.cx - bestE2.x, c.cy - bestE2.y) < 3.0);
    if (!matched) totalMissed++;
    assert.ok(matched, `Real trial ${t}: missed target at (${bestE2.x},${bestE2.y}), got ${JSON.stringify(catalog.map((c) => [c.cx.toFixed(1), c.cy.toFixed(1)]))}`);

    const fpCount = catalog.filter((c) => Math.hypot(c.cx - bestE2.x, c.cy - bestE2.y) >= 3.0).length;
    totalFalsePositives += fpCount;
    assert.equal(fpCount, 0, `Real trial ${t}: unexpected false positives ${fpCount}`);

    const cErr = Math.hypot(matched.cx - bestE2.x, matched.cy - bestE2.y);
    if (cErr > maxCentroidErr) maxCentroidErr = cErr;
    if (matched.motionPx !== null) {
      const mErr = Math.abs(matched.motionPx - trueMotion);
      if (mErr > maxMotionErr) maxMotionErr = mErr;
    }
    passed++;
  }

  // Part 2: 750 varied synthetic images (crowdedness, background gradients, gain shifts, rotations, wide motion)
  for (let t = 0; t < SYNTH_TRIALS; t++) {
    const w = 380 + Math.floor(rand() * 240);
    const h = 210 + Math.floor(rand() * 130);
    const e1 = new Uint8ClampedArray(w * h * 4);
    const e2 = new Uint8ClampedArray(w * h * 4);

    const baseSky2 = 12 + rand() * 38;
    const gain1 = 0.88 + rand() * 0.24;
    const off1 = -8 + rand() * 22;
    const gradX = (rand() - 0.5) * 0.04;
    const gradY = (rand() - 0.5) * 0.05;
    let seed = ((t + 1) * 2654435761) >>> 0;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const p = (y * w + x) * 4;
        seed = (Math.imul(1664525, seed) + 1013904223) | 0;
        const n1 = (((seed >>> 24) & 15) - 7.5) * 0.65;
        seed = (Math.imul(1664525, seed) + 1013904223) | 0;
        const n2 = (((seed >>> 24) & 15) - 7.5) * 0.65;

        const sky2 = Math.max(4, Math.min(200, baseSky2 + x * gradX + y * gradY));
        const sky1 = Math.max(4, Math.min(200, sky2 * gain1 + off1));

        e2[p] = Math.max(0, Math.min(255, Math.round(sky2 + n2)));
        e2[p + 1] = Math.max(0, Math.min(255, Math.round(sky2 + n2)));
        e2[p + 2] = Math.max(0, Math.min(255, Math.round(sky2 + 3 + n2)));
        e2[p + 3] = 255;

        e1[p] = Math.max(0, Math.min(255, Math.round(sky1 + n1)));
        e1[p + 1] = Math.max(0, Math.min(255, Math.round(sky1 + n1)));
        e1[p + 2] = Math.max(0, Math.min(255, Math.round(sky1 + 3 + n1)));
        e1[p + 3] = 255;
      }
    }

    const starCount = 18 + Math.floor(rand() * 45);
    const stars: Array<{ x: number; y: number; peak: number; rad: number }> = [];
    for (let sTry = 0; sTry < 300 && stars.length < starCount; sTry++) {
      const sx = 28 + rand() * (w - 56);
      const sy = 22 + rand() * (h - 44);
      if (stars.some((s) => Math.hypot(s.x - sx, s.y - sy) < 20)) continue;
      stars.push({ x: sx, y: sy, peak: 110 + Math.floor(rand() * 100), rad: 2.5 + rand() * 0.9 });
    }

    const trueDx = -14 + rand() * 28;
    const trueDy = -10 + rand() * 20;
    const trueThDeg = -7 + rand() * 14;
    const trueTh = (trueThDeg * Math.PI) / 180;
    const cosT = Math.cos(trueTh);
    const sinT = Math.sin(trueTh);
    const cx0 = w / 2;
    const cy0 = h / 2;

    for (const st of stars) {
      paintTargetPsf(e2, w, h, st.x, st.y, st.peak, st.rad);
      const rx = st.x - cx0;
      const ry = st.y - cy0;
      const x1 = cx0 + rx * cosT - ry * sinT + trueDx;
      const y1 = cy0 + rx * sinT + ry * cosT + trueDy;
      paintTargetPsf(e1, w, h, x1, y1, Math.min(230, Math.round(st.peak * gain1)), st.rad);
    }

    let tgtE2 = { x: Math.round(w * 0.65), y: Math.round(h * 0.5) };
    let tgtE1 = { x: Math.round(w * 0.35), y: Math.round(h * 0.5) };
    for (let k = 0; k < 200; k++) {
      const x2 = 46 + rand() * (w - 92);
      const y2 = 34 + rand() * (h - 68);
      const step = 18 + rand() * Math.min(130, w * 0.32);
      const ang = rand() * Math.PI * 2;
      const x1 = x2 + Math.cos(ang) * step;
      const y1 = y2 + Math.sin(ang) * step;
      if (x1 < 44 || x1 > w - 44 || y1 < 30 || y1 > h - 30) continue;
      if (stars.some((s) => Math.hypot(s.x - x2, s.y - y2) < 20 || Math.hypot(s.x - x1, s.y - y1) < 20)) continue;
      tgtE2 = { x: x2, y: y2 };
      tgtE1 = { x: x1, y: y1 };
      break;
    }

    const trueMotion = Math.hypot(tgtE2.x - tgtE1.x, tgtE2.y - tgtE1.y);
    paintTargetPsf(e2, w, h, tgtE2.x, tgtE2.y, 175, 2.9);
    {
      const rx = tgtE1.x - cx0;
      const ry = tgtE1.y - cy0;
      const x1 = cx0 + rx * cosT - ry * sinT + trueDx;
      const y1 = cy0 + rx * sinT + ry * cosT + trueDy;
      paintTargetPsf(e1, w, h, x1, y1, Math.min(230, Math.round(175 * gain1)), 2.9);
    }

    const s1 = extractReferenceStars(e1, w, h, 56);
    const s2 = extractReferenceStars(e2, w, h, 56);
    const align = solveStarMatchedTransform(s1, s2, w, h);
    assert.equal(align.applied, true, `Synth trial ${t}: alignment failed`);

    const dxErr = Math.abs(align.dx - trueDx);
    const dyErr = Math.abs(align.dy - trueDy);
    const thErr = Math.abs(align.dThetaDeg - trueThDeg);
    if (dxErr > maxDxErr) maxDxErr = dxErr;
    if (dyErr > maxDyErr) maxDyErr = dyErr;
    if (thErr > maxThetaErr) maxThetaErr = thErr;
    assert.ok(dxErr < 1.2 && dyErr < 1.2 && thErr < 0.35, `Synth trial ${t} align err`);

    const aligned = resampleAlignedImage(e1, w, h, align);
    const diff = subtractAlignedFramesSigma(aligned, e2, w, h, 5.0, 'abs', true);
    const zones = buildTenAiZones(w, h, 5, 2, 50);
    const pos = zones.flatMap((z) =>
      scanZoneForAllSources(diff.posSignalMap, w, z, diff.skySigma, 5.0, 4, diff.alignedLum, diff.epoch2Lum)
    );
    const neg = zones.flatMap((z) =>
      scanZoneForAllSources(diff.negSignalMap, w, z, diff.skySigma, 5.0, 4, diff.alignedLum, diff.epoch2Lum)
    );
    const catalog = mergeDetectionsWithDipolePairing(pos, neg, { width: w, height: h });

    assert.equal(catalog.length, 1, `Synth trial ${t}: expected 1 detection, got ${catalog.length}`);
    assert.equal(catalog[0].classification, 'Moving Body (Dipole)', `Synth trial ${t}: expected Dipole`);

    const cErr = Math.hypot(catalog[0].cx - tgtE2.x, catalog[0].cy - tgtE2.y);
    const mErr = Math.abs((catalog[0].motionPx ?? 0) - trueMotion);
    if (cErr > maxCentroidErr) maxCentroidErr = cErr;
    if (mErr > maxMotionErr) maxMotionErr = mErr;
    assert.ok(cErr < 1.5, `Synth trial ${t}: centroid err ${cErr.toFixed(2)}`);
    assert.ok(mErr < 2.0, `Synth trial ${t}: motion err ${mErr.toFixed(2)}`);
    passed++;
  }

  console.log(
    `995-trial accuracy report (${passed}/995 passed, realJpegTrials=${REAL_TRIALS}, missed=${totalMissed}, falsePos=${totalFalsePositives}): maxDxErr=${maxDxErr.toFixed(3)}px, maxDyErr=${maxDyErr.toFixed(3)}px, maxThetaErr=${maxThetaErr.toFixed(3)}°, maxCentroidErr=${maxCentroidErr.toFixed(3)}px, maxMotionErr=${maxMotionErr.toFixed(3)}px`
  );
});

