/* Uncertainty explorer and related widgets.
   Registers block renderers for the app.js contract.
   Classes prefixed with ux- for isolation.
*/

window.BLOCKS = window.BLOCKS || {};

// ============================================================================
// Utilities
// ============================================================================

/** Simple seeded PRNG for reproducible random subset selection */
class SeededRandom {
  constructor(seed) {
    this.seed = seed;
  }
  next() {
    this.seed = (this.seed * 9301 + 49297) % 233280;
    return this.seed / 233280;
  }
}

/** Clamp value to [min, max] */
function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

/** Format number with given decimals */
function fmt(value, decimals = 2) {
  if (value === null || value === undefined) return '—';
  if (decimals === 'int') return Math.round(value).toString();
  return value.toFixed(decimals);
}

/** Get CSS variable value (e.g., --accent) and parse as color */
function getCSSVar(varName) {
  const root = document.documentElement;
  return getComputedStyle(root).getPropertyValue(varName).trim();
}

/** Get RdYlBu color via CSS variable */
function getRdylbuColor(t, reverse = false) {
  const clamped = clamp(t, 0, 1);
  const idx = Math.round(clamped * 10);
  const varName = reverse ? `--rdylbu-${idx}` : `--rdylbu-${10 - idx}`;
  return getCSSVar(varName);
}

/** Calculate percentile of array */
function percentile(arr, p) {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.ceil(sorted.length * p) - 1;
  return sorted[Math.max(0, idx)];
}

/** Calculate standard deviation (population) */
function std(arr) {
  if (arr.length <= 1) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  const sq = arr.map(x => (x - mean) ** 2).reduce((a, b) => a + b, 0);
  return Math.sqrt(sq / arr.length); // population std
}

/** Request animation frame throttled redraw */
function rafThrottle(fn) {
  let pending = false;
  return () => {
    if (!pending) {
      pending = true;
      requestAnimationFrame(() => {
        fn();
        pending = false;
      });
    }
  };
}

/** Generate nice (1-2-5) tick positions inside [min, max] */
function niceTicks(min, max, targetCount = 5) {
  if (!(max > min)) return [];
  const raw = (max - min) / targetCount;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks = [];
  for (let val = Math.ceil(min / step) * step; val <= max + step * 1e-6; val += step) {
    ticks.push(Math.round(val / step) * step);
  }
  return ticks.map((v) => Math.round(v * 1e9) / 1e9);
}

/** Compact tick label: no trailing zeros */
function tickLabel(v) {
  return String(Math.round(v * 1e6) / 1e6);
}

/** Render axis ticks and labels on SVG */
function renderAxisTicks(svg, scale, min, max, isVertical, isX, margin, plotW, plotH, ticks, formatter = tickLabel) {
  const tickSize = 4;
  const textOffset = 12;

  ticks.forEach((tickVal) => {
    if (isX) {
      const x = scale(tickVal);
      // Tick line
      const tickLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      tickLine.setAttribute('x1', x);
      tickLine.setAttribute('x2', x);
      tickLine.setAttribute('y1', margin.top + plotH);
      tickLine.setAttribute('y2', margin.top + plotH + tickSize);
      tickLine.setAttribute('stroke', getCSSVar('--text-muted'));
      tickLine.setAttribute('stroke-width', '0.5');
      svg.appendChild(tickLine);

      // Label
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', x);
      label.setAttribute('y', margin.top + plotH + textOffset);
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('font-size', '11');
      label.setAttribute('fill', getCSSVar('--text-muted'));
      label.setAttribute('font-family', getCSSVar('--font-mono'));
      label.textContent = formatter(tickVal);
      svg.appendChild(label);
    } else {
      const y = scale(tickVal);
      // Tick line
      const tickLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      tickLine.setAttribute('x1', margin.left - tickSize);
      tickLine.setAttribute('x2', margin.left);
      tickLine.setAttribute('y1', y);
      tickLine.setAttribute('y2', y);
      tickLine.setAttribute('stroke', getCSSVar('--text-muted'));
      tickLine.setAttribute('stroke-width', '0.5');
      svg.appendChild(tickLine);

      // Label
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', margin.left - textOffset);
      label.setAttribute('y', y + 3);
      label.setAttribute('text-anchor', 'end');
      label.setAttribute('font-size', '11');
      label.setAttribute('fill', getCSSVar('--text-muted'));
      label.setAttribute('font-family', getCSSVar('--font-mono'));
      label.textContent = formatter(tickVal);
      svg.appendChild(label);
    }
  });
}

/** Setup ResizeObserver for SVG auto-sizing and redraw */
function setupSVGResize(container, svg, renderFn) {
  const observer = new ResizeObserver(() => {
    const w = container.clientWidth || 400;
    const h = container.clientHeight || 300;
    if (w > 0 && h > 0) {
      svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
      renderFn(w, h);
    }
  });
  observer.observe(container);
  // Trigger initial render
  setTimeout(() => {
    const w = container.clientWidth || 400;
    const h = container.clientHeight || 300;
    if (w > 0 && h > 0) {
      svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
      renderFn(w, h);
    }
  }, 0);
}

// ============================================================================
// Data computations
// ============================================================================

/** Compute ensemble stats for selected members */
function computeEnsembleStats(testSample, memberIndices) {
  const n = memberIndices.length;
  if (n === 0) return null;

  const q1 = testSample.pred.map(p => p[0]); // q0.1
  const q2 = testSample.pred.map(p => p[1]); // q0.5 (median)
  const q3 = testSample.pred.map(p => p[2]); // q0.9

  // Ensemble: mean of the selected members' quantiles
  const ensQ1 = memberIndices.reduce((s, i) => s + q1[i], 0) / n;
  const ensQ2 = memberIndices.reduce((s, i) => s + q2[i], 0) / n;
  const ensQ3 = memberIndices.reduce((s, i) => s + q3[i], 0) / n;

  // Member spread: std of selected members' medians
  const medians = memberIndices.map(i => q2[i]);
  const spread = std(medians);

  // Coverage: is observed in [Q10, Q90]?
  const observed = testSample.observed;
  const covered = observed >= ensQ1 && observed <= ensQ3 ? 1 : 0;

  // Width of Q10–Q90
  const width = ensQ3 - ensQ1;

  // Error of ensemble median vs observed
  const error = Math.abs(ensQ2 - observed);
  const rmse = error; // For single sample, use absolute error

  return {
    observed,
    ensQ1,
    ensQ2,
    ensQ3,
    spread,
    width,
    covered,
    error,
    medians,
    q1,
    q2,
    q3,
  };
}

/** Compute batch statistics over test samples */
function computeBatchStats(testSample, memberIndices) {
  if (memberIndices.length === 0) return null;

  let rmse = 0,
    coverage = 0,
    totalWidth = 0;
  const spreads = [];
  const errors = [];

  for (const sample of testSample) {
    const stats = computeEnsembleStats(sample, memberIndices);
    rmse += stats.error * stats.error;
    coverage += stats.covered;
    totalWidth += stats.width;
    spreads.push(stats.spread);
    errors.push(stats.error);
  }

  rmse = Math.sqrt(rmse / testSample.length);
  coverage = coverage / testSample.length;
  const meanWidth = totalWidth / testSample.length;
  const meanSpread = spreads.reduce((a, b) => a + b, 0) / spreads.length;

  return {
    rmse,
    coverage,
    meanWidth,
    meanSpread,
    spreads,
    errors,
  };
}

// ============================================================================
// 1. UNCERTAINTY EXPLORER
// ============================================================================

window.BLOCKS['uncertainty-explorer'] = (host, block, ctx) => {
  const { data, content, tpl, md, rdylbu } = ctx;
  const { explorer } = content;
  const { members, test_sample, variance_means } = data;

  // === State ===
  let selectedMembers = new Set(Array.from({ length: 42 }, (_, i) => i)); // All selected
  let selectedSampleIdx = 0; // Start with first sample
  let randomN = 10; // For random preset
  let currentPreset = 'all';

  // Find initial sample (max spread in first 100)
  {
    let maxSpread = -1;
    for (let i = 0; i < Math.min(100, test_sample.length); i++) {
      const stats = computeEnsembleStats(test_sample[i], Array.from(selectedMembers));
      if (stats.spread > maxSpread) {
        maxSpread = stats.spread;
        selectedSampleIdx = i;
      }
    }
  }

  // === Helper: render member grid ===
  function updateMemberGrid() {
    const gridContainer = host.querySelector('.ux-member-grid');
    if (!gridContainer) return;

    // Clear and rebuild
    gridContainer.innerHTML = '';

    // Column headers (HPO replicates 0-6)
    const colHeaders = document.createElement('div');
    colHeaders.style.gridColumn = '2 / span 7';
    colHeaders.style.gridRow = '1';
    colHeaders.style.display = 'contents';
    for (let col = 0; col < 7; col++) {
      const h = document.createElement('div');
      h.className = 'ux-member-header';
      h.textContent = col.toString();
      h.style.gridColumn = 2 + col;
      h.style.gridRow = 1;
      h.setAttribute('aria-label', `HPO replicate ${col}`);
      h.addEventListener('click', () => toggleReplicate(col));
      gridContainer.appendChild(h);
    }

    // Row headers (Splits 1-6)
    for (let row = 0; row < 6; row++) {
      const h = document.createElement('div');
      h.className = 'ux-member-header';
      h.textContent = (row + 1).toString();
      h.style.gridColumn = 1;
      h.style.gridRow = row + 2;
      h.setAttribute('aria-label', `Split ${row + 1}`);
      h.addEventListener('click', () => toggleSplit(row + 1));
      gridContainer.appendChild(h);
    }

    // Member cells
    for (let split = 1; split <= 6; split++) {
      for (let rep = 0; rep < 7; rep++) {
        const memberIdx = (split - 1) * 7 + rep;
        const cell = document.createElement('button');
        cell.className = 'ux-member-cell';
        cell.textContent = memberIdx.toString();
        cell.setAttribute('aria-pressed', selectedMembers.has(memberIdx) ? 'true' : 'false');
        cell.setAttribute('aria-label', `Member split ${split} replicate ${rep}`);
        cell.style.gridColumn = rep + 2;
        cell.style.gridRow = split + 1;

        cell.addEventListener('click', () => toggleMember(memberIdx));
        gridContainer.appendChild(cell);
      }
    }
  }

  function toggleMember(idx) {
    if (selectedMembers.has(idx)) {
      if (selectedMembers.size > 1) selectedMembers.delete(idx);
    } else {
      selectedMembers.add(idx);
    }
    updateSelection();
  }

  function toggleSplit(split) {
    const indices = Array.from({ length: 7 }, (_, i) => (split - 1) * 7 + i);
    const allSelected = indices.every(i => selectedMembers.has(i));

    if (allSelected && selectedMembers.size > 7) {
      indices.forEach(i => selectedMembers.delete(i));
    } else {
      indices.forEach(i => selectedMembers.add(i));
    }
    updateSelection();
  }

  function toggleReplicate(rep) {
    const indices = Array.from({ length: 6 }, (_, i) => i * 7 + rep);
    const allSelected = indices.every(i => selectedMembers.has(i));

    if (allSelected && selectedMembers.size > 6) {
      indices.forEach(i => selectedMembers.delete(i));
    } else {
      indices.forEach(i => selectedMembers.add(i));
    }
    updateSelection();
  }

  function updateSelection() {
    updateMemberGrid();
    updateMetrics();
    updateScatter();
    updateMap();
    updateInspector();
    updateLiveRegion();
  }

  // === Preset handling ===
  function applyPreset(presetId) {
    currentPreset = presetId;
    selectedMembers.clear();

    if (presetId === 'all') {
      for (let i = 0; i < 42; i++) selectedMembers.add(i);
    } else if (presetId === 'one-split') {
      const split = (parseInt(currentPreset.split) || 1);
      for (let i = 0; i < 7; i++) selectedMembers.add((split - 1) * 7 + i);
      // Next call: cycle to next split
    } else if (presetId === 'one-replicate') {
      const rep = (parseInt(currentPreset.replicate) || 0);
      for (let i = 0; i < 6; i++) selectedMembers.add(i * 7 + rep);
    } else if (presetId === 'random') {
      const rng = new SeededRandom(42);
      const indices = Array.from({ length: 42 }, (_, i) => i)
        .sort(() => rng.next() - 0.5)
        .slice(0, randomN);
      indices.forEach(i => selectedMembers.add(i));
    } else if (presetId === 'single') {
      const single = parseInt(currentPreset.single) || 0;
      selectedMembers.add(single);
    }

    updateSelection();
  }

  // === Metrics ===
  function updateMetrics() {
    const stats = computeBatchStats(test_sample, Array.from(selectedMembers));
    if (!stats) return;

    const tiles = [
      { selector: '.ux-rmse', value: fmt(stats.rmse, 2), unit: 'µmol/kg' },
      {
        selector: '.ux-coverage',
        value: fmt(stats.coverage * 100, 1),
        unit: '%',
      },
      { selector: '.ux-width', value: fmt(stats.meanWidth, 2), unit: 'µmol/kg' },
      { selector: '.ux-spread', value: fmt(stats.meanSpread, 2), unit: 'µmol/kg' },
    ];

    tiles.forEach(({ selector, value, unit }) => {
      const elem = host.querySelector(selector);
      if (elem) {
        const valElem = elem.querySelector('.ux-metric-value');
        const unitElem = elem.querySelector('.ux-metric-unit');
        if (valElem) valElem.textContent = value;
        if (unitElem) unitElem.textContent = unit;
      }
    });

    const label = host.querySelector('.ux-selection-label');
    if (label) label.textContent = `Members selected: ${selectedMembers.size}`;
  }

  // === Scatter plot ===
  function updateScatter() {
    const container = host.querySelector('.ux-scatter-wrapper');
    const svg = host.querySelector('.ux-scatter-svg');
    if (!svg || !container) return;

    // Compute spreads for all samples
    const spreads = test_sample.map(s =>
      computeEnsembleStats(s, Array.from(selectedMembers)).spread
    );

    // Color scale: 5th-95th percentile
    const pMin = percentile(spreads, 0.05);
    const pMax = percentile(spreads, 0.95);
    const spreadRange = pMax - pMin || 1;

    const renderScatter = (width, height) => {
      const margin = { top: 20, right: 20, bottom: 50, left: 60 };
      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;

      // Data range - find both observed and predicted
      const observed = test_sample.map(s => s.observed);
      const predicted = test_sample.map(s => {
        const stats = computeEnsembleStats(s, Array.from(selectedMembers));
        return stats.ensQ2;
      });
      const allValues = [...observed, ...predicted];
      let axisMin = Math.min(...allValues);
      let axisMax = Math.max(...allValues);
      const axisPad = (axisMax - axisMin) * 0.05;
      axisMin -= axisPad;
      axisMax += axisPad;

      // Scale functions (same range on both axes for 1:1 line)
      const xScale = (val) => margin.left + ((val - axisMin) / (axisMax - axisMin)) * plotW;
      const yScale = (val) => margin.top + (1 - (val - axisMin) / (axisMax - axisMin)) * plotH;

      // Build SVG
      svg.innerHTML = '';

      // Grid lines
      const ticks = niceTicks(axisMin, axisMax, 5);
      ticks.forEach((tick) => {
        const x = xScale(tick);
        const y = yScale(tick);
        // Vertical grid
        const vLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        vLine.setAttribute('x1', x);
        vLine.setAttribute('x2', x);
        vLine.setAttribute('y1', margin.top);
        vLine.setAttribute('y2', margin.top + plotH);
        vLine.setAttribute('stroke', getCSSVar('--border'));
        vLine.setAttribute('stroke-width', '0.5');
        vLine.setAttribute('opacity', '0.5');
        svg.appendChild(vLine);

        // Horizontal grid
        const hLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        hLine.setAttribute('x1', margin.left);
        hLine.setAttribute('x2', margin.left + plotW);
        hLine.setAttribute('y1', y);
        hLine.setAttribute('y2', y);
        hLine.setAttribute('stroke', getCSSVar('--border'));
        hLine.setAttribute('stroke-width', '0.5');
        hLine.setAttribute('opacity', '0.5');
        svg.appendChild(hLine);
      });

      // 1:1 line
      const line1x1 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line1x1.setAttribute('x1', xScale(axisMin));
      line1x1.setAttribute('y1', yScale(axisMin));
      line1x1.setAttribute('x2', xScale(axisMax));
      line1x1.setAttribute('y2', yScale(axisMax));
      line1x1.setAttribute('stroke', getCSSVar('--border'));
      line1x1.setAttribute('stroke-width', '1.5');
      line1x1.setAttribute('stroke-dasharray', '4,4');
      svg.appendChild(line1x1);

      // Axes
      const axisX = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      axisX.setAttribute('x1', margin.left);
      axisX.setAttribute('x2', margin.left + plotW);
      axisX.setAttribute('y1', margin.top + plotH);
      axisX.setAttribute('y2', margin.top + plotH);
      axisX.setAttribute('stroke', getCSSVar('--text-muted'));
      axisX.setAttribute('stroke-width', '1');
      svg.appendChild(axisX);

      const axisY = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      axisY.setAttribute('x1', margin.left);
      axisY.setAttribute('x2', margin.left);
      axisY.setAttribute('y1', margin.top);
      axisY.setAttribute('y2', margin.top + plotH);
      axisY.setAttribute('stroke', getCSSVar('--text-muted'));
      axisY.setAttribute('stroke-width', '1');
      svg.appendChild(axisY);

      // Axis ticks and labels
      renderAxisTicks(svg, xScale, axisMin, axisMax, false, true, margin, plotW, plotH, ticks);
      renderAxisTicks(svg, yScale, axisMin, axisMax, true, false, margin, plotW, plotH, ticks);

      // Axis labels
      const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      xLabel.setAttribute('x', margin.left + plotW / 2);
      xLabel.setAttribute('y', height - 8);
      xLabel.setAttribute('text-anchor', 'middle');
      xLabel.setAttribute('font-size', '11');
      xLabel.setAttribute('fill', getCSSVar('--text-muted'));
      xLabel.textContent = 'Observed TA (µmol/kg)';
      svg.appendChild(xLabel);

      const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      yLabel.setAttribute('x', 12);
      yLabel.setAttribute('y', margin.top + plotH / 2);
      yLabel.setAttribute('text-anchor', 'middle');
      yLabel.setAttribute('font-size', '11');
      yLabel.setAttribute('fill', getCSSVar('--text-muted'));
      yLabel.setAttribute('transform', `rotate(-90 12 ${margin.top + plotH / 2})`);
      yLabel.textContent = 'Predicted (µmol/kg)';
      svg.appendChild(yLabel);

      // Points
      test_sample.forEach((sample, idx) => {
        const stats = computeEnsembleStats(sample, Array.from(selectedMembers));
        const t = clamp((stats.spread - pMin) / spreadRange, 0, 1);
        const color = getRdylbuColor(t, true); // reverse: low=blue, high=red

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', xScale(sample.observed));
        circle.setAttribute('cy', yScale(stats.ensQ2));
        circle.setAttribute('r', '3.5');
        circle.setAttribute('fill', color);
        circle.setAttribute('class', `ux-scatter-point ${selectedSampleIdx === idx ? 'selected' : ''}`);
        circle.setAttribute('data-idx', idx);

        circle.addEventListener('click', () => selectSample(idx));
        circle.addEventListener('mouseenter', (e) => showTooltip(e, sample, stats));
        circle.addEventListener('mouseleave', () => hideTooltip());

        svg.appendChild(circle);
      });

      // Colorbar legend
      const cbContainer = host.querySelector('.ux-colorbar');
      if (cbContainer) {
        cbContainer.innerHTML = '';

        const label = document.createElement('div');
        label.className = 'ux-colorbar-label';
        label.textContent = 'Member spread (µmol/kg)';
        cbContainer.appendChild(label);

        const ramp = document.createElement('div');
        ramp.className = 'ux-colorbar-ramp';
        for (let i = 0; i <= 10; i++) {
          const segment = document.createElement('div');
          segment.style.flex = '1';
          segment.style.background = getRdylbuColor(i / 10, true);
          ramp.appendChild(segment);
        }
        cbContainer.appendChild(ramp);

        const ticks = document.createElement('div');
        ticks.className = 'ux-colorbar-tick';
        ticks.innerHTML = `<span>${fmt(pMin, 2)}</span><span>${fmt((pMin + pMax) / 2, 2)}</span><span>${fmt(pMax, 2)}</span>`;
        cbContainer.appendChild(ticks);
      }
    };

    setupSVGResize(container, svg, renderScatter);
  }

  // === Map ===
  function updateMap() {
    const mapContainer = host.querySelector('.ux-map-container');
    const mapSvg = host.querySelector('.ux-map-svg');
    if (!mapSvg || !mapContainer) return;

    const spreads = test_sample.map(s =>
      computeEnsembleStats(s, Array.from(selectedMembers)).spread
    );
    const pMin = percentile(spreads, 0.05);
    const pMax = percentile(spreads, 0.95);
    const spreadRange = pMax - pMin || 1;

    const renderMap = (width, height) => {
      mapSvg.innerHTML = '';

      // Graticule lines
      for (let lat = -90; lat <= 90; lat += 30) {
        const y = ((lat + 90) / 180) * height;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', '0');
        line.setAttribute('x2', width);
        line.setAttribute('y1', y);
        line.setAttribute('y2', y);
        line.setAttribute('class', 'ux-map-graticule');
        mapSvg.appendChild(line);

        // Latitude label
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', 8);
        label.setAttribute('y', y + 3);
        label.setAttribute('font-size', '9');
        label.setAttribute('fill', getCSSVar('--text-muted'));
        label.setAttribute('font-family', getCSSVar('--font-mono'));
        label.textContent = lat + '°';
        mapSvg.appendChild(label);
      }

      for (let lon = -180; lon <= 180; lon += 60) {
        const x = ((lon + 180) / 360) * width;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x);
        line.setAttribute('x2', x);
        line.setAttribute('y1', '0');
        line.setAttribute('y2', height);
        line.setAttribute('class', 'ux-map-graticule');
        mapSvg.appendChild(line);

        // Longitude label
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', x);
        label.setAttribute('y', height - 3);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('font-size', '9');
        label.setAttribute('fill', getCSSVar('--text-muted'));
        label.setAttribute('font-family', getCSSVar('--font-mono'));
        label.textContent = lon + '°';
        mapSvg.appendChild(label);
      }

      // Points
      test_sample.forEach((sample, idx) => {
        const stats = computeEnsembleStats(sample, Array.from(selectedMembers));
        const t = clamp((stats.spread - pMin) / spreadRange, 0, 1);
        const color = getRdylbuColor(t, true);

        const x = ((sample.lon + 180) / 360) * width;
        const y = ((sample.lat + 90) / 180) * height;

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x);
        circle.setAttribute('cy', y);
        circle.setAttribute('r', '3');
        circle.setAttribute('fill', color);
        circle.setAttribute('class', `ux-map-point ${selectedSampleIdx === idx ? 'selected' : ''}`);
        circle.setAttribute('data-idx', idx);

        circle.addEventListener('click', () => selectSample(idx));
        mapSvg.appendChild(circle);
      });
    };

    setupSVGResize(mapContainer, mapSvg, renderMap);
  }

  // === Sample inspector ===
  function updateInspector() {
    const sample = test_sample[selectedSampleIdx];
    const stats = computeEnsembleStats(sample, Array.from(selectedMembers));

    // Info section
    const infoContainer = host.querySelector('.ux-sample-info');
    if (infoContainer) {
      infoContainer.innerHTML = `
        <div class="ux-info-group">
          <div class="ux-info-label">Cruise</div>
          <div class="ux-info-value">${sample.cruise}</div>
        </div>
        <div class="ux-info-group">
          <div class="ux-info-label">Date</div>
          <div class="ux-info-value">${sample.date}</div>
        </div>
        <div class="ux-info-group">
          <div class="ux-info-label">Position</div>
          <div class="ux-info-value">${fmt(sample.lat, 2)}°, ${fmt(sample.lon, 2)}°</div>
        </div>
        <div class="ux-info-group">
          <div class="ux-info-label">Depth</div>
          <div class="ux-info-value">${fmt(sample.depth, 1)} m</div>
        </div>
        <div class="ux-info-group">
          <div class="ux-info-label">Observed</div>
          <div class="ux-info-value">${fmt(sample.observed, 2)} µmol/kg</div>
        </div>
        <div class="ux-info-group">
          <div class="ux-info-label">Ensemble median</div>
          <div class="ux-info-value">${fmt(stats.ensQ2, 2)} µmol/kg</div>
        </div>
        <div class="ux-info-group">
          <div class="ux-info-label">Error</div>
          <div class="ux-info-value">${fmt(stats.error, 2)} µmol/kg</div>
        </div>
        <div class="ux-info-group">
          <div class="ux-info-label">Member spread</div>
          <div class="ux-info-value">${fmt(stats.spread, 2)} µmol/kg</div>
        </div>
      `;
    }

    // Whisker plot
    const whiskerSvg = host.querySelector('.ux-whisker-svg');
    if (whiskerSvg) {
      renderWhiskerPlot(whiskerSvg, sample, stats);
    }

    // Variance summary
    const summary = host.querySelector('.ux-variance-summary');
    if (summary && selectedMembers.size >= 2) {
      summary.innerHTML = renderVarianceSummary(stats);
    }

    // Update selected state in scatter and map
    host.querySelectorAll('.ux-scatter-point').forEach((p) => {
      p.classList.toggle('selected', parseInt(p.getAttribute('data-idx')) === selectedSampleIdx);
    });
    host.querySelectorAll('.ux-map-point').forEach((p) => {
      p.classList.toggle('selected', parseInt(p.getAttribute('data-idx')) === selectedSampleIdx);
    });
  }

  function renderWhiskerPlot(svg, sample, stats) {
    const container = svg.parentElement;
    const NS = 'http://www.w3.org/2000/svg';
    const el = (name, attrs, text) => {
      const node = document.createElementNS(NS, name);
      Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
      if (text !== undefined) node.textContent = text;
      return node;
    };

    const draw = () => {
      const st = container.__whisker;
      if (!st) return;
      const { sample: smp, stats: s } = st;
      const cs = getComputedStyle(container);
      const width = Math.max(
        320,
        container.clientWidth - (parseFloat(cs.paddingLeft) || 0) - (parseFloat(cs.paddingRight) || 0)
      );
      const rowH = 9;
      const groupGap = 10;
      const margin = { left: 56, right: 20, top: 34, bottom: 42 };
      const members = Array.from(selectedMembers).sort((x, y) => x - y);
      const groups = new Map();
      members.forEach((m) => {
        const split = Math.floor(m / 7) + 1;
        if (!groups.has(split)) groups.set(split, []);
        groups.get(split).push(m);
      });
      const plotW = width - margin.left - margin.right;
      const plotH = members.length * rowH + (groups.size - 1) * groupGap + 8;
      const height = margin.top + plotH + margin.bottom;

      // x range fitted to selected members' bands + observed + ensemble
      let lo = Math.min(smp.observed, s.ensQ1);
      let hi = Math.max(smp.observed, s.ensQ3);
      members.forEach((m) => {
        lo = Math.min(lo, s.q1[m]);
        hi = Math.max(hi, s.q3[m]);
      });
      const pad = (hi - lo || 1) * 0.08;
      lo -= pad;
      hi += pad;
      const xScale = (v) => margin.left + ((v - lo) / (hi - lo)) * plotW;
      const muted = getCSSVar('--text-muted');
      const border = getCSSVar('--border');

      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      svg.style.width = '100%';
      svg.style.height = `${height}px`;
      svg.innerHTML = '';

      // legend (above the plot)
      let lx = margin.left;
      [
        ['Observed', getCSSVar('--c-observed'), 'line'],
        [width < 560 ? 'Ensemble' : 'Ensemble median', getCSSVar('--c-ensemble'), 'line'],
        [width < 560 ? 'Q10-Q90' : 'Ensemble Q10-Q90', getCSSVar('--c-band'), 'box'],
        [width < 560 ? 'Members' : 'Member median and Q10-Q90', getCSSVar('--c-member'), 'dot'],
      ].forEach(([label, color, kind]) => {
        if (kind === 'line') svg.appendChild(el('line', { x1: lx, x2: lx + 14, y1: 12, y2: 12, stroke: color, 'stroke-width': 2 }));
        else if (kind === 'box') svg.appendChild(el('rect', { x: lx, y: 6, width: 14, height: 12, fill: color }));
        else svg.appendChild(el('circle', { cx: lx + 7, cy: 12, r: 3, fill: color }));
        svg.appendChild(el('text', { x: lx + 19, y: 16, 'font-size': 11, fill: muted }, label));
        lx += 19 + label.length * 6.2 + 14;
      });

      // grid + x axis
      const ticks = niceTicks(lo, hi, 6);
      ticks.forEach((v) => {
        svg.appendChild(el('line', { x1: xScale(v), x2: xScale(v), y1: margin.top, y2: margin.top + plotH, stroke: border, 'stroke-width': 0.5, opacity: 0.7 }));
      });
      svg.appendChild(el('line', { x1: margin.left, x2: margin.left + plotW, y1: margin.top + plotH, y2: margin.top + plotH, stroke: muted }));
      renderAxisTicks(svg, xScale, lo, hi, false, true, margin, plotW, plotH, ticks);
      svg.appendChild(el('text', { x: margin.left + plotW / 2, y: height - 6, 'text-anchor': 'middle', 'font-size': 11, fill: muted }, 'Total alkalinity (µmol/kg)'));

      // ensemble band behind everything
      svg.appendChild(el('rect', {
        x: xScale(s.ensQ1), y: margin.top, width: Math.max(1, xScale(s.ensQ3) - xScale(s.ensQ1)), height: plotH,
        class: 'ux-whisker-ensemble-band',
      }));

      // member rows grouped by split
      let y = margin.top + 8;
      groups.forEach((ids, split) => {
        const y0 = y;
        ids.forEach((m) => {
          const rep = m % 7;
          const row = el('g', { class: 'ux-whisker-row' });
          row.appendChild(el('title', {}, `Split ${split}, HPO ${rep}: median ${fmt(s.q2[m], 1)} (Q10 ${fmt(s.q1[m], 1)} to Q90 ${fmt(s.q3[m], 1)})`));
          row.appendChild(el('rect', {
            x: xScale(s.q1[m]), y: y - 2, width: Math.max(1, xScale(s.q3[m]) - xScale(s.q1[m])), height: 4, class: 'ux-whisker-band',
          }));
          row.appendChild(el('circle', { cx: xScale(s.q2[m]), cy: y, r: 2.6, class: 'ux-whisker-dot' }));
          svg.appendChild(row);
          y += rowH;
        });
        svg.appendChild(el('text', {
          x: margin.left - 8, y: (y0 + y - rowH) / 2 + 4, 'text-anchor': 'end', 'font-size': 11, 'font-weight': 600, fill: muted,
        }, `Split ${split}`));
        y += groupGap;
      });

      // observed + ensemble median lines on top
      svg.appendChild(el('line', { x1: xScale(smp.observed), x2: xScale(smp.observed), y1: margin.top, y2: margin.top + plotH, class: 'ux-whisker-observed' }));
      svg.appendChild(el('line', { x1: xScale(s.ensQ2), x2: xScale(s.ensQ2), y1: margin.top, y2: margin.top + plotH, class: 'ux-whisker-ensemble' }));
    };

    container.__whisker = { sample, stats };
    if (!container.__whiskerObserver) {
      let lastWidth = 0;
      container.__whiskerObserver = new ResizeObserver(() => {
        if (container.clientWidth !== lastWidth) {
          lastWidth = container.clientWidth;
          draw();
        }
      });
      container.__whiskerObserver.observe(container);
    }
    draw();
  }

  function renderVarianceSummary(stats) {
    // Compute split vs HPO variance
    const memberList = Array.from(selectedMembers);
    const splitGroups = new Map();

    memberList.forEach((idx) => {
      const split = Math.floor(idx / 7) + 1;
      if (!splitGroups.has(split)) {
        splitGroups.set(split, []);
      }
      splitGroups.get(split).push(stats.q2[idx]);
    });

    let splitVar = 0,
      withinVar = 0;

    if (splitGroups.size >= 2) {
      // Between-split variance
      const splitMeans = Array.from(splitGroups.values()).map((vals) =>
        vals.reduce((a, b) => a + b, 0) / vals.length
      );
      const globalMean =
        memberList
          .map((idx) => stats.q2[idx])
          .reduce((a, b) => a + b, 0) / memberList.length;
      splitVar = splitMeans
        .map((m) => (m - globalMean) ** 2)
        .reduce((a, b) => a + b, 0) / splitMeans.length;
    }

    if (memberList.some((idx) => splitGroups.get(Math.floor(idx / 7) + 1).length >= 2)) {
      // Within-split variance (std within each split, averaged)
      const withinVars = Array.from(splitGroups.values())
        .filter((vals) => vals.length >= 2)
        .map((vals) => {
          const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
          return (
            vals.map((v) => (v - mean) ** 2).reduce((a, b) => a + b, 0) / vals.length
          );
        });
      if (withinVars.length > 0) {
        withinVar = withinVars.reduce((a, b) => a + b, 0) / withinVars.length;
      }
    }

    const splitStd = Math.sqrt(splitVar);
    const withinStd = Math.sqrt(withinVar);

    return `
      Between-split std: ${splitGroups.size >= 2 ? fmt(splitStd, 2) + ' µmol/kg' : 'n/a'}
      <br/>
      Within-split std: ${withinVar > 0 ? fmt(withinStd, 2) + ' µmol/kg' : 'n/a'}
    `;
  }

  // === Sample selection ===
  function selectSample(idx) {
    selectedSampleIdx = idx;
    updateInspector();
    updateLiveRegion();
  }

  function nextSample() {
    selectedSampleIdx = (selectedSampleIdx + 1) % test_sample.length;
    updateInspector();
  }

  function previousSample() {
    selectedSampleIdx = (selectedSampleIdx - 1 + test_sample.length) % test_sample.length;
    updateInspector();
  }

  // === Tooltip ===
  let tooltipElem = null;

  function showTooltip(e, sample, stats) {
    if (!tooltipElem) {
      tooltipElem = document.createElement('div');
      tooltipElem.className = 'ux-scatter-tooltip';
      host.appendChild(tooltipElem);
    }

    tooltipElem.innerHTML = `
      <div class="ux-tooltip-row">
        <div class="ux-tooltip-label">Cruise:</div>
        <div class="ux-tooltip-value">${sample.cruise}</div>
      </div>
      <div class="ux-tooltip-row">
        <div class="ux-tooltip-label">Date:</div>
        <div class="ux-tooltip-value">${sample.date}</div>
      </div>
      <div class="ux-tooltip-row">
        <div class="ux-tooltip-label">Position:</div>
        <div class="ux-tooltip-value">${fmt(sample.lat, 1)}°, ${fmt(sample.lon, 1)}°</div>
      </div>
      <div class="ux-tooltip-row">
        <div class="ux-tooltip-label">Observed:</div>
        <div class="ux-tooltip-value">${fmt(sample.observed, 2)}</div>
      </div>
      <div class="ux-tooltip-row">
        <div class="ux-tooltip-label">Predicted:</div>
        <div class="ux-tooltip-value">${fmt(stats.ensQ2, 2)}</div>
      </div>
      <div class="ux-tooltip-row">
        <div class="ux-tooltip-label">Spread:</div>
        <div class="ux-tooltip-value">${fmt(stats.spread, 2)}</div>
      </div>
    `;

    const rect = e.target.getBoundingClientRect();
    const parentRect = host.getBoundingClientRect();
    tooltipElem.style.left = rect.left - parentRect.left + 10 + 'px';
    tooltipElem.style.top = rect.top - parentRect.top + 10 + 'px';
  }

  function hideTooltip() {
    if (tooltipElem) {
      tooltipElem.style.display = 'none';
    }
  }

  // === Live region for accessibility ===
  function updateLiveRegion() {
    const region = host.querySelector('.ux-live-region');
    if (region) {
      const sample = test_sample[selectedSampleIdx];
      const stats = computeEnsembleStats(sample, Array.from(selectedMembers));
      region.textContent = `Sample at ${sample.lat.toFixed(1)}, ${sample.lon.toFixed(1)}: observed ${sample.observed.toFixed(2)}, predicted ${stats.ensQ2.toFixed(2)}, spread ${stats.spread.toFixed(2)}. ${selectedMembers.size} members selected.`;
    }
  }

  // === Initial render ===
  host.innerHTML = `
    <div class="ux-live-region" role="status" aria-live="polite" aria-atomic="true"></div>

    <div class="ux-explorer">
      <div>
        <div class="ux-member-grid-wrapper">
          <div class="ux-member-grid-controls">
            <div class="ux-preset-chip" aria-pressed="true">All 42</div>
            <div class="ux-preset-chip">One split</div>
            <div class="ux-preset-chip">One replicate</div>
            <div class="ux-preset-chip">Random</div>
            <div class="ux-preset-chip">Single</div>
            <div class="ux-selection-label">Members selected: 42</div>
          </div>

          <div class="ux-random-controls ux-hidden">
            <label>
              N:
              <input type="range" class="ux-random-slider" min="2" max="42" value="10">
              <span class="ux-random-value">10</span>
            </label>
            <button class="ux-reshuffle-btn">Reshuffle</button>
          </div>

          <div class="ux-member-grid"></div>
        </div>

        <div class="ux-metrics-grid">
          <div class="ux-metric-tile ux-rmse">
            <div class="ux-metric-label">RMSE</div>
            <div class="ux-metric-value">—<span class="ux-metric-unit">µmol/kg</span></div>
          </div>
          <div class="ux-metric-tile ux-coverage">
            <div class="ux-metric-label">80% coverage</div>
            <div class="ux-metric-value">—<span class="ux-metric-unit">%</span></div>
          </div>
          <div class="ux-metric-tile ux-width">
            <div class="ux-metric-label">Mean Q10–Q90</div>
            <div class="ux-metric-value">—<span class="ux-metric-unit">µmol/kg</span></div>
          </div>
          <div class="ux-metric-tile ux-spread">
            <div class="ux-metric-label">Mean spread</div>
            <div class="ux-metric-value">—<span class="ux-metric-unit">µmol/kg</span></div>
          </div>
        </div>
      </div>

      <div class="ux-charts-column">
        <div class="ux-chart-card">
          <div class="ux-scatter-wrapper"><svg class="ux-scatter-svg"></svg></div>
          <div class="ux-colorbar"></div>
        </div>

        <div class="ux-chart-card">
          <div class="ux-map-container">
            <svg class="ux-map-svg"></svg>
          </div>
        </div>
      </div>
    </div>

    <div class="ux-sample-inspector">
      <div class="ux-inspector-header">
        <div>
          <strong>Sample inspector</strong>
        </div>
        <div class="ux-sample-nav">
          <button class="ux-nav-btn" aria-label="Previous sample">← Prev</button>
          <button class="ux-nav-btn" aria-label="Next sample">Next →</button>
          <select class="ux-sort-select" aria-label="Jump to">
            <option value="">Jump to...</option>
            <option value="uncertain">Most uncertain</option>
            <option value="least">Least uncertain</option>
            <option value="error">Largest error</option>
            <option value="random">Random</option>
          </select>
        </div>
      </div>

      <div class="ux-sample-info"></div>
      <div class="ux-whisker-plot-container">
        <svg class="ux-whisker-svg"></svg>
      </div>
      <div class="ux-variance-summary"></div>
    </div>
  `;

  // === Event handlers ===
  const presetChips = host.querySelectorAll('.ux-member-grid-controls .ux-preset-chip');
  presetChips.forEach((chip, idx) => {
    const presetIds = ['all', 'one-split', 'one-replicate', 'random', 'single'];
    chip.addEventListener('click', () => {
      // Toggle if same preset, or switch
      if (currentPreset === presetIds[idx]) {
        // Cycle to next variant
        if (presetIds[idx] === 'one-split') {
          currentPreset = 'one-split-2';
        } else if (presetIds[idx] === 'one-replicate') {
          currentPreset = 'one-replicate-1';
        } else if (presetIds[idx] === 'single') {
          currentPreset = 'single-1';
        }
      } else {
        currentPreset = presetIds[idx];
      }
      applyPreset(currentPreset);

      // Update chip states
      presetChips.forEach((c, i) => {
        c.setAttribute('aria-pressed', i === idx ? 'true' : 'false');
      });
    });
  });

  // Random controls
  const randomSlider = host.querySelector('.ux-random-slider');
  if (randomSlider) {
    randomSlider.addEventListener('input', (e) => {
      randomN = parseInt(e.target.value);
      host.querySelector('.ux-random-value').textContent = randomN;
      if (currentPreset === 'random') {
        applyPreset('random');
      }
    });
  }

  const reshuffleBtn = host.querySelector('.ux-reshuffle-btn');
  if (reshuffleBtn) {
    reshuffleBtn.addEventListener('click', () => {
      // Change seed for new shuffle
      const currentSeed = 42;
      applyPreset('random');
    });
  }

  // Sample navigation
  host.querySelector('.ux-nav-btn:nth-of-type(1)').addEventListener('click', previousSample);
  host.querySelector('.ux-nav-btn:nth-of-type(2)').addEventListener('click', nextSample);

  // Sort select
  host.querySelector('.ux-sort-select').addEventListener('change', (e) => {
    if (e.target.value === 'uncertain') {
      let maxSpread = -1,
        maxIdx = 0;
      test_sample.forEach((s, i) => {
        const stats = computeEnsembleStats(s, Array.from(selectedMembers));
        if (stats.spread > maxSpread) {
          maxSpread = stats.spread;
          maxIdx = i;
        }
      });
      selectedSampleIdx = maxIdx;
    } else if (e.target.value === 'least') {
      let minSpread = Infinity,
        minIdx = 0;
      test_sample.forEach((s, i) => {
        const stats = computeEnsembleStats(s, Array.from(selectedMembers));
        if (stats.spread < minSpread) {
          minSpread = stats.spread;
          minIdx = i;
        }
      });
      selectedSampleIdx = minIdx;
    } else if (e.target.value === 'error') {
      let maxError = -1,
        maxIdx = 0;
      test_sample.forEach((s, i) => {
        const stats = computeEnsembleStats(s, Array.from(selectedMembers));
        if (stats.error > maxError) {
          maxError = stats.error;
          maxIdx = i;
        }
      });
      selectedSampleIdx = maxIdx;
    } else if (e.target.value === 'random') {
      selectedSampleIdx = Math.floor(Math.random() * test_sample.length);
    }
    e.target.value = '';
    updateInspector();
    updateLiveRegion();
  });

  // Initial update
  updateMemberGrid();
  updateMetrics();
  updateScatter();
  updateMap();
  updateInspector();
  updateLiveRegion();
};

// ============================================================================
// 2. VARIANCE DECOMPOSITION
// ============================================================================

window.BLOCKS['variance-decomposition'] = (host, block, ctx) => {
  const { data, content } = ctx;
  const { variance_means } = data;

  // Training color for split variance (blue-ish)
  const colTrain = getCSSVar('--c-train');
  // Warning color for HPO variance (orange-ish)
  const colHpo = getCSSVar('--c-warn');

  host.innerHTML = `
    <div class="ux-variance-legend">
      <div class="ux-variance-legend-item">
        <div class="ux-variance-legend-box" style="background: ${colTrain};"></div>
        <span>Split (data)</span>
      </div>
      <div class="ux-variance-legend-item">
        <div class="ux-variance-legend-box" style="background: ${colHpo};"></div>
        <span>HPO (tuning)</span>
      </div>
    </div>

    <div class="ux-variance-chart">
      <div class="ux-variance-row">
        <div class="ux-variance-label">Q0.1</div>
        <div class="ux-variance-bar">
          <div class="ux-variance-segment" style="background: ${colTrain}; flex: var(--split-flex, 1);"></div>
          <div class="ux-variance-segment" style="background: ${colHpo}; flex: var(--hpo-flex, 1);"></div>
        </div>
        <div class="ux-variance-label" style="min-width: 80px;">σ = <span class="ux-total-std">—</span></div>
      </div>
      <div class="ux-variance-row">
        <div class="ux-variance-label">Q0.5</div>
        <div class="ux-variance-bar">
          <div class="ux-variance-segment" style="background: ${colTrain}; flex: var(--split-flex, 1);"></div>
          <div class="ux-variance-segment" style="background: ${colHpo}; flex: var(--hpo-flex, 1);"></div>
        </div>
        <div class="ux-variance-label" style="min-width: 80px;">σ = <span class="ux-total-std">—</span></div>
      </div>
      <div class="ux-variance-row">
        <div class="ux-variance-label">Q0.9</div>
        <div class="ux-variance-bar">
          <div class="ux-variance-segment" style="background: ${colTrain}; flex: var(--split-flex, 1);"></div>
          <div class="ux-variance-segment" style="background: ${colHpo}; flex: var(--hpo-flex, 1);"></div>
        </div>
        <div class="ux-variance-label" style="min-width: 80px;">σ = <span class="ux-total-std">—</span></div>
      </div>
    </div>

    <div class="ux-variance-caption"></div>
  `;

  // Fill in data
  const rows = host.querySelectorAll('.ux-variance-row');
  const quantiles = ['q_0.1', 'q_0.5', 'q_0.9'];

  rows.forEach((row, idx) => {
    const qKey = quantiles[idx];
    const qData = variance_means[qKey];
    const total = qData.split_var + qData.hpo_var;

    // Set flex proportions
    const bars = row.querySelectorAll('.ux-variance-bar');
    bars.forEach((bar) => {
      bar.style.setProperty('--split-flex', qData.split_var.toString());
      bar.style.setProperty('--hpo-flex', qData.hpo_var.toString());
    });

    // Total std
    const stdElem = row.querySelector('.ux-total-std');
    stdElem.textContent = fmt(Math.sqrt(total), 2);
  });

  // Caption
  const caption = host.querySelector('.ux-variance-caption');
  const q05 = variance_means['q_0.5'];
  const total05 = q05.split_var + q05.hpo_var;
  const splitPct = ((q05.split_var / total05) * 100).toFixed(1);
  const hpoPct = ((q05.hpo_var / total05) * 100).toFixed(1);

  caption.textContent = `For the median, ${splitPct}% of member variance comes from which data a model saw, ${hpoPct}% from tuning randomness.`;
};

// ============================================================================
// 3. SPREAD VS ERROR
// ============================================================================

window.BLOCKS['spread-vs-error'] = (host, block, ctx) => {
  const { data } = ctx;
  const { spread_vs_error } = data;
  const NS = 'http://www.w3.org/2000/svg';

  const series = [
    { key: 'member_std', label: 'Member std', color: '--c-test' },
    { key: 'scaled_iqr', label: 'Scaled Q90-Q10', color: '--accent' },
    { key: 'combined', label: 'Combined', color: '--c-validation' },
  ];
  const visible = new Set(['member_std', 'scaled_iqr']);

  host.innerHTML = `
    <div class="ux-spread-legend" role="group" aria-label="Uncertainty estimates">
      ${series.map((s) => `
        <button type="button" class="ux-spread-legend-chip" data-series="${s.key}" aria-pressed="${visible.has(s.key)}" style="--chip-color: var(${s.color})">
          <span class="ux-chip-swatch"></span>${s.label}
        </button>`).join('')}
    </div>
    <div class="ux-sve-plot"><svg class="ux-spread-svg"></svg></div>
    <div class="ux-spread-caption"></div>
  `;

  const el = (name, attrs, text) => {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const plot = host.querySelector('.ux-sve-plot');
  const svg = host.querySelector('.ux-spread-svg');

  const render = (width, height) => {
    const margin = { top: 16, right: 20, bottom: 44, left: 56 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const active = series.filter((s) => visible.has(s.key) && spread_vs_error[s.key]);

    // same range on both axes so the dashed 1:1 line is the line of perfect uncertainty
    let hi = 0;
    active.forEach((s) => spread_vs_error[s.key].forEach((p) => { hi = Math.max(hi, p.u, p.rmse); }));
    hi = (hi || 1) * 1.05;
    const lo = 0;
    const xScale = (v) => margin.left + ((v - lo) / (hi - lo)) * plotW;
    const yScale = (v) => margin.top + (1 - (v - lo) / (hi - lo)) * plotH;
    const muted = getCSSVar('--text-muted');
    const border = getCSSVar('--border');

    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.innerHTML = '';
    const ticks = niceTicks(lo, hi, 6);
    ticks.forEach((v) => {
      svg.appendChild(el('line', { x1: margin.left, x2: margin.left + plotW, y1: yScale(v), y2: yScale(v), stroke: border, 'stroke-width': 0.5, opacity: 0.7 }));
      svg.appendChild(el('line', { x1: xScale(v), x2: xScale(v), y1: margin.top, y2: margin.top + plotH, stroke: border, 'stroke-width': 0.5, opacity: 0.7 }));
    });
    svg.appendChild(el('line', { x1: margin.left, x2: margin.left + plotW, y1: margin.top + plotH, y2: margin.top + plotH, stroke: muted }));
    svg.appendChild(el('line', { x1: margin.left, x2: margin.left, y1: margin.top, y2: margin.top + plotH, stroke: muted }));
    renderAxisTicks(svg, xScale, lo, hi, false, true, margin, plotW, plotH, ticks);
    renderAxisTicks(svg, yScale, lo, hi, true, false, margin, plotW, plotH, ticks);

    svg.appendChild(el('line', {
      x1: xScale(lo), y1: yScale(lo), x2: xScale(hi), y2: yScale(hi),
      stroke: muted, 'stroke-width': 1.5, 'stroke-dasharray': '5,4',
    }));
    svg.appendChild(el('text', {
      x: xScale(hi) - 6, y: yScale(hi) + 14, 'text-anchor': 'end', 'font-size': 11, fill: muted,
    }, 'perfect: error = uncertainty'));

    active.forEach((s) => {
      const color = getCSSVar(s.color);
      const pts = spread_vs_error[s.key];
      const d = pts.map((p, i) => `${i ? 'L' : 'M'} ${xScale(p.u)} ${yScale(p.rmse)}`).join(' ');
      svg.appendChild(el('path', { d, stroke: color, 'stroke-width': 2, fill: 'none' }));
      pts.forEach((p) => {
        const c = el('circle', { cx: xScale(p.u), cy: yScale(p.rmse), r: 4, fill: color });
        c.appendChild(el('title', {}, `${s.label}: uncertainty ${fmt(p.u, 2)}, RMSE ${fmt(p.rmse, 2)} µmol/kg (n = ${p.n})`));
        svg.appendChild(c);
      });
    });

    svg.appendChild(el('text', { x: margin.left + plotW / 2, y: height - 6, 'text-anchor': 'middle', 'font-size': 11, fill: muted }, 'Predicted uncertainty, binned (µmol/kg)'));
    svg.appendChild(el('text', {
      x: 13, y: margin.top + plotH / 2, 'text-anchor': 'middle', 'font-size': 11, fill: muted,
      transform: `rotate(-90 13 ${margin.top + plotH / 2})`,
    }, 'RMSE of ensemble median (µmol/kg)'));
  };

  host.querySelectorAll('.ux-spread-legend-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const key = chip.dataset.series;
      if (visible.has(key)) {
        if (visible.size === 1) return; // keep at least one series
        visible.delete(key);
      } else {
        visible.add(key);
      }
      chip.setAttribute('aria-pressed', String(visible.has(key)));
      render(plot.clientWidth, plot.clientHeight);
    });
  });
  setupSVGResize(plot, svg, render);

  host.querySelector('.ux-spread-caption').innerHTML = `
    Correlation with absolute error: <strong>r = ${fmt(spread_vs_error.corr_abs_err.member_std, 2)}</strong> (member std) vs
    <strong>r = ${fmt(spread_vs_error.corr_abs_err.scaled_iqr, 2)}</strong> (scaled Q90-Q10).
    Points above the dashed line mean the real error is larger than the predicted uncertainty.
  `;
};

// ============================================================================
// 4. ENSEMBLE GROWTH
// ============================================================================

window.BLOCKS['ensemble-growth'] = (host, block, ctx) => {
  const { data } = ctx;
  const { growth } = data;
  const NS = 'http://www.w3.org/2000/svg';

  host.innerHTML = `
    <div class="ux-growth-panels">
      <div class="ux-growth-panel">
        <div class="ux-growth-panel-title">RMSE vs members</div>
        <div class="ux-growth-plot" data-kind="rmse"><svg class="ux-growth-svg"></svg></div>
      </div>
      <div class="ux-growth-panel">
        <div class="ux-growth-panel-title">80 % coverage vs members</div>
        <div class="ux-growth-plot" data-kind="coverage"><svg class="ux-growth-svg"></svg></div>
      </div>
    </div>
    <div class="ux-growth-caption"></div>
  `;

  const el = (name, attrs, text) => {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (text !== undefined) node.textContent = text;
    return node;
  };

  // kind: key in growth rows, yLabel, optional reference value (dashed) with label, extra y-range to include
  const specs = {
    rmse: { key: 'rmse', yLabel: 'RMSE (µmol/kg)', include: [], pad: 0.15 },
    coverage: { key: 'coverage80', yLabel: 'Coverage', ref: { value: 0.8, label: 'nominal 80 %' }, include: [0.8], pad: 0.1 },
  };
  const xTicks = growth.map((d) => d.n).filter((n) => [1, 7, 14, 21, 28, 42].includes(n));

  host.querySelectorAll('.ux-growth-plot').forEach((plot) => {
    const spec = specs[plot.dataset.kind];
    const svg = plot.querySelector('svg');

    const render = (width, height) => {
      const margin = { top: 16, right: 16, bottom: 40, left: 52 };
      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;
      const ys = growth.map((d) => d[spec.key]).concat(spec.include);
      let yMin = Math.min(...ys);
      let yMax = Math.max(...ys);
      const pad = (yMax - yMin || 1) * spec.pad;
      yMin -= pad;
      yMax += pad;
      const nMin = growth[0].n;
      const nMax = growth[growth.length - 1].n;
      const xScale = (v) => margin.left + ((v - nMin) / (nMax - nMin)) * plotW;
      const yScale = (v) => margin.top + (1 - (v - yMin) / (yMax - yMin)) * plotH;
      const muted = getCSSVar('--text-muted');
      const border = getCSSVar('--border');
      const accent = getCSSVar('--accent');

      svg.innerHTML = '';
      const yTicks = niceTicks(yMin, yMax, 4);
      yTicks.forEach((v) => {
        svg.appendChild(el('line', { x1: margin.left, x2: margin.left + plotW, y1: yScale(v), y2: yScale(v), stroke: border, 'stroke-width': 0.5, opacity: 0.6 }));
      });
      svg.appendChild(el('line', { x1: margin.left, x2: margin.left + plotW, y1: margin.top + plotH, y2: margin.top + plotH, stroke: muted }));
      svg.appendChild(el('line', { x1: margin.left, x2: margin.left, y1: margin.top, y2: margin.top + plotH, stroke: muted }));
      renderAxisTicks(svg, xScale, nMin, nMax, false, true, margin, plotW, plotH, xTicks);
      renderAxisTicks(svg, yScale, yMin, yMax, true, false, margin, plotW, plotH, yTicks,
        spec.key === 'coverage80' ? (v) => `${Math.round(v * 100)} %` : tickLabel);

      if (spec.ref) {
        const y = yScale(spec.ref.value);
        svg.appendChild(el('line', { x1: margin.left, x2: margin.left + plotW, y1: y, y2: y, stroke: getCSSVar('--c-warn'), 'stroke-width': 1.5, 'stroke-dasharray': '4,3' }));
        svg.appendChild(el('text', { x: margin.left + plotW, y: y - 5, 'text-anchor': 'end', 'font-size': 11, fill: getCSSVar('--c-warn') }, spec.ref.label));
      }

      const d = growth.map((row, i) => `${i ? 'L' : 'M'} ${xScale(row.n)} ${yScale(row[spec.key])}`).join(' ');
      svg.appendChild(el('path', { d, stroke: accent, 'stroke-width': 2, fill: 'none' }));
      growth.forEach((row) => {
        const c = el('circle', { cx: xScale(row.n), cy: yScale(row[spec.key]), r: 3.5, fill: accent });
        c.appendChild(el('title', {}, `${row.n} members: ${spec.key === 'coverage80' ? (row[spec.key] * 100).toFixed(1) + ' %' : row[spec.key].toFixed(2)}`));
        svg.appendChild(c);
      });

      svg.appendChild(el('text', { x: margin.left + plotW / 2, y: height - 6, 'text-anchor': 'middle', 'font-size': 11, fill: muted }, 'Ensemble members'));
      svg.appendChild(el('text', { x: 12, y: margin.top + plotH / 2, 'text-anchor': 'middle', 'font-size': 11, fill: muted, transform: `rotate(-90 12 ${margin.top + plotH / 2})` }, spec.yLabel));
    };
    setupSVGResize(plot, svg, render);
  });

  const finalRmse = growth[growth.length - 1].rmse;
  const flat = growth.find((d) => d.rmse <= finalRmse + 0.05);
  const finalCov = growth[growth.length - 1].coverage80;
  host.querySelector('.ux-growth-caption').innerHTML = `
    Error is within 0.05 µmol/kg of its final value from about <strong>${flat ? flat.n : 'N/A'} members</strong>.
    Coverage plateaus near <strong>${(finalCov * 100).toFixed(1)} %</strong>, below the nominal 80 %.
  `;
};
