// High-resolution TA methods page renderer
// Fetches content.json and data.json, renders the page with block handlers

const app = {
  data: null,
  content: null,
  animateOnScroll: new Map(),
  selectedMember: 0,

  async init() {
    try {
      const [dataResp, contentResp] = await Promise.all([
        fetch('data.json'),
        fetch('content.json')
      ]);

      if (!dataResp.ok || !contentResp.ok) throw new Error('Failed to fetch');

      this.data = await dataResp.json();
      this.content = await contentResp.json();

      this.render();
      this.setupScrollSpy();
      this.setupAnimations();
      this.setupThemeToggle();
    } catch (err) {
      console.error('Failed to load data:', err);
      const errorEl = document.getElementById('load-error');
      errorEl.innerHTML = `
        <strong>Could not load the page.</strong>
        <p>The app needs to be served over HTTP (file:// doesn't work with fetch):</p>
        <p><code>cd /net/sea/work/gregorl/projects/highres_TA/web && python -m http.server 8000</code></p>
        <p>Then open <code>http://localhost:8000</code></p>
      `;
      errorEl.hidden = false;
    }
  },

  render() {
    // Create helper context
    const ctx = {
      data: this.data,
      content: this.content,
      tpl: this.tpl.bind(this),
      md: this.md.bind(this),
      rdylbu: this.rdylbu.bind(this)
    };

    // Brand
    const brand = document.getElementById('brand');
    brand.textContent = this.content.site.short_title;

    // Nav
    const nav = document.getElementById('nav');
    this.content.site.nav.forEach(item => {
      const a = document.createElement('a');
      a.href = `#${item.id}`;
      a.textContent = item.label;
      a.className = 'nav-link';
      nav.appendChild(a);
    });

    // Hero
    const hero = document.getElementById('hero');
    const heroHTML = `
      <div class="section__inner">
        <div class="hero-content">
          <div class="hero-eyebrow">${this.escape(this.content.hero.eyebrow)}</div>
          <h1 class="hero-headline">${this.escape(this.content.hero.headline)}</h1>
          <p class="hero-lead">${this.md(this.content.hero.lead, ctx)}</p>
          <div class="hero-stats">
            ${this.content.hero.stats.map(stat => `
              <div class="stat-card">
                <div class="stat-label">${this.escape(stat.label)}</div>
                <div class="stat-value">${this.tpl(stat.value, ctx)}</div>
                <div class="stat-note">${this.tpl(this.escape(stat.note), ctx)}</div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `;
    hero.innerHTML = heroHTML;

    // Content sections
    const contentDiv = document.getElementById('content');
    this.content.sections.forEach(section => {
      const sectionEl = document.createElement('section');
      sectionEl.className = 'section';
      sectionEl.id = section.id;

      const inner = document.createElement('div');
      inner.className = 'section__inner';

      const sectionHeader = document.createElement('div');
      sectionHeader.className = 'section-header';
      sectionHeader.innerHTML = `
        <div class="kicker">${this.escape(section.kicker)}</div>
        <h2 class="section-title">${this.escape(section.title)}</h2>
        <p class="section-lead">${this.md(section.lead, ctx)}</p>
      `;
      inner.appendChild(sectionHeader);

      // Render blocks
      section.blocks.forEach(block => {
        try {
          const blockEl = this.renderBlock(block, ctx);
          if (blockEl) {
            inner.appendChild(blockEl);
            this.animateOnScroll.set(blockEl, true);
          }
        } catch (err) {
          console.error(`Error rendering block type "${block.type}":`, err);
          const errorNote = document.createElement('div');
          errorNote.className = 'block-error';
          errorNote.textContent = `Error rendering ${block.type} block`;
          inner.appendChild(errorNote);
        }
      });

      sectionEl.appendChild(inner);
      contentDiv.appendChild(sectionEl);
    });

    // Footer
    const footer = document.getElementById('footer');
    footer.innerHTML = `
      <div class="section__inner">
        <p>${this.escape(this.content.site.footer)}</p>
      </div>
    `;
  },

  renderBlock(block, ctx) {
    const container = document.createElement('div');
    container.className = `block block--${block.type}`;

    // Add title if present
    if (block.title) {
      const title = document.createElement('h3');
      title.className = 'block-title';
      title.textContent = block.title;
      container.appendChild(title);
    }

    // Route to renderer
    switch (block.type) {
      case 'text':
        return this.renderText(container, block, ctx);
      case 'cards':
        return this.renderCards(container, block, ctx);
      case 'steps':
        return this.renderSteps(container, block, ctx);
      case 'table':
        return this.renderTable(container, block, ctx);
      case 'callout':
        return this.renderCallout(container, block, ctx);
      case 'equation':
        return this.renderEquation(container, block, ctx);
      case 'funnel':
        return this.renderFunnel(container, block, ctx);
      case 'split-explorer':
        return this.renderSplitExplorer(container, block, ctx);
      case 'member-grid':
        return this.renderMemberGrid(container, block, ctx);
      default:
        // Try to call external renderer
        if (window.BLOCKS && window.BLOCKS[block.type]) {
          window.BLOCKS[block.type](container, block, ctx);
          return container;
        }
        console.warn(`Unknown block type: ${block.type}`);
        return null;
    }
  },

  renderText(container, block, ctx) {
    const p = document.createElement('p');
    p.className = 'block-text';
    p.innerHTML = this.md(block.body, ctx);
    container.appendChild(p);
    return container;
  },

  renderCards(container, block, ctx) {
    const grid = document.createElement('div');
    grid.className = `cards-grid cards-${block.columns}col`;

    block.items.forEach(item => {
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        ${item.tag ? `<div class="card-tag">${this.escape(item.tag)}</div>` : ''}
        <h4 class="card-title">${this.escape(item.title)}</h4>
        <p class="card-body">${this.md(item.body, ctx)}</p>
      `;
      grid.appendChild(card);
    });

    container.appendChild(grid);
    return container;
  },

  renderSteps(container, block, ctx) {
    const timeline = document.createElement('ol');
    timeline.className = 'steps-timeline';

    block.items.forEach((item, idx) => {
      const li = document.createElement('li');
      li.className = 'step-item';
      li.innerHTML = `
        <div class="step-number">${idx + 1}</div>
        <div class="step-content">
          <h4 class="step-title">${this.escape(item.title)}</h4>
          <p class="step-body">${this.md(item.body, ctx)}</p>
        </div>
      `;
      timeline.appendChild(li);
    });

    container.appendChild(timeline);
    return container;
  },

  renderTable(container, block, ctx) {
    const wrapper = document.createElement('div');
    wrapper.className = 'table-wrapper';

    const table = document.createElement('table');
    table.className = 'data-table';

    // Header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    block.columns.forEach(col => {
      const th = document.createElement('th');
      th.textContent = col;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    const tbody = document.createElement('tbody');
    block.rows.forEach((row, idx) => {
      const tr = document.createElement('tr');
      if (idx % 2 === 1) tr.className = 'zebra';
      row.forEach(cell => {
        const td = document.createElement('td');
        td.innerHTML = this.md(cell, ctx);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    wrapper.appendChild(table);
    container.appendChild(wrapper);
    return container;
  },

  renderCallout(container, block, ctx) {
    container.className += ` callout callout--${block.tone}`;
    container.innerHTML = `
      <div class="callout-bar"></div>
      <div class="callout-content">
        ${block.title ? `<h4 class="callout-title">${this.escape(block.title)}</h4>` : ''}
        <p class="callout-body">${this.md(block.body, ctx)}</p>
      </div>
    `;
    return container;
  },

  renderEquation(container, block, ctx) {
    const code = document.createElement('pre');
    code.className = 'equation';
    code.innerHTML = block.items.map(item => this.escape(item)).join('\n\n');
    container.appendChild(code);
    return container;
  },

  renderFunnel(container, block, ctx) {
    const funnel = this.data.funnel;
    const maxRows = funnel[0].rows;

    const funnelDiv = document.createElement('div');
    funnelDiv.className = 'funnel';

    funnel.forEach((step, idx) => {
      const ratio = step.rows / maxRows;
      const retained = ((step.rows / maxRows) * 100).toFixed(1);
      const rampStops = [10, 8, 3, 2, 0]; // RdYlBu without the pale yellow stops
      const stopIdx = rampStops[Math.round(idx * (rampStops.length - 1) / Math.max(1, funnel.length - 1))];
      const color = getComputedStyle(document.documentElement).getPropertyValue(`--rdylbu-${stopIdx}`).trim();

      const bar = document.createElement('div');
      bar.className = 'funnel-bar';
      bar.style.setProperty('--width', `${ratio * 100}%`);
      bar.style.setProperty('--color', color);
      bar.innerHTML = `
        <div class="funnel-label">
          <div class="funnel-step">${this.escape(step.step)}</div>
          <div class="funnel-stats">
            <span>${this.formatInt(step.rows)} rows</span>
            <span>${retained}% retained</span>
            <span>${step.cruises} cruises</span>
          </div>
        </div>
      `;
      funnelDiv.appendChild(bar);
    });

    container.appendChild(funnelDiv);
    return container;
  },

  renderSplitExplorer(container, block, ctx) {
    const blocks = this.data.blocks;

    const explorer = document.createElement('div');
    explorer.className = 'split-explorer';

    // Split selector
    const splitControl = document.createElement('div');
    splitControl.className = 'segmented-control';
    splitControl.innerHTML = '<label>Split</label>';
    for (let i = 1; i <= 6; i++) {
      const btn = document.createElement('button');
      btn.textContent = `${i}`;
      btn.className = i === 1 ? 'active' : '';
      btn.setAttribute('aria-pressed', i === 1 ? 'true' : 'false');
      btn.addEventListener('click', () => this.updateSplitExplorer(i, 1, explorer));
      splitControl.appendChild(btn);
    }
    explorer.appendChild(splitControl);

    // Blocks visualization
    const visualization = document.createElement('div');
    visualization.className = 'split-visualization';

    const blockRow = document.createElement('div');
    blockRow.className = 'blocks-row';
    blockRow.id = 'split-blocks';
    visualization.appendChild(blockRow);

    // Inner fold selector
    const innerControl = document.createElement('div');
    innerControl.className = 'segmented-control';
    innerControl.innerHTML = '<label>Inner fold</label>';
    for (let i = 1; i <= 5; i++) {
      const btn = document.createElement('button');
      btn.textContent = `${i}`;
      btn.className = i === 1 ? 'active' : '';
      btn.setAttribute('aria-pressed', i === 1 ? 'true' : 'false');
      btn.addEventListener('click', () => this.updateSplitExplorer(1, i, explorer));
      innerControl.appendChild(btn);
    }
    visualization.appendChild(innerControl);

    // Inner blocks row
    const innerBlockRow = document.createElement('div');
    innerBlockRow.className = 'blocks-row inner-blocks';
    innerBlockRow.id = 'inner-blocks';
    visualization.appendChild(innerBlockRow);

    // Summary text
    const summary = document.createElement('p');
    summary.className = 'split-summary';
    summary.id = 'split-summary';
    visualization.appendChild(summary);

    explorer.appendChild(visualization);
    container.appendChild(explorer);

    // Initial render
    this.updateSplitExplorer(1, 1, explorer);

    return container;
  },

  updateSplitExplorer(split, innerFold, container) {
    const blocks = this.data.blocks;
    const testBlock = 0;
    const validationBlock = split;

    // Update split buttons
    const splitBtns = container.querySelectorAll('.segmented-control')[0].querySelectorAll('button');
    splitBtns.forEach((btn, idx) => {
      if (idx + 1 === split) {
        btn.classList.add('active');
        btn.setAttribute('aria-pressed', 'true');
      } else {
        btn.classList.remove('active');
        btn.setAttribute('aria-pressed', 'false');
      }
    });

    // Render main blocks
    const blockRow = container.querySelector('#split-blocks');
    blockRow.innerHTML = '';
    blocks.forEach((block, idx) => {
      const tile = document.createElement('div');
      const role = idx === testBlock ? 'test' : idx === validationBlock ? 'validation' : 'train';
      tile.className = `block-tile block-tile--${role}`;
      tile.innerHTML = `
        <div class="block-label">Block ${idx}</div>
        <div class="block-info">
          <div>${this.formatInt(block.observations)} obs</div>
          <div>${block.cruises} cruises</div>
        </div>
      `;
      blockRow.appendChild(tile);
    });

    // Update inner fold buttons
    const innerBtns = container.querySelectorAll('.segmented-control')[1].querySelectorAll('button');
    innerBtns.forEach((btn, idx) => {
      if (idx + 1 === innerFold) {
        btn.classList.add('active');
        btn.setAttribute('aria-pressed', 'true');
      } else {
        btn.classList.remove('active');
        btn.setAttribute('aria-pressed', 'false');
      }
    });

    // Render inner blocks (training blocks only)
    const innerBlockRow = container.querySelector('#inner-blocks');
    innerBlockRow.innerHTML = '';
    const trainingBlocks = blocks.filter((_, idx) => idx !== testBlock && idx !== validationBlock);
    trainingBlocks.forEach((block, idx) => {
      const heldOutBlock = (idx + 1) === innerFold ? idx + 1 : null;
      const isHeldOut = (idx + 1) === innerFold;
      const tile = document.createElement('div');
      tile.className = `block-tile block-tile--train ${isHeldOut ? 'held-out' : 'fit'}`;
      const role = isHeldOut ? 'held out for early stopping' : 'fit';
      tile.innerHTML = `
        <div class="block-label">Block ${block.block} <span class="role">${role}</span></div>
        <div class="block-info">
          <div>${this.formatInt(block.observations)} obs</div>
        </div>
      `;
      innerBlockRow.appendChild(tile);
    });

    // Update summary
    const summary = container.querySelector('#split-summary');
    const trainBlocks = blocks.filter((_, idx) => idx !== testBlock && idx !== validationBlock).map(b => b.block).join(', ');
    const trainObs = blocks.filter((_, idx) => idx !== testBlock && idx !== validationBlock).reduce((sum, b) => sum + b.observations, 0);
    summary.textContent = `Split ${split} · trains on blocks ${trainBlocks} (${this.formatInt(trainObs)} obs) · validates on block ${validationBlock} · tests on block ${testBlock}`;
  },

  renderMemberGrid(container, block, ctx) {
    const members = this.data.members;
    const splits = 6;
    const replicates = 7;

    const gridContainer = document.createElement('div');
    gridContainer.className = 'member-grid-container';

    // Color legend
    const rmsValues = members.map(m => m.test.rmse);
    const minRmse = Math.min(...rmsValues);
    const maxRmse = Math.max(...rmsValues);

    const legend = document.createElement('div');
    legend.className = 'rmse-legend';
    legend.innerHTML = `
      <div class="legend-label">RMSE</div>
      <div class="legend-gradient">
        <div class="legend-tick" style="left: 0%">${minRmse.toFixed(1)}</div>
        <div class="legend-tick" style="left: 100%">${maxRmse.toFixed(1)}</div>
      </div>
    `;
    gridContainer.appendChild(legend);

    // Summary stats
    const rmseSorted = [...rmsValues].sort((a, b) => a - b);
    const medianRmse = rmseSorted[Math.floor(rmseSorted.length / 2)];
    const stats = document.createElement('div');
    stats.className = 'member-summary-stats';
    stats.innerHTML = `
      <div>Min RMSE: ${minRmse.toFixed(1)}</div>
      <div>Median RMSE: ${medianRmse.toFixed(1)}</div>
      <div>Max RMSE: ${maxRmse.toFixed(1)}</div>
    `;
    gridContainer.appendChild(stats);

    // Grid wrapper with scroll
    const gridWrapper = document.createElement('div');
    gridWrapper.className = 'member-grid-wrapper';

    const gridContent = document.createElement('div');
    gridContent.className = 'member-grid';

    // Headers
    const headerRowTop = document.createElement('div');
    headerRowTop.className = 'grid-row header-row';
    const cornerCell = document.createElement('div');
    cornerCell.className = 'grid-cell corner';
    headerRowTop.appendChild(cornerCell);
    for (let r = 0; r < replicates; r++) {
      const cell = document.createElement('div');
      cell.className = 'grid-cell header';
      cell.textContent = `HPO ${r}`;
      headerRowTop.appendChild(cell);
    }
    gridContent.appendChild(headerRowTop);

    // Data rows
    for (let s = 1; s <= splits; s++) {
      const row = document.createElement('div');
      row.className = 'grid-row';

      // Row header
      const rowHeader = document.createElement('div');
      rowHeader.className = 'grid-cell row-header';
      rowHeader.textContent = `Split ${s}`;
      row.appendChild(rowHeader);

      // Data cells
      for (let r = 0; r < replicates; r++) {
        const member = members.find(m => m.split === s && m.replicate === r);
        if (member) {
          const rmse = member.test.rmse;
          const color = this.rdylbu((rmse - minRmse) / (maxRmse - minRmse), true);
          const cell = document.createElement('button');
          cell.className = 'grid-cell data-cell';
          cell.style.setProperty('--cell-color', color);
          cell.innerHTML = `<span>${rmse.toFixed(1)}</span>`;
          cell.addEventListener('click', () => this.selectMember(member, gridContainer));
          cell.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.selectMember(member, gridContainer);
          });
          row.appendChild(cell);
        }
      }

      gridContent.appendChild(row);
    }

    gridWrapper.appendChild(gridContent);
    gridContainer.appendChild(gridWrapper);

    // Detail panel
    const detailPanel = document.createElement('div');
    detailPanel.className = 'member-detail-panel';
    detailPanel.id = 'member-detail';
    gridContainer.appendChild(detailPanel);

    // Initial selection
    this.selectMember(members[0], gridContainer);

    container.appendChild(gridContainer);
    return container;
  },

  selectMember(member, container) {
    this.selectedMember = member;

    const detailPanel = container.querySelector('#member-detail');
    detailPanel.innerHTML = `
      <div class="detail-header">
        <h4>${this.escape(member.name)}</h4>
        <div class="detail-meta">Seed: ${member.seed}</div>
      </div>

      <div class="detail-section">
        <h5>Hyperparameters</h5>
        <dl class="detail-list">
          <dt>Depth</dt><dd>${member.params.depth.toFixed(1)}</dd>
          <dt>L2 leaf reg</dt><dd>${member.params.l2_leaf_reg.toFixed(4)}</dd>
          <dt>Learning rate</dt><dd>${member.params.learning_rate.toFixed(4)}</dd>
          <dt>Min data in leaf</dt><dd>${member.params.min_data_in_leaf.toFixed(0)}</dd>
          <dt>Random strength</dt><dd>${member.params.random_strength.toFixed(4)}</dd>
        </dl>
      </div>

      <div class="detail-section">
        <h5>Training</h5>
        <dl class="detail-list">
          <dt>Iterations</dt><dd>${member.iterations}</dd>
          <dt>Inner CV loss</dt><dd>${member.inner_cv_loss.toFixed(3)}</dd>
          <dt>Training rows</dt><dd>${this.formatInt(member.n_train)}</dd>
          <dt>Training cruises</dt><dd>${member.cruises_train}</dd>
          <dt>Validation rows</dt><dd>${this.formatInt(member.n_validation)}</dd>
        </dl>
      </div>

      <div class="detail-section">
        <h5>Test scores</h5>
        <dl class="detail-list">
          <dt>RMSE</dt><dd>${member.test.rmse.toFixed(2)} µmol/kg</dd>
          <dt>MAE</dt><dd>${member.test.mae.toFixed(2)} µmol/kg</dd>
          <dt>CRPS</dt><dd>${member.test.crps.toFixed(3)}</dd>
          <dt>80% coverage</dt><dd>${(member.test.coverage80 * 100).toFixed(1)}%</dd>
          <dt>Validation RMSE</dt><dd>${member.validation.rmse.toFixed(2)} µmol/kg</dd>
        </dl>
      </div>
    `;
  },

  // Helper functions
  tpl(str, ctx) {
    // Resolve {{path.to.value|fmt}} placeholders
    return str.replace(/\{\{([^|]+)(?:\|([^}]+))?\}\}/g, (match, path, fmt) => {
      const keys = path.trim().split('.');
      let value = ctx.data;

      for (const key of keys) {
        if (value && typeof value === 'object' && key in value) {
          value = value[key];
        } else {
          return 'n/a';
        }
      }

      if (fmt) {
        fmt = fmt.trim();
        if (fmt === 'int') return this.formatInt(value);
        if (fmt === 'pct') return (value * 100).toFixed(1) + ' %';
        // Numeric format: N decimals
        const decimals = parseInt(fmt);
        if (!isNaN(decimals)) return value.toFixed(decimals);
      }

      return String(value);
    });
  },

  md(str, ctx) {
    // First apply template substitutions
    let html = this.tpl(str, ctx);
    // HTML escape
    html = this.escape(html);
    // Then parse markdown-like syntax
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    return html;
  },

  rdylbu(t, reverse = false) {
    // Interpolate across 11 RdYlBu stops
    const stops = [];
    for (let i = 0; i <= 10; i++) {
      const varName = `--rdylbu-${i}`;
      const color = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
      stops.push(color);
    }

    if (reverse) t = 1 - t;
    t = Math.max(0, Math.min(1, t));
    const idx = t * 10;
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);
    const frac = idx - lower;

    if (lower === upper) return stops[lower];

    // Interpolate colors
    const c1 = this.parseColor(stops[lower]);
    const c2 = this.parseColor(stops[upper]);
    return this.mixColors(c1, c2, frac);
  },

  parseColor(color) {
    // Parse hex color
    const hex = color.trim();
    const r = parseInt(hex.substr(1, 2), 16);
    const g = parseInt(hex.substr(3, 2), 16);
    const b = parseInt(hex.substr(5, 2), 16);
    return { r, g, b };
  },

  mixColors(c1, c2, t) {
    const r = Math.round(c1.r * (1 - t) + c2.r * t);
    const g = Math.round(c1.g * (1 - t) + c2.g * t);
    const b = Math.round(c1.b * (1 - t) + c2.b * t);
    return `rgb(${r}, ${g}, ${b})`;
  },

  escape(str) {
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
  },

  formatInt(num) {
    // Swiss apostrophe thousands separator
    return Math.round(num).toString().replace(/\B(?=(\d{3})+(?!\d))/g, "'");
  },

  setupThemeToggle() {
    const toggle = document.getElementById('theme-toggle');
    const html = document.documentElement;

    // Load saved theme
    try {
      const saved = localStorage.getItem('theme');
      if (saved) html.setAttribute('data-theme', saved);
    } catch (e) {
      // localStorage may not be available
    }

    toggle.addEventListener('click', () => {
      const current = html.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', next);

      try {
        localStorage.setItem('theme', next);
      } catch (e) {
        // localStorage may not be available
      }
    });
  },

  setupScrollSpy() {
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('.nav-link');

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${entry.target.id}`) {
              link.classList.add('active');
            }
          });
        }
      });
    }, { threshold: 0.3 });

    sections.forEach(section => observer.observe(section));
  },

  setupAnimations() {
    // Animate elements on scroll with fade/slide-up
    if (!('IntersectionObserver' in window)) return;

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting && this.animateOnScroll.has(entry.target)) {
          if (window.matchMedia('(prefers-reduced-motion: no-preference)').matches) {
            entry.target.style.animation = 'fadeSlideUp 0.5s ease-out backwards';
          }
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 });

    this.animateOnScroll.forEach((_, el) => {
      observer.observe(el);
    });
  }
};

// Initialize on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => app.init());
} else {
  app.init();
}
