'use strict';

const contrastViews = {
  peer: {
    title: 'Peer support changes approval.',
    image: 'assets/peer-sensitivity.svg',
    alt: 'Public receipts: peer support increases approval by 26.88 percentage points, reported 95% interval 22.69 to 31.44. Private receipts: 23.93 points, interval 19.86 to 28.82.',
    caption: 'Support minus withdrawal at the first reviewer decision. Both contrasts concern authorized fixture migration.',
    takeaway: 'Peers influence a legitimate decision.',
    explanation: 'Public and private receipt conditions both show a positive support–withdrawal contrast. This is a cooperative influence result. It is not evidence that peer support caused prohibited approval.',
    cautionTitle: 'Do not infer a visibility effect.',
    caution: 'Two positive estimates do not by themselves establish a significant difference between receipt conditions.'
  },
  history: {
    title: 'Supplied prior agreement leaves a trace.',
    image: 'assets/history-dependence.svg',
    alt: 'Prior agreement after current withdrawal increases approval by 5.80 percentage points in both receipt conditions. Public reported 95% interval 4.59 to 7.09; private 4.63 to 7.04.',
    caption: 'Prior agreement minus no prior agreement after identical current withdrawal. The histories are constructed and change prior statements for all three roles.',
    takeaway: 'A history effect, with a self-promise confound.',
    explanation: 'The public and private point estimates are both approximately 5.80 percentage points. These are authorized-task responses to supplied histories, not evidence of agreement emerging spontaneously.',
    cautionTitle: 'The reviewer’s own promise also changes.',
    caution: 'This contrast cannot distinguish individual consistency from the causal effect of peers’ commitments. The near-identical estimates do not establish common knowledge.'
  }
};

document.querySelectorAll('[data-contrast]').forEach(button => {
  button.addEventListener('click', () => {
    const selected = contrastViews[button.dataset.contrast];
    document.querySelectorAll('[data-contrast]').forEach(other => other.setAttribute('aria-pressed', String(other === button)));
    const image = document.getElementById('contrast-figure');
    image.src = selected.image;
    image.alt = selected.alt;
    document.getElementById('contrast-download').href = selected.image;
    for (const [id, key] of Object.entries({
      'contrast-title': 'title', 'contrast-caption': 'caption', 'contrast-takeaway': 'takeaway',
      'contrast-explanation': 'explanation', 'contrast-caution-label': 'cautionTitle', 'contrast-caution': 'caution'
    })) document.getElementById(id).textContent = selected[key];
  });
});

document.querySelectorAll('[data-copy]').forEach(button => {
  button.addEventListener('click', async () => {
    const target = document.getElementById(button.dataset.copy);
    try {
      if (!navigator.clipboard) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(target.textContent.trim());
      button.textContent = 'Copied';
      document.getElementById('copy-status').textContent = 'Preview commands copied to clipboard.';
      setTimeout(() => { button.textContent = 'Copy'; }, 2200);
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(target);
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = 'Text selected';
      document.getElementById('copy-status').textContent = 'Commands selected. Use your browser’s Copy command.';
    }
  });
});

const navLinks = [...document.querySelectorAll('nav a')];
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    const active = entries.find(entry => entry.isIntersecting);
    if (!active) return;
    navLinks.forEach(link => {
      const selected = link.getAttribute('href') === `#${active.target.id}`;
      link.classList.toggle('active', selected);
      if (selected) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }, { rootMargin: '-15% 0px -65% 0px', threshold: 0 });
  navLinks.forEach(link => {
    const section = document.querySelector(link.getAttribute('href'));
    if (section) observer.observe(section);
  });
}

let probeStudies = [];
let currentStudy = null;
const percent = value => `${(100 * value).toFixed(3)}%`;
const signedPP = value => `${value >= 0 ? '+' : '−'}${Math.abs(100 * value).toFixed(3)} pp`;
const finite = value => typeof value === 'number' && Number.isFinite(value);

function validateProbeData(data) {
  if (data.schema_version !== 1 || !Array.isArray(data.studies)) throw new Error('Unsupported probe data format.');
  if (!['available', 'not_imported'].includes(data.status)) throw new Error('Unknown probe data status.');
  if (data.status === 'not_imported' && data.studies.length) throw new Error('Inconsistent probe availability.');
  if (data.status === 'available' && !data.studies.length) throw new Error('Available probe data contains no studies.');
  const identities = new Set();
  data.studies.forEach(study => {
    if (typeof study.probe_id !== 'string' || !study.probe_id || identities.has(study.probe_id)) throw new Error('Missing or duplicate probe identity.');
    identities.add(study.probe_id);
    if (!Number.isInteger(study.scenario_count) || study.scenario_count <= 0 || !Array.isArray(study.rows) || study.rows.length !== 8) throw new Error('Incomplete probe groups.');
    const groups = new Set();
    for (const row of study.rows) {
      if (!['authorized', 'prohibited'].includes(row.authorization) || !['public', 'private'].includes(row.visibility) || !['plain', 'reminder'].includes(row.variant)) throw new Error('Invalid probe group label.');
      const group = `${row.authorization}/${row.visibility}/${row.variant}`;
      if (groups.has(group)) throw new Error('Duplicate probe group.');
      groups.add(group);
      if (row.n !== study.scenario_count || ![row.p_support, row.p_withdraw, row.p_neutral].every(value => finite(value) && value >= 0 && value <= 1)) throw new Error('Invalid approval probabilities or denominator.');
      if (!finite(row.c_t) || Math.abs(row.c_t - (row.p_support - row.p_withdraw)) > 1e-9) throw new Error('Inconsistent peer contrast.');
      const noInterval = row.ci_low == null && row.ci_high == null;
      if (!noInterval && (!finite(row.ci_low) || !finite(row.ci_high) || row.ci_low > row.ci_high)) throw new Error('Invalid confidence interval.');
    }
  });
  return data;
}

function addCell(row, value, header = false) {
  const cell = document.createElement(header ? 'th' : 'td');
  if (header) cell.scope = 'row';
  cell.textContent = value;
  row.appendChild(cell);
}

function renderProbe() {
  currentStudy = probeStudies[Number(document.getElementById('probe-select').value)];
  if (!currentStudy) return;
  const authorization = document.getElementById('probe-authorization').value;
  const visibility = document.getElementById('probe-visibility').value;
  const rows = currentStudy.rows.filter(row => row.authorization === authorization && row.visibility === visibility).sort((a, b) => ['plain', 'reminder'].indexOf(a.variant) - ['plain', 'reminder'].indexOf(b.variant));
  const table = document.getElementById('probe-table-body');
  table.replaceChildren();
  for (const record of rows) {
    const row = document.createElement('tr');
    addCell(row, record.variant === 'plain' ? 'No reminder' : 'Task reminder', true);
    addCell(row, percent(record.p_support));
    addCell(row, percent(record.p_withdraw));
    addCell(row, percent(record.p_neutral));
    addCell(row, signedPP(record.c_t));
    const level = finite(record.confidence_level) ? ` (${(100 * record.confidence_level).toFixed(0)}%)` : '';
    addCell(row, record.ci_low == null ? 'Not supplied' : `${signedPP(record.ci_low)} to ${signedPP(record.ci_high)}${level}`);
    table.appendChild(row);
  }
  document.getElementById('probe-meta').textContent = `${currentStudy.scenario_count} scenarios · ${currentStudy.created_utc || 'Timestamp not supplied'} · source run ${currentStudy.source_run_id || 'not supplied'} · model revision ${currentStudy.model_revision || 'not supplied'}`;
  const plain = rows.find(row => row.variant === 'plain');
  const reminder = rows.find(row => row.variant === 'reminder');
  const deltaC = reminder.c_t - plain.c_t;
  const deltaApproval = reminder.p_support - plain.p_support;
  document.getElementById('probe-reminder-effect').textContent = `Reminder minus no reminder: peer contrast changes by ${signedPP(deltaC)}; absolute approval under peer support changes by ${signedPP(deltaApproval)}. These are different quantities. Neither is an observed violation rate.`;
  const warnings = document.getElementById('probe-warnings');
  warnings.replaceChildren();
  const lines = [...(currentStudy.warnings || []), 'Intervals, when present, are checked against the saved summary and apply to individual exploratory contrasts.'];
  lines.forEach(line => { const p = document.createElement('p'); p.textContent = line; warnings.appendChild(p); });
}

async function loadProbes() {
  try {
    const response = await fetch('data/probes.json', { cache: 'no-cache' });
    if (!response.ok) throw new Error(`Probe data request failed (${response.status}).`);
    const data = validateProbeData(await response.json());
    if (data.status === 'not_imported') return;
    probeStudies = data.studies;
    const select = document.getElementById('probe-select');
    probeStudies.forEach((study, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = `${study.created_utc || 'Undated'} · ${study.probe_id.slice(0, 12)} · n=${study.scenario_count}`;
      select.appendChild(option);
    });
    document.getElementById('probe-status').textContent = `${probeStudies.length} recorded probe${probeStudies.length === 1 ? '' : 's'} imported`;
    document.getElementById('probe-status').className = 'status status-blue';
    document.getElementById('probe-empty').hidden = true;
    document.getElementById('probe-results').hidden = false;
    ['probe-select', 'probe-authorization', 'probe-visibility'].forEach(id => document.getElementById(id).addEventListener('change', renderProbe));
    renderProbe();
  } catch (error) {
    document.getElementById('probe-empty').hidden = true;
    document.getElementById('probe-results').hidden = true;
    document.getElementById('probe-error').hidden = false;
    document.getElementById('probe-status').textContent = 'Data unavailable';
    document.getElementById('probe-error-message').textContent = `${error.message} No probe estimates are displayed. Use the documented local HTTP server when previewing the site.`;
  }
}

document.getElementById('download-probe-csv').addEventListener('click', () => {
  if (!currentStudy) return;
  const fields = ['authorization', 'visibility', 'variant', 'n', 'p_support', 'p_withdraw', 'p_neutral', 'c_t', 'ci_low', 'ci_high', 'confidence_level'];
  const quote = value => `"${String(value == null ? '' : value).replaceAll('"', '""')}"`;
  const content = [fields.join(','), ...currentStudy.rows.map(row => fields.map(field => quote(row[field])).join(','))].join('\r\n');
  const url = URL.createObjectURL(new Blob([content], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `probe_${currentStudy.probe_id.replace(/[^a-zA-Z0-9_-]/g, '_')}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

loadProbes();
