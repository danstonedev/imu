document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData();
    // Append scalar fields explicitly to control ordering
    formData.append('height_m', document.getElementById('height').value || '1.75');
    formData.append('mass_kg', document.getElementById('mass').value || '100');
    // Ensure baseline_mode is sent even for uploads
    const baselineSel = document.getElementById('baseline-mode');
    formData.append('baseline_mode', baselineSel?.value || 'linear');

    // Unified input: accepts .csv, .zip, or a folder
    const dataInput = document.getElementById('data-input');
    const chosen = (dataInput && dataInput.files) ? Array.from(dataInput.files) : [];
    if (chosen.length === 1 && chosen[0].name.toLowerCase().endsWith('.zip')) {
        formData.append('archive', chosen[0]);
    } else if (chosen.length > 0) {
        // Treat as files/folder selection
        chosen.forEach(f => formData.append('files', f));
        chosen.map(f => f.webkitRelativePath || f.name).forEach(p => formData.append('paths', p));
    }

    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error');

    loadingDiv.classList.remove('hidden');
    resultsDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');

    try {
    const response = await fetch('/api/analyze/', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Analysis failed');
        }

        const payload = await response.json();
        if (payload && Array.isArray(payload.batch)) {
            // Batch mode: populate dataset selector, show first dataset
            setupDatasetPicker(payload.batch);
            if (payload.batch.length) displayResults(payload.batch[0].results);
        } else {
            // Single result
            hideDatasetPicker();
            displayResults(payload);
        }

    } catch (e) {
        errorDiv.classList.remove('hidden');
    } finally {
        loadingDiv.classList.add('hidden');
    }
});

let lastResults = null;
let cycleRendered = false;
let lastBatch = null;

function setupDatasetPicker(batch) {
    lastBatch = batch;
    const picker = document.getElementById('dataset-picker');
    const select = document.getElementById('dataset-select');
    if (!picker || !select) return;
    picker.classList.remove('hidden');
    select.innerHTML = '';
    batch.forEach((b, i) => {
        const opt = document.createElement('option');
        opt.value = String(i);
        opt.textContent = b.name || `Dataset ${i+1}`;
        select.appendChild(opt);
    });
    select.onchange = () => {
        const idx = parseInt(select.value, 10);
        if (!isNaN(idx) && lastBatch && lastBatch[idx]) {
            displayResults(lastBatch[idx].results);
        }
    };
}

function hideDatasetPicker() {
    const picker = document.getElementById('dataset-picker');
    const select = document.getElementById('dataset-select');
    if (picker) picker.classList.add('hidden');
    if (select) select.innerHTML = '';
    lastBatch = null;
}

function displayResults(results) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.classList.remove('hidden');
    lastResults = results;

    // Tabs
    setupTabs();

    // Downloads
    const leftCsvLink = document.getElementById('left-csv-link');
    const rightCsvLink = document.getElementById('right-csv-link');
    leftCsvLink.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(results.left_csv);
    rightCsvLink.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(results.right_csv);
    const leftCycleCsvLink = document.getElementById('left-cycle-csv-link');
    const rightCycleCsvLink = document.getElementById('right-cycle-csv-link');
    if (results.left_cycle_csv) leftCycleCsvLink.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(results.left_cycle_csv);
    if (results.right_cycle_csv) rightCycleCsvLink.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(results.right_cycle_csv);
    // Angle CSV downloads
    const leftAnglesCsv = document.getElementById('left-angles-csv-link');
    const rightAnglesCsv = document.getElementById('right-angles-csv-link');
    if (leftAnglesCsv && results.left_angles_csv) leftAnglesCsv.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(results.left_angles_csv);
    if (rightAnglesCsv && results.right_angles_csv) rightAnglesCsv.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(results.right_angles_csv);
    const lHipCyc = document.getElementById('left-hip-cycle-angles-csv-link');
    const rHipCyc = document.getElementById('right-hip-cycle-angles-csv-link');
    const lKneeCyc = document.getElementById('left-knee-cycle-angles-csv-link');
    const rKneeCyc = document.getElementById('right-knee-cycle-angles-csv-link');
    if (lHipCyc && results.left_hip_cycle_csv) lHipCyc.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(results.left_hip_cycle_csv);
    if (rHipCyc && results.right_hip_cycle_csv) rHipCyc.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(results.right_hip_cycle_csv);
    if (lKneeCyc && results.left_knee_cycle_csv) lKneeCyc.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(results.left_knee_cycle_csv);
    if (rKneeCyc && results.right_knee_cycle_csv) rKneeCyc.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(results.right_knee_cycle_csv);

    // Time series charts (Mx/My/Mz/|M| with stance shading)
    renderTimeCharts(results);

    // Cycle averages (mean ± SD) -> defer until Cycle tab is opened to avoid hidden-width sizing issues
    cycleRendered = false;

    // Metrics
    renderMetrics(results);
}

let charts = {};

function setupTabs() {
    const buttons = document.querySelectorAll('.tab-btn');
    const panels = document.querySelectorAll('.tab-panel');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            panels.forEach(p => p.classList.add('hidden'));
            btn.classList.add('active');
            document.getElementById('tab-' + btn.dataset.tab).classList.remove('hidden');
            // Re-render charts when their tab is activated so sizes are correct
            if (btn.dataset.tab === 'time' && lastResults) {
                renderTimeCharts(lastResults);
            }
            if (btn.dataset.tab === 'cycle' && lastResults) {
                renderCycleCharts(lastResults);
                cycleRendered = true;
            }
            if (btn.dataset.tab === 'angles-time' && lastResults) {
                renderAnglesTimeCharts(lastResults);
            }
            if (btn.dataset.tab === 'angles-cycle' && lastResults) {
                renderAnglesCycleCharts(lastResults);
            }
        });
    });
    // If Cycle tab is already active by default when results appear, render it once
    const active = document.querySelector('.tab-btn.active');
    if (active && active.dataset.tab === 'cycle' && lastResults && !cycleRendered) {
        renderCycleCharts(lastResults);
        cycleRendered = true;
    }
}

let timeChartListenersBound = false;
function renderTimeCharts(results) {
    const host = document.getElementById('time-overlay');
    host.innerHTML = '';
    const sideSel = document.getElementById('time-side');
    const width = host.clientWidth || 800;
    const height = host.clientHeight || 380;
    const margin = { top: 20, right: 20, bottom: 30, left: 45 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(host).append('svg')
        .attr('width', width)
        .attr('height', height);
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    // Clip data to plotting area to prevent overspill at ends
    const clipId = 'clip-ts';
    svg.append('defs').append('clipPath').attr('id', clipId)
        .append('rect').attr('x', 0).attr('y', 0).attr('width', innerW).attr('height', innerH);
    const gData = g.append('g').attr('clip-path', `url(#${clipId})`);

    const normalize = true;
    const timeL = results.time_L || [];
    const timeR = results.time_R || [];
    const picks = {
        Mx: document.getElementById('show-mx').checked,
        My: document.getElementById('show-my').checked,
        Mz: document.getElementById('show-mz').checked,
        Mmag: document.getElementById('show-mmag').checked,
    };
    const sideMode = sideSel?.value || 'both';
    const L = { Mx: results.L_mx, My: results.L_my, Mz: results.L_mz, Mmag: results.L_Mmag };
    const R = { Mx: results.R_mx, My: results.R_my, Mz: results.R_mz, Mmag: results.R_Mmag };
    const stanceL = results.stance_L||[];
    const stanceR = results.stance_R||[];

    // Build flat series arrays [{t,y,side,comp}]
    const series = [];
    const push = (side, comp, times, arr, color) => {
        if (!picks[comp]) return;
        if (!(sideMode === 'both' || sideMode === side)) return;
        const line = d3.line().x(d=>x(d.t)).y(d=>y(d.y));
        const data = (arr||[]).map((v,i)=> ({ t: times[i], y: v }));
        series.push({ side, comp, color, data, line });
    };
    const colors = {
        L: { Mx: '#2196f3', My: '#4caf50', Mz: '#f44336', Mmag: '#9e9e9e' },
        R: { Mx: 'rgba(33,150,243,0.8)', My: 'rgba(76,175,80,0.8)', Mz: 'rgba(244,67,54,0.8)', Mmag: '#636363' }
    };

    // Scales (compute domain from selected data)
    const allTimes = [];
    const allVals = [];
    ['Mx','My','Mz','Mmag'].forEach(c => {
        if (!picks[c]) return;
        if (sideMode !== 'R') { allTimes.push(...timeL); allVals.push(...(L[c]||[])); }
        if (sideMode !== 'L') { allTimes.push(...timeR); allVals.push(...(R[c]||[])); }
    });
    const tMin = Math.floor(d3.min(allTimes) ?? 0);
    const tMax = Math.ceil(d3.max(allTimes) ?? 1);
    const vMin = Math.min(0, d3.min(allVals) ?? 0);
    const vMax = Math.max(1, d3.max(allVals) ?? 1);

    const x = d3.scaleLinear().domain([tMin, tMax]).range([0, innerW]);
    const y = d3.scaleLinear().domain([vMin, vMax]).nice().range([innerH, 0]);

    // Grid and axes with per-second minor ticks
    const roundTo = (v, d) => { const f = Math.pow(10, d); return Math.round(v * f) / f; };
    const computeSteps = (domain) => {
        const range = Math.max(0.0001, domain[1]-domain[0]);
        if (range <= 3.5) return { gridStep: 0.1, labelStep: 0.1, decimals: 1 };
        if (range <= 60) return { gridStep: 1, labelStep: 1, decimals: 0 };
        return { gridStep: 1, labelStep: 5, decimals: 0 };
    };
    const steps = computeSteps([tMin, tMax]);
    const buildValues = (d0, d1, step, decimals) => {
        const start = roundTo(Math.ceil(d0/step)*step, decimals);
        const end = roundTo(Math.floor(d1/step)*step, decimals);
        const vals = [];
        for (let v = start; v <= end + 1e-8; v = roundTo(v + step, decimals)) vals.push(roundTo(v, decimals));
        return vals;
    };
    const gridVals = buildValues(tMin, tMax, steps.gridStep, steps.decimals);
    const labelVals = buildValues(tMin, tMax, steps.labelStep, steps.decimals);
    const xAxis = g.append('g').attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(x).tickValues(labelVals).tickFormat(d => steps.decimals ? d.toFixed(1) : d));
    const gridGroup = g.append('g').attr('class','grid');
    gridGroup.selectAll('line.gridline')
        .data(gridVals, d=>d)
        .join('line')
        .attr('class','gridline')
        .attr('x1', d=>x(d)).attr('x2', d=>x(d))
        .attr('y1', 0).attr('y2', innerH)
        .attr('stroke', 'rgba(0,0,0,0.08)');
    g.append('g').call(d3.axisLeft(y));
    svg.append('text').attr('x', margin.left + innerW/2).attr('y', height-5).attr('text-anchor','middle').text('Time (s)');
    svg.append('text').attr('transform',`translate(12,${margin.top+innerH/2}) rotate(-90)`).attr('text-anchor','middle').text('Torque (Nm)');

    // Quick hover: tooltip + crosshair
    // reuse a single tooltip across renders to avoid duplicates stacking
    let tooltip = d3.select('body').select('div.d3-tooltip');
    if (tooltip.empty()) tooltip = d3.select('body').append('div').attr('class','d3-tooltip').style('opacity',0);
    const cross = g.append('line').attr('class','crosshair-x').attr('y1',0).attr('y2',innerH).attr('stroke','rgba(0,0,0,0.25)').style('display','none');
    const bisect = d3.bisector(d=>d).left;
    const valueAt = (arr, idx) => (idx>=0 && idx < arr.length ? arr[idx] : undefined);
    const onMove = (ev, zx) => {
        const [mx] = d3.pointer(ev, g.node());
        const xScale = zx || x;
        const t = xScale.invert(mx);
        cross.attr('x1', mx).attr('x2', mx).style('display','block');
        const sLocal = computeSteps(xScale.domain());
        const iL = timeL.length ? Math.max(0, Math.min(timeL.length-1, bisect(timeL, t))) : -1;
        const iR = timeR.length ? Math.max(0, Math.min(timeR.length-1, bisect(timeR, t))) : -1;
        const rows = [];
        rows.push(`<div><b>t=${sLocal.decimals?roundTo(t,1).toFixed(1):Math.round(t)}s</b></div>`);
        const pushRow = (side,label,val,color)=>{ if(val!==undefined){ rows.push(`<div><span style=\"display:inline-block;width:10px;height:10px;background:${color};margin-right:6px\"></span>${side} ${label}: ${val.toFixed(1)}</div>`);} };
        if (sideMode !== 'R' && iL>=0){
            if (picks.Mmag) pushRow('L','|M|', valueAt(L.Mmag,iL), colors.L.Mmag);
            if (picks.Mx) pushRow('L','Mx', valueAt(L.Mx,iL), colors.L.Mx);
            if (picks.My) pushRow('L','My', valueAt(L.My,iL), colors.L.My);
            if (picks.Mz) pushRow('L','Mz', valueAt(L.Mz,iL), colors.L.Mz);
        }
        if (sideMode !== 'L' && iR>=0){
            if (picks.Mmag) pushRow('R','|M|', valueAt(R.Mmag,iR), colors.R.Mmag);
            if (picks.Mx) pushRow('R','Mx', valueAt(R.Mx,iR), colors.R.Mx);
            if (picks.My) pushRow('R','My', valueAt(R.My,iR), colors.R.My);
            if (picks.Mz) pushRow('R','Mz', valueAt(R.Mz,iR), colors.R.Mz);
        }
        tooltip.html(rows.join('')).style('opacity',1).style('left', (ev.pageX + 12) + 'px').style('top', (ev.pageY + 12) + 'px');
    };
    const onLeave = () => { cross.style('display','none'); tooltip.style('opacity',0); };
    svg.on('mousemove', (ev)=> onMove(ev, null));
    svg.on('mouseleave', onLeave);

    // Stance shading (as small markers at y baseline to mimic prior)
    const drawStance = (times, stance, color) => {
        const points = [];
        for (let i=0;i<stance.length;i++) if (stance[i]) points.push(times[i]);
    gData.selectAll(null)
            .data(points)
            .join('rect')
            .attr('class','stance-mark')
            .attr('x', d=>x(d)-0.5)
            .attr('y', y(0)-2)
            .attr('width', 1)
            .attr('height', 4)
            .attr('fill', color);
    };
    if (sideMode !== 'R') drawStance(timeL, stanceL, 'rgba(0,0,255,0.25)');
    if (sideMode !== 'L') drawStance(timeR, stanceR, 'rgba(255,0,0,0.25)');

    // Lines
    if (sideMode !== 'R') {
        Object.keys(picks).forEach(c => push('L', c, timeL, L[c], colors.L[c]));
    }
    if (sideMode !== 'L') {
        Object.keys(picks).forEach(c => push('R', c, timeR, R[c], colors.R[c]));
    }
    const makeLine = (sx) => d3.line().x(pt=>sx(pt.t)).y(pt=>y(pt.y));
    const pathSel = gData.selectAll('path.series-path')
        .data(series)
        .join('path')
        .attr('class','series-path')
        .attr('fill','none')
        .attr('stroke', s=>s.color)
        .attr('stroke-width', 1.5)
        .attr('d', s => makeLine(x)(s.data));

    // remove old hover implementation (replaced below)

    // Zoom & pan with D3 zoom, re-render axes/grid/lines on transform
    const rescale = (transform) => {
        const zx = transform.rescaleX(x);
        const [d0, d1] = zx.domain();
        const s = computeSteps([d0, d1]);
        const gridValsZ = buildValues(d0, d1, s.gridStep, s.decimals);
        const labelValsZ = buildValues(d0, d1, s.labelStep, s.decimals);
        xAxis.call(d3.axisBottom(zx).tickValues(labelValsZ).tickFormat(d => s.decimals ? roundTo(d,1).toFixed(1) : Math.round(d)));
        gridGroup.selectAll('line.gridline')
            .data(gridValsZ, d=>d)
            .join(
                enter => enter.append('line').attr('class','gridline').attr('y1',0).attr('y2',innerH).attr('stroke','rgba(0,0,0,0.08)').attr('x1', d=>zx(d)).attr('x2', d=>zx(d)),
                update => update.attr('x1', d=>zx(d)).attr('x2', d=>zx(d)),
                exit => exit.remove()
            );
        pathSel.attr('d', s => makeLine(zx)(s.data));
        gData.selectAll('rect.stance-mark').attr('x', d=>zx(d)-0.5);
    // keep hover aligned after zoom
    svg.on('mousemove', (ev)=> onMove(ev, zx));
    svg.on('mouseleave', onLeave);
    };
    const zoom = d3.zoom()
        .scaleExtent([1, 40])
        .translateExtent([[0,0],[innerW, innerH]])
        .extent([[0,0],[innerW, innerH]])
        .on('zoom', (ev) => rescale(ev.transform));
    svg.call(zoom);
    document.getElementById('reset-zoom').onclick = () => {
        svg.transition().duration(200).call(zoom.transform, d3.zoomIdentity);
    };

    if (!timeChartListenersBound) {
        ['show-mx','show-my','show-mz','show-mmag'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', () => renderTimeCharts(results));
        });
        if (sideSel) sideSel.addEventListener('change', () => renderTimeCharts(results));
        timeChartListenersBound = true;
    }
}

// D3 cycle compare chart (mean ± SD, Left vs Right)
function renderCycleCharts(results) {
    const compSel = document.getElementById('cycle-component');
    const magModeSel = document.getElementById('mag-mode');
    const anchorSel = document.getElementById('cycle-anchor');
    const host = document.getElementById('cycle-compare-d3');
    if (!host) return;
    host.innerHTML = '';

    const width = host.clientWidth || 800;
    const height = host.clientHeight || 380;
    // Extra top margin to fit the stance/swing info rail
    const margin = { top: 64, right: 24, bottom: 40, left: 52 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(host).append('svg')
        .attr('width', width)
        .attr('height', height);
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const comp = compSel?.value || 'Mmag';
    const magMode = magModeSel?.value || 'mean_mag';
    const magWrap = document.getElementById('mag-mode-wrap');
    if (magWrap) magWrap.style.display = (comp === 'Mmag') ? 'inline-flex' : 'none';
    const anchorKey = anchorSel?.value || 'independent';
    const normalize = true;

    let left = { mean: [], sd: [] };
    let right = { mean: [], sd: [] };
    // Helper to derive alternative magnitude definitions from component means/SD if selected
    const deriveMagVariants = (Lsrc, Rsrc) => {
        // Build arrays from per-component means when available
        const Lmx = results?.cycles?.L?.Mx?.mean || [];
        const Lmy = results?.cycles?.L?.My?.mean || [];
        const Lmz = results?.cycles?.L?.Mz?.mean || [];
        const Rmx = results?.cycles?.R?.Mx?.mean || [];
        const Rmy = results?.cycles?.R?.My?.mean || [];
        const Rmz = results?.cycles?.R?.Mz?.mean || [];
        const n = Math.min(Lmx.length, Lmy.length, Lmz.length);
        const m = Math.min(Rmx.length, Rmy.length, Rmz.length);
        if (magMode === 'mag_of_mean') {
            const LmagOfMean = Array.from({length:n}, (_,i)=> Math.hypot(Lmx[i]||0, Lmy[i]||0, Lmz[i]||0));
            const RmagOfMean = Array.from({length:m}, (_,i)=> Math.hypot(Rmx[i]||0, Rmy[i]||0, Rmz[i]||0));
            Lsrc = { mean: LmagOfMean, sd: new Array(n).fill(0) };
            Rsrc = { mean: RmagOfMean, sd: new Array(m).fill(0) };
        } else if (magMode === 'rms_mag') {
            // RMS(|M|) ~= sqrt(E[Mx^2] + E[My^2] + E[Mz^2]). Without per-stride data, approximate via SD if provided:
            // E[X^2] = Var(X) + (E[X])^2. Use sd^2 + mean^2 per component.
            const Lmx_sd = results?.cycles?.L?.Mx?.sd || [];
            const Lmy_sd = results?.cycles?.L?.My?.sd || [];
            const Lmz_sd = results?.cycles?.L?.Mz?.sd || [];
            const Rmx_sd = results?.cycles?.R?.Mx?.sd || [];
            const Rmy_sd = results?.cycles?.R?.My?.sd || [];
            const Rmz_sd = results?.cycles?.R?.Mz?.sd || [];
            const n2 = Math.min(n, Lmx_sd.length, Lmy_sd.length, Lmz_sd.length);
            const m2 = Math.min(m, Rmx_sd.length, Rmy_sd.length, Rmz_sd.length);
            const Lrms = Array.from({length:n2}, (_,i)=> Math.sqrt(
                (Lmx_sd[i]||0)**2 + (Lmy_sd[i]||0)**2 + (Lmz_sd[i]||0)**2 +
                (Lmx[i]||0)**2   + (Lmy[i]||0)**2   + (Lmz[i]||0)**2
            ));
            const Rrms = Array.from({length:m2}, (_,i)=> Math.sqrt(
                (Rmx_sd[i]||0)**2 + (Rmy_sd[i]||0)**2 + (Rmz_sd[i]||0)**2 +
                (Rmx[i]||0)**2   + (Rmy[i]||0)**2   + (Rmz[i]||0)**2
            ));
            Lsrc = { mean: Lrms, sd: new Array(n2).fill(0) };
            Rsrc = { mean: Rrms, sd: new Array(m2).fill(0) };
        }
        return { left: Lsrc, right: Rsrc };
    };

    if (anchorKey === 'independent') {
        left = results?.cycles?.L?.[comp] || left;
        right = results?.cycles?.R?.[comp] || right;
        if (comp === 'Mmag' && magMode !== 'mean_mag') {
            const derived = deriveMagVariants(left, right);
            left = derived.left; right = derived.right;
        }
        // Phase-align Right to Left (independent mode) so the chart reflects the same alignment as metrics
        const Lm = Array.isArray(left?.mean) ? left.mean.slice() : [];
        const Rm = Array.isArray(right?.mean) ? right.mean.slice() : [];
        const Rs = Array.isArray(right?.sd) ? right.sd.slice() : [];
        const nAlign = Math.min(Lm.length, Rm.length);
        let deltaPhasePct = null;
        if (nAlign >= 3) {
            const circShift = (arr, k) => {
                const n = arr.length; if (!n) return [];
                const s = ((k % n) + n) % n;
                return arr.slice(s).concat(arr.slice(0, s));
            };
            let best = { k: 0, r: -Infinity };
            for (let k = 0; k < nAlign; k++) {
                const yk = circShift(Rm.slice(0, nAlign), k);
                const stats = regressAndCorr(Lm.slice(0, nAlign), yk);
                if (stats.r > best.r) best = { k, r: stats.r };
            }
            // Compute signed shortest shift in percent [-50, 50]
            let signedShift = best.k;
            if (signedShift > nAlign/2) signedShift = signedShift - nAlign;
            deltaPhasePct = (signedShift / nAlign) * 100;
            // Apply shift to Right mean and SD for plotting
            const RmAligned = circShift(Rm, best.k);
            const RsAligned = circShift(Rs, best.k);
            right = { mean: RmAligned, sd: RsAligned };
            // Stash for later label
            renderCycleCharts._deltaPhasePct = deltaPhasePct;
        } else {
            renderCycleCharts._deltaPhasePct = null;
        }
    } else {
        const cmp = results?.cycles_compare?.[anchorKey]?.[comp] || null;
        left = cmp?.L || left; right = cmp?.R || right;
        if (comp === 'Mmag' && magMode !== 'mean_mag') {
            // For anchored modes, approximate variants from component anchored means/SD if present
            const Lmx = results?.cycles_compare?.[anchorKey]?.Mx?.L?.mean || [];
            const Lmy = results?.cycles_compare?.[anchorKey]?.My?.L?.mean || [];
            const Lmz = results?.cycles_compare?.[anchorKey]?.Mz?.L?.mean || [];
            const Rmx = results?.cycles_compare?.[anchorKey]?.Mx?.R?.mean || [];
            const Rmy = results?.cycles_compare?.[anchorKey]?.My?.R?.mean || [];
            const Rmz = results?.cycles_compare?.[anchorKey]?.Mz?.R?.mean || [];
            const n = Math.min(Lmx.length, Lmy.length, Lmz.length);
            const m = Math.min(Rmx.length, Rmy.length, Rmz.length);
            if (magMode === 'mag_of_mean') {
                left = { mean: Array.from({length:n}, (_,i)=> Math.hypot(Lmx[i]||0, Lmy[i]||0, Lmz[i]||0)), sd: new Array(n).fill(0) };
                right = { mean: Array.from({length:m}, (_,i)=> Math.hypot(Rmx[i]||0, Rmy[i]||0, Rmz[i]||0)), sd: new Array(m).fill(0) };
            } else if (magMode === 'rms_mag') {
                const Lmx_sd = results?.cycles_compare?.[anchorKey]?.Mx?.L?.sd || [];
                const Lmy_sd = results?.cycles_compare?.[anchorKey]?.My?.L?.sd || [];
                const Lmz_sd = results?.cycles_compare?.[anchorKey]?.Mz?.L?.sd || [];
                const Rmx_sd = results?.cycles_compare?.[anchorKey]?.Mx?.R?.sd || [];
                const Rmy_sd = results?.cycles_compare?.[anchorKey]?.My?.R?.sd || [];
                const Rmz_sd = results?.cycles_compare?.[anchorKey]?.Mz?.R?.sd || [];
                const n2 = Math.min(n, Lmx_sd.length, Lmy_sd.length, Lmz_sd.length);
                const m2 = Math.min(m, Rmx_sd.length, Rmy_sd.length, Rmz_sd.length);
                left = { mean: Array.from({length:n2}, (_,i)=> Math.sqrt(
                    (Lmx_sd[i]||0)**2 + (Lmy_sd[i]||0)**2 + (Lmz_sd[i]||0)**2 +
                    (Lmx[i]||0)**2   + (Lmy[i]||0)**2   + (Lmz[i]||0)**2
                )), sd: new Array(n2).fill(0) };
                right = { mean: Array.from({length:m2}, (_,i)=> Math.sqrt(
                    (Rmx_sd[i]||0)**2 + (Rmy_sd[i]||0)**2 + (Rmz_sd[i]||0)**2 +
                    (Rmx[i]||0)**2   + (Rmy[i]||0)**2   + (Rmz[i]||0)**2
                )), sd: new Array(m2).fill(0) };
            }
        }
        renderCycleCharts._deltaPhasePct = null;
    }
    const n = Math.max(left.mean?.length || 0, right.mean?.length || 0);
    if (!n) return;
    const pct = i => (n>1 ? (i / (n-1)) * 100 : 0);

    const build = (S) => {
        const pts = [];
        const mean = S.mean || []; const sd = S.sd || [];
        for (let i=0;i<n;i++) {
            const m = mean[i] ?? 0; const d = sd[i] ?? 0;
            pts.push({ x: pct(i), y: m, y0: m - d, y1: m + d });
        }
        return pts;
    };
    const L = build(left);
    const R = build(right);

    const allY = [...L, ...R].flatMap(p => [p.y0, p.y1, 0]);
    const y = d3.scaleLinear().domain(d3.extent(allY)).nice().range([innerH, 0]);
    const x = d3.scaleLinear().domain([0, 100]).range([0, innerW]);

    // axes
    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(10));
    g.append('g').call(d3.axisLeft(y));
    svg.append('text').attr('x', margin.left + innerW/2).attr('y', height-6).attr('text-anchor','middle').text('Gait phase (%)');
    svg.append('text').attr('transform',`translate(14,${margin.top+innerH/2}) rotate(-90)`).attr('text-anchor','middle').text('Nm');

    // zero line
    g.append('line').attr('x1',0).attr('x2',innerW).attr('y1',y(0)).attr('y2',y(0)).attr('stroke','#aaa').attr('stroke-dasharray','3,3');

    // Top info rail: HS (0%), TO (toe-off), HS (100%) + Stance/Swing labels
    const numberish = (v) => (typeof v === 'number' && isFinite(v));
    const toFromKey = (key) => {
        const meta = results?.cycles_compare?.[key]?.meta;
        return numberish(meta?.to_percent) ? Math.max(0, Math.min(100, meta.to_percent)) : null;
    };
    let toPct = null;
    if (anchorKey === 'independent') {
        const cands = [toFromKey('anchor_L'), toFromKey('anchor_R')].filter(v => v!=null);
        if (cands.length) toPct = cands.reduce((a,b)=>a+b,0) / cands.length;
    } else {
        toPct = toFromKey(anchorKey);
    }
    
    // Validate toe-off percentage - should be between 25-70% for various walking speeds
    // If outside this range, use typical value but note it's estimated
    let isEstimated = false;
    if (!numberish(toPct) || toPct < 25 || toPct > 70) {
        toPct = 60; // typical adult walking toe-off
        isEstimated = true;
    }
    const rail = svg.append('g').attr('transform', `translate(${margin.left},${margin.top - 36})`);
    
    // Colored phase bars
    const phaseBarHeight = 8;
    const phaseY = 28;
    
    // Stance phase bar (light blue)
    rail.append('rect')
        .attr('x', x(0))
        .attr('y', phaseY)
        .attr('width', x(toPct) - x(0))
        .attr('height', phaseBarHeight)
        .attr('fill', '#87CEEB')
        .attr('stroke', '#5DADE2')
        .attr('stroke-width', 1);
    
    // Swing phase bar (light orange)
    rail.append('rect')
        .attr('x', x(toPct))
        .attr('y', phaseY)
        .attr('width', x(100) - x(toPct))
        .attr('height', phaseBarHeight)
        .attr('fill', '#FFB366')
        .attr('stroke', '#FF8C42')
        .attr('stroke-width', 1);
    
    // Key events with tick marks
    const events = [ 
        {p:0, l:'HS (IC)', desc:'Heel Strike (Initial Contact)'}, 
        {p:50, l:'Opp IC', desc:'Opposite Initial Contact'}, 
        {p:toPct, l:'TO', desc:'Toe Off'}, 
        {p:100, l:'HS', desc:'Heel Strike'} 
    ];
    
    // Event tick marks
    rail.selectAll('line.rail')
        .data(events)
        .join('line')
        .attr('class','rail')
        .attr('x1', d=>x(d.p)).attr('x2', d=>x(d.p))
        .attr('y1', 18).attr('y2', 26)
        .attr('stroke', d => d.p === 50 ? '#999' : '#888')
        .attr('stroke-width', d => d.p === 50 ? 1 : 2);
    
    // Event labels
    rail.selectAll('text.rail')
        .data(events)
        .join('text')
        .attr('class','rail')
        .attr('x', d=>x(d.p)).attr('y', 12)
        .attr('text-anchor','middle')
        .attr('font-size', d => d.p === 50 ? 10 : 11)
        .attr('fill', d => d.p === 50 ? '#777' : '#555')
        .text(d=>d.l);
    
    // Phase labels with percentage indicators
    const midStance = (0 + toPct) / 2;
    const midSwing = (toPct + 100) / 2;
    const stanceLabel = isEstimated ? `Stance (≈${Math.round(toPct)}%*)` : `Stance (≈${Math.round(toPct)}%)`;
    const swingLabel = isEstimated ? `Swing (≈${Math.round(100-toPct)}%*)` : `Swing (≈${Math.round(100-toPct)}%)`;
    
    rail.append('text')
        .attr('x', x(midStance)).attr('y', 0)
        .attr('text-anchor','middle').attr('font-size', 12).attr('fill','#444').attr('font-weight', 'bold')
        .text(stanceLabel);
    rail.append('text')
        .attr('x', x(midSwing)).attr('y', 0)
        .attr('text-anchor','middle').attr('font-size', 12).attr('fill','#444').attr('font-weight', 'bold')
        .text(swingLabel);
        
    // Add note about estimated values if needed
    if (isEstimated) {
        rail.append('text')
            .attr('x', x(95)).attr('y', 40)
            .attr('text-anchor','end').attr('font-size', 9).attr('fill','#777')
            .text('*Estimated - data outside normal range');
    }

    // series
    const colors = { L: '#1f77b4', R: '#d62728' };
    const area = d3.area().x(d=>x(d.x)).y0(d=>y(d.y0)).y1(d=>y(d.y1));
    const line = d3.line().x(d=>x(d.x)).y(d=>y(d.y));
    g.append('path').datum(L).attr('fill', colors.L).attr('opacity',0.14).attr('stroke','none').attr('d', area);
    g.append('path').datum(R).attr('fill', colors.R).attr('opacity',0.14).attr('stroke','none').attr('d', area);
    g.append('path').datum(L).attr('fill','none').attr('stroke', colors.L).attr('stroke-width', 2).attr('d', line);
    g.append('path').datum(R).attr('fill','none').attr('stroke', colors.R).attr('stroke-width', 2).attr('d', line);

    // simple legend
    const legend = svg.append('g').attr('transform', `translate(${margin.left},${margin.top-8})`);
    const items = [
        { name:'Left mean', color: colors.L },
        { name:'Right mean', color: colors.R },
    ];
    const li = legend.selectAll('g.item').data(items).join('g').attr('class','item').attr('transform',(d,i)=>`translate(${i*140},0)`);
    li.append('line').attr('x1',0).attr('x2',20).attr('y1',0).attr('y2',0).attr('stroke',d=>d.color).attr('stroke-width',3);
    li.append('text').attr('x',26).attr('y',4).text(d=>d.name).attr('font-size',12).attr('fill','#111');
    // Show Δphase for independent mode if available
    if (anchorKey === 'independent' && typeof renderCycleCharts._deltaPhasePct === 'number' && isFinite(renderCycleCharts._deltaPhasePct)) {
        svg.append('text')
            .attr('x', margin.left + 300)
            .attr('y', margin.top - 8)
            .attr('font-size', 12)
            .attr('fill', '#444')
            .text(`Δphase (R→L): ${renderCycleCharts._deltaPhasePct.toFixed(1)}%`);
    }

    // Show compare counts when using anchored mode
    if (anchorKey !== 'independent') {
        const cmp = results?.cycles_compare?.[anchorKey]?.[comp] || null;
        const used = cmp?.count_used ?? 0;
        const total = cmp?.count_total ?? 0;
        svg.append('text')
            .attr('x', margin.left)
            .attr('y', margin.top - 24)
            .attr('font-size', 12)
            .attr('fill', '#666')
            .text(`Segments used/total: ${used}/${total}`);
    }

    // hover
    let tooltip = d3.select('body').select('div.d3-tooltip');
    if (tooltip.empty()) tooltip = d3.select('body').append('div').attr('class','d3-tooltip').style('opacity',0);
    const cross = g.append('line').attr('y1',0).attr('y2',innerH).attr('stroke','rgba(0,0,0,0.25)').style('display','none');
    const mL = g.append('circle').attr('r',3).attr('fill', colors.L).attr('stroke','#fff').attr('stroke-width',1).style('display','none');
    const mR = g.append('circle').attr('r',3).attr('fill', colors.R).attr('stroke','#fff').attr('stroke-width',1).style('display','none');
    const onMove = (ev) => {
        const [mx] = d3.pointer(ev, g.node());
        const p = Math.max(0, Math.min(100, x.invert(mx)));
        const idx = Math.round((p/100) * (n-1));
        const lval = L[idx]?.y; const rval = R[idx]?.y;
        cross.attr('x1', mx).attr('x2', mx).style('display','block');
        if (lval!=null) { mL.attr('cx', x(pct(idx))).attr('cy', y(lval)).style('display','block'); }
        if (rval!=null) { mR.attr('cx', x(pct(idx))).attr('cy', y(rval)).style('display','block'); }
        const rows = [`<div><b>${Math.round(p)}%</b></div>`];
        if (lval!=null) rows.push(`<div><span style="display:inline-block;width:10px;height:10px;background:${colors.L};margin-right:6px"></span>L mean: ${lval.toFixed(1)}</div>`);
    if (rval!=null) rows.push(`<div><span style="display:inline-block;width:10px;height:10px;background:${colors.R};margin-right:6px"></span>R mean: ${rval.toFixed(1)}</div>`);
        tooltip.html(rows.join('')).style('opacity',1).style('left', (ev.pageX + 12) + 'px').style('top', (ev.pageY + 12) + 'px');
    };
    const onLeave = () => { cross.style('display','none'); mL.style('display','none'); mR.style('display','none'); tooltip.style('opacity',0); };
    g.append('rect').attr('x',0).attr('y',0).attr('width',innerW).attr('height',innerH).attr('fill','transparent')
        .on('mousemove', onMove).on('mouseleave', onLeave);

    // metrics below
    renderCycleCorrelation(results);
    if (compSel) compSel.onchange = () => renderCycleCharts(results);
    if (magModeSel) magModeSel.onchange = () => renderCycleCharts(results);
    if (anchorSel) anchorSel.onchange = () => renderCycleCharts(results);
}

// Compute L/R correlation with phase alignment for Independent anchor
function renderCycleCorrelation(results) {
    const host = document.getElementById('cycle-corr');
    if (!host) return;
    // Use Independent anchor means as-is; Right-side sign normalization is handled in the backend
    const compKeys = ['Mmag','Mx','My','Mz'];
    const labels = { Mmag: '|M|', Mx: 'Mx', My: 'My', Mz: 'Mz' };
    const rows = [];
    // Helper: circularly shift array by k (k>=0 shifts left); returns new array
    const circShift = (arr, k) => {
        const n = arr.length; if (!n) return [];
        const s = ((k % n) + n) % n;
        return arr.slice(s).concat(arr.slice(0, s));
    };
    // Find phase shift that maximizes Pearson r between x and y
    const bestPhaseAlign = (x, y) => {
        const n = Math.min(x.length, y.length);
        if (n < 3) return { shift: 0, yAligned: y.slice(0, n), r: 0 };
        let best = { shift: 0, r: -Infinity, yAligned: y.slice(0, n) };
        for (let k = 0; k < n; k++) {
            const yk = circShift(y.slice(0, n), k);
            const r = regressAndCorr(x.slice(0, n), yk).r;
            if (r > best.r) best = { shift: k, r, yAligned: yk };
        }
        // Convert shift to signed percent in range [-50, 50]
        let signedShift = best.shift;
        if (signedShift > n/2) signedShift = signedShift - n; // prefer shortest direction
        const phasePct = (signedShift / n) * 100;
        return { shift: signedShift, yAligned: best.yAligned, r: best.r, phasePct };
    };
    compKeys.forEach(k => {
        const L = results?.cycles?.L?.[k]?.mean || [];
        const R = results?.cycles?.R?.[k]?.mean || [];
        const n = Math.min(L.length, R.length);
        if (!n || n < 3) {
            rows.push(`<tr><td>${labels[k]}</td><td colspan="5">N/A</td></tr>`);
            return;
        }
        const x = L.slice(0, n);
        const y = R.slice(0, n);
        const aligned = bestPhaseAlign(x, y);
        const stats = regressAndCorr(x, aligned.yAligned);
        rows.push(`<tr><td>${labels[k]}</td><td>${stats.r.toFixed(3)}</td><td>${stats.r2.toFixed(3)}</td><td>${stats.slope.toFixed(3)}</td><td>${stats.intercept.toFixed(3)}</td><td>${aligned.phasePct.toFixed(1)}%</td></tr>`);
    });
    host.innerHTML = `
        <table class="metrics-table">
            <thead>
                <tr><th colspan="6">Cycle Similarity (Independent anchor; phase-aligned)</th></tr>
                <tr><th>Component</th><th>r</th><th>R²</th><th>Slope</th><th>Intercept</th><th>Δphase</th></tr>
            </thead>
            <tbody>${rows.join('')}</tbody>
        </table>
    `;
}

// Helper: Pearson correlation and OLS slope/intercept
function regressAndCorr(x, y) {
    const n = Math.min(x.length, y.length);
    let sx=0, sy=0, sxx=0, syy=0, sxy=0;
    for (let i=0;i<n;i++) {
        const xi = x[i];
        const yi = y[i];
        sx += xi; sy += yi; sxx += xi*xi; syy += yi*yi; sxy += xi*yi;
    }
    const mx = sx / n; const my = sy / n;
    const cov = sxy / n - mx*my;
    const vx = sxx / n - mx*mx;
    const vy = syy / n - my*my;
    const r = (vx>0 && vy>0) ? (cov / Math.sqrt(vx*vy)) : 0;
    const slope = (vx>0) ? (cov / vx) : 0;
    const intercept = my - slope * mx;
    const r2 = r*r;
    return { r, r2, slope, intercept };
}

function renderMetrics(results) {
    const el = document.getElementById('metrics');
    const peakAbs = arr => (Array.isArray(arr) && arr.length) ? Math.max(...arr.map(v => Math.abs(v))) : 0;
    const symIdx = (L, R) => (L>0 && R>0) ? (100 * (1 - Math.abs(L - R) / ((L + R)/2))) : NaN;
    const fmt = (v, d=1) => (typeof v === 'number' && isFinite(v)) ? v.toFixed(d) : 'N/A';
    const fmtPct = v => (typeof v === 'number' && isFinite(v)) ? v.toFixed(1)+'%' : 'N/A';
    const fmtVec = v => (Array.isArray(v) ? `[${v.map(x => fmt(x,1)).join(', ')}]` : 'N/A');

    // Component arrays
    const comps = [
        { key: 'Mmag', label: '|M|', L: results.L_Mmag, R: results.R_Mmag },
        { key: 'Mx', label: 'Mx', L: results.L_mx, R: results.R_mx },
        { key: 'My', label: 'My', L: results.L_my, R: results.R_my },
        { key: 'Mz', label: 'Mz', L: results.L_mz, R: results.R_mz },
    ];
    const peaks = comps.map(c => {
        const Lp = peakAbs(c.L||[]);
        const Rp = peakAbs(c.R||[]);
        return { label: c.label, Lp, Rp, sym: symIdx(Lp, Rp) };
    });

    // Cycles and normalization
    const usedL = results.cycles?.L?.count_used ?? null;
    const totalL = results.cycles?.L?.count_total ?? null;
    const usedR = results.cycles?.R?.count_used ?? null;
    const totalR = results.cycles?.R?.count_total ?? null;
    const normTime = true; // normalization locked on
    const normCyc = true;  // normalization locked on

    // Baseline and toe-off
    const baseline = results?.meta?.baseline_mode || 'linear';
    const toL = results?.cycles_compare?.anchor_L?.meta?.to_percent;
    const toR = results?.cycles_compare?.anchor_R?.meta?.to_percent;
    const bL0 = results?.baseline_JCS?.L?.start || null;
    const bL1 = results?.baseline_JCS?.L?.end || null;
    const bR0 = results?.baseline_JCS?.R?.start || null;
    const bR1 = results?.baseline_JCS?.R?.end || null;

    // Calibration windows
    const cws = Array.isArray(results.cal_windows) ? results.cal_windows : [];
    const cwCount = cws.length;
    const cwDur = cws.reduce((s,w)=> s + Math.max(0, (w.end_s||0)-(w.start_s||0)), 0);

    // Build HTML tables
    const overview = `
        <table class="metrics-table">
            <thead><tr><th colspan="2">Overview</th></tr></thead>
            <tbody>
                <tr><td>Baseline Mode</td><td>${baseline}</td></tr>
                <tr><td>Toe-off % (Left-ref)</td><td>${fmtPct(toL)}</td></tr>
                <tr><td>Toe-off % (Right-ref)</td><td>${fmtPct(toR)}</td></tr>
                <tr><td>Normalized Signs</td><td>time=on (locked), cycles=on (locked)</td></tr>
                <tr><td>Calibration Windows</td><td>${cwCount} (total ${fmt(cwDur,2)} s)</td></tr>
            </tbody>
        </table>
    `;

    const counts = `
        <table class="metrics-table">
            <thead><tr><th colspan="2">Cycle Counts</th></tr></thead>
            <tbody>
                <tr><td>Left Cycles Used / Total</td><td>${usedL??'N/A'} / ${totalL??'N/A'}</td></tr>
                <tr><td>Right Cycles Used / Total</td><td>${usedR??'N/A'} / ${totalR??'N/A'}</td></tr>
            </tbody>
        </table>
    `;

    const peaksTblRows = peaks.map(p => (
        `<tr><td>${p.label}</td><td>${fmt(p.Lp)}</td><td>${fmt(p.Rp)}</td><td>${fmt(p.sym,1)}%</td></tr>`
    )).join('');
    const peaksTbl = `
        <table class="metrics-table">
            <thead><tr><th colspan="4">Peaks (abs)</th></tr>
            <tr><th>Component</th><th>Left (Nm)</th><th>Right (Nm)</th><th>Symmetry (%)</th></tr></thead>
            <tbody>${peaksTblRows}</tbody>
        </table>
    `;

    const baseTbl = `
        <table class="metrics-table">
            <thead><tr><th colspan="3">Baseline (JCS, per side)</th></tr>
            <tr><th>Side</th><th>Start [Mx, My, Mz] (Nm)</th><th>End [Mx, My, Mz] (Nm)</th></tr></thead>
            <tbody>
                <tr><td>Left</td><td>${fmtVec(bL0)}</td><td>${fmtVec(bL1)}</td></tr>
                <tr><td>Right</td><td>${fmtVec(bR0)}</td><td>${fmtVec(bR1)}</td></tr>
            </tbody>
        </table>
    `;

    el.innerHTML = `${overview}${counts}${peaksTbl}${baseTbl}`;
}

// Angles time-series overlay (hip/knee; flex/add/rot)
let anglesTimeListenersBound = false;
function renderAnglesTimeCharts(results) {
    const host = document.getElementById('angles-time-chart');
    if (!host) return;
    host.innerHTML = '';
    const jointSel = document.getElementById('angle-time-joint');
    const sideSel = document.getElementById('time-angle-side');
    const joint = jointSel?.value || 'hip';
    const timeL = results.time_L || [];
    const timeR = results.time_R || [];
    const stanceL = results.stance_L || [];
    const stanceR = results.stance_R || [];
    const Lsrc = joint === 'hip' ? (results.L_hip_angles_deg || []) : (results.L_knee_angles_deg || []);
    const Rsrc = joint === 'hip' ? (results.R_hip_angles_deg || []) : (results.R_knee_angles_deg || []);
    const picks = {
        flex: document.getElementById('show-ang-flex')?.checked ?? true,
        add: document.getElementById('show-ang-add')?.checked ?? true,
        rot: document.getElementById('show-ang-rot')?.checked ?? true,
    };
    const compIndex = { flex: 0, add: 1, rot: 2 };
    const compColors = { flex: '#1f77b4', add: '#2ca02c', rot: '#ff7f0e' };
    const sideMode = sideSel?.value || 'both';

    const width = host.clientWidth || 800;
    const height = host.clientHeight || 380;
    const margin = { top: 20, right: 20, bottom: 30, left: 45 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(host).append('svg').attr('width', width).attr('height', height);
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    const clipId = 'clip-ang-ts';
    svg.append('defs').append('clipPath').attr('id', clipId)
        .append('rect').attr('x', 0).attr('y', 0).attr('width', innerW).attr('height', innerH);
    const gData = g.append('g').attr('clip-path', `url(#${clipId})`);

    // Build series: [{side, comp, data:[{t,y}], color}]
    const series = [];
    const pushSeries = (side, comp, times, arr2d, color) => {
        if (!picks[comp]) return;
        if (!(sideMode === 'both' || sideMode === side)) return;
        const j = compIndex[comp];
        const data = [];
        for (let i=0;i<times.length;i++) {
            const row = arr2d?.[i];
            if (!row || row.length <= j) continue;
            data.push({ t: times[i], y: row[j] });
        }
        series.push({ side, comp, color, data });
    };
    const colorsL = compColors;
    const colorsR = { flex: 'rgba(31,119,180,0.7)', add: 'rgba(44,160,44,0.7)', rot: 'rgba(255,127,14,0.7)' };
    if (sideMode !== 'R') {
        Object.keys(picks).forEach(c => pushSeries('L', c, timeL, Lsrc, colorsL[c]));
    }
    if (sideMode !== 'L') {
        Object.keys(picks).forEach(c => pushSeries('R', c, timeR, Rsrc, colorsR[c]));
    }

    // Scales
    const allTimes = [];
    const allVals = [];
    series.forEach(s => { s.data.forEach(d => { allTimes.push(d.t); allVals.push(d.y); }); });
    const tMin = Math.floor(d3.min(allTimes) ?? 0);
    const tMax = Math.ceil(d3.max(allTimes) ?? 1);
    const vMin = Math.min(0, d3.min(allVals) ?? -50);
    const vMax = Math.max(0, d3.max(allVals) ?? 50);
    const x = d3.scaleLinear().domain([tMin, tMax]).range([0, innerW]);
    const y = d3.scaleLinear().domain([vMin, vMax]).nice().range([innerH, 0]);

    // Axes and grid
    const roundTo = (v, d) => { const f = Math.pow(10, d); return Math.round(v * f) / f; };
    const computeSteps = (domain) => {
        const range = Math.max(0.0001, domain[1]-domain[0]);
        if (range <= 3.5) return { gridStep: 0.1, labelStep: 0.1, decimals: 1 };
        if (range <= 60) return { gridStep: 1, labelStep: 1, decimals: 0 };
        return { gridStep: 1, labelStep: 5, decimals: 0 };
    };
    const buildValues = (d0, d1, step, decimals) => {
        const start = roundTo(Math.ceil(d0/step)*step, decimals);
        const end = roundTo(Math.floor(d1/step)*step, decimals);
        const vals = [];
        for (let v = start; v <= end + 1e-8; v = roundTo(v + step, decimals)) vals.push(roundTo(v, decimals));
        return vals;
    };
    const steps = computeSteps([tMin, tMax]);
    const xAxis = g.append('g').attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(x).tickValues(buildValues(tMin, tMax, steps.labelStep, steps.decimals)).tickFormat(d => steps.decimals ? d.toFixed(1) : d));
    const gridGroup = g.append('g').attr('class','grid');
    gridGroup.selectAll('line.gridline')
        .data(buildValues(tMin, tMax, steps.gridStep, steps.decimals), d=>d)
        .join('line')
        .attr('class','gridline')
        .attr('x1', d=>x(d)).attr('x2', d=>x(d))
        .attr('y1', 0).attr('y2', innerH)
        .attr('stroke', 'rgba(0,0,0,0.08)');
    g.append('g').call(d3.axisLeft(y));
    svg.append('text').attr('x', margin.left + innerW/2).attr('y', height-5).attr('text-anchor','middle').text('Time (s)');
    svg.append('text').attr('transform',`translate(12,${margin.top+innerH/2}) rotate(-90)`).attr('text-anchor','middle').text('Angle (deg)');

    // Stance marks
    const drawStance = (times, stance, color) => {
        const points = [];
        for (let i=0;i<stance.length;i++) if (stance[i]) points.push(times[i]);
        gData.selectAll(null)
            .data(points)
            .join('rect')
            .attr('class','stance-mark')
            .attr('x', d=>x(d)-0.5)
            .attr('y', y(0)-2)
            .attr('width', 1)
            .attr('height', 4)
            .attr('fill', color);
    };
    if (sideMode !== 'R') drawStance(timeL, stanceL, 'rgba(0,0,255,0.25)');
    if (sideMode !== 'L') drawStance(timeR, stanceR, 'rgba(255,0,0,0.25)');

    // Lines
    const makeLine = (sx) => d3.line().x(pt=>sx(pt.t)).y(pt=>y(pt.y));
    const pathSel = gData.selectAll('path.ang-series-path')
        .data(series)
        .join('path')
        .attr('class','ang-series-path')
        .attr('fill','none')
        .attr('stroke', s=>s.color)
        .attr('stroke-width', 1.5)
        .attr('d', s => makeLine(x)(s.data));

    // Hover tooltip
    let tooltip = d3.select('body').select('div.d3-tooltip');
    if (tooltip.empty()) tooltip = d3.select('body').append('div').attr('class','d3-tooltip').style('opacity',0);
    const cross = g.append('line').attr('class','crosshair-x').attr('y1',0).attr('y2',innerH).attr('stroke','rgba(0,0,0,0.25)').style('display','none');
    const bisect = d3.bisector(d=>d).left;
    const valueAt = (mat, idx, j) => {
        const row = Array.isArray(mat) ? mat[idx] : null; return (row && row.length>j) ? row[j] : undefined;
    };
    const onMove = (ev, zx) => {
        const [mx] = d3.pointer(ev, g.node());
        const xScale = zx || x;
        const t = xScale.invert(mx);
        cross.attr('x1', mx).attr('x2', mx).style('display','block');
        const sLocal = computeSteps(xScale.domain());
        const iL = timeL.length ? Math.max(0, Math.min(timeL.length-1, bisect(timeL, t))) : -1;
        const iR = timeR.length ? Math.max(0, Math.min(timeR.length-1, bisect(timeR, t))) : -1;
        const rows = [];
        rows.push(`<div><b>t=${sLocal.decimals?roundTo(t,1).toFixed(1):Math.round(t)}s (${joint})</b></div>`);
        const pushRow = (side,label,val,color)=>{ if(val!==undefined){ rows.push(`<div><span style=\"display:inline-block;width:10px;height:10px;background:${color};margin-right:6px\"></span>${side} ${label}: ${val.toFixed(1)}°</div>`);} };
        if (sideMode !== 'R' && iL>=0){
            if (picks.flex) pushRow('L','flex', valueAt(Lsrc, iL, 0), colorsL.flex);
            if (picks.add)  pushRow('L','add',  valueAt(Lsrc, iL, 1), colorsL.add);
            if (picks.rot)  pushRow('L','rot',  valueAt(Lsrc, iL, 2), colorsL.rot);
        }
        if (sideMode !== 'L' && iR>=0){
            if (picks.flex) pushRow('R','flex', valueAt(Rsrc, iR, 0), colorsR.flex);
            if (picks.add)  pushRow('R','add',  valueAt(Rsrc, iR, 1), colorsR.add);
            if (picks.rot)  pushRow('R','rot',  valueAt(Rsrc, iR, 2), colorsR.rot);
        }
        tooltip.html(rows.join('')).style('opacity',1).style('left', (ev.pageX + 12) + 'px').style('top', (ev.pageY + 12) + 'px');
    };
    const onLeave = () => { cross.style('display','none'); tooltip.style('opacity',0); };
    svg.on('mousemove', (ev)=> onMove(ev, null));
    svg.on('mouseleave', onLeave);

    // Zoom
    const rescale = (transform) => {
        const zx = transform.rescaleX(x);
        const [d0, d1] = zx.domain();
        const s = computeSteps([d0, d1]);
        xAxis.call(d3.axisBottom(zx).tickValues(buildValues(d0, d1, s.labelStep, s.decimals)).tickFormat(d => s.decimals ? roundTo(d,1).toFixed(1) : Math.round(d)));
        gridGroup.selectAll('line.gridline')
            .data(buildValues(d0, d1, s.gridStep, s.decimals), d=>d)
            .join(
                enter => enter.append('line').attr('class','gridline').attr('y1',0).attr('y2',innerH).attr('stroke','rgba(0,0,0,0.08)').attr('x1', d=>zx(d)).attr('x2', d=>zx(d)),
                update => update.attr('x1', d=>zx(d)).attr('x2', d=>zx(d)),
                exit => exit.remove()
            );
        gData.selectAll('path.ang-series-path').attr('d', s => d3.line().x(pt=>zx(pt.t)).y(pt=>y(pt.y))(s.data));
        gData.selectAll('rect.stance-mark').attr('x', d=>zx(d)-0.5);
        svg.on('mousemove', (ev)=> onMove(ev, zx));
        svg.on('mouseleave', onLeave);
    };
    const zoom = d3.zoom().scaleExtent([1, 40]).translateExtent([[0,0],[innerW, innerH]]).extent([[0,0],[innerW, innerH]]).on('zoom', (ev)=> rescale(ev.transform));
    svg.call(zoom);
    const resetBtn = document.getElementById('reset-angles-zoom');
    if (resetBtn) resetBtn.onclick = () => svg.transition().duration(200).call(zoom.transform, d3.zoomIdentity);

    if (!anglesTimeListenersBound) {
        ['show-ang-flex','show-ang-add','show-ang-rot','angle-time-joint','time-angle-side'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', () => renderAnglesTimeCharts(results));
        });
        anglesTimeListenersBound = true;
    }
}

// Angles cycle chart (mean ± SD, Left vs Right)
let anglesCycleListenersBound = false;
function renderAnglesCycleCharts(results) {
    const host = document.getElementById('angles-cycle-chart');
    if (!host) return;
    host.innerHTML = '';
    const jointSel = document.getElementById('angle-cycle-joint');
    const compSel = document.getElementById('angle-cycle-component');
    const joint = jointSel?.value || 'hip';
    const comp = compSel?.value || 'flex';
    const cycles = results?.angle_cycles?.[joint];
    if (!cycles) return;
    const L = cycles?.L?.[comp] || { mean: [], sd: [] };
    const R = cycles?.R?.[comp] || { mean: [], sd: [] };
    const n = Math.max(L.mean?.length || 0, R.mean?.length || 0);
    if (!n) return;
    const pct = i => (n>1 ? (i / (n-1)) * 100 : 0);

    const build = (S) => {
        const pts = [];
        const mean = S.mean || []; const sd = S.sd || [];
        for (let i=0;i<n;i++) {
            const m = mean[i] ?? 0; const d = sd[i] ?? 0;
            pts.push({ x: pct(i), y: m, y0: m - d, y1: m + d });
        }
        return pts;
    };
    const dL = build(L);
    const dR = build(R);

    const width = host.clientWidth || 800;
    const height = host.clientHeight || 380;
    const margin = { top: 24, right: 24, bottom: 40, left: 52 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const svg = d3.select(host).append('svg').attr('width', width).attr('height', height);
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const allY = [...dL, ...dR].flatMap(p => [p.y0, p.y1, 0]);
    const y = d3.scaleLinear().domain(d3.extent(allY)).nice().range([innerH, 0]);
    const x = d3.scaleLinear().domain([0, 100]).range([0, innerW]);
    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(10));
    g.append('g').call(d3.axisLeft(y));
    svg.append('text').attr('x', margin.left + innerW/2).attr('y', height-6).attr('text-anchor','middle').text('Gait phase (%)');
    svg.append('text').attr('transform',`translate(14,${margin.top+innerH/2}) rotate(-90)`).attr('text-anchor','middle').text('deg');
    g.append('line').attr('x1',0).attr('x2',innerW).attr('y1',y(0)).attr('y2',y(0)).attr('stroke','#aaa').attr('stroke-dasharray','3,3');

    const colors = { L: '#1f77b4', R: '#d62728' };
    const area = d3.area().x(d=>x(d.x)).y0(d=>y(d.y0)).y1(d=>y(d.y1));
    const line = d3.line().x(d=>x(d.x)).y(d=>y(d.y));
    g.append('path').datum(dL).attr('fill', colors.L).attr('opacity',0.14).attr('stroke','none').attr('d', area);
    g.append('path').datum(dR).attr('fill', colors.R).attr('opacity',0.14).attr('stroke','none').attr('d', area);
    g.append('path').datum(dL).attr('fill','none').attr('stroke', colors.L).attr('stroke-width', 2).attr('d', line);
    g.append('path').datum(dR).attr('fill','none').attr('stroke', colors.R).attr('stroke-width', 2).attr('d', line);

    // Hover
    let tooltip = d3.select('body').select('div.d3-tooltip');
    if (tooltip.empty()) tooltip = d3.select('body').append('div').attr('class','d3-tooltip').style('opacity',0);
    const cross = g.append('line').attr('y1',0).attr('y2',innerH).attr('stroke','rgba(0,0,0,0.25)').style('display','none');
    const mL = g.append('circle').attr('r',3).attr('fill', colors.L).attr('stroke','#fff').attr('stroke-width',1).style('display','none');
    const mR = g.append('circle').attr('r',3).attr('fill', colors.R).attr('stroke','#fff').attr('stroke-width',1).style('display','none');
    const onMove = (ev) => {
        const [mx] = d3.pointer(ev, g.node());
        const p = Math.max(0, Math.min(100, x.invert(mx)));
        const idx = Math.round((p/100) * (n-1));
        const lval = dL[idx]?.y; const rval = dR[idx]?.y;
        cross.attr('x1', mx).attr('x2', mx).style('display','block');
        if (lval!=null) { mL.attr('cx', x(pct(idx))).attr('cy', y(lval)).style('display','block'); }
        if (rval!=null) { mR.attr('cx', x(pct(idx))).attr('cy', y(rval)).style('display','block'); }
        const rows = [`<div><b>${Math.round(p)}% (${joint} ${comp})</b></div>`];
        if (lval!=null) rows.push(`<div><span style="display:inline-block;width:10px;height:10px;background:${colors.L};margin-right:6px"></span>L mean: ${lval.toFixed(1)}°</div>`);
        if (rval!=null) rows.push(`<div><span style="display:inline-block;width:10px;height:10px;background:${colors.R};margin-right:6px"></span>R mean: ${rval.toFixed(1)}°</div>`);
        tooltip.html(rows.join('')).style('opacity',1).style('left', (ev.pageX + 12) + 'px').style('top', (ev.pageY + 12) + 'px');
    };
    const onLeave = () => { cross.style('display','none'); mL.style('display','none'); mR.style('display','none'); tooltip.style('opacity',0); };
    g.append('rect').attr('x',0).attr('y',0).attr('width',innerW).attr('height',innerH).attr('fill','transparent')
        .on('mousemove', onMove).on('mouseleave', onLeave);

    if (!anglesCycleListenersBound) {
        ['angle-cycle-joint','angle-cycle-component'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', () => renderAnglesCycleCharts(results));
        });
        anglesCycleListenersBound = true;
    }
}

// Convenience: Run with sample data by omitting files
document.getElementById('run-sample').addEventListener('click', async function() {
    const form = document.getElementById('upload-form');
    const fd = new FormData();
    const h = document.getElementById('height').value || '1.75';
    const m = document.getElementById('mass').value || '100';
    fd.append('height_m', h);
    fd.append('mass_kg', m);
    const baseline = document.getElementById('baseline-mode')?.value || 'linear';
    fd.append('baseline_mode', baseline);

    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error');
    loadingDiv.classList.remove('hidden');
    resultsDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');

    try {
        const response = await fetch('/api/analyze/', { method: 'POST', body: fd });
        if (!response.ok) throw new Error('Analysis failed');
        const results = await response.json();
        displayResults(results);
    } catch (e) {
        errorDiv.classList.remove('hidden');
    } finally {
        loadingDiv.classList.add('hidden');
    }
});

// Ensure defaults present on load
document.addEventListener('DOMContentLoaded', () => {
    const h = document.getElementById('height');
    const m = document.getElementById('mass');
    if (!h.value) h.value = '1.75';
    if (!m.value) m.value = '100';
});
