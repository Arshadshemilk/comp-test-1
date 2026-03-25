/* Red-Team Tool — Frontend Logic */

// ===== STATE =====
const state = {
    currentTab: 'lab',
    modelLoaded: false,
    modelInfo: null,
    apiKeySet: false,
    taxonomy: { attackTypes: [], riskCategories: [], riskSubcategories: [] },
    lastRun: null,
    lastValidation: null,
    breaks: [],
    history: [],
    _optimizedPrompt: null,
    _lastTranslation: null,
};

// ===== INIT =====
document.addEventListener('DOMContentLoaded', init);

async function init() {
    setupTabs();
    setupEventListeners();
    await Promise.all([
        loadTaxonomy(),
        checkModelStatus(),
        checkApiKeyStatus(),
        loadBreaks(),
        loadHistory(),
        populateModelSelector(),
        loadOpenAIModel(),
    ]);
}

// ===== TAB NAVIGATION =====
function setupTabs() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
}

function switchTab(tab) {
    state.currentTab = tab;
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`.nav-btn[data-tab="${tab}"]`).classList.add('active');
    document.querySelectorAll('.tab-content').forEach(s => s.classList.add('hidden'));
    document.getElementById(`tab-${tab}`).classList.remove('hidden');
    if (tab === 'breaks') loadBreaks();
    if (tab === 'lab') loadHistory();
}

// ===== EVENT LISTENERS =====
function setupEventListeners() {
    // Model bar
    $('btn-load-model').addEventListener('click', () => loadModel('sel-model'));
    // Settings
    $('btn-save-apikey').addEventListener('click', saveApiKey);
    $('btn-save-openai-model').addEventListener('click', saveOpenAIModel);
    $('btn-load-model-settings').addEventListener('click', () => loadModel('sel-model-settings'));
    // Sync model selectors
    $('sel-model').addEventListener('change', () => { $('sel-model-settings').value = $('sel-model').value; });
    $('sel-model-settings').addEventListener('change', () => { $('sel-model').value = $('sel-model-settings').value; });
    // Prompt Lab
    $('btn-run').addEventListener('click', runPrompt);
    $('btn-validate').addEventListener('click', validatePrompt);
    $('btn-optimize').addEventListener('click', optimizePrompt);
    $('btn-translate').addEventListener('click', translatePrompt);
    $('btn-translate-to-local').addEventListener('click', translateToLocal);
    $('btn-save-break').addEventListener('click', openSaveModal);
    // Cascading dropdown
    $('sel-risk-category').addEventListener('change', updateSubcategories);
    // Modal
    $('btn-confirm-save').addEventListener('click', confirmSaveBreak);
    $('btn-cancel-save').addEventListener('click', closeSaveModal);
    // Export
    $('btn-preview-export').addEventListener('click', loadExportPreview);
    $('btn-download-export').addEventListener('click', downloadExport);
}

// ===== API HELPER =====
async function api(path, opts = {}) {
    const res = await fetch(path, {
        headers: { 'Content-Type': 'application/json', ...opts.headers },
        ...opts,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
    return data;
}

function $(id) { return document.getElementById(id); }
function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text || '';
    return d.innerHTML;
}
function renderMd(text) {
    if (typeof marked !== 'undefined') {
        try { return marked.parse(text || ''); } catch (_) {}
    }
    return `<pre class="response-text">${escapeHtml(text)}</pre>`;
}

// ===== TOAST =====
function showToast(msg, type = 'success') {
    const t = document.createElement('div');
    t.className = `toast toast-${type}`;
    t.textContent = msg;
    document.body.appendChild(t);
    requestAnimationFrame(() => t.classList.add('show'));
    setTimeout(() => {
        t.classList.remove('show');
        setTimeout(() => t.remove(), 300);
    }, 3000);
}

// ===== SETTINGS — API KEY =====
async function saveApiKey() {
    const key = $('input-apikey').value.trim();
    if (!key) return showToast('Enter an API key', 'error');
    try {
        await api('/api/settings/apikey', { method: 'POST', body: JSON.stringify({ api_key: key }) });
        state.apiKeySet = true;
        localStorage.setItem('_rtk', key);
        updateApiIndicator(true);
        showToast('API key saved');
    } catch (e) { showToast(e.message, 'error'); }
}

async function checkApiKeyStatus() {
    try {
        const d = await api('/api/settings/apikey/status');
        state.apiKeySet = d.is_set;
        updateApiIndicator(d.is_set);
        if (!d.is_set) {
            const saved = localStorage.getItem('_rtk');
            if (saved) {
                await api('/api/settings/apikey', { method: 'POST', body: JSON.stringify({ api_key: saved }) });
                state.apiKeySet = true;
                updateApiIndicator(true);
            }
        }
    } catch (e) { /* ignore */ }
}

function updateApiIndicator(ok) {
    const el = $('api-status');
    el.querySelector('.status-dot').className = `status-dot ${ok ? 'online' : 'offline'}`;
    el.querySelector('span:last-child').textContent = ok ? 'API key set' : 'No API key';
}

// ===== SETTINGS — OPENAI MODEL =====
async function saveOpenAIModel() {
    const model = $('sel-openai-model').value;
    try {
        await api('/api/settings/openai-model', { method: 'POST', body: JSON.stringify({ model }) });
        $('openai-model-status').textContent = `Current: ${model}`;
        showToast(`OpenAI model set to ${model}`);
    } catch (e) { showToast(e.message, 'error'); }
}

async function loadOpenAIModel() {
    try {
        const d = await api('/api/settings/openai-model');
        $('sel-openai-model').value = d.model;
        $('openai-model-status').textContent = `Current: ${d.model}`;
    } catch (e) { /* ignore */ }
}

// ===== SETTINGS — MODEL =====
async function populateModelSelector() {
    try {
        const models = await api('/api/models');
        const selectors = [$('sel-model'), $('sel-model-settings')];
        selectors.forEach(sel => {
            sel.innerHTML = '<option value="">Select model...</option>';
            for (const [k, v] of Object.entries(models)) {
                const opt = document.createElement('option');
                opt.value = k;
                opt.textContent = `${v.name} (${v.language})`;
                sel.appendChild(opt);
            }
        });
    } catch (e) { /* ignore */ }
}

async function loadModel(selId) {
    const key = $(selId).value;
    if (!key) return showToast('Select a model first', 'error');
    // Disable both load buttons
    const btns = [$('btn-load-model'), $('btn-load-model-settings')];
    btns.forEach(b => { b.disabled = true; b.textContent = 'Loading...'; });
    try {
        const d = await api('/api/models/load', { method: 'POST', body: JSON.stringify({ model_key: key }) });
        state.modelLoaded = true;
        state.modelInfo = d;
        updateModelIndicator(d);
        $('model-info').innerHTML = `<span class="text-green">&#10003;</span> ${d.model} &mdash; ${d.vram_gb} GB VRAM`;
        showToast(`Model loaded: ${d.model}`);
    } catch (e) { showToast(e.message, 'error'); }
    finally { btns.forEach(b => { b.disabled = false; b.textContent = 'Load'; }); $('btn-load-model-settings').textContent = 'Load Model'; }
}

async function checkModelStatus() {
    try {
        const d = await api('/api/models/status');
        state.modelLoaded = d.loaded;
        if (d.loaded) {
            state.modelInfo = d;
            updateModelIndicator(d);
            const info = $('model-info');
            if (info) info.innerHTML = `<span class="text-green">&#10003;</span> ${d.model_name} &mdash; ${d.vram_gb} GB VRAM`;
        }
    } catch (e) { /* ignore */ }
}

function updateModelIndicator(d) {
    const dot = $('model-bar-dot');
    const label = $('model-bar-label');
    const detail = $('model-bar-detail');
    if (d && d.loaded !== false) {
        dot.classList.add('active');
        label.classList.add('active');
        label.textContent = d.model_name || d.model || 'Model loaded';
        detail.textContent = `${d.language || ''} \u2022 ${d.vram_gb || '?'} GB VRAM`;
    } else {
        dot.classList.remove('active');
        label.classList.remove('active');
        label.textContent = 'No model loaded';
        detail.textContent = '';
    }
}

// ===== TAXONOMY =====
async function loadTaxonomy() {
    try {
        const [at, rc, rsc] = await Promise.all([
            api('/api/taxonomy/attack-types'),
            api('/api/taxonomy/risk-categories'),
            api('/api/taxonomy/risk-subcategories'),
        ]);
        state.taxonomy = { attackTypes: at, riskCategories: rc, riskSubcategories: rsc };
        populateDropdowns();
    } catch (e) { console.error('Taxonomy load failed', e); }
}

function populateDropdowns() {
    const atSel = $('sel-attack-type');
    const rcSel = $('sel-risk-category');
    atSel.innerHTML = '<option value="">Attack type...</option>';
    state.taxonomy.attackTypes.forEach(t => {
        const o = document.createElement('option');
        o.value = t['Attack Type'];
        o.textContent = `${t['Attack Type']}`;
        atSel.appendChild(o);
    });
    rcSel.innerHTML = '<option value="">Risk category...</option>';
    state.taxonomy.riskCategories.forEach(c => {
        const o = document.createElement('option');
        o.value = c['Risk Category'];
        o.textContent = `${c['Risk Category']}`;
        rcSel.appendChild(o);
    });
    updateSubcategories();
}

function updateSubcategories() {
    const cat = $('sel-risk-category').value;
    const sub = $('sel-risk-subcategory');
    sub.innerHTML = '<option value="">Subcategory...</option>';
    const list = cat
        ? state.taxonomy.riskSubcategories.filter(s => s['Risk Category'] === cat)
        : state.taxonomy.riskSubcategories;
    list.forEach(s => {
        const o = document.createElement('option');
        o.value = s['Risk Subcategory'];
        o.textContent = s['Risk Subcategory'];
        sub.appendChild(o);
    });
}

// ===== PROMPT LAB — GENERATION =====
function getGenParams() {
    return {
        max_new_tokens: 2048,
        temperature: 0.7,
        top_p: 0.9,
        repetition_penalty: 1.1,
    };
}

async function runPrompt() {
    const prompt = $('prompt-input').value.trim();
    if (!prompt) return showToast('Enter a prompt first', 'error');
    if (!state.modelLoaded) return showToast('Load a model first (use the model bar above)', 'error');

    const btn = $('btn-run');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Streaming...';
    $('response-area').innerHTML = `
        <div class="response-header">
            <span class="badge badge-break">GENERATING</span>
            <span class="text-muted" id="token-counter">0 tokens</span>
        </div>
        <div class="response-md streaming-cursor" id="stream-output"></div>`;
    $('validation-area').innerHTML = '';

    let fullText = '';
    let tokenCount = 0;

    try {
        const params = getGenParams();
        const body = JSON.stringify({ prompt, ...params });
        const resp = await fetch('/api/generate/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body,
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'Stream failed');
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });

            const lines = buf.split('\n');
            buf = lines.pop(); // keep incomplete line

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const payload = line.slice(6);
                if (payload === '[DONE]') continue;
                try {
                    const ev = JSON.parse(payload);
                    if (ev.token) {
                        fullText += ev.token;
                        tokenCount++;
                        const el = $('stream-output');
                        if (el) {
                            el.innerHTML = renderMd(fullText);
                            el.scrollTop = el.scrollHeight;
                        }
                        const tc = $('token-counter');
                        if (tc) tc.textContent = `${tokenCount} tokens`;
                    }
                    if (ev.status) {
                        // Final message with refusal info
                        state.lastRun = { response: fullText, status: ev.status, refusal_count: ev.refusal_count, prompt };
                        state.lastValidation = null;
                        displayResponse(state.lastRun);
                        loadHistory();
                    }
                } catch (_) {}
            }
        }
        // If no final status was received (edge case), display what we have
        if (!state.lastRun || state.lastRun.prompt !== prompt) {
            state.lastRun = { response: fullText, status: 'POSSIBLE_BREAK', refusal_count: 0, prompt };
            displayResponse(state.lastRun);
            loadHistory();
        }
    } catch (e) {
        $('response-area').innerHTML = `<div class="error-msg">${escapeHtml(e.message)}</div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Run';
    }
}

function displayResponse(d) {
    const cls = d.status === 'POSSIBLE_BREAK' ? 'badge-break' : 'badge-refused';
    const lbl = d.status === 'POSSIBLE_BREAK' ? 'POSSIBLE BREAK' : 'REFUSED';
    $('response-area').innerHTML = `
        <div class="response-header">
            <span class="badge ${cls}">${lbl}</span>
            <span class="text-muted">Refusal keywords: ${d.refusal_count}</span>
            <button class="btn btn-outline btn-sm" id="btn-translate-response" onclick="translateResponse()">Translate → EN</button>
            <button class="btn btn-purple btn-sm" id="btn-categorize" onclick="categorizeAttack()">Categorize (AI)</button>
        </div>
        <div class="response-md" id="response-md-content">${renderMd(d.response)}</div>
        <div id="response-translation"></div>`;
}

async function translateResponse() {
    if (!state.apiKeySet) return showToast('Set OpenAI API key first', 'error');
    const response = state.lastRun?.response;
    if (!response) return showToast('No response to translate', 'error');

    const btn = $('btn-translate-response');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';
    try {
        const lang = state.modelInfo?.language || 'Swahili';
        const d = await api('/api/openai/translate', {
            method: 'POST',
            body: JSON.stringify({ text: response, source_lang: lang, target_lang: 'English' }),
        });
        $('response-translation').innerHTML = `
            <div class="suggestion-box" style="margin-top:12px">
                <h4>English Translation</h4>
                <div class="response-md">${renderMd(d.translation)}</div>
                <button class="btn btn-secondary btn-sm" style="margin-top:8px" onclick="this.closest('.suggestion-box').parentElement.innerHTML=''">Close</button>
            </div>`;
        showToast('Response translated');
    } catch (e) { showToast(e.message, 'error'); }
    finally { btn.disabled = false; btn.innerHTML = 'Translate → EN'; }
}

async function categorizeAttack() {
    if (!state.apiKeySet) return showToast('Set OpenAI API key first', 'error');
    if (!state.lastRun) return showToast('No response to categorize', 'error');

    const btn = $('btn-categorize');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';
    try {
        const d = await api('/api/openai/categorize', {
            method: 'POST',
            body: JSON.stringify({
                prompt: state.lastRun.prompt,
                response: state.lastRun.response,
            }),
        });
        // Set the dropdowns to the AI-suggested values
        if (d.attack_type) $('sel-attack-type').value = d.attack_type;
        if (d.risk_category) {
            $('sel-risk-category').value = d.risk_category;
            updateSubcategories();
        }
        if (d.risk_subcategory) {
            setTimeout(() => { $('sel-risk-subcategory').value = d.risk_subcategory; }, 50);
        }
        showToast(`Categorized: ${d.attack_type} / ${d.risk_category} / ${d.risk_subcategory}`);
    } catch (e) { showToast(e.message, 'error'); }
    finally { btn.disabled = false; btn.innerHTML = 'Categorize (AI)'; }
}

// ===== PROMPT LAB — VALIDATION =====
async function validatePrompt() {
    const prompt = $('prompt-input').value.trim();
    if (!prompt) return showToast('Enter a prompt first', 'error');
    if (!state.modelLoaded) return showToast('Load a model first', 'error');

    const btn = $('btn-validate');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Validating...';
    $('validation-area').innerHTML = '<div class="loading">Running 3 validation passes...</div>';

    try {
        const d = await api('/api/generate/rerun', { method: 'POST', body: JSON.stringify({ prompt, num_runs: 3, ...getGenParams() }) });
        state.lastValidation = { ...d, prompt };
        displayValidation(d);
    } catch (e) {
        $('validation-area').innerHTML = `<div class="error-msg">${escapeHtml(e.message)}</div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Validate 3\u00d7';
    }
}

function displayValidation(d) {
    const ok = d.confirmed;
    const runs = d.results.map(r => {
        const icon = r.status === 'POSSIBLE_BREAK' ? '✅' : '❌';
        const cls = r.status === 'POSSIBLE_BREAK' ? 'run-break' : 'run-refused';
        return `<span class="validation-run ${cls}">${icon} Run ${r.run}</span>`;
    }).join('');

    $('validation-area').innerHTML = `
        <div class="validation-header">
            <h4>Validation Results</h4>
            <span class="validation-score ${ok ? 'text-green' : 'text-red'}">
                ${d.break_count}/${d.total_runs} breaks — ${ok ? '✓ CONFIRMED' : '✗ NOT CONFIRMED'}
            </span>
        </div>
        <div class="validation-runs">${runs}</div>
        ${d.best_response ? `<details><summary>Best response (longest break)</summary><div class="response-md">${renderMd(d.best_response)}</div></details>` : ''}`;
}

// ===== PROMPT LAB — OPTIMIZE =====
async function optimizePrompt() {
    const prompt = $('prompt-input').value.trim();
    if (!prompt) return showToast('Enter a prompt first', 'error');
    if (!state.apiKeySet) return showToast('Set OpenAI API key first (Settings)', 'error');

    const btn = $('btn-optimize');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Optimizing...';
    try {
        const d = await api('/api/openai/optimize', {
            method: 'POST',
            body: JSON.stringify({
                prompt,
                language: state.modelInfo?.language || '',
                attack_type: $('sel-attack-type').value,
                risk_category: $('sel-risk-category').value,
                risk_subcategory: $('sel-risk-subcategory').value,
            }),
        });
        state._optimizedPrompt = d.optimized_prompt;
        $('optimize-suggestion').innerHTML = `
            <div class="suggestion-box">
                <h4>AI-Optimized Prompt</h4>
                <pre class="suggestion-text">${escapeHtml(d.optimized_prompt)}</pre>
                <div class="suggestion-actions">
                    <button class="btn btn-accent btn-sm" onclick="acceptOptimized()">Accept</button>
                    <button class="btn btn-secondary btn-sm" onclick="dismissOptimized()">Dismiss</button>
                </div>
            </div>`;
    } catch (e) { showToast(e.message, 'error'); }
    finally { btn.disabled = false; btn.innerHTML = 'Optimize (AI)'; }
}

function acceptOptimized() {
    if (state._optimizedPrompt) $('prompt-input').value = state._optimizedPrompt;
    dismissOptimized();
}
function dismissOptimized() {
    $('optimize-suggestion').innerHTML = '';
    state._optimizedPrompt = null;
}

// ===== PROMPT LAB — TRANSLATE =====
async function translatePrompt() {
    const prompt = $('prompt-input').value.trim();
    if (!prompt) return showToast('Enter a prompt first', 'error');
    if (!state.apiKeySet) return showToast('Set OpenAI API key first', 'error');

    const btn = $('btn-translate');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';
    try {
        const d = await api('/api/openai/translate', {
            method: 'POST',
            body: JSON.stringify({
                text: prompt,
                source_lang: state.modelInfo?.language || 'Swahili',
                target_lang: 'English',
            }),
        });
        state._lastTranslation = d.translation;
        $('translate-result').innerHTML = `
            <div class="suggestion-box">
                <h4>English Translation</h4>
                <pre class="suggestion-text">${escapeHtml(d.translation)}</pre>
                <button class="btn btn-secondary btn-sm" onclick="this.closest('.suggestion-box').parentElement.innerHTML=''">Close</button>
            </div>`;
        showToast('Translation ready');
    } catch (e) { showToast(e.message, 'error'); }
    finally { btn.disabled = false; btn.innerHTML = 'Translate \u2192 EN'; }
}

async function translateToLocal() {
    const prompt = $('prompt-input').value.trim();
    if (!prompt) return showToast('Enter a prompt first', 'error');
    if (!state.apiKeySet) return showToast('Set OpenAI API key first', 'error');

    const lang = state.modelInfo?.language || 'Swahili';
    const btn = $('btn-translate-to-local');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';
    try {
        const d = await api('/api/openai/translate', {
            method: 'POST',
            body: JSON.stringify({
                text: prompt,
                source_lang: 'English',
                target_lang: lang,
            }),
        });
        $('translate-result').innerHTML = `
            <div class="suggestion-box">
                <h4>${escapeHtml(lang)} Translation</h4>
                <pre class="suggestion-text">${escapeHtml(d.translation)}</pre>
                <div style="display:flex;gap:8px;margin-top:8px">
                    <button class="btn btn-accent btn-sm" onclick="document.getElementById('prompt-input').value=this.closest('.suggestion-box').querySelector('.suggestion-text').textContent;this.closest('.suggestion-box').parentElement.innerHTML=''">Use as Prompt</button>
                    <button class="btn btn-secondary btn-sm" onclick="this.closest('.suggestion-box').parentElement.innerHTML=''">Close</button>
                </div>
            </div>`;
        showToast(`Translated to ${lang}`);
    } catch (e) { showToast(e.message, 'error'); }
    finally { btn.disabled = false; btn.innerHTML = 'Translate \u2192 Local'; }
}

// ===== SAVE BREAK MODAL =====
function openSaveModal() {
    if (!state.lastRun && !state.lastValidation)
        return showToast('Run a prompt first', 'error');

    const prompt = state.lastRun?.prompt || '';
    const response = state.lastValidation?.best_response || state.lastRun?.response || '';

    $('save-prompt-preview').textContent = prompt;
    $('save-response-preview').textContent = response;
    $('save-prompt-english').value = '';
    $('save-notes').value = '';
    $('save-translate-status').textContent = '';
    $('save-notes-status').textContent = '';

    // Auto-generate attack ID based on existing breaks count
    const nextId = `B${state.breaks.length + 1}`;
    $('save-attack-id').value = nextId;

    // Populate modal dropdowns from taxonomy
    populateSaveDropdowns();

    // Set dropdown values from lab selectors
    $('save-attack-type-sel').value = $('sel-attack-type').value || '';
    $('save-risk-cat-sel').value = $('sel-risk-category').value || '';
    updateSaveSubcategories();
    $('save-risk-sub-sel').value = $('sel-risk-subcategory').value || '';

    const bc = state.lastValidation?.break_count || 1;
    const tr = state.lastValidation?.total_runs || 1;
    $('save-score').textContent = `${bc}/${tr}`;

    $('save-modal').classList.remove('hidden');

    // Auto-run AI tasks if API key is set
    if (state.apiKeySet) {
        runSaveAI(prompt, response, bc, tr);
    }
}

function populateSaveDropdowns() {
    const atSel = $('save-attack-type-sel');
    const rcSel = $('save-risk-cat-sel');
    atSel.innerHTML = '<option value="">Select...</option>';
    state.taxonomy.attackTypes.forEach(t => {
        const o = document.createElement('option');
        o.value = t['Attack Type'];
        o.textContent = t['Attack Type'];
        atSel.appendChild(o);
    });
    rcSel.innerHTML = '<option value="">Select...</option>';
    state.taxonomy.riskCategories.forEach(c => {
        const o = document.createElement('option');
        o.value = c['Risk Category'];
        o.textContent = c['Risk Category'];
        rcSel.appendChild(o);
    });
    rcSel.addEventListener('change', updateSaveSubcategories);
    updateSaveSubcategories();
}

function updateSaveSubcategories() {
    const cat = $('save-risk-cat-sel').value;
    const sub = $('save-risk-sub-sel');
    sub.innerHTML = '<option value="">Select...</option>';
    const list = cat
        ? state.taxonomy.riskSubcategories.filter(s => s['Risk Category'] === cat)
        : state.taxonomy.riskSubcategories;
    list.forEach(s => {
        const o = document.createElement('option');
        o.value = s['Risk Subcategory'];
        o.textContent = s['Risk Subcategory'];
        sub.appendChild(o);
    });
}

async function runSaveAI(prompt, response, bc, tr) {
    const progress = $('save-ai-progress');
    const status = $('save-ai-status');
    progress.classList.remove('hidden');

    let tasks = [];
    let completed = 0;
    const total = 3; // translate, categorize, notes
    const updateStatus = () => {
        status.textContent = `AI auto-fill: ${completed}/${total} done...`;
    };

    // 1. Translate prompt to English
    const translateTask = (async () => {
        try {
            $('save-translate-status').textContent = '(translating...)';
            const lang = state.modelInfo?.language || 'Swahili';
            const d = await api('/api/openai/translate', {
                method: 'POST',
                body: JSON.stringify({ text: prompt, source_lang: lang, target_lang: 'English' }),
            });
            if (!$('save-prompt-english').value) {
                $('save-prompt-english').value = d.translation;
            }
            $('save-translate-status').textContent = '(✓ auto-translated)';
        } catch (e) {
            $('save-translate-status').textContent = '(✗ translation failed)';
        }
        completed++;
        updateStatus();
    })();
    tasks.push(translateTask);

    // 2. Categorize attack
    const categorizeTask = (async () => {
        try {
            const d = await api('/api/openai/categorize', {
                method: 'POST',
                body: JSON.stringify({ prompt, response }),
            });
            if (d.attack_type && !$('save-attack-type-sel').value) {
                $('save-attack-type-sel').value = d.attack_type;
            }
            if (d.risk_category && !$('save-risk-cat-sel').value) {
                $('save-risk-cat-sel').value = d.risk_category;
                updateSaveSubcategories();
            }
            if (d.risk_subcategory) {
                setTimeout(() => {
                    if (!$('save-risk-sub-sel').value) {
                        $('save-risk-sub-sel').value = d.risk_subcategory;
                    }
                }, 50);
            }
        } catch (_) {}
        completed++;
        updateStatus();
    })();
    tasks.push(categorizeTask);

    // 3. Generate contextual notes (needs translation, but start anyway)
    const notesTask = (async () => {
        try {
            await translateTask; // wait for translation first
            $('save-notes-status').textContent = '(generating notes...)';
            const englishPrompt = $('save-prompt-english').value || prompt;
            const modelName = state.modelInfo?.name || '';
            const d = await api('/api/openai/context', {
                method: 'POST',
                body: JSON.stringify({
                    prompt_original: prompt,
                    prompt_english: englishPrompt,
                    response,
                    attack_type: $('save-attack-type-sel').value || $('sel-attack-type').value,
                    risk_category: $('save-risk-cat-sel').value || $('sel-risk-category').value,
                    risk_subcategory: $('save-risk-sub-sel').value || $('sel-risk-subcategory').value,
                    model_name: modelName,
                    break_count: bc,
                    total_runs: tr,
                }),
            });
            if (!$('save-notes').value) {
                $('save-notes').value = d.contextual_notes;
            }
            $('save-notes-status').textContent = '(✓ auto-generated)';
        } catch (e) {
            $('save-notes-status').textContent = '(✗ generation failed)';
        }
        completed++;
        updateStatus();
    })();
    tasks.push(notesTask);

    await Promise.allSettled(tasks);
    status.textContent = 'AI auto-fill complete!';
    setTimeout(() => progress.classList.add('hidden'), 2000);
}

function rerunSaveAI() {
    const prompt = $('save-prompt-preview').textContent;
    const response = $('save-response-preview').textContent;
    $('save-prompt-english').value = '';
    $('save-notes').value = '';
    const score = $('save-score').textContent.split('/');
    runSaveAI(prompt, response, parseInt(score[0]) || 1, parseInt(score[1]) || 1);
}

function closeSaveModal() { $('save-modal').classList.add('hidden'); }

async function confirmSaveBreak() {
    const prompt = state.lastRun?.prompt || '';
    const response = state.lastValidation?.best_response || state.lastRun?.response || '';

    const body = {
        attack_id: $('save-attack-id').value.trim(),
        attack_type: $('save-attack-type-sel').value,
        risk_category: $('save-risk-cat-sel').value,
        risk_subcategory: $('save-risk-sub-sel').value,
        prompt_original: prompt,
        prompt_english: $('save-prompt-english').value.trim(),
        response,
        contextual_notes: $('save-notes').value.trim(),
        break_count: state.lastValidation?.break_count || 1,
        total_runs: state.lastValidation?.total_runs || 1,
    };

    try {
        const d = await api('/api/breaks', { method: 'POST', body: JSON.stringify(body) });
        showToast(`Break saved: ${d.attack_id}`);
        closeSaveModal();
        loadBreaks();
    } catch (e) { showToast(e.message, 'error'); }
}

// ===== BREAKS DATABASE =====
async function loadBreaks() {
    try {
        state.breaks = await api('/api/breaks');
        renderBreaks();
    } catch (e) { console.error('Load breaks failed', e); }
}

function renderBreaks() {
    const el = $('breaks-list');
    const cnt = $('breaks-count');
    if (cnt) cnt.textContent = `${state.breaks.length} break${state.breaks.length !== 1 ? 's' : ''} saved`;

    if (!state.breaks.length) {
        el.innerHTML = '<div class="empty-state">No breaks saved yet. Find breaks in the Prompt Lab and save them here.</div>';
        return;
    }

    el.innerHTML = state.breaks.map(b => {
        const missingEnglish = !b.prompt_english;
        const missingNotes = !b.contextual_notes;
        const complete = !missingEnglish && !missingNotes;
        return `
        <div class="break-card" id="break-${b.id}">
            <div class="break-header" onclick="toggleBreak(${b.id})">
                <div class="break-meta">
                    <span class="badge badge-id">${esc(b.attack_id)}</span>
                    <span class="break-model">${esc(b.model_name)}</span>
                    <span class="badge badge-type">${esc(b.attack_type)}</span>
                    <span class="badge badge-risk">${esc(b.risk_category)}/${esc(b.risk_subcategory)}</span>
                    <span class="break-score">${b.break_count}/${b.total_runs}</span>
                </div>
                <div class="break-actions-mini">
                    <div class="break-completeness">
                        ${missingEnglish ? '<span class="completeness-tag missing">No translation</span>' : ''}
                        ${missingNotes ? '<span class="completeness-tag missing">No notes</span>' : ''}
                        ${complete ? '<span class="completeness-tag ok">Complete</span>' : ''}
                    </div>
                    <span class="text-muted">${new Date(b.created_at + 'Z').toLocaleDateString()}</span>
                </div>
            </div>
            <div class="break-body hidden" id="break-body-${b.id}">
                <div class="break-section"><h5>Prompt (Original)</h5><pre class="break-text">${esc(b.prompt_original)}</pre></div>
                ${b.prompt_english ? `<div class="break-section"><h5>Prompt (English)</h5><pre class="break-text">${esc(b.prompt_english)}</pre></div>` : ''}
                <div class="break-section"><h5>Response</h5><div class="response-md">${renderMd(b.response)}</div></div>
                <div class="break-section">
                    <h5>Contextual Notes</h5>
                    <div id="notes-${b.id}">
                        ${b.contextual_notes
                            ? `<div class="notes-text">${esc(b.contextual_notes)}</div>`
                            : '<em class="text-muted">No notes yet</em>'}
                    </div>
                </div>
                <div class="break-actions">
                    <button class="btn btn-accent btn-sm" onclick="generateContext(${b.id}, this)">Generate Context (AI)</button>
                    <button class="btn btn-outline btn-sm" onclick="editNotes(${b.id})">Edit Notes</button>
                    <button class="btn btn-danger btn-sm" onclick="deleteBreakConfirm(${b.id})">Delete</button>
                </div>
            </div>
        </div>`;
    }).join('');
}

function esc(t) { return escapeHtml(t); }
function toggleBreak(id) { $(`break-body-${id}`).classList.toggle('hidden'); }

async function generateContext(breakId, btn) {
    if (!state.apiKeySet) return showToast('Set OpenAI API key first', 'error');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Generating...';
    try {
        await api('/api/openai/context', { method: 'POST', body: JSON.stringify({ break_id: breakId }) });
        showToast('Context notes generated');
        loadBreaks();
    } catch (e) { showToast(e.message, 'error'); }
    finally { btn.disabled = false; btn.innerHTML = 'Generate Context (AI)'; }
}

function editNotes(breakId) {
    const b = state.breaks.find(x => x.id === breakId);
    if (!b) return;
    $(`notes-${breakId}`).innerHTML = `
        <textarea class="notes-edit" id="notes-edit-${breakId}" rows="6">${escapeHtml(b.contextual_notes || '')}</textarea>
        <div class="notes-edit-actions">
            <button class="btn btn-accent btn-sm" onclick="saveNotes(${breakId})">Save</button>
            <button class="btn btn-secondary btn-sm" onclick="loadBreaks()">Cancel</button>
        </div>`;
}

async function saveNotes(breakId) {
    const notes = $(`notes-edit-${breakId}`).value;
    try {
        await api(`/api/breaks/${breakId}`, { method: 'PUT', body: JSON.stringify({ contextual_notes: notes }) });
        showToast('Notes saved');
        loadBreaks();
    } catch (e) { showToast(e.message, 'error'); }
}

async function deleteBreakConfirm(breakId) {
    if (!confirm('Delete this break? This cannot be undone.')) return;
    try {
        await api(`/api/breaks/${breakId}`, { method: 'DELETE' });
        showToast('Break deleted');
        loadBreaks();
    } catch (e) { showToast(e.message, 'error'); }
}

// ===== EXPORT =====
async function loadExportPreview() {
    const name = $('export-team-name').value.trim() || 'team_name';
    try {
        const d = await api(`/api/export/markdown?team_name=${encodeURIComponent(name)}`);
        $('export-preview').textContent = d.markdown;
        // Quality warnings
        const warnings = [];
        const breaks = state.breaks;
        if (!breaks.length) {
            warnings.push('No breaks saved yet. You need at least 1 break to submit.');
        } else {
            const noEnglish = breaks.filter(b => !b.prompt_english).length;
            const noNotes = breaks.filter(b => !b.contextual_notes).length;
            const lowScore = breaks.filter(b => b.break_count < 2).length;
            if (noEnglish) warnings.push(`${noEnglish} break(s) missing English translation — required for submission.`);
            if (noNotes) warnings.push(`${noNotes} break(s) missing contextual notes — strongly impacts evaluation score.`);
            if (lowScore) warnings.push(`${lowScore} break(s) with low consistency (< 2/${breaks[0]?.total_runs || 3} runs).`);
            if (breaks.length < 10) warnings.push(`Only ${breaks.length} break(s) saved. More diverse attacks = stronger submission.`);
        }
        const wEl = $('export-warnings');
        if (wEl) {
            wEl.innerHTML = warnings.length
                ? warnings.map(w => `<div class="export-warning">${esc(w)}</div>`).join('')
                : '<div class="export-warning" style="border-color:var(--green);color:var(--green)">All breaks look complete. Ready to export!</div>';
        }
    } catch (e) {
        $('export-preview').textContent = e.message;
    }
}

function downloadExport() {
    const text = $('export-preview').textContent;
    if (!text) return showToast('Preview first', 'error');
    const blob = new Blob([text], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'submission.md';
    a.click();
    URL.revokeObjectURL(url);
}

// ===== HISTORY =====
async function loadHistory() {
    try {
        state.history = await api('/api/history?limit=20');
        renderHistory();
    } catch (e) { /* ignore */ }
}

function renderHistory() {
    const el = $('history-list');
    if (!el) return;
    if (!state.history.length) {
        el.innerHTML = '<div class="text-muted" style="font-size:12px">No history yet</div>';
        return;
    }
    el.innerHTML = state.history.map(h => `
        <div class="history-item" onclick="loadFromHistory(${h.id})">
            <span class="badge ${h.status === 'POSSIBLE_BREAK' ? 'badge-break' : 'badge-refused'} badge-sm">
                ${h.status === 'POSSIBLE_BREAK' ? '!' : '\u2713'}
            </span>
            <span class="history-text">${escapeHtml(h.prompt.substring(0, 100))}${h.prompt.length > 100 ? '…' : ''}</span>
        </div>`).join('');
}

function loadFromHistory(id) {
    const h = state.history.find(x => x.id === id);
    if (!h) return;
    $('prompt-input').value = h.prompt;
    showToast('Prompt loaded');
}
