/* Finder controls are isolated from game commands, game state, and board coordinates. */
(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const read = key => { try { return localStorage.getItem(key); } catch { return null; } };
  const save = (key, value) => { try { localStorage.setItem(key, String(value)); } catch {} };
  const player = side => side === 0 ? 'A' : 'B';
  const number = value => Number.isFinite(value) ? value.toLocaleString() : '—';
  let library = null, selected = read('congklak.finder.run') || '', cert = null;
  let step = Number(read(`congklak.finder.step.${selected}`)) || 0;
  let visible = false, fetching = false, sending = false, loading = null, epoch = 0, listKey = '';
  const certificates = new Map();
  const savedN = Number(read('congklak.finder.n'));
  if (Number.isInteger(savedN) && savedN >= 1 && savedN <= 64) el('finder-n').value = savedN;
  const savedGoal = read('congklak.finder.goal'), savedStarter = read('congklak.finder.starter');
  if (['majority','refill_threshold'].includes(savedGoal)) el('finder-goal').value = savedGoal;
  if (['0','1'].includes(savedStarter)) el('finder-starter').value = savedStarter;

  function view(finder) {
    visible = finder;
    el('play-view').hidden = finder;
    el('finder-view').hidden = !finder;
    el('play-view-tab').setAttribute('aria-pressed', String(!finder));
    el('finder-view-tab').setAttribute('aria-pressed', String(finder));
    save('congklak.view', finder ? 'finder' : 'play');
    if (finder) poll();
  }
  el('play-view-tab').addEventListener('click', () => view(false));
  el('finder-view-tab').addEventListener('click', () => view(true));

  function error(message = '') {
    el('finder-error').textContent = message;
    el('finder-error').hidden = !message;
  }
  function sizeNote() {
    const n = Number(el('finder-n').value);
    el('finder-size-note').textContent = Number.isInteger(n) && n >= 1 && n <= 64
      ? `${n} pits × ${n} seeds · ${number(2*n*n)} seeds in total`
      : 'Choose an integer from 1 to 64.';
  }
  el('finder-n').addEventListener('input', sizeNote);

  function select(id) {
    selected = id; cert = null; loading = null;
    step = Number(read(`congklak.finder.step.${id}`)) || 0;
    save('congklak.finder.run', id);
    render();
  }
  el('finder-runs').addEventListener('change', () => select(el('finder-runs').value));

  function render() {
    if (!library) return;
    const runs = library.runs;
    if (!runs.some(run => run.run_id === selected)) {
      selected = library.active_run || (runs[0] && runs[0].run_id) || '';
      cert = null; step = Number(read(`congklak.finder.step.${selected}`)) || 0;
      save('congklak.finder.run', selected);
    }
    const run = runs.find(item => item.run_id === selected);
    const options = runs.map(item => {
      const option = document.createElement('option');
      option.value = item.run_id;
      const status = item.active ? (item.pausing ? 'pausing' : 'running') : item.error ? 'needs attention' : item.outcome === 'guaranteed_win' ? 'verified' : item.state === 'exhausted' ? 'no witness' : 'paused';
      option.textContent = `n=${item.n ?? '?'} · ${player(item.starter)} · ${item.goal === 'refill_threshold' ? 'stronger target' : 'majority'} · ${status} · ${item.run_id.slice(0,6)}`;
      return option;
    });
    if (!options.length) {
      const option = document.createElement('option'); option.value = ''; option.textContent = 'No runs yet'; options.push(option);
    }
    const newListKey = JSON.stringify(options.map(option => [option.value,option.textContent]));
    if (newListKey !== listKey) { el('finder-runs').replaceChildren(...options); listKey = newListKey; }
    el('finder-runs').value = selected;
    el('finder-start').disabled = sending || !!library.active_run || runs.length >= library.run_limit;
    el('finder-pause').disabled = sending || !run?.active || run.pausing;
    el('finder-resume').disabled = sending || !!library.active_run || !run?.resumable || (['found','exhausted'].includes(run?.state) && !run?.error);
    if (!run) { replay(null); return; }
    const won = run.outcome === 'guaranteed_win' && !run.error;
    el('finder-badge').textContent = run.active ? run.pausing ? 'Pausing…' : run.state === 'pending_verification' ? 'Verifying…' : 'Searching…' : run.error ? 'Needs attention' : won ? 'Verified win' : run.state === 'exhausted' ? 'Search complete' : 'Paused';
    el('finder-badge').className = `badge${run.active ? ' running' : ''}`;
    el('finder-progress-title').textContent = `Dakon-${run.n ?? '?'} · Player ${player(run.starter)} starts`;
    el('finder-status').textContent = run.error ? run.error : won
      ? 'A winning line passed the independent verifier. Replay every choice below.'
      : run.active ? run.pausing ? 'Finishing the current safe checkpoint before pausing.' : run.state === 'pending_verification' ? 'Target reached. Checking the entire line seed by seed before certifying it.' : 'Testing opening choices. Only genuine store bonuses extend the first turn.'
      : run.state === 'exhausted' ? 'No uninterrupted first-turn witness for this target. This does not prove that the whole game is lost.'
      : `Checkpoint saved. ${run.stop_reason === 'time_limit' ? 'The work-session time limit was reached. ' : ''}Resume to continue from this frontier.`;
    el('finder-banked').textContent = `${number(run.best_store)} / ${number(run.required)}`;
    el('finder-actions').textContent = number(run.simulated_actions);
    el('finder-positions').textContent = number(run.expanded_positions);
    el('finder-checkpoint').textContent = number(run.generation);
    el('finder-depth').textContent = `Frontier depth ${number(run.depth)} · cache hits ${number(run.cache_hits)}${run.recovered_from_older_checkpoint ? ' · recovered previous valid checkpoint' : ''}${run.active ? ` · ${number(run.session_seconds)}s this session` : ''}`;
    if (won) {
      const cached = certificates.get(selected);
      if (cached) { if (cert !== cached) { cert = cached; replay(cert); } }
      else { replay(null); loadCertificate(selected); }
    } else { cert = null; replay(null); }
  }

  async function loadCertificate(id) {
    if (loading === id) return;
    loading = id;
    try {
      const response = await fetch(`/api/finder/${id}/certificate`);
      const data = await response.json();
      if (!data.ok || !data.verification?.passed || data.run_id !== id) throw new Error(data.error?.message || 'The sequence could not be verified.');
      certificates.set(id, data.certificate.body);
      if (selected === id) { cert = data.certificate.body; replay(cert); }
    } catch (exc) { if (selected === id) { cert = null; replay(null); error(exc.message); } }
    finally { if (loading === id) loading = null; }
  }

  function cup(side, value, pit, highlighted) {
    const node = document.createElement('div'); node.className = `finder-pit${highlighted ? ' chosen' : ''}`;
    const count = document.createElement('strong'); count.textContent = value;
    const label = document.createElement('span'); label.textContent = `${player(side)}${pit}`;
    node.append(count, label); return node;
  }
  function store(side, value) {
    const node = document.createElement('div'); node.className = `store store-${player(side).toLowerCase()}`;
    const label = document.createElement('span'); label.textContent = player(side);
    const count = document.createElement('strong'); count.textContent = value;
    const caption = document.createElement('small'); caption.textContent = 'STORE';
    node.append(label, count, caption); return node;
  }
  function replay(body) {
    el('finder-replay-empty').hidden = !!body;
    el('finder-replay-content').hidden = !body;
    el('finder-verified').hidden = !body;
    if (!body) { el('finder-board').replaceChildren(); el('finder-sequence').replaceChildren(); return; }
    const {n, starter} = body.recipe, count = body.actions.length;
    step = Math.max(0, Math.min(Math.trunc(step) || 0, count));
    save(`congklak.finder.step.${selected}`, step);
    const cells = step ? body.steps[step-1].cells : [...Array(n).fill(n),0,...Array(n).fill(n),0];
    const next = step < count ? body.actions[step] : null;
    const rows = document.createElement('div'); rows.className = 'finder-rows';
    for (const side of [1,0]) {
      const row = document.createElement('div'); row.className = 'finder-row';
      for (let display = 0; display < n; display++) {
        const pit = side === 0 ? n-1-display : display;
        row.append(cup(side, cells[side*(n+1)+pit], n-pit, side === starter && pit === next));
      }
      rows.append(row);
    }
    el('finder-board').replaceChildren(store(0, cells[n]), rows, store(1, cells[2*n+1]));
    el('finder-proof').textContent = `${player(starter)} banks ${number(body.proof.starter_store)} of ${number(body.proof.total)} seeds in ${count} choices. The opponent can finish with at most ${number(body.proof.opponent_max_total)}. No opponent choice occurs before the target is reached.`;
    el('finder-step').textContent = step ? `Choice ${step} of ${count}` : `Initial position · ${count} choices`;
    el('finder-slider').max = count; el('finder-slider').value = step;
    el('finder-first').disabled = el('finder-prev').disabled = step === 0;
    el('finder-next').disabled = step === count;
    const detail = step ? body.steps[step-1] : null;
    const ending = {own_store:'Last seed in the store: another choice.', own_capture:'Capture: the turn ends.', own_empty:'Empty own pit: the turn ends.', opponent_empty:'Empty opposing pit: the turn ends.'};
    el('finder-step-detail').textContent = detail
      ? `${player(starter)} chose ${player(starter)}${n-body.actions[step-1]}. ${number(detail.deposits)} seed drops, ${number(detail.relays)} relays${detail.captured ? `, ${number(detail.captured)} seeds captured` : ''}. ${ending[detail.reason]}${step === count ? ' Winning target reached.' : ` Next choose ${player(starter)}${n-body.actions[step]}.`}`
      : `${n} seeds in each pit. Highlighted pit ${player(starter)}${n-body.actions[0]} is the first choice. Stores stay on the left from each player’s seat.`;
    const buttons = body.actions.map((action, i) => {
      const button = document.createElement('button'); button.type = 'button'; button.className = 'finder-choice';
      button.textContent = `${i+1}. ${player(starter)}${n-action}`;
      button.setAttribute('aria-label', `Replay after choice ${i+1}: ${player(starter)}${n-action}`);
      button.setAttribute('aria-current', String(step === i+1));
      button.addEventListener('click', () => seek(i+1)); return button;
    });
    el('finder-sequence').replaceChildren(...buttons);
  }
  function seek(value) { if (cert) { step = value; replay(cert); } }
  el('finder-first').addEventListener('click', () => seek(0));
  el('finder-prev').addEventListener('click', () => seek(step-1));
  el('finder-next').addEventListener('click', () => seek(step+1));
  el('finder-slider').addEventListener('input', () => seek(Number(el('finder-slider').value)));

  async function command(req) {
    if (sending || !library) return;
    sending = true; epoch++; error(); render();
    try {
      const response = await fetch('/api/finder/command', {method:'POST', headers:{'Content-Type':'application/json','X-Congklak-Token':library.token}, body:JSON.stringify(req)});
      const data = await response.json();
      if (!data.ok) throw new Error(data.error?.message || 'Finder command failed.');
      library = data;
      if (req.op === 'start') select(data.active_run || data.runs[0]?.run_id || '');
    } catch (exc) { error(`${exc.message} Refreshing the run library will recover any saved progress.`); }
    finally { sending = false; render(); }
  }
  el('finder-form').addEventListener('submit', event => {
    event.preventDefault();
    const n = Number(el('finder-n').value);
    if (!Number.isInteger(n) || n < 1 || n > 64) { error('Choose an integer from 1 to 64.'); return; }
    save('congklak.finder.n', n); save('congklak.finder.goal', el('finder-goal').value);
    save('congklak.finder.starter', el('finder-starter').value);
    command({op:'start', n, goal:el('finder-goal').value, starter:Number(el('finder-starter').value)});
  });
  el('finder-pause').addEventListener('click', () => command({op:'pause',run_id:selected}));
  el('finder-resume').addEventListener('click', () => command({op:'resume',run_id:selected}));

  async function poll() {
    if (fetching || sending) return;
    fetching = true;
    const requestEpoch = epoch;
    try {
      const response = await fetch('/api/finder'); const data = await response.json();
      if (!data.ok) throw new Error(data.error?.message || 'Finder is unavailable.');
      // A command response is newer than a poll that started before it.
      if (!sending && requestEpoch === epoch) { library = data; render(); }
    } catch (exc) { error(`${exc.message} Saved checkpoints remain on this computer.`); }
    finally { fetching = false; }
  }
  sizeNote();
  view(read('congklak.view') === 'finder');
  if (!visible) poll();
  setInterval(() => { if (visible || library?.active_run) poll(); }, 1000);
})();
