// Board page for Olimpia. chess.js handles the rules on this side. Every reply comes from the API;
// if the engine cannot be reached the page says so and never makes a move up.
import { Chess } from './vendor/chess.js';

const API = 'https://olimpia-api-production.up.railway.app';
const WAKE_TIMEOUT_S = 240;      // how long to keep showing "waking" before giving up
const REQUEST_TIMEOUT_MS = 25000;

const GLYPH = { k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟' };
const NAMES = { k: 'king', q: 'queen', r: 'rook', b: 'bishop', n: 'knight', p: 'pawn' };
const FILES = 'abcdefgh';

const $ = (id) => document.getElementById(id);
const boardEl = $('board');
const statusEl = $('status');
const noteEl = $('engine-note');
const promotionEl = $('promotion');

const game = new Chess();
let humanColor = 'w';
let flipped = false;
let selected = null;
let focusSquare = 'e2';
let lastMove = null;
let thinking = false;
let gameToken = 0;        // bumps on new game or take back, so a late reply gets ignored
let engineReady = false;
let squares = {};

// --- Drawing --------------------------------------------------------------------------------

function orderedSquares() {
  const list = [];
  for (let rank = 8; rank >= 1; rank--) {
    for (let file = 0; file < 8; file++) list.push(FILES[file] + rank);
  }
  return flipped ? list.reverse() : list;
}

function buildBoard() {
  boardEl.textContent = '';
  squares = {};
  orderedSquares().forEach((name, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    const fileIndex = FILES.indexOf(name[0]);
    const rank = Number(name[1]);
    button.className = `sq ${(fileIndex + rank) % 2 === 0 ? 'light' : 'dark'}`;
    button.dataset.square = name;
    button.setAttribute('role', 'gridcell');
    button.addEventListener('click', () => onSquare(name));
    button.addEventListener('keydown', (e) => onKey(e, name));
    boardEl.appendChild(button);
    squares[name] = button;
    if ((index + 1) % 8 === 0) addCoord(button, 'rank', name[1]);
    if (index >= 56) addCoord(button, 'file', name[0]);
  });
  // rank labels on the left edge read better, so move them there
  for (const el of boardEl.querySelectorAll('.coord.rank')) {
    const row = el.parentElement;
    const i = [...boardEl.children].indexOf(row);
    boardEl.children[i - 7].appendChild(el);
  }
  render();
}

function addCoord(button, kind, text) {
  const span = document.createElement('span');
  span.className = `coord ${kind}`;
  span.textContent = text;
  span.setAttribute('aria-hidden', 'true');
  button.appendChild(span);
}

function render() {
  const targets = selected ? game.moves({ square: selected, verbose: true }).map((m) => m.to) : [];
  let checkedKing = null;
  if (game.inCheck()) {
    for (const row of game.board()) {
      for (const p of row) if (p && p.type === 'k' && p.color === game.turn()) checkedKing = p.square;
    }
  }
  for (const [name, button] of Object.entries(squares)) {
    const piece = game.get(name);
    let pieceEl = button.querySelector('.piece');
    if (piece) {
      if (!pieceEl) {
        pieceEl = document.createElement('span');
        button.prepend(pieceEl);
      }
      pieceEl.className = `piece ${piece.color}`;
      pieceEl.textContent = `${GLYPH[piece.type]}︎`;
    } else if (pieceEl) {
      pieceEl.remove();
    }
    button.classList.toggle('selected', selected === name);
    button.classList.toggle('target', targets.includes(name));
    button.classList.toggle('occupied', !!piece);
    button.classList.toggle('last', !!lastMove && (lastMove.from === name || lastMove.to === name));
    button.classList.toggle('check', checkedKing === name);
    button.tabIndex = name === focusSquare ? 0 : -1;
    const label = piece ? `${piece.color === 'w' ? 'white' : 'black'} ${NAMES[piece.type]}` : 'empty';
    button.setAttribute('aria-label', `${name}, ${label}${selected === name ? ', selected' : ''}${targets.includes(name) ? ', legal move' : ''}`);
  }
  renderMoves();
  $('undo').disabled = thinking || game.history().length < 2;
}

function renderMoves() {
  const list = $('moves');
  list.textContent = '';
  const history = game.history();
  for (let i = 0; i < history.length; i += 2) {
    const li = document.createElement('li');
    const white = document.createElement('span');
    white.textContent = history[i];
    li.appendChild(white);
    if (history[i + 1]) {
      const black = document.createElement('span');
      black.textContent = history[i + 1];
      li.appendChild(black);
    }
    list.appendChild(li);
  }
  list.scrollTop = list.scrollHeight;
}

function setStatus(text, kind = '') {
  statusEl.textContent = text;
  statusEl.className = `status ${kind}`;
}

// --- Input ----------------------------------------------------------------------------------

function onKey(event, name) {
  const steps = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
  const step = steps[event.key];
  if (!step) return;
  event.preventDefault();
  const direction = flipped ? -1 : 1;
  const file = FILES.indexOf(name[0]) + step[0] * direction;
  const rank = Number(name[1]) + step[1] * direction;
  if (file < 0 || file > 7 || rank < 1 || rank > 8) return;
  focusSquare = FILES[file] + rank;
  render();
  squares[focusSquare].focus();
}

function onSquare(name) {
  focusSquare = name;
  if (thinking || game.isGameOver() || game.turn() !== humanColor) return;
  const piece = game.get(name);
  if (selected) {
    const legal = game.moves({ square: selected, verbose: true }).filter((m) => m.to === name);
    if (legal.length) {
      if (legal[0].promotion) {
        askPromotion(selected, name);
      } else {
        playHuman(selected, name);
      }
      return;
    }
  }
  selected = piece && piece.color === humanColor && selected !== name ? name : null;
  render();
}

function askPromotion(from, to) {
  promotionEl.textContent = '';
  for (const type of ['q', 'r', 'b', 'n']) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = `${GLYPH[type]}︎`;
    button.setAttribute('aria-label', `Promote to ${NAMES[type]}`);
    button.className = `piece ${humanColor}`;
    button.style.pointerEvents = 'auto';
    button.addEventListener('click', () => {
      promotionEl.hidden = true;
      playHuman(from, to, type);
    });
    promotionEl.appendChild(button);
  }
  promotionEl.hidden = false;
  promotionEl.firstChild.focus();
}

function playHuman(from, to, promotion) {
  const move = game.move({ from, to, promotion });
  selected = null;
  lastMove = move;
  render();
  if (!announceGameEnd()) requestEngineMove();
}

function announceGameEnd() {
  if (!game.isGameOver()) return false;
  let text;
  if (game.isCheckmate()) {
    text = game.turn() === humanColor ? 'Checkmate. Olimpia wins.' : 'Checkmate. You win.';
  } else if (game.isStalemate()) {
    text = 'Draw by stalemate.';
  } else if (game.isThreefoldRepetition()) {
    text = 'Draw by threefold repetition.';
  } else if (game.isInsufficientMaterial()) {
    text = 'Draw by insufficient material.';
  } else {
    text = 'Draw by the fifty-move rule.';
  }
  setStatus(text);
  return true;
}

// --- Engine ---------------------------------------------------------------------------------

async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Railway sleeps the service when idle. A request wakes it, then Numba compiles the engine. While
// that happens the health check says "warming", or the platform answers without CORS headers, so
// every failure in this window counts as "still waking".
let wakePromise = null;
function wakeEngine() {
  wakePromise ??= (async () => {
    const started = performance.now();
    for (;;) {
      const seconds = Math.round((performance.now() - started) / 1000);
      try {
        const response = await fetchWithTimeout(`${API}/healthz`);
        const body = await response.json();
        if (body.engine === 'ready') {
          engineReady = true;
          $('budget').textContent = String(body.think_ms);
          noteEl.textContent = 'Engine awake.';
          wakePromise = null;
          return true;
        }
        if (body.engine === 'failed') break;
      } catch {
        // the platform is still bringing the container up
      }
      if (seconds > WAKE_TIMEOUT_S) break;
      noteEl.textContent = `Waking the engine: ${seconds} s. It sleeps when nobody is playing, and compiles its search on the way up, which can take a minute or two.`;
      if (thinking) setStatus(`Waking the engine (${seconds} s). Your move is in, and Olimpia replies once it is up.`, 'waiting');
      await wait(2000);
    }
    wakePromise = null;
    noteEl.textContent = '';
    return false;
  })();
  return wakePromise;
}

function showEngineDown(token, message) {
  thinking = false;
  render();
  if (token !== gameToken) return;
  setStatus(message, 'error');
  const retry = document.createElement('button');
  retry.type = 'button';
  retry.textContent = 'Try again';
  retry.className = 'retry';
  retry.addEventListener('click', () => requestEngineMove());
  noteEl.textContent = '';
  noteEl.appendChild(retry);
}

async function requestEngineMove() {
  const token = gameToken;
  const fen = game.fen();
  thinking = true;
  render();
  setStatus('Olimpia is thinking...', 'waiting');

  for (let attempt = 0; attempt < 8; attempt++) {
    if (!engineReady) {
      const awake = await wakeEngine();
      if (token !== gameToken) return;
      if (!awake) {
        showEngineDown(token, 'The engine could not be reached, so there is no reply to show.');
        return;
      }
      setStatus('Olimpia is thinking...', 'waiting');
    }
    let response;
    try {
      response = await fetchWithTimeout(`${API}/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen }),
      });
    } catch {
      // the server went back to sleep or dropped the connection
      engineReady = false;
      continue;
    }
    if (token !== gameToken) return;
    if (response.status === 200) {
      applyEngineMove(await response.json());
      return;
    }
    if (response.status === 429) {
      const delay = Number(response.headers.get('Retry-After')) || 30;
      setStatus(`The demo's rate limit was reached. Retrying in ${delay} s.`, 'waiting');
      await wait(delay * 1000);
    } else if (response.status === 503) {
      const body = await response.json().catch(() => ({}));
      if (body.error === 'engine warming') engineReady = false;
      else await wait(2000);
    } else {
      const body = await response.json().catch(() => ({}));
      showEngineDown(token, `The engine refused this position: ${body.error ?? response.status}.`);
      return;
    }
    if (token !== gameToken) return;
  }
  showEngineDown(token, 'The engine is busy or not responding, so there is no reply to show.');
}

function applyEngineMove(result) {
  const uci = result.move;
  let move;
  try {
    move = game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
  } catch {
    showEngineDown(gameToken, `The engine returned a move the board rejects: ${uci}.`);
    return;
  }
  thinking = false;
  lastMove = move;
  showStats(result);
  render();
  if (!announceGameEnd()) {
    setStatus(`Olimpia played ${move.san}. Your move.`);
  }
}

function showStats({ stats, think_ms: budget }) {
  const number = new Intl.NumberFormat('en');
  $('s-nps').textContent = number.format(stats.nps);
  $('s-nodes').textContent = number.format(stats.nodes);
  $('s-depth').textContent = String(stats.depth);
  $('s-time').textContent = `${stats.time_ms.toFixed(0)} ms`;
  if (stats.mate_in_plies !== null) {
    $('s-score').textContent = stats.mate_in_plies > 0 ? `mate in ${stats.mate_in_plies} plies` : `mated in ${-stats.mate_in_plies} plies`;
  } else {
    const pawns = stats.score_cp / 100;
    $('s-score').textContent = `${pawns >= 0 ? '+' : ''}${pawns.toFixed(2)} for Olimpia`;
  }
  $('budget').textContent = String(budget);
  noteEl.textContent = stats.source === 'fallback' ? 'This move came from the engine\'s Python fallback search, not the compiled one.' : '';
}

// --- Buttons --------------------------------------------------------------------------------

function newGame(color) {
  gameToken += 1;
  game.reset();
  humanColor = color;
  flipped = color === 'b';
  selected = null;
  lastMove = null;
  thinking = false;
  promotionEl.hidden = true;
  noteEl.textContent = engineReady ? '' : noteEl.textContent;
  buildBoard();
  if (color === 'b') {
    requestEngineMove();
  } else {
    setStatus('Your move.');
  }
}

$('new-white').addEventListener('click', () => newGame('w'));
$('new-black').addEventListener('click', () => newGame('b'));
$('flip').addEventListener('click', () => {
  flipped = !flipped;
  buildBoard();
});
$('undo').addEventListener('click', () => {
  if (thinking || game.history().length < 2) return;
  gameToken += 1;
  game.undo();
  game.undo();
  const previous = game.history({ verbose: true }).at(-1);
  lastMove = previous ? { from: previous.from, to: previous.to } : null;
  selected = null;
  render();
  setStatus('Your move.');
});

buildBoard();
setStatus('Your move.');
// Start waking the server as soon as the page opens, so the first move rarely has to wait long.
wakeEngine().then((awake) => {
  if (!awake && !thinking) noteEl.textContent = 'The engine could not be reached right now. You can still try a move.';
});
