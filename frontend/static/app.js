const fields = [
  "current_model",
  "best_model",
  "replay_size",
  "training_step",
  "latest_loss",
  "latest_elo",
  "train_steps_per_sec",
  "games_per_min",
  "parallel_self_play_games",
  "heuristic_policy_moves",
  "model_policy_moves",
  "heuristic_bootstrap_games",
  "avg_game_moves",
  "steps_per_cycle",
  "batch_size",
];

let gameSize = 15;
let gameWinner = 0;
let statusHoldUntilMs = 0;
let heldStatus = "";

const TRAINING_STATUS_HOLD_MS = 2000;

const STATUS_LABELS = {
  boot: "启动中",
  idle: "空闲",
  paused: "已暂停",
  self_play: "自我对弈",
  training: "训练中",
  arena: "擂台评估",
  unknown: "未知",
};

function resolveDisplayStatus(rawStatus) {
  const now = Date.now();
  if (rawStatus === "training") {
    heldStatus = "training";
    statusHoldUntilMs = now + TRAINING_STATUS_HOLD_MS;
    return "training";
  }

  if (rawStatus === "paused" || rawStatus === "boot" || rawStatus === "arena") {
    heldStatus = "";
    statusHoldUntilMs = 0;
    return rawStatus;
  }

  if (heldStatus === "training" && now < statusHoldUntilMs && rawStatus === "idle") {
    return "training";
  }

  if (now >= statusHoldUntilMs) {
    heldStatus = "";
    statusHoldUntilMs = 0;
  }
  return rawStatus;
}

function updateStatus(data) {
  const rawStatus = data.status || "unknown";
  const displayStatus = resolveDisplayStatus(rawStatus);
  document.getElementById("status-pill").textContent = STATUS_LABELS[displayStatus] || displayStatus;
  document.getElementById("device-pill").textContent = `运算设备: ${data.device || "检测中"}`;
  const policyNode = document.getElementById("policySource");
  if (policyNode) {
    const currentModel = `${data.current_model || "bootstrap"}`;
    const bestModel = `${data.best_model || "bootstrap"}`;
    if (currentModel !== "bootstrap") {
      policyNode.textContent = `对战策略: 已恢复 ${currentModel}，最佳模型 ${bestModel}`;
    } else {
      policyNode.textContent = "对战策略: 等待恢复训练模型";
    }
  }

  for (const key of fields) {
    const node = document.getElementById(key);
    if (!node) continue;

    const value = data[key];
    if (key === "latest_loss" && typeof value === "number") {
      node.textContent = value.toFixed(4);
    } else if (key === "latest_elo" && typeof value === "number") {
      node.textContent = value.toFixed(1);
    } else if ((key === "train_steps_per_sec" || key === "games_per_min") && typeof value === "number") {
      node.textContent = value.toFixed(1);
    } else {
      node.textContent = `${value ?? "-"}`;
    }
  }

  // avg_game_ms 单独带单位
  const msNode = document.getElementById("avg_game_ms");
  if (msNode && typeof data.avg_game_ms === "number") {
    msNode.textContent = data.avg_game_ms >= 1000
      ? `${(data.avg_game_ms / 1000).toFixed(2)} s`
      : `${data.avg_game_ms.toFixed(0)} ms`;
  }

  const modelMoves = Number(data.model_policy_moves || 0);
  const heuristicMoves = Number(data.heuristic_policy_moves || 0);
  const totalPolicyMoves = modelMoves + heuristicMoves;
  const modelRatio = totalPolicyMoves > 0 ? (modelMoves * 100) / totalPolicyMoves : 0;
  const heuristicRatio = totalPolicyMoves > 0 ? (heuristicMoves * 100) / totalPolicyMoves : 0;

  const modelRatioNode = document.getElementById("model_policy_ratio");
  if (modelRatioNode) {
    modelRatioNode.textContent = `${modelRatio.toFixed(1)}%`;
  }
  const heuristicRatioNode = document.getElementById("heuristic_policy_ratio");
  if (heuristicRatioNode) {
    heuristicRatioNode.textContent = `${heuristicRatio.toFixed(1)}%`;
  }

  updateParallelButtons(Number(data.target_parallel_self_play_games || data.parallel_self_play_games || 0));
  renderSelfPlayBoards(Array.isArray(data.active_games) ? data.active_games : []);
  renderQuickEvalBoards(Array.isArray(data.quick_eval_games) ? data.quick_eval_games : []);

  const generationInput = document.getElementById("quickEvalGeneration");
  const baselineInput = document.getElementById("quickEvalBaseline");
  const deployedGeneration = Number(data.deployed_generation || 0);
  if (generationInput) {
    generationInput.max = String(deployedGeneration);
    if (!generationInput.dataset.initialized) {
      generationInput.value = String(deployedGeneration);
      generationInput.dataset.initialized = "true";
    }
    if (document.activeElement !== generationInput) {
      generationInput.placeholder = `当前 g${deployedGeneration}`;
    }
  }
  if (baselineInput) {
    baselineInput.max = String(deployedGeneration);
    if (!baselineInput.dataset.initialized) {
      baselineInput.value = "0";
      baselineInput.dataset.initialized = "true";
    }
  }
}

function formatElapsed(ms) {
  if (ms >= 60_000) return `${(ms / 60_000).toFixed(2)} min`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${Math.max(0, ms).toFixed(0)} ms`;
}

function renderSelfPlayBoards(games) {
  const grid = document.getElementById("selfPlayGrid");
  if (!grid) return;

  if (games.length === 0) {
    grid.innerHTML = "<div class='eval-result'>暂无训练棋局快照</div>";
    return;
  }

  grid.innerHTML = "";
  for (const game of games) {
    const card = document.createElement("div");
    card.className = "selfplay-card";

    const title = document.createElement("div");
    title.className = "selfplay-title";
    const workerId = Number(game.worker_id ?? 0);
    const moveCount = Number(game.move_count ?? 0);
    const winner = Number(game.winner ?? 0);
    const done = Boolean(game.done);
    const elapsed = Number(game.elapsed_ms ?? 0);
    let statusText = done ? "已结束" : "进行中";
    if (winner === 1) statusText = `${statusText} 黑胜`;
    if (winner === -1) statusText = `${statusText} 白胜`;
    title.textContent = `并行局 #${workerId} | 手数 ${moveCount} | 用时 ${formatElapsed(elapsed)} | ${statusText}`;
    card.appendChild(title);

    const board = document.createElement("div");
    board.className = "live-board";
    const matrix = Array.isArray(game.board) ? game.board : [];
    const size = matrix.length > 0 ? matrix.length : 15;
    board.style.gridTemplateColumns = `repeat(${size}, 1fr)`;

    for (let r = 0; r < size; r += 1) {
      for (let c = 0; c < size; c += 1) {
        const value = Number((matrix[r] || [])[c] ?? 0);
        const cell = document.createElement("div");
        cell.className = "live-cell";
        if (value !== 0) {
          const stone = document.createElement("span");
          stone.className = `stone ${value === 1 ? "black" : "white"}`;
          cell.appendChild(stone);
        }
        board.appendChild(cell);
      }
    }

    card.appendChild(board);
    grid.appendChild(card);
  }
}

function renderQuickEvalBoards(games) {
  const grid = document.getElementById("quickEvalGrid");
  if (!grid) return;

  if (games.length === 0) {
    grid.innerHTML = "<div class='eval-result'>暂无快速验证局面</div>";
    return;
  }

  grid.innerHTML = "";
  for (const game of games) {
    const card = document.createElement("div");
    card.className = "selfplay-card";

    const title = document.createElement("div");
    title.className = "selfplay-title";
    const gameId = Number(game.game_id ?? 0);
    const moveCount = Number(game.move_count ?? 0);
    const winner = Number(game.winner ?? 0);
    const done = Boolean(game.done);
    const elapsed = Number(game.elapsed_ms ?? 0);
    const generation = Number(game.generation ?? 0);
    const baseline = Number(game.baseline_generation ?? 0);
    let statusText = done ? "已结束" : "进行中";
    if (winner === 1) statusText = `${statusText} 黑胜`;
    if (winner === -1) statusText = `${statusText} 白胜`;
    title.textContent = `验证局 #${gameId} | g${generation} vs g${baseline} | 手数 ${moveCount} | 用时 ${formatElapsed(elapsed)} | ${statusText}`;
    card.appendChild(title);

    const board = document.createElement("div");
    board.className = "live-board";
    const matrix = Array.isArray(game.board) ? game.board : [];
    const size = matrix.length > 0 ? matrix.length : 15;
    board.style.gridTemplateColumns = `repeat(${size}, 1fr)`;

    for (let r = 0; r < size; r += 1) {
      for (let c = 0; c < size; c += 1) {
        const value = Number((matrix[r] || [])[c] ?? 0);
        const cell = document.createElement("div");
        cell.className = "live-cell";
        if (value !== 0) {
          const stone = document.createElement("span");
          stone.className = `stone ${value === 1 ? "black" : "white"}`;
          cell.appendChild(stone);
        }
        board.appendChild(cell);
      }
    }

    card.appendChild(board);
    grid.appendChild(card);
  }
}

function updateParallelButtons(target) {
  const buttons = document.querySelectorAll(".parallel-btn");
  buttons.forEach((btn) => {
    const value = Number(btn.dataset.parallel || "0");
    if (value === target) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });
}

function updatePolicySource(aiPolicy, generation) {
  const policyNode = document.getElementById("policySource");
  if (!policyNode) return;

  if (aiPolicy === "trained_model") {
    policyNode.textContent = `对战策略: 训练模型推理（已部署代次 g${generation ?? 0}）`;
    return;
  }
  policyNode.textContent = `对战策略: 启发式策略（代次 g${generation ?? 0}，模型暂不可用）`;
}

async function refresh() {
  const res = await fetch("/api/status");
  const data = await res.json();
  updateStatus(data);
}

function connectSocket() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${window.location.host}/ws/status`);

  ws.onmessage = (ev) => {
    try {
      updateStatus(JSON.parse(ev.data));
    } catch (err) {
      console.error(err);
    }
  };

  ws.onclose = () => setTimeout(connectSocket, 1000);
}

function buildBoard(size = 15) {
  gameSize = size;
  const board = document.getElementById("board");
  board.style.gridTemplateColumns = `repeat(${size}, 1fr)`;
  board.innerHTML = "";
  for (let i = 0; i < size * size; i += 1) {
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.dataset.index = String(i);
    cell.addEventListener("click", () => playMove(i));
    board.appendChild(cell);
  }
}

function renderBoard(grid) {
  const board = document.getElementById("board");
  const cells = board.querySelectorAll(".cell");

  for (let r = 0; r < grid.length; r += 1) {
    for (let c = 0; c < grid[r].length; c += 1) {
      const idx = r * gameSize + c;
      const cell = cells[idx];
      if (!cell) continue;

      cell.innerHTML = "";
      const value = grid[r][c];
      if (value === 0) continue;

      const stone = document.createElement("span");
      stone.className = `stone ${value === 1 ? "black" : "white"}`;
      cell.appendChild(stone);
    }
  }
}

function setGameMessage(text) {
  const node = document.getElementById("gameMessage");
  node.textContent = text;
}

function showIngestBanner(msg) {
  const banner = document.getElementById("gameIngest");
  banner.textContent = msg;
  banner.style.display = "block";
}

function hideIngestBanner() {
  const banner = document.getElementById("gameIngest");
  banner.style.display = "none";
  banner.textContent = "";
}

async function fetchGameState() {
  const res = await fetch("/api/game/state");
  const data = await res.json();
  updatePolicySource(data.ai_policy, data.deployed_generation);
  if (data.size !== gameSize) {
    buildBoard(data.size);
  }
  renderBoard(data.board);
  gameWinner = data.winner;
  if (data.winner === 1) {
    setGameMessage("你赢了！黑棋获胜。");
  } else if (data.winner === -1) {
    setGameMessage("AI 获胜！白棋获胜。");
  } else {
    setGameMessage("你执黑棋，点击棋盘落子。");
  }
}

async function playMove(move) {
  if (gameWinner !== 0) return;

  const res = await fetch("/api/game/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ move }),
  });

  if (!res.ok) {
    const payload = await res.json().catch(() => ({}));
    setGameMessage(payload.detail || "Invalid move.");
    return;
  }

  const data = await res.json();
  renderBoard(data.board);
  gameWinner = data.winner;
  updatePolicySource(data.ai_policy, data.deployed_generation);

  if (data.winner === 1) {
    setGameMessage("你赢了！黑棋获胜。");
    showIngestBanner("✅ 本局对弈数据已以更高权重加入训练集，AI 将从你的走法中学习。");
  } else if (data.winner === -1) {
    setGameMessage("AI 获胜！白棋获胜。");
    showIngestBanner("✅ 本局对弈数据已以更高权重加入训练集，AI 将持续强化此类策略。");
  } else if (data.legal_moves === 0) {
    setGameMessage("平局！棋盘已满。");
    showIngestBanner("✅ 本局对弈数据已以更高权重加入训练集。");
  } else {
    setGameMessage(`AI 已落子，请继续。`);
  }
}

async function resetGame() {
  await fetch("/api/game/reset", { method: "POST" });
  gameWinner = 0;
  hideIngestBanner();
  await fetchGameState();
}

async function control(action) {
  // User-triggered control should immediately drop any stale training hold state.
  heldStatus = "";
  statusHoldUntilMs = 0;
  await fetch(`/api/control/${action}`, { method: "POST" });
  await refresh();
}

async function setParallelGames(count) {
  const resultNode = document.getElementById("parallelSwitchResult");
  if (resultNode) resultNode.textContent = `切换到 ${count} 并行中...`;

  try {
    const res = await fetch(`/api/control/parallel/${count}`, { method: "POST" });
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      throw new Error(payload.detail || "切换失败");
    }
    if (resultNode) resultNode.textContent = `已切换目标并行数: ${count}`;
    await refresh();
  } catch (err) {
    console.error(err);
    if (resultNode) resultNode.textContent = "并行数切换失败";
  }
}

async function runQuickEval(games = 30) {
  const resultNode = document.getElementById("quickEvalResult");
  const btn = document.getElementById("quickEvalBtn");
  const generationInput = document.getElementById("quickEvalGeneration");
  const baselineInput = document.getElementById("quickEvalBaseline");
  const generation = Number(generationInput?.value || 0);
  const baselineGeneration = Number(baselineInput?.value || 0);
  if (resultNode) resultNode.textContent = "评测中，请稍候...";
  if (btn) btn.disabled = true;

  try {
    const res = await fetch("/api/eval/quick", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ games, generation, baseline_generation: baselineGeneration }),
    });
    if (!res.ok) {
      throw new Error("评测接口调用失败");
    }
    const data = await res.json();
    const wr = (Number(data.win_rate || 0) * 100).toFixed(1);
    const avgMs = Number(data.avg_game_ms || 0).toFixed(1);
    const avgMoves = Number(data.avg_moves || 0).toFixed(1);
    if (resultNode) {
      resultNode.textContent = `g${data.generation} vs g${data.baseline_generation}: ${data.wins}-${data.losses}-${data.draws}，胜率 ${wr}% | 平均 ${avgMoves} 手/${avgMs}ms 每局`;
    }
  } catch (err) {
    console.error(err);
    if (resultNode) resultNode.textContent = "评测失败，请稍后重试";
  } finally {
    if (btn) btn.disabled = false;
  }
}

document.getElementById("pauseBtn").addEventListener("click", () => control("pause"));
document.getElementById("resumeBtn").addEventListener("click", () => control("resume"));
document.getElementById("refreshBtn").addEventListener("click", refresh);
document.getElementById("newGameBtn").addEventListener("click", resetGame);
document.getElementById("newGameBtn").addEventListener("click", hideIngestBanner);
document.getElementById("quickEvalBtn").addEventListener("click", () => runQuickEval(32));
document.querySelectorAll(".parallel-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const count = Number(btn.dataset.parallel || "8");
    setParallelGames(count);
  });
});

buildBoard();
refresh();
connectSocket();
fetchGameState();
