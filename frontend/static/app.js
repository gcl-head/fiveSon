const fields = [
  "current_model",
  "best_model",
  "replay_size",
  "training_step",
  "latest_loss",
  "latest_elo",
  "train_steps_per_sec",
  "games_per_min",
  "avg_game_moves",
  "steps_per_cycle",
  "batch_size",
];

let gameSize = 15;
let gameWinner = 0;

const STATUS_LABELS = {
  boot: "启动中",
  idle: "训练中",
  paused: "已暂停",
  self_play: "自我对弈",
  training: "梯度更新",
  arena: "擂台评估",
  unknown: "未知",
};

function updateStatus(data) {
  const rawStatus = data.status || "unknown";
  document.getElementById("status-pill").textContent = STATUS_LABELS[rawStatus] || rawStatus;
  document.getElementById("device-pill").textContent = `运算设备: ${data.device || "检测中"}`;

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
}

function updatePolicySource(aiPolicy, generation) {
  const policyNode = document.getElementById("policySource");
  if (!policyNode) return;

  if (aiPolicy === "trained_model") {
    policyNode.textContent = `对战策略: 训练模型推理（已部署代次 g${generation ?? 0}）`;
    return;
  }
  policyNode.textContent = `对战策略: 启发式回退（代次 g${generation ?? 0}，模型暂不可用）`;
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
  await fetch(`/api/control/${action}`, { method: "POST" });
  await refresh();
}

async function runQuickEval(games = 30) {
  const resultNode = document.getElementById("quickEvalResult");
  const btn = document.getElementById("quickEvalBtn");
  if (resultNode) resultNode.textContent = "评测中，请稍候...";
  if (btn) btn.disabled = true;

  try {
    const res = await fetch("/api/eval/quick", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ games }),
    });
    if (!res.ok) {
      throw new Error("评测接口调用失败");
    }
    const data = await res.json();
    const wr = (Number(data.win_rate || 0) * 100).toFixed(1);
    const avgMs = Number(data.avg_game_ms || 0).toFixed(1);
    const avgMoves = Number(data.avg_moves || 0).toFixed(1);
    if (resultNode) {
      resultNode.textContent = `当前代 g${data.generation} vs 基线 g${data.baseline_generation}: ${data.wins}-${data.losses}-${data.draws}，胜率 ${wr}% | 平均 ${avgMoves} 手/${avgMs}ms 每局`;
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
document.getElementById("quickEvalBtn").addEventListener("click", () => runQuickEval(30));

buildBoard();
refresh();
connectSocket();
fetchGameState();
