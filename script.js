/**
 * Jogo da Velha (Tic-Tac-Toe) com Modos Avançados e Três Níveis de IA.
 *
 * Níveis de Dificuldade:
 * - random: Fácil (Jogadas Aleatórias)
 * - mcts: Intermediário (Monte Carlo Tree Search)
 * - minimax: Difícil (Invencível / Minimax)
 */

// ====================================================================
// CONSTANTES GLOBAIS
// ====================================================================

const PLAYER_X = 'X';
const PLAYER_O = 'O';
const WINNING_COMBOS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // Linhas
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // Colunas
    [0, 4, 8], [2, 4, 6]             // Diagonais
];

// ====================================================================
// FUNÇÕES AUXILIARES DE JOGO
// ====================================================================

function checkWin(board) {
    for (const combo of WINNING_COMBOS) {
        const [a, b, c] = combo;
        if (board[a] && board[a] === board[b] && board[a] === board[c]) {
            return board[a];
        }
    }
    return null;
}

function checkTie(board) {
    return !checkWin(board) && board.every(cell => cell !== null);
}

function getAvailableMoves(board) {
    return board.map((val, idx) => val === null ? idx : null).filter(val => val !== null);
}

// ====================================================================
// AGENTES DE IA
// ====================================================================

/** Agente para o Nível Fácil (Aleatório). */
class RandomAgent {
    selectMove(board) {
        const availableMoves = getAvailableMoves(board);
        if (availableMoves.length === 0) return -1;
        return availableMoves[Math.floor(Math.random() * availableMoves.length)];
    }
}

/** Agente para o Nível Difícil (Minimax). */
class MinimaxAgent {
    evaluate(board) {
        const winner = checkWin(board);
        if (winner === PLAYER_O) return 10;
        if (winner === PLAYER_X) return -10;
        return 0;
    }

    minimax(board, isMaximizing, depth) {
        const result = checkWin(board);
        if (result !== null || checkTie(board)) {
            const score = this.evaluate(board);
            if (score > 0) return score - depth;
            if (score < 0) return score + depth;
            return 0;
        }

        const playerToMove = isMaximizing ? PLAYER_O : PLAYER_X;
        let bestScore = isMaximizing ? -Infinity : Infinity;

        for (const move of getAvailableMoves(board)) {
            const newBoard = [...board];
            newBoard[move] = playerToMove;
            const score = this.minimax(newBoard, !isMaximizing, depth + 1);
            bestScore = isMaximizing ? Math.max(bestScore, score) : Math.min(bestScore, score);
        }
        return bestScore;
    }

    selectMove(board, player) {
        let bestScore = player === PLAYER_O ? -Infinity : Infinity;
        let bestMove = -1;
        const isMaximizing = player === PLAYER_O;

        for (const move of getAvailableMoves(board)) {
            const newBoard = [...board];
            newBoard[move] = player;
            const score = this.minimax(newBoard, !isMaximizing, 0);
            if ((isMaximizing && score > bestScore) || (!isMaximizing && score < bestScore)) {
                bestScore = score;
                bestMove = move;
            }
        }
        return bestMove;
    }
}

// ====================================================================
// MONTE CARLO TREE SEARCH (MCTS) — VERSÃO CORRIGIDA
// ====================================================================

class MCTSNode {
    constructor(board, playerToMove, parent = null, move = null) {
        this.board = [...board];
        this.playerToMove = playerToMove;
        this.parent = parent;
        this.move = move;
        this.wins = 0;
        this.visits = 0;
        this.children = [];
        this.unexploredMoves = getAvailableMoves(this.board);
    }

    getUCTScore(c = 0.9) { // constante ajustada para equilíbrio exploração/exploração
        if (this.visits === 0) return Infinity;
        if (!this.parent || this.parent.visits === 0) return Infinity;
        return (this.wins / this.visits) + c * Math.sqrt(Math.log(this.parent.visits) / this.visits);
    }
}


class MCTS_Agent {
  constructor(iterations = 5000) {
    this.iterations = iterations;
  }

  selectNode(node) {
    while (node.unexploredMoves.length === 0 && node.children.length > 0) {
      node = node.children.reduce((best, child) =>
        child.getUCTScore() > best.getUCTScore() ? child : best
      );
    }
    return node;
  }

  expandNode(node) {
    if (node.unexploredMoves.length === 0) return null;

    const move = node.unexploredMoves.pop();
    const newBoard = [...node.board];

    // O jogador que vai jogar neste nó (pai)
    const playerWhoMoved = node.playerToMove;
    newBoard[move] = playerWhoMoved;

    const nextPlayer = playerWhoMoved === PLAYER_X ? PLAYER_O : PLAYER_X;
    const child = new MCTSNode(newBoard, nextPlayer, node, move);

    // Guarda quem fez o movimento, para usar na simulação
    child.playerWhoMoved = playerWhoMoved;
    node.children.push(child);
    return child;
  }

  rollout(boardState, playerWhoMoved) {
    let tempBoard = [...boardState];
    let currentPlayer = playerWhoMoved === PLAYER_X ? PLAYER_O : PLAYER_X;

    while (checkWin(tempBoard) === null && !checkTie(tempBoard)) {
      const moves = getAvailableMoves(tempBoard);
      const move = moves[Math.floor(Math.random() * moves.length)];
      tempBoard[move] = currentPlayer;
      currentPlayer = currentPlayer === PLAYER_X ? PLAYER_O : PLAYER_X;
    }

    const winner = checkWin(tempBoard);
    if (winner === playerWhoMoved) return 1;
    if (winner === null) return 0;
    return -1;
  }

  backpropagate(node, result) {
    let current = node;
    while (current !== null) {
      current.visits++;
      current.wins += result > 0 ? 1 : result === 0 ? 0.5 : 0;
      result = -result; // alterna perspectiva a cada nível
      current = current.parent;
    }
  }

  selectMove(board, player) {
    const root = new MCTSNode(board, player);

    for (let i = 0; i < this.iterations; i++) {
      let node = this.selectNode(root);

      if (checkWin(node.board) === null && !checkTie(node.board)) {
        const expandedNode = this.expandNode(node);
        if (expandedNode) {
          const result = this.rollout(expandedNode.board, expandedNode.playerWhoMoved);
          this.backpropagate(expandedNode, result);
        }
      } else {
        const winner = checkWin(node.board);
        let result = 0;
        if (winner === player) result = 1;
        else if (winner) result = -1;
        this.backpropagate(node, result);
      }
    }

    let bestMove = -1;
    let bestScore = -Infinity;
    for (const child of root.children) {
      if (child.visits > 0) {
        const score = child.wins / child.visits;
        if (score > bestScore) {
          bestScore = score;
          bestMove = child.move;
        }
      }
    }
    return bestMove !== -1 ? bestMove : getAvailableMoves(board)[0];
  }
}


// ====================================================================
// CLASSE PRINCIPAL DO JOGO
// ====================================================================

class TicTacToe {
    constructor() {
        this.dom = {
            board: document.getElementById('game-board'),
            message: document.getElementById('message'),
            startButton: document.getElementById('start-button'),
            modeSelect: document.getElementById('mode-select'),
            levelSelect: document.getElementById('level-select'),
            starterSelectGroup: document.querySelector('.starter-select-group'),
            starterSelect: document.getElementById('starter-select'),
            labelX: document.getElementById('label-x'),
            labelO: document.getElementById('label-o'),
            scoreX: document.getElementById('score-x'),
            scoreO: document.getElementById('score-o'),
            currentMode: document.getElementById('current-mode'),
            iaXSelect: document.getElementById('ia-x-select'),
            iaOSelect: document.getElementById('ia-o-select'),
            iaXGroup: document.getElementById('ia-x-group'),
            iaOGroup: document.getElementById('ia-o-group'),
        };

        this.agents = {
            'random': new RandomAgent(),
            'mcts': new MCTS_Agent(5000),
            'minimax': new MinimaxAgent(),
        };

        this.board = Array(9).fill(null);
        this.currentPlayer = PLAYER_X;
        this.gameActive = false;
        this.scores = { 'X': 0, 'O': 0, 'T': 0 };

        this.init();
    }

    init() {
        this.setupBoard();
        this.addEventListeners();
        this.updateLabels();
        this.dom.message.textContent = "Selecione o modo e clique em 'Iniciar Novo Jogo'.";
    }

    setupBoard() {
        this.dom.board.innerHTML = '';
        for (let i = 0; i < 9; i++) {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            cell.dataset.index = i;
            cell.addEventListener('click', this.handleCellClick.bind(this));
            this.dom.board.appendChild(cell);
        }
    }

    addEventListeners() {
        this.dom.startButton.addEventListener('click', this.startGame.bind(this));
        const selects = ['mode-select', 'level-select', 'starter-select', 'ia-x-select', 'ia-o-select'];
        selects.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', () => this.updateLabels());
        });
    }

    updateLabels() {
        const mode = this.dom.modeSelect.value;
        const level = this.dom.levelSelect.value;
        const levelText = { 'random': 'FÁCIL', 'mcts': 'INTERMEDIÁRIO', 'minimax': 'DIFÍCIL' };

        this.dom.iaXGroup.style.display = 'none';
        this.dom.iaOGroup.style.display = 'none';

        if (mode === 'humano-humano') {
            this.dom.levelSelect.disabled = true;
            this.dom.starterSelectGroup.style.display = 'flex';
            this.dom.labelX.textContent = "Vitórias Jogador 1 (X):";
            this.dom.labelO.textContent = "Vitórias Jogador 2 (O):";
            this.dom.currentMode.textContent = "Modo Atual: Humano (X) vs. Humano (O)";
        } else if (mode === 'humano-ia') {
            this.dom.levelSelect.disabled = false;
            this.dom.starterSelectGroup.style.display = 'flex';
            const starter = this.dom.starterSelect.value;
            if (starter === 'humano') {
                this.dom.labelX.textContent = "Vitórias Humano (X):";
                this.dom.labelO.textContent = `Vitórias Máquina (O) [${levelText[level]}]:`;
            } else {
                this.dom.labelX.textContent = `Vitórias Máquina (X) [${levelText[level]}]:`;
                this.dom.labelO.textContent = "Vitórias Humano (O):";
            }
            this.dom.currentMode.textContent = `Modo Atual: ${starter === 'humano' ? 'Humano (X) vs. Máquina (O)' : 'Máquina (X) vs. Humano (O)'} [${levelText[level]}]`;
        } else {
            this.dom.levelSelect.disabled = true;
            this.dom.starterSelectGroup.style.display = 'none';
            this.dom.iaXGroup.style.display = 'flex';
            this.dom.iaOGroup.style.display = 'flex';
            const iaX = this.dom.iaXSelect.value;
            const iaO = this.dom.iaOSelect.value;
            this.dom.labelX.textContent = `Vitórias Máquina (X) [${levelText[iaX]}]:`;
            this.dom.labelO.textContent = `Vitórias Máquina (O) [${levelText[iaO]}]:`;
            this.dom.currentMode.textContent = `Modo Atual: Máquina (X) [${levelText[iaX]}] vs. Máquina (O) [${levelText[iaO]}]`;
        }

        document.getElementById('score-tie').textContent = this.scores.T;
    }

    startGame() {
        this.board.fill(null);
        this.gameActive = true;
        this.currentPlayer = PLAYER_X;
        this.dom.board.querySelectorAll('.cell').forEach(c => {
            c.textContent = '';
            c.classList.remove('x', 'o');
            c.style.cursor = 'pointer';
        });
        this.updateLabels();
        this.dom.message.textContent = `É a vez de ${this.currentPlayer}`;
        const mode = this.dom.modeSelect.value;
        const starter = this.dom.starterSelect.value;
        if (mode === 'ia-ia' || (mode === 'humano-ia' && starter === 'maquina')) {
            setTimeout(() => this.iaTurn(), 150);
        }
    }

    handleCellClick(e) {
        if (!this.gameActive) return;
        const mode = this.dom.modeSelect.value;
        const index = parseInt(e.target.dataset.index);
        if (this.board[index] !== null || mode === 'ia-ia') return;
        if (this.getAgent(this.currentPlayer)) return;
        this.makeMove(e.target, index);
    }

    makeMove(cell, index) {
        this.board[index] = this.currentPlayer;
        cell.textContent = this.currentPlayer;
        cell.classList.add(this.currentPlayer.toLowerCase());
        const winner = checkWin(this.board);
        if (winner || checkTie(this.board)) {
            this.gameActive = false;
            if (winner) this.updateScores(winner);
            else this.updateScores(null);
            this.dom.message.textContent = winner ? `${winner} venceu!` : 'Empate!';
            return;
        }
        this.currentPlayer = this.currentPlayer === PLAYER_X ? PLAYER_O : PLAYER_X;
        const agent = this.getAgent(this.currentPlayer);
        if (agent) setTimeout(() => this.iaTurn(), 300);
    }

    getAgent(player) {
        const mode = this.dom.modeSelect.value;
        if (mode === 'humano-humano') return null;
        const level = this.dom.levelSelect.value;
        const starter = this.dom.starterSelect.value;
        if (mode === 'ia-ia') {
            const iaX = this.dom.iaXSelect.value;
            const iaO = this.dom.iaOSelect.value;
            return player === PLAYER_X ? this.agents[iaX] : this.agents[iaO];
        }
        const xIsMachine = starter === 'maquina';
        if ((player === PLAYER_X && xIsMachine) || (player === PLAYER_O && !xIsMachine))
            return this.agents[level];
        return null;
    }

    iaTurn() {
        if (!this.gameActive) return;
        const player = this.currentPlayer;
        const agent = this.getAgent(player);
        if (!agent) return;
        const move = agent.selectMove(this.board, player);
        if (move !== -1) {
            const cell = this.dom.board.querySelector(`[data-index="${move}"]`);
            this.makeMove(cell, move);
        }
    }

    updateScores(winner) {
        if (winner === 'X') this.scores.X++;
        else if (winner === 'O') this.scores.O++;
        else this.scores.T++;
        this.dom.scoreX.textContent = this.scores.X;
        this.dom.scoreO.textContent = this.scores.O;
        document.getElementById('score-tie').textContent = this.scores.T;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.game = new TicTacToe();
});
