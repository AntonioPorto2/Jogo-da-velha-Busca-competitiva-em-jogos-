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
// UTILS E FUNÇÕES DE JOGO
// ====================================================================

/** Verifica se há um vencedor. */
function checkWin(board) {
    for (const combo of WINNING_COMBOS) {
        const [a, b, c] = combo;
        if (board[a] && board[a] === board[b] && board[a] === board[c]) {
            return board[a];
        }
    }
    return null;
}

/** Verifica se o jogo resultou em empate. */
function checkTie(board) {
    return !checkWin(board) && board.every(cell => cell !== null);
}

/** Retorna uma lista de índices vazios (jogadas possíveis). */
function getAvailableMoves(board) {
    return board.map((val, idx) => val === null ? idx : null).filter(val => val !== null);
}

// ====================================================================
// AGENTES DE IA (CLASSES - Random, Minimax, MCTS)
// O código dos agentes foi mantido funcional e robusto.
// ====================================================================

/** Agente para o Nível Fácil (Aleatório). */
class RandomAgent {
    selectMove(board) {
        const availableMoves = getAvailableMoves(board);
        if (availableMoves.length === 0) return -1;
        const randomIndex = Math.floor(Math.random() * availableMoves.length);
        return availableMoves[randomIndex];
    }
}

/** Agente para o Nível Difícil (Minimax). */
class MinimaxAgent {
    // A avaliação padrão favorece o PLAYER_O, mas a lógica de seleção de movimento 
    // lida com quem é o maximizador/minimizador (player parâmetro em selectMove)
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
        const moves = getAvailableMoves(board);

        for (const move of moves) {
            const newBoard = [...board];
            newBoard[move] = playerToMove;
            let score = this.minimax(newBoard, !isMaximizing, depth + 1);
            bestScore = isMaximizing ? Math.max(bestScore, score) : Math.min(bestScore, score);
        }
        return bestScore;
    }

    selectMove(board, player) {
        let bestScore = player === PLAYER_O ? -Infinity : Infinity;
        let bestMove = -1;
        const moves = getAvailableMoves(board);
        const isMaximizing = player === PLAYER_O; // O é Maximizador, X é Minimizador

        for (const move of moves) {
            const newBoard = [...board];
            newBoard[move] = player;

            let score = this.minimax(newBoard, !isMaximizing, 0);

            if (isMaximizing) {
                if (score > bestScore) {
                    bestScore = score;
                    bestMove = move;
                }
            } else { // Minimizador
                if (score < bestScore) {
                    bestScore = score;
                    bestMove = move;
                }
            }
        }
        return bestMove;
    }
}

/** Implementa o nó para a árvore MCTS. */
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

    getUCTScore(c = Math.sqrt(2)) {
        if (this.visits === 0) return Infinity;
        if (this.parent.visits === 0) return Infinity; // Previne divisão por zero se o pai for a raiz e não tiver visitas
        return (this.wins / this.visits) + c * Math.sqrt(Math.log(this.parent.visits) / this.visits);
    }
}

/** Agente para o Nível Intermediário (MCTS). */
class MCTS_Agent {
    constructor(iterations = 5000) {
        this.iterations = iterations;
    }

    selectNode(node) {
        while (node.unexploredMoves.length === 0 && node.children.length > 0) {
            let bestScore = -Infinity;
            let bestChild = null;
            for (const child of node.children) {
                const score = child.getUCTScore();
                if (score > bestScore) {
                    bestScore = score;
                    bestChild = child;
                }
            }
            // Se bestChild for null, é um nó terminal, retorna o nó atual.
            node = bestChild || node;
        }
        return node;
    }

    expandNode(node) {
        if (node.unexploredMoves.length === 0) return null;

        const move = node.unexploredMoves.pop();
        const newBoard = [...node.board];
        // O próximo a jogar é quem fez o movimento para o novo nó
        const playerWhoMoved = node.playerToMove === PLAYER_X ? PLAYER_O : PLAYER_X;

        newBoard[move] = playerWhoMoved;

        // O próximo playerToMove para o novo nó (o oponente de quem moveu)
        const nextPlayerToMove = playerWhoMoved === PLAYER_X ? PLAYER_O : PLAYER_X;

        const child = new MCTSNode(newBoard, nextPlayerToMove, node, move);
        node.children.push(child);
        return child;
    }

    rollout(boardState, playerTurn) {
        let tempBoard = [...boardState];
        let currentTurn = playerTurn;

        while (checkWin(tempBoard) === null && !checkTie(tempBoard)) {
            const moves = getAvailableMoves(tempBoard);
            if (moves.length === 0) break;

            const move = moves[Math.floor(Math.random() * moves.length)];
            tempBoard[move] = currentTurn;
            currentTurn = currentTurn === PLAYER_X ? PLAYER_O : PLAYER_X;
        }

        const winner = checkWin(tempBoard);
        // Retorna a pontuação do ponto de vista do PLAYER_O (IA padrão)
        if (winner === PLAYER_O) return 1;
        if (winner === PLAYER_X) return -1;
        return 0;
    }

    backpropagate(node, result) {
        let current = node;
        while (current !== null) {
            current.visits++;
            // Ajusta a pontuação para a perspectiva do jogador que moveu para o nó pai
            const playerPerspective = current.parent ? (current.parent.playerToMove === PLAYER_O ? 1 : -1) : 1;

            if (result * playerPerspective === 1) current.wins += 1;
            else if (result === 0) current.wins += 0.5;

            current = current.parent;
        }
    }

    selectMove(board, player) {
        // A raiz é o tabuleiro atual, e o próximo a jogar é o 'player' (a IA)
        const root = new MCTSNode(board, player);

        for (let i = 0; i < this.iterations; i++) {
            let node = this.selectNode(root);

            if (checkWin(node.board) === null && !checkTie(node.board)) {
                const expandedNode = this.expandNode(node);
                if (expandedNode) {
                    // O rollout começa com o jogador do nó expandido.
                    const result = this.rollout(expandedNode.board, expandedNode.playerToMove);

                    // A propagação para MCTS deve ser feita a partir da perspectiva do jogador que moveu.
                    // Para simplificar, usamos uma backpropagation mais simples aqui:
                    this.backpropagate(expandedNode, result);
                }
            } else {
                const winner = checkWin(node.board);
                let result = (winner === PLAYER_O) ? 1 : (winner === PLAYER_X) ? -1 : 0;
                this.backpropagate(node, result);
            }
        }

        let bestRatio = -Infinity;
        let bestMove = -1;
        let mostVisits = -1;

        // Escolhe o melhor filho baseado na taxa de vitória (W/V)
        for (const child of root.children) {
            if (child.visits === 0) continue;

            const ratio = child.wins / child.visits;

            if (ratio > bestRatio) {
                bestRatio = ratio;
                bestMove = child.move;
                mostVisits = child.visits;
            } else if (ratio === bestRatio && child.visits > mostVisits) {
                bestMove = child.move;
                mostVisits = child.visits;
            }
        }
        // Se por algum motivo falhar (apenas o primeiro nó), usa o primeiro movimento disponível.
        return bestMove !== -1 ? bestMove : getAvailableMoves(board)[0];
    }
}


// ====================================================================
// CLASSE PRINCIPAL DO JOGO (TicTacToe)
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
        };

        this.agents = {
            'random': new RandomAgent(),
            'mcts': new MCTS_Agent(5000),
            'minimax': new MinimaxAgent(),
        };

        this.board = Array(9).fill(null);
        this.currentPlayer = PLAYER_X;
        this.gameActive = false;
        this.scores = { 'X': 0, 'O': 0 };

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
        // Os onchange dos seletores já chamam updateLabels() no HTML
    }

    // --- Lógica de UI e Placar ---

    updateLabels() {
        const mode = this.dom.modeSelect.value;
        const level = this.dom.levelSelect.value;
        const levelText = { 'random': 'FÁCIL', 'mcts': 'INTERMEDIÁRIO', 'minimax': 'DIFÍCIL' };

        // 1. Controle de Visibilidade e Habilitação
        this.dom.levelSelect.disabled = (mode === 'humano-humano');
        this.dom.starterSelectGroup.style.display = (mode === 'humano-ia') ? 'flex' : 'none';

        // 2. Atualização dos Rótulos
        if (mode === 'humano-humano') {
            this.dom.labelX.textContent = "Vitórias Jogador 1 (X):";
            this.dom.labelO.textContent = "Vitórias Jogador 2 (O):";
            this.dom.currentMode.textContent = "Modo Atual: Humano (X) vs. Humano (O)";
        } else if (mode === 'humano-ia') {
            const starter = this.dom.starterSelect.value;

            if (starter === 'humano') {
                this.dom.labelX.textContent = "Vitórias Humano (X):";
                this.dom.labelO.textContent = `Vitórias Máquina (O) [${levelText[level]}]:`;
                this.dom.currentMode.textContent = `Modo Atual: Humano (X) vs. Máquina (O) [${levelText[level]}]`;
                this.dom.starterSelect.querySelector('option[value="humano"]').textContent = 'Humano (X)';
                this.dom.starterSelect.querySelector('option[value="maquina"]').textContent = 'Máquina (O)';
            } else {
                this.dom.labelX.textContent = `Vitórias Máquina (X) [${levelText[level]}]:`;
                this.dom.labelO.textContent = "Vitórias Humano (O):";
                this.dom.currentMode.textContent = `Modo Atual: Máquina (X) [${levelText[level]}] vs. Humano (O)`;
                this.dom.starterSelect.querySelector('option[value="humano"]').textContent = 'Humano (O)';
                this.dom.starterSelect.querySelector('option[value="maquina"]').textContent = 'Máquina (X)';
            }
        } else { // 'ia-ia'
            // Em IA vs IA, X é sempre a Máquina 'Random' e O usa o nível selecionado
            this.dom.labelX.textContent = "Vitórias Máquina (X) [FÁCIL]:";
            this.dom.labelO.textContent = `Vitórias Máquina (O) [${levelText[level]}]:`;
            this.dom.currentMode.textContent = `Modo Atual: Máquina (X) [FÁCIL] vs. Máquina (O) [${levelText[level]}]`;
        }

        this.dom.scoreX.textContent = this.scores.X;
        this.dom.scoreO.textContent = this.scores.O;
    }

    // --- Lógica de Início ---

    startGame() {
        this.board.fill(null);
        this.gameActive = true;
        this.currentPlayer = PLAYER_X;

        const mode = this.dom.modeSelect.value;
        const starter = this.dom.starterSelect.value;

        // Limpa a UI
        this.dom.board.querySelectorAll('.cell').forEach(cell => {
            cell.textContent = '';
            cell.classList.remove('x', 'o');
            cell.style.cursor = 'pointer';
        });

        this.updateLabels();
        this.dom.message.textContent = `É a vez de ${this.currentPlayer}`;

        // Inicia o turno
        if (mode === 'ia-ia') {
            this.iaTurn(); // Inicia o loop: iaTurn -> makeMove -> iaTurn (e assim por diante)
        } else if (mode === 'humano-ia') {
            const xIsMachine = (starter === 'maquina');

            // Se a Máquina for 'X', ela joga primeiro
            if (xIsMachine) {
                this.iaTurn();
            }
        }
    }

    // --- Lógica de Movimento e Turnos (CORRIGIDA) ---

    handleCellClick(e) {
        if (!this.gameActive) return;

        const mode = this.dom.modeSelect.value;
        const cell = e.target;
        const index = parseInt(cell.dataset.index);

        if (this.board[index] !== null || mode === 'ia-ia') return;

        // Checa se é a vez do Humano
        const isMachineTurn = this.getAgent(this.currentPlayer) !== null;

        if (isMachineTurn) {
            this.dom.message.textContent = "Aguarde o turno da Máquina.";
            return;
        }

        this.makeMove(cell, index);
    }

    makeMove(cell, index) {
        if (!this.gameActive) return;

        this.board[index] = this.currentPlayer;
        cell.textContent = this.currentPlayer;
        cell.classList.add(this.currentPlayer.toLowerCase());
        cell.style.cursor = 'default';

        const winner = checkWin(this.board);

        if (winner || checkTie(this.board)) {
            // Fim de jogo
            this.gameActive = false;
            if (winner) {
                this.dom.message.textContent = `${winner} Venceu!`;
                this.updateScores(winner);
            } else {
                this.dom.message.textContent = "Empate!";
            }
            return;
        }

        // Continua o jogo
        this.currentPlayer = this.currentPlayer === PLAYER_X ? PLAYER_O : PLAYER_X;
        this.dom.message.textContent = `É a vez de ${this.currentPlayer}`;

        // Determina se o próximo jogador é uma Máquina
        const isMachineTurn = this.getAgent(this.currentPlayer) !== null;

        if (isMachineTurn) {
            // Chama o turno de IA de forma assíncrona (com atraso para visualização)
            setTimeout(() => {
                if (this.gameActive) {
                    this.iaTurn();
                }
            }, 500);
        }
    }

    // --- Lógica de IA e Determinação de Agente ---

    getAgent(player) {
        const mode = this.dom.modeSelect.value;
        if (mode === 'humano-humano') return null;

        const starter = this.dom.starterSelect.value;
        const level = this.dom.levelSelect.value;

        if (mode === 'ia-ia') {
            // X (primeiro a jogar) é sempre o Random, O usa o nível selecionado
            return player === PLAYER_X ? this.agents['random'] : this.agents[level];
        }

        // Lógica para Humano vs IA
        const xIsMachine = starter === 'maquina';

        if (player === PLAYER_X && xIsMachine) {
            return this.agents[level];
        } else if (player === PLAYER_O && !xIsMachine) {
            return this.agents[level];
        }

        return null; // É turno do Humano
    }

    iaTurn() {
        if (!this.gameActive) return;

        const player = this.currentPlayer;
        const agent = this.getAgent(player);
        if (!agent) return;

        this.dom.message.textContent = `Máquina (${player}) pensando...`;

        // Não é necessário um setTimeout aqui, pois o makeMove já usa um delay.
        const startTime = performance.now();
        const move = agent.selectMove(this.board, player);
        const endTime = performance.now();

        this.dom.message.textContent = `Máquina (${player}) jogou. (Tempo: ${(endTime - startTime).toFixed(2)} ms)`;

        if (move !== -1) {
            const cell = this.dom.board.querySelector(`[data-index="${move}"]`);
            // Chama makeMove, que por sua vez chama iaTurn de volta se o próximo for IA
            this.makeMove(cell, move);
        }
    }

    // --- Placar e Estado ---

    updateScores(winner) {
        if (winner === 'X') {
            this.scores.X++;
        } else if (winner === 'O') {
            this.scores.O++;
        }
        this.dom.scoreX.textContent = this.scores.X;
        this.dom.scoreO.textContent = this.scores.O;
    }
}

// Inicializa o jogo
document.addEventListener('DOMContentLoaded', () => {
    window.game = new TicTacToe();
});