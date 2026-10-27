# CodSoft Artificial Intelligence Internship - Task 2: Tic-Tac-Toe AI

A lightweight, terminal-based Tic-Tac-Toe game powered by an unbeatable **Minimax algorithm** with **Alpha-Beta Pruning**. Tested on **Python 3.13.1** on Windows, this project demonstrates classic game theory, adversarial search trees, and optimal heuristic evaluation without any external libraries.

---

## 📌 How the Minimax Algorithm Works

Minimax is a recursive backtracking algorithm used in two-player, turn-based, zero-sum games of perfect information (such as Tic-Tac-Toe, Connect Four, or Chess).

### 1. Two Opposing Players
- **Maximizer (AI):** Aims to choose moves that achieve the highest possible score.
- **Minimizer (Human):** Assumed to play optimally, choosing moves that minimize the AI's score.

### 2. Terminal State Scoring
When evaluating leaf nodes in the game tree:
- **AI Win:** `+10 - depth` (penalizing depth incentivizes finding the fastest path to victory).
- **Human Win:** `-10 + depth` (rewarding depth incentivizes resisting and delaying loss as long as possible).
- **Draw:** `0` (neutral evaluation).

### 3. Alpha-Beta Pruning Optimization
Standard Minimax evaluates every branch in the game tree. Alpha-Beta pruning optimizes this by maintaining two cutoff thresholds:
- $\alpha$ (Alpha): The highest score guaranteed to the Maximizer so far.
- $\beta$ (Beta): The lowest score guaranteed to the Minimizer so far.

Whenever $\beta \le \alpha$, the opposing player already has a guaranteed path that makes the current branch irrelevant. Exploring further moves in that subtree is pruned (skipped), dramatically cutting execution time without altering the optimal result.

### 4. Move Ordering
Moves are explored in order of strategic dominance (Center $\rightarrow$ Corners $\rightarrow$ Edges), triggering pruning earlier in the recursion.

### 5. Mathematical Guarantee
Because Tic-Tac-Toe is a solved finite game, an optimal Minimax agent **cannot be defeated**. The match will always end in either an **AI victory** or a **draw**.

---

## ✨ Features

- **Unbeatable Minimax AI:** Plays strictly optimal moves; never loses a game.
- **Move Validation:** Guards against out-of-bounds indices, occupied cells, and non-numeric inputs.
- **Player Flexibility:** Choose to play as `X` (moves first) or `O` (moves second).
- **Visual Coordinate Helper:** Empty cells display numbers `1` through `9` for intuitive input.
- **Zero External Dependencies:** Built solely using Python's standard library (`math`, `random`, `sys`, `unittest`).
- **Comprehensive Unit Tests:** 20+ automated tests covering board state logic, horizontal/vertical/diagonal wins, draws, terminal states, immediate threats/wins, and 100 simulated games against random players with zero losses.

---

## 📁 Project Directory Structure

```text
Task-2-Tic-Tac-Toe-AI/
├── .venv/               # Isolated Python virtual environment (ignored by git)
├── game.py              # Tic-Tac-Toe board state, validation, and win/draw rules
├── ai.py                # Minimax algorithm with Alpha-Beta pruning & move ordering
├── main.py              # Interactive CLI game loop and player interface
├── test_game.py         # Automated unit tests using Python's unittest framework
├── requirements.txt     # Dependency specification (Python Standard Library only)
└── README.md            # Comprehensive project documentation
```

---

## 🚀 Setup & Execution (Windows)

All commands are tailored for Windows PowerShell and Command Prompt.

### 1. Navigate to the Task Directory

```powershell
cd D:\CODSOFT_TASKSNO\Task-2-Tic-Tac-Toe-AI
```

### 2. Set Up Virtual Environment

If the virtual environment is not already created:

```powershell
py -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\activate
```

*(You will see `(.venv)` in your terminal prompt).*

### 3. Verify Dependencies

Because this implementation uses the Python Standard Library exclusively:

```powershell
pip install -r requirements.txt
```

---

## 🎮 Running the Game

With the virtual environment activated:

```powershell
python main.py
```

Or run directly without manual activation:

```powershell
.\.venv\Scripts\python.exe main.py
```

*(or `py main.py`)*

### Example Gameplay

```text
============================================================
      CodSoft Artificial Intelligence Internship
             Task 2: Tic-Tac-Toe AI Agent        
============================================================
  * Architecture: Unbeatable Minimax with Alpha-Beta Pruning
  * Coordinates: Enter numbers 1 through 9 to choose a cell:
       1 | 2 | 3
      ---+---+---
       4 | 5 | 6
      ---+---+---
       7 | 8 | 9
============================================================

Choose your marker ('X' plays first, 'O' plays second) [X/O, default: X]: X

You are playing as [X]. AI is playing as [O].
Let the game begin!


   1 | 2 | 3 
  ---+---+---
   4 | 5 | 6 
  ---+---+---
   7 | 8 | 9 

Your turn [X]:
Enter your move (1-9) or 'q' to quit: 1

   X | 2 | 3 
  ---+---+---
   4 | 5 | 6 
  ---+---+---
   7 | 8 | 9 

AI [O] is thinking...
AI chose cell 5.


   X | 2 | 3 
  ---+---+---
   4 | O | 6 
  ---+---+---
   7 | 8 | 9 

Your turn [X]:
Enter your move (1-9) or 'q' to quit: 9
...
----------------------------------------
  Result: IT'S A DRAW! Well played!
----------------------------------------
```

---

## 🧪 Running Unit Tests

Automated unit tests in `test_game.py` verify:
1. **Board Logic:** Initial board state, valid moves, out-of-bounds rejection, occupied cell guards, undo operations, clone isolation, and coordinate board rendering.
2. **Win & Draw Engine:** All 3 row wins, all 3 column wins, both diagonal wins, full-board draw detection, and non-terminal state identification.
3. **Minimax & AI Optimality:**
   - Immediate winning moves (taking winning cell in row, column, or diagonal).
   - Immediate blocking moves (denying human opponent immediate winning threats).
   - Preference for winning over blocking when both exist simultaneously.
   - Optimal opening move on empty board.
   - Minimax vs Minimax self-play (always ends in a draw).
   - 100 simulated matches against randomized opponents (50 games as X, 50 games as O) with **0 losses** (100% win or draw rate).

To execute the test suite:

```powershell
.\.venv\Scripts\python.exe -m unittest test_game.py -v
```

or with an activated environment:

```powershell
python -m unittest test_game.py -v
```
