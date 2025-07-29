# Tic-Tac-Toe AI: From Zero to Superhuman

A comprehensive implementation showcasing the progression of AI algorithms for Tic-Tac-Toe, from simple rule-based systems to sophisticated search algorithms that achieve near-superhuman performance.

## 🎯 Project Overview

This project demonstrates fundamental AI concepts through practical implementation, starting with basic heuristics and evolving to advanced algorithms capable of perfect play on any board size. Each implementation builds upon the previous one, illustrating key concepts in game AI development.

## 🚀 Features

- **Multiple AI Algorithms**: Six different bot implementations with increasing sophistication
- **Scalable Design**: Works on any board size (3×3 to 10×10) with customizable win conditions
- **Interactive GUI**: Tkinter-based interface for human vs. AI and bot vs. bot gameplay
- **Performance Analysis**: Built-in timing and statistics for algorithm comparison
- **Educational Focus**: Clear progression from basic to advanced AI techniques

## 🤖 Algorithms Implemented

### 1. Rule-Based Bot
- Strategic heuristics: win, block, take center, take corners
- Demonstrates classical expert systems from the 1960s-70s

### 2. Minimax V0 (Basic)
- Pure minimax algorithm with depth limiting
- Perfect mathematical play within search depth

### 3. Minimax V1 (Iterative Deepening)
- Time-managed progressive search
- Optimal use of available thinking time

### 4. Minimax V2 (Alpha-Beta Pruning)
- Eliminates unnecessary search branches
- Significant performance improvement while maintaining optimality

### 5. Minimax V3 (Alpha-Beta + Iterative)
- Combines pruning with time management
- Gold standard for perfect-information games

### 6. Minimax V4 (+ Transposition Tables)
- Memoization of previously computed positions
- Avoids redundant calculations across different move sequences

### 7. Monte Carlo Tree Search (MCTS)
- Statistical sampling instead of exhaustive search
- Handles larger boards where minimax becomes impractical
- Uses UCT formula for exploration vs. exploitation balance

## 📋 Requirements

```
Python 3.6+
tkinter (usually included with Python)
```

## 🏃‍♂️ Quick Start

### Basic Usage
```bash
# Clone the repository
git clone https://github.com/yourusername/tictactoe-ai.git
cd tictactoe-ai

# Run with default 4×4 board
python tictactoe_nxn.py
```

### Custom Configuration
```python
# 5×5 board with 4-in-a-row to win
TicTacToe(size=5, win_len=4).run()

# Classic 3×3
TicTacToe(size=3, win_len=3).run()
```

## 🎮 How to Play

1. **Launch the game**: Run `python tictactoe_nxn.py`
2. **Select players**: Choose from dropdown menus (Human or various AI bots)
3. **Play**: Click cells to make moves, or watch bots play each other
4. **Experiment**: Try different board sizes and bot combinations

## 🏗️ Project Structure

```
├── tictactoe_nxn.py          # Main game interface and logic
├── bots_nxn.py               # All AI bot implementations
└── README.md                 # This file
```

## 🧠 Educational Value

This project illustrates key AI concepts:

- **Search Algorithms**: Tree search, backtracking, pruning
- **Optimization**: Alpha-beta pruning, iterative deepening, memoization
- **Statistical Methods**: Monte Carlo simulation, UCT selection
- **Algorithm Analysis**: Time complexity, space complexity, trade-offs

## 📊 Performance Comparison

| Algorithm | 3×3 Performance | 4×4 Performance | 5×5+ Performance |
|-----------|----------------|----------------|------------------|
| Rule-Based | Perfect | Good | Moderate |
| Minimax V0 | Perfect | Limited depth | Too slow |
| Minimax V2 | Perfect | Strong | Limited |
| Minimax V4 | Perfect | Very strong | Good |
| MCTS | Perfect | Near-perfect | very strong |

## 🎥 Video Tutorial

Watch the complete development process and algorithm explanations in my YouTube video: [Zero to Superhuman: Tic-Tac-Toe AI](https://www.youtube.com/watch/your-video-link)

## 🔍 Algorithm Details

### Minimax with Alpha-Beta Pruning
The minimax algorithm explores the complete game tree, assuming both players play optimally. Alpha-beta pruning eliminates branches that cannot affect the final decision, dramatically improving efficiency.

### Monte Carlo Tree Search
MCTS uses statistical sampling to evaluate positions, making it particularly effective for larger boards where exhaustive search becomes impractical. It balances exploration of new moves with exploitation of promising positions.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new AI algorithms
- Implement better evaluation functions
- Improve the GUI
- Add performance benchmarks
- Enhance documentation

## 🔗 Links

- **Play Online**: https://users.ugent.be/~saheyndr/tictactoe_ai.html
- **YouTube Channel**: [Educational AI Videos](https://youtube.com/@your-channel)
