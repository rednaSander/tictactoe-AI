# tictactoe_nxn.py
import tkinter as tk
from tkinter import messagebox, ttk
import random
from bots_nxn import RuleBasedBot, RandomBot, MinimaxBot_V0, MinimaxBot_V1, MinimaxBot_V2, MinimaxBot_V3, MinimaxBot_V4, MonteCarloTreeSearchBot  
class TicTacToe:
    def __init__(self, size=4, win_len=4):
        self.size = size
        self.win_len = win_len
        self.window = tk.Tk()
        self.window.title(f'{size}×{size} Tic Tac Toe')
        self.window.geometry(f"{size*120}x{size*120+200}")

        # game state
        self.current_player = random.choice(["X", "O"])
        self.board = [""] * (size*size)
        self.buttons = []

        # Score tracking
        self.score_x = 0
        self.score_o = 0

        # precompute all winning lines
        self.win_patterns = self._make_win_patterns()
        print([line for line in self.win_patterns])

        # Bot configuration
        self.player_x = None
        self.player_o = None

        self.create_bot_selection()
        self.create_score_display()
        self.create_board()
        self.check_bot_turn()

    def _make_win_patterns(self):
        """Generate all possible winning patterns for the board"""
        N, K = self.size, self.win_len
        idx = lambda r, c: r * N + c
        lines = []

        # Horizontal lines (rows) - slide the window across each row
        for r in range(N):
            for start_col in range(N - K + 1):
                line = [idx(r, start_col + i) for i in range(K)]
                lines.append(line)

        # Vertical lines (columns) - slide the window down each column
        for c in range(N):
            for start_row in range(N - K + 1):
                line = [idx(start_row + i, c) for i in range(K)]
                lines.append(line)

        # Diagonal lines (↘ direction) - top-left to bottom-right
        for start_row in range(N - K + 1):
            for start_col in range(N - K + 1):
                line = [idx(start_row + i, start_col + i) for i in range(K)]
                lines.append(line)

        # Anti-diagonal lines (↗ direction) - top-right to bottom-left
        for start_row in range(N - K + 1):
            for start_col in range(K - 1, N):
                line = [idx(start_row + i, start_col - i) for i in range(K)]
                lines.append(line)

        return lines

    def run(self):
        self.window.mainloop()

    def create_bot_selection(self):
        frame = tk.Frame(self.window)
        frame.grid(row=0, column=0, columnspan=self.size, pady=10, sticky="ew")
        
        # Updated bot options to include MCTS
        bot_options = ("Human", "Random Bot", "Rule-based Bot", "MinimaxBot_V0", "MinimaxBot_V1", 
                      "MinimaxBot_V2", "MinimaxBot_V3", "MinimaxBot_V4", "MonteCarloTreeSearch")
        
        # X selector
        tk.Label(frame, text="Player X:", font=("Arial", 10)).grid(row=0, column=0, padx=5)
        self.x_var = tk.StringVar(value="Human")
        x_combo = ttk.Combobox(frame, textvariable=self.x_var, 
                              values=bot_options, 
                              width=15)
        x_combo.grid(row=0, column=1, padx=5)
        x_combo.bind('<<ComboboxSelected>>', self.on_player_change)
        
        # O selector
        tk.Label(frame, text="Player O:", font=("Arial", 10)).grid(row=0, column=2, padx=5)
        self.o_var = tk.StringVar(value="Human")
        o_combo = ttk.Combobox(frame, textvariable=self.o_var, 
                              values=bot_options, 
                              width=15)
        o_combo.grid(row=0, column=3, padx=5)
        o_combo.bind('<<ComboboxSelected>>', self.on_player_change)

    def create_score_display(self):
        score_frame = tk.Frame(self.window)
        score_frame.grid(row=1, column=0, columnspan=self.size, pady=10)
        
        self.score_label = tk.Label(score_frame, text=self._score_text(), font=("Arial", 14))
        self.score_label.pack(side=tk.LEFT, padx=10)
        
        reset_btn = tk.Button(score_frame, text="Reset Score", command=self.reset_score)
        reset_btn.pack(side=tk.LEFT, padx=10)

    def _score_text(self):
        return f"Score – X: {self.score_x} | O: {self.score_o}"

    def on_player_change(self, event=None):
        mapping = {
            "Human": None,
            "Random Bot": RandomBot,
            "Rule-based Bot": RuleBasedBot,
            "MinimaxBot_V0": MinimaxBot_V0,
            "MinimaxBot_V1": MinimaxBot_V1,
            "MinimaxBot_V2": MinimaxBot_V2,
            "MinimaxBot_V3": MinimaxBot_V3,
            "MinimaxBot_V4": MinimaxBot_V4,
            "MonteCarloTreeSearch": MonteCarloTreeSearchBot,  # Add MCTS bot
        }
        
        cls_x = mapping[self.x_var.get()]
        # Try to pass the game configuration to the bots, fall back to basic constructor
        if cls_x:
            try:
                self.player_x = cls_x("X", self.size, self.win_len, self.win_patterns)
            except TypeError:
                # Fallback for bots that don't accept all parameters
                self.player_x = cls_x("X")
                # Manually set the attributes
                self.player_x.size = self.size
                self.player_x.win_len = self.win_len
                self.player_x.win_patterns = self.win_patterns
        else:
            self.player_x = None
        
        cls_o = mapping[self.o_var.get()]
        if cls_o:
            try:
                self.player_o = cls_o("O", self.size, self.win_len, self.win_patterns)
            except TypeError:
                # Fallback for bots that don't accept all parameters
                self.player_o = cls_o("O")
                # Manually set the attributes
                self.player_o.size = self.size
                self.player_o.win_len = self.win_len
                self.player_o.win_patterns = self.win_patterns
        else:
            self.player_o = None
        
        self.check_bot_turn()

    def create_board(self):
        # Create the game board buttons
        for r in range(self.size):
            for c in range(self.size):
                idx = r * self.size + c
                btn = tk.Button(self.window, text="", font=("Arial", 16, "bold"),
                               width=3, height=1,
                               command=lambda i=idx: self.human_move(i))
                btn.grid(row=r+2, column=c, padx=1, pady=1, sticky="nsew")
                self.buttons.append(btn)

        # Configure grid weights for proper resizing
        for i in range(self.size):
            self.window.grid_rowconfigure(i+2, weight=1)
            self.window.grid_columnconfigure(i, weight=1)

        # Status label
        self.status_label = tk.Label(self.window, text=f"Player {self.current_player}'s turn", 
                                    font=("Arial", 12))
        self.status_label.grid(row=self.size+2, column=0, columnspan=self.size, pady=10)
        
        # New game button
        new_game_btn = tk.Button(self.window, text="New Game", font=("Arial", 12),
                                command=self.reset_game)
        new_game_btn.grid(row=self.size+3, column=0, columnspan=self.size, pady=10)

    def human_move(self, idx):
        """Handle human player moves"""
        bot = self.player_x if self.current_player == "X" else self.player_o
        if bot: 
            return  # It's a bot's turn, ignore human clicks
        self.make_move(idx)

    def make_move(self, idx):
        """Make a move at the given index"""
        if self.board[idx]:  # Position already taken
            return False
        
        # Make the move
        self.board[idx] = self.current_player
        self.buttons[idx].config(text=self.current_player)

        # Check for win
        if self.check_winner():
            if self.current_player == "X": 
                self.score_x += 1
            else:                   
                self.score_o += 1
            self.update_score_display()
            messagebox.showinfo("Game Over", f"Player {self.current_player} wins!")
            self.reset_game(preserve_score=True)
            return True

        # Check for tie
        if all(self.board):
            messagebox.showinfo("Game Over", "It's a tie!")
            self.reset_game(preserve_score=True)
            return True

        # Switch players
        self.current_player = "O" if self.current_player == "X" else "X"
        self.status_label.config(text=f"Player {self.current_player}'s turn")
        self.window.after(300, self.check_bot_turn)  # Slightly longer delay for 4x4
        return True

    def check_bot_turn(self):
        """Check if it's a bot's turn and make the move"""
        bot = self.player_x if self.current_player == "X" else self.player_o
        if bot:
            move = bot.make_move(self.board.copy())
            if move is not None:
                self.make_move(move)

    def check_winner(self):
        """Check if current player has won"""
        for pattern in self.win_patterns:
            # Get the values at the pattern positions
            values = [self.board[i] for i in pattern]
            # Check if all positions have the same non-empty value
            if values[0] and all(v == values[0] for v in values):
                if values[0] == self.current_player:
                    return True
        return False

    def update_score_display(self):
        """Update the score display"""
        self.score_label.config(text=self._score_text())

    def reset_score(self):
        """Reset the score to 0-0"""
        self.score_x = self.score_o = 0
        self.update_score_display()
        messagebox.showinfo("Score Reset", "Score reset to 0–0!")

    def reset_game(self, preserve_score=False):
        """Reset the game to initial state"""
        self.board = [""] * (self.size * self.size)
        for btn in self.buttons:
            btn.config(text="")
        
        if not preserve_score:
            self.score_x = self.score_o = 0
            self.update_score_display()
        
        self.current_player = random.choice(["X", "O"])
        self.status_label.config(text=f"Player {self.current_player}'s turn")
        # Recreate bots with current configuration when game resets
        self.on_player_change()
        self.window.after(300, self.check_bot_turn)

if __name__ == "__main__":
    TicTacToe(size=4, win_len=4).run()



