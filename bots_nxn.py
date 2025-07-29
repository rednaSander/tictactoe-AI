# bots_nxn.py
import random
import math

class Bot:
    """Base Class for all bots"""
    def __init__(self, symbol, size=3, win_len=3, win_patterns=None):
        self.symbol = symbol  # 'X' or 'O'
        self.size = size  # Board size (3 for 3x3, 4 for 4x4, etc.)
        self.win_len = win_len  # Length needed to win
        self.win_patterns = win_patterns or []  # Winning patterns from the game

    def make_move(self, board):
        """Takes the current board state and returns the chosen move position"""
        raise NotImplementedError("Subclasses must implement make_move")

class RuleBasedBot(Bot):
    """Bot that follows strategic rules (adapted for any grid size)"""
    
    def __init__(self, symbol, size=3, win_len=3, win_patterns=None):
        super().__init__(symbol, size, win_len, win_patterns)

    def make_move(self, board):
        # Rule 1: If we can make a winning move, make it
        move = self._find_winning_move(board, self.symbol)
        if move is not None:
            return move
        
        # Rule 2: If we can block the opponent from winning, block the opponent
        opponent = "O" if self.symbol == 'X' else 'X'
        move = self._find_winning_move(board, opponent)
        if move is not None:
            return move
        
        # Rule 3: If center available, take it (works for any odd-sized grid)
        if self.size % 2 == 1:
            center_pos = self.size * self.size // 2
            if board[center_pos] == "":
                return center_pos
        
        # Rule 4: If corners are available, take one
        corners = self._get_corners()
        available_corners = [pos for pos in corners if board[pos] == ""]
        if available_corners:
            return random.choice(available_corners)
        
        # Rule 5: Take any remaining spot
        available = [i for i in range(len(board)) if board[i] == ""]
        return random.choice(available) if available else None
    
    def _get_corners(self):
        """Get corner positions for any grid size"""
        return [
            0,                              # top-left
            self.size - 1,                  # top-right
            self.size * (self.size - 1),    # bottom-left
            self.size * self.size - 1       # bottom-right
        ]
    
    def _find_winning_move(self, board, player):
        """Find a move that would win the game for the given player"""
        opponent = "O" if player == 'X' else 'X'
        
        for pattern in self.win_patterns:
            # Count how many positions the player already has in this winning pattern
            player_count = 0
            empty_positions = []
            
            for pos in pattern:
                if board[pos] == player:
                    player_count += 1
                elif board[pos] == "":
                    empty_positions.append(pos)

            # If player has (win_len-1) in a row and there's 1 empty spot, that's the winning move
            if player_count == (self.win_len - 1) and len(empty_positions) == 1:
                return empty_positions[0]
            
        return None

class RandomBot(Bot):
    """A bot that makes random moves (works for any grid size)"""
    
    def __init__(self, symbol, size=3, win_len=3, win_patterns=None):
        super().__init__(symbol, size, win_len, win_patterns)

    def make_move(self, board):
        available = [i for i in range(len(board)) if board[i] == ""]
        return random.choice(available) if available else None

# Copy your exact MinimaxBot_V0 code here and let's add debugging
class MinimaxBot_V0(Bot):
    """
    Exact copy of your original bot with step-by-step debugging
    """
    
    def __init__(self, symbol, size=5, win_len=5, win_patterns=None, max_depth=None):
        self.symbol = symbol
        self.size = size
        self.win_len = win_len
        self.win_patterns = win_patterns
        self.verbose = False
        
        # Set default max_depth based on board size if not specified
        if max_depth is None:
            if size <= 3:
                self.max_depth = 9  # Can search full tree for 3x3
            elif size == 4:
                self.max_depth = 5 # Reasonable for 4x4
            elif size == 5:
                self.max_depth = 4  # Conservative for 5x5
            else:
                self.max_depth = 1
        else:
            self.max_depth = max_depth
        
        print(f"=== Bot initialized ===")
        print(f"Symbol: {self.symbol}")
        print(f"Size: {self.size}")
        print(f"Win length: {self.win_len}")
        print(f"Max depth: {self.max_depth}")
        print(f"Win patterns: {self.win_patterns}")
        print()

    def make_move(self, board):
        print(f"\n=== {self.symbol}'s turn ===")
        print(f"Board: {board}")  # Just show first 5 positions
        
        self.nodes_evaluated = 0 
        score, move = self.minimax(board, self.symbol, depth=0)
        
        print(f"Chosen: {move} (score: {score})")
        return move

    def minimax(self, board, current_player, depth, verbose=True):
        """Your original minimax with minimal logging"""
        self.nodes_evaluated += 1
        
        # Terminal checks
        winner = self.check_winner(board)
        if winner:
            result = 1 if winner == self.symbol else -1
            if depth <= 1 and verbose:
                print(f"  Depth {depth}: Winner {winner} → score {result}")
            return result, None
        
        # Check if board is full (tie)
        if "" not in board:
            return 0, None
        
        # Check if we've reached depth limit
        if depth >= self.max_depth:
            if depth <= 1 and verbose:
                print(f"  Depth {depth}: Hit depth limit → score 0")
            return 0, None

        best_move = None
        available_moves = [i for i in range(len(board)) if board[i] == ""]
        
        # If it's our turn, we maximize; else we minimize
        if current_player == self.symbol:
            best_score = -float('inf')
            for move in available_moves:
                new_board = board.copy()
                new_board[move] = current_player
                next_player = "O" if current_player == "X" else "X"
                
                score, _ = self.minimax(new_board, next_player, depth + 1)
                
                if depth == 0:  # Only log top level
                    if verbose: 
                        print(f"Move {move}: score {score}")
                
                if score > best_score:
                    best_score, best_move = score, move
        else:
            best_score = float('inf')
            for move in available_moves:
                new_board = board.copy()
                new_board[move] = current_player
                next_player = "O" if current_player == "X" else "X"
                
                score, _ = self.minimax(new_board, next_player, depth + 1)
                
                if score < best_score:
                    best_score, best_move = score, move

        return best_score, best_move

    def check_winner(self, board):
        """Your original check_winner with minimal debugging"""
        if not self.win_patterns:
            print("❌ No win patterns!")
            return None
            
        for pattern in self.win_patterns:
            line_values = [board[pos] for pos in pattern]
            if line_values[0] and all(v == line_values[0] for v in line_values):
                return line_values[0]
        return None

    def print_board(self, board):
        """Helper to visualize the board"""
        for r in range(self.size):
            row = []
            for c in range(self.size):
                idx = r * self.size + c
                cell = board[idx] if board[idx] else "."
                row.append(f"{cell:2}")
            print(f"  {' '.join(row)}")
    
# only iterative deepening
import time

class MinimaxBot_V1(Bot):
    """
    Bot that uses basic minimax with iterative deepening
    """
    
    def __init__(self, symbol, size=5, win_len=5, win_patterns=None, time_limit=5.0):
        super().__init__(symbol, size, win_len, win_patterns)

        self.verbose = True
        self.time_limit = time_limit
        
        if self.verbose:
            print(f"MinimaxBot_V1 initialized with time_limit={self.time_limit}s")

    def make_move(self, board):
        """Make a move using iterative deepening"""
        self.nodes_evaluated = 0
        start_time = time.time()
        
        best_move = None
        best_score = -float('inf')
        max_depth = 1
        
        # Keep searching deeper until time runs out
        while True:
            try:
                # Check if we have enough time for another iteration
                elapsed = time.time() - start_time
                if elapsed >= self.time_limit * 0.8:  # Use 80% of time limit as margin
                    break
                
                # Search at current depth
                score, move = self.minimax_with_timeout(board, self.symbol, max_depth, start_time)
                
                # Update best move if we found a better one
                if move is not None:
                    best_move = move
                    best_score = score
                    
                    if self.verbose:
                        elapsed = time.time() - start_time
                        print(f"Depth {max_depth}: score={best_score}, move={best_move}, time={elapsed:.3f}s, nodes={self.nodes_evaluated}")
                    
                    # If we found a winning move, no need to search deeper
                    if best_score == 1:
                        break
                    
                    # Check if we've reached all terminal positions 
                    empty_positions = len([pos for pos, val in enumerate(board) if val == ""])
                    if max_depth > empty_positions:
                        if self.verbose:
                            print(f"Searched deeper than remaining moves ({empty_positions}), stopping search")
                        break
                
                max_depth += 1
                
            except TimeoutError:
                # Time limit reached
                break
        
        if self.verbose:
            total_time = time.time() - start_time
            print(f"MinimaxBot_V1 chose position {best_move} (depth reached: {max_depth}, total time: {total_time:.3f}s, total nodes: {self.nodes_evaluated})")
        
        return best_move

    def minimax_with_timeout(self, board, current_player, max_depth, start_time, depth=0):
        """Basic minimax with timeout check"""
        # Check timeout
        if time.time() - start_time > self.time_limit:
            raise TimeoutError("Time limit exceeded")
        
        self.nodes_evaluated += 1 
        
        # Terminal checks
        winner = self.check_winner(board)
        if winner:
            return 1 if winner == self.symbol else -1, None
        
        # Check if board is full (tie) or depth limit reached
        if "" not in board or depth >= max_depth:
            return 0, None

        # Determine if we're maximizing or minimizing
        is_maximizing = current_player == self.symbol
        best_score = -float('inf') if is_maximizing else float('inf')
        best_move = None
        
        available_moves = [i for i in range(len(board)) if board[i] == ""]
        next_player = "O" if current_player == "X" else "X"
        
        for move in available_moves:
            new_board = board.copy()
            new_board[move] = current_player
            
            score, _ = self.minimax_with_timeout(new_board, next_player, max_depth, start_time, depth + 1)
            
            if is_maximizing:
                if score > best_score:
                    best_score, best_move = score, move
            else:
                if score < best_score:
                    best_score, best_move = score, move

        return best_score, best_move

    def check_winner(self, board):
        """Check if there's a winner on the board using the provided win patterns"""
        for pattern in self.win_patterns:
            line_values = [board[pos] for pos in pattern]
            if line_values[0] and all(v == line_values[0] for v in line_values):
                return line_values[0]
        return None

    
# + alpha beta pruning 

class MinimaxBot_V2(Bot):
    """Bot that uses the minimax with alpha-beta pruning"""
    
    def __init__(self, symbol, size=5, win_len=5, win_patterns=None, max_depth=None):
        super().__init__(symbol, size, win_len, win_patterns)
        
        self.verbose = False

        # Set default max_depth based on board size if not specified
        if max_depth is None:
            if size <= 3:
                self.max_depth = 9  # Can search full tree for 3x3
            elif size == 4:
                self.max_depth = 9  # Reasonable for 4x4
            elif size == 5:
                self.max_depth = 7  # Conservative for 5x5
            else:
                self.max_depth = 3  # Very conservative for larger boards
        else:
            self.max_depth = max_depth
        if self.verbose: 
            print(f"MinimaxBot_abp initialized with max_depth={self.max_depth}")

    def make_move(self, board):
        self.nodes_evaluated = 0  # Reset counter
        _, move = self.minimax(board, self.symbol, depth=0)
        if self.verbose:
            print(f"MinimaxBot_abp chose position {move} (evaluated {self.nodes_evaluated} game states)")
        return move

    def minimax(self, board, current_player, alpha=-float('inf'), beta=float('inf'), depth=0):
        self.nodes_evaluated += 1
        
        # Terminal checks
        winner = self.check_winner(board)
        if winner:
            # +1 for our win, -1 for opponent's win
            return (1 if winner == self.symbol else -1), None
        if "" not in board:
            return 0, None  # tie
        
        # Depth limit check
        if depth >= self.max_depth:
            return 0, None  # Return neutral score when depth limit reached

        best_move = None
        is_maximizing = current_player == self.symbol
        best_score = -float('inf') if is_maximizing else float('inf')
        
        # Use correct board size instead of hard-coded 9
        board_size = len(board)
        for i in range(board_size):
            if board[i] == "":
                new_board = board.copy()
                new_board[i] = current_player
                next_player = "O" if current_player == "X" else "X"
                score, _ = self.minimax(new_board, next_player, alpha, beta, depth + 1)

                if is_maximizing:
                    if score > best_score:
                        best_score, best_move = score, i
                    alpha = max(alpha, score)
                    # Early termination: found the best possible score
                    if score == 1:
                        break
                else:
                    if score < best_score:
                        best_score, best_move = score, i
                    beta = min(beta, score)
                    # Early termination: found the worst possible score
                    if score == -1:
                        break
                
                # Alpha-beta cutoff
                if beta <= alpha:
                    break

        return best_score, best_move

    def check_winner(self, board):
        """Check if there's a winner on the board using the provided win patterns"""
        for pattern in self.win_patterns:
            line_values = [board[pos] for pos in pattern]
            if line_values[0] and all(v == line_values[0] for v in line_values):
                return line_values[0]
        return None
    

    def minimax(self, board, current_player, alpha=-float('inf'), beta=float('inf'), depth=0):
        self.nodes_evaluated += 1
        
        # Terminal checks
        winner = self.check_winner(board)
        if winner:
            # +1 for our win, -1 for opponent's win
            return (1 if winner == self.symbol else -1), None
        if "" not in board:
            return 0, None  # tie
        
        # Depth limit check
        if depth >= self.max_depth:
            return 0, None  # Return neutral score when depth limit reached

        best_move = None
        is_maximizing = current_player == self.symbol
        best_score = -float('inf') if is_maximizing else float('inf')
        
        # Use correct board size instead of hard-coded 9
        board_size = len(board)
        for i in range(board_size):
            if board[i] == "":
                new_board = board.copy()
                new_board[i] = current_player
                next_player = "O" if current_player == "X" else "X"
                score, _ = self.minimax(new_board, next_player, alpha, beta, depth + 1)

                if is_maximizing:
                    if score > best_score:
                        best_score, best_move = score, i
                    alpha = max(alpha, score)
                    # Early termination: found the best possible score
                    if score == 1:
                        break
                else:
                    if score < best_score:
                        best_score, best_move = score, i
                    beta = min(beta, score)
                    # Early termination: found the worst possible score
                    if score == -1:
                        break
                
                # Alpha-beta cutoff
                if beta <= alpha:
                    break

        return best_score, best_move
    
import time

# + iterative deepening

class MinimaxBot_V3(Bot):
    """Bot that uses minimax with alpha-beta pruning and iterative deepening"""
    
    def __init__(self, symbol, size=5, win_len=5, win_patterns=None, time_limit=5.0):
        super().__init__(symbol, size, win_len, win_patterns)
        
        self.verbose = False
        self.time_limit = time_limit  # Time limit in seconds
        self.nodes_evaluated = 0

        if self.verbose: 
            print(f"MinimaxBot_V3 initialized with time_limit={self.time_limit}s")

    def make_move(self, board):
        """Make a move using iterative deepening"""
        self.nodes_evaluated = 0
        start_time = time.time()
        
        best_move = None
        best_score = -float('inf')
        depth = 1
        
        # Keep searching deeper until time runs out
        while True:
            try:
                # Check if we have enough time for another iteration
                elapsed = time.time() - start_time
                if elapsed >= self.time_limit * 0.8:  # Use 80% of time limit as safety margin
                    break
                
                # Track nodes before this iteration
                nodes_before = self.nodes_evaluated
                
                # Search at current depth
                score, move = self.minimax_with_timeout(board, self.symbol, depth, start_time)
                
                # Calculate nodes explored in this iteration
                nodes_this_iteration = self.nodes_evaluated - nodes_before
                
                # Update best move if we found a better one
                if move is not None:
                    best_move = move
                    best_score = score
                    
                    if self.verbose:
                        elapsed = time.time() - start_time
                        print(f"Depth {depth}: score={score}, move={move}, time={elapsed:.3f}s, nodes={self.nodes_evaluated}")
                    
                    # If we found a winning move, no need to search deeper
                    if score == 1:
                        break
                    
                    # Check if we've reached all terminal positions 
                    # When near endgame, if we're searching deeper than remaining moves, we're done
                    empty_positions = len([pos for pos, val in enumerate(board) if val == ""])
                    if depth > empty_positions:
                        if self.verbose:
                            print(f"Searched deeper than remaining moves ({empty_positions}), stopping search")
                        break
                    
                    # Also check if very few new nodes were explored (tree is exhausted)
                    if depth > 1 and nodes_this_iteration <= empty_positions:
                        if self.verbose:
                            print(f"Reached all terminal positions at depth {depth}, stopping search")
                        break
                
                depth += 1
                
            except TimeoutError:
                # Time limit reached
                break
        
        if self.verbose:
            total_time = time.time() - start_time
            print(f"MinimaxBot_V3 chose position {best_move} (depth reached: {depth-1}, total time: {total_time:.3f}s, total nodes: {self.nodes_evaluated})")
        
        return best_move

    def minimax_with_timeout(self, board, current_player, max_depth, start_time, alpha=-float('inf'), beta=float('inf'), depth=0):
        """Minimax with timeout check"""
        # Check timeout
        if time.time() - start_time > self.time_limit:
            raise TimeoutError("Time limit exceeded")
        
        self.nodes_evaluated += 1
        
        # Terminal checks
        winner = self.check_winner(board)
        if winner:
            return (1 if winner == self.symbol else -1), None
        if "" not in board:
            return 0, None  # tie
        
        # Depth limit check (only for current iteration)
        if depth >= max_depth:
            return self.evaluate_position(board), None

        best_move = None
        is_maximizing = current_player == self.symbol
        best_score = -float('inf') if is_maximizing else float('inf')
        
        board_size = len(board)
        for i in range(board_size):
            if board[i] == "":
                new_board = board.copy()
                new_board[i] = current_player
                next_player = "O" if current_player == "X" else "X"
                
                try:
                    score, _ = self.minimax_with_timeout(new_board, next_player, max_depth, start_time, alpha, beta, depth + 1)
                except TimeoutError:
                    # If we timeout during this iteration, return what we have so far
                    raise

                if is_maximizing:
                    if score > best_score:
                        best_score, best_move = score, i
                    alpha = max(alpha, score)
                    # Early termination: found the best possible score
                    if score == 1:
                        break
                else:
                    if score < best_score:
                        best_score, best_move = score, i
                    beta = min(beta, score)
                    # Early termination: found the worst possible score
                    if score == -1:
                        break
                
                # Alpha-beta cutoff
                if beta <= alpha:
                    break

        return best_score, best_move

    def evaluate_position(self, board):
        """Simple position evaluation for non-terminal positions"""
        # For now, return neutral score
        # This could be enhanced with a more sophisticated evaluation function
        return 0

    def check_winner(self, board):
        """Check if there's a winner on the board using the provided win patterns"""
        for pattern in self.win_patterns:
            line_values = [board[pos] for pos in pattern]
            if line_values[0] and all(v == line_values[0] for v in line_values):
                return line_values[0]
        return None
    

# + transposition talbe: if the 2 players have the same board but this same board could have been created by a different order of moves then for the different orders we only have to look it up in our cache.
import time

class MinimaxBot_V4(Bot):
    """Bot that uses minimax with alpha-beta pruning, iterative deepening, and transposition table"""
    
    def __init__(self, symbol, size=5, win_len=5, win_patterns=None, time_limit=5.0):
        super().__init__(symbol, size, win_len, win_patterns)
        
        self.verbose = True
        self.time_limit = time_limit  # Time limit in seconds
        self.nodes_evaluated = 0
        self.transposition_table = {}  # Hash table for memoization
        self.cache_hits = 0  # Track how often we use cached results

        if self.verbose: 
            print(f"MinimaxBot_V4 initialized with time_limit={self.time_limit}s")

    def make_move(self, board):
        """Make a move using iterative deepening with transposition table"""
        self.nodes_evaluated = 0
        self.cache_hits = 0
        self.transposition_table.clear()  # Clear cache for new move
        start_time = time.time()
        
        best_move = None
        best_score = -float('inf')
        depth = 1
        
        # Keep searching deeper until time runs out
        while True:
            try:
                # Check if we have enough time for another iteration
                elapsed = time.time() - start_time
                if elapsed >= self.time_limit * 0.8:  # Use 80% of time limit as safety margin
                    break
                
                # Search at current depth
                score, move = self.minimax_with_timeout(board, self.symbol, depth, start_time)
                
                # Update best move if we found a better one
                if move is not None:
                    best_move = move
                    best_score = score
                    
                    if self.verbose:
                        elapsed = time.time() - start_time
                        print(f"Depth {depth}: score={score}, move={move}, time={elapsed:.3f}s, nodes={self.nodes_evaluated}, cache_hits={self.cache_hits}")
                    
                    # If we found a winning move, no need to search deeper
                    if score == 1:
                        break
                    
                    # Check if we've reached all terminal positions 
                    # When near endgame, if we're searching deeper than remaining moves, we're done
                    empty_positions = len([pos for pos, val in enumerate(board) if val == ""])
                    if depth > empty_positions:
                        if self.verbose:
                            print(f"Searched deeper than remaining moves ({empty_positions}), stopping search")
                        break
                
                depth += 1
                
            except TimeoutError:
                # Time limit reached
                break
        
        if self.verbose:
            total_time = time.time() - start_time
            cache_hit_rate = (self.cache_hits / max(self.nodes_evaluated, 1)) * 100
            print(f"MinimaxBot_V4 chose position {best_move} (depth reached: {depth-1}, total time: {total_time:.3f}s, total nodes: {self.nodes_evaluated}, cache hit rate: {cache_hit_rate:.1f}%)")
        
        return best_move

    def get_board_hash(self, board):
        """Generate a hash key for the current board position"""
        # Convert board to a tuple so it can be used as dictionary key
        return tuple(board)

    def minimax_with_timeout(self, board, current_player, max_depth, start_time, alpha=-float('inf'), beta=float('inf'), depth=0):
        """Minimax with timeout check and transposition table"""
        # Check timeout
        if time.time() - start_time > self.time_limit:
            raise TimeoutError("Time limit exceeded")
        
        # Generate hash for this position
        board_hash = self.get_board_hash(board)
        
        # Check transposition table
        if board_hash in self.transposition_table:
            cached_depth, cached_score, cached_move = self.transposition_table[board_hash]
            # Only use cached result if it was computed at equal or greater depth
            if cached_depth >= max_depth - depth:
                self.cache_hits += 1
                return cached_score, cached_move
        
        self.nodes_evaluated += 1
        
        # Terminal checks
        winner = self.check_winner(board)
        if winner:
            score = (1 if winner == self.symbol else -1)
            # Store terminal positions in cache
            self.transposition_table[board_hash] = (float('inf'), score, None)
            return score, None
        if "" not in board:
            # Store tie positions in cache
            self.transposition_table[board_hash] = (float('inf'), 0, None)
            return 0, None  # tie
        
        # Depth limit check (only for current iteration)
        if depth >= max_depth:
            score = self.evaluate_position(board)
            # Store evaluation in cache
            self.transposition_table[board_hash] = (max_depth - depth, score, None)
            return score, None

        best_move = None
        is_maximizing = current_player == self.symbol
        best_score = -float('inf') if is_maximizing else float('inf')
        
        board_size = len(board)
        for i in range(board_size):
            if board[i] == "":
                new_board = board.copy()
                new_board[i] = current_player
                next_player = "O" if current_player == "X" else "X"
                
                try:
                    score, _ = self.minimax_with_timeout(new_board, next_player, max_depth, start_time, alpha, beta, depth + 1)
                except TimeoutError:
                    # If we timeout during this iteration, return what we have so far
                    raise

                if is_maximizing:
                    if score > best_score:
                        best_score, best_move = score, i
                    alpha = max(alpha, score)
                    # Early termination: found the best possible score
                    if score == 1:
                        break
                else:
                    if score < best_score:
                        best_score, best_move = score, i
                    beta = min(beta, score)
                    # Early termination: found the worst possible score
                    if score == -1:
                        break
                
                # Alpha-beta cutoff
                if beta <= alpha:
                    break

        # Store result in transposition table
        self.transposition_table[board_hash] = (max_depth - depth, best_score, best_move)
        
        return best_score, best_move

    def evaluate_position(self, board):
        """Simple position evaluation for non-terminal positions"""
        # For now, return neutral score
        # This could be enhanced with a more sophisticated evaluation function
        return 0

    def check_winner(self, board):
        """Check if there's a winner on the board using the provided win patterns"""
        for pattern in self.win_patterns:
            line_values = [board[pos] for pos in pattern]
            if line_values[0] and all(v == line_values[0] for v in line_values):
                return line_values[0]
        return None
    

class MCTSNode:
    """A node in the Monte Carlo Tree Search tree."""
    
    def __init__(self, board, current_player, parent=None, move=None):
        """Initialize a new MCTS node.
        
        Args:
            board: Current board state
            current_player: Player to move in this state
            parent: Parent node (None for root)
            move: Move that led to this node
        """
        self.board = board.copy()
        self.current_player = current_player
        self.parent = parent
        self.move = move
        
        # Statistics for MCTS
        self.visits = 0
        self.wins = 0
        self.children = []
        self.untried_moves = self._get_available_moves()
    
    def _get_available_moves(self):
        """Return list of empty positions on the board."""
        return [i for i, cell in enumerate(self.board) if cell == ""]
    
    def is_fully_expanded(self):
        """Check if all possible moves from this position have been tried."""
        return len(self.untried_moves) == 0
    
    def is_game_over(self, win_patterns):
        """Check if the game is over (win or tie)."""
        return self._check_winner(win_patterns) is not None or "" not in self.board
    
    def _check_winner(self, win_patterns):
        """Return the winner ('X' or 'O') or None if no winner."""
        for pattern in win_patterns:
            cells = [self.board[pos] for pos in pattern]
            if cells[0] != "" and all(cell == cells[0] for cell in cells):
                return cells[0]
        return None
    
    def UCT(self, exploration_constant=1.6):
        """Calculate Upper Confidence Bound for Trees (UCT) value.
        
        UCT balances exploitation (choosing good moves) with exploration
        (trying new moves). Higher values indicate more promising nodes.
        """
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have highest priority
        
        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def select_best_child(self):
        """Select the child with highest UCT value."""
        return max(self.children, key=lambda child: child.UCT())
    
    def add_child(self, move):
        """Create and add a new child node for the given move."""
        # Create new board state
        new_board = self.board.copy()
        new_board[move] = self.current_player
        
        # Determine next player
        next_player = "O" if self.current_player == "X" else "X"
        
        # Create and add child
        child = MCTSNode(new_board, next_player, parent=self, move=move)
        self.children.append(child)
        self.untried_moves.remove(move)
        
        return child
    
    def simulate_random_game(self, win_patterns, bot_symbol):
        """Play out a random game from this position and return the result.
        
        Returns:
            1 if bot wins, -1 if bot loses, 0 for tie
        """
        # Copy board to avoid modifying node state
        sim_board = self.board.copy()
        sim_player = self.current_player
        
        while True:
            # Check for winner
            winner = self._check_winner_on_board(sim_board, win_patterns)
            if winner:
                return 1 if winner == bot_symbol else -1
            
            # Check for tie
            empty_cells = [i for i, cell in enumerate(sim_board) if cell == ""]
            if not empty_cells:
                return 0
            
            # Make random move
            move = random.choice(empty_cells)
            sim_board[move] = sim_player
            sim_player = "O" if sim_player == "X" else "X"
    
    def _check_winner_on_board(self, board, win_patterns):
        """Check for winner on a specific board state."""
        for pattern in win_patterns:
            cells = [board[pos] for pos in pattern]
            if cells[0] != "" and all(cell == cells[0] for cell in cells):
                return cells[0]
        return None
    
    def update_statistics(self, result, bot_symbol):
        """Update node statistics based on simulation result.
        
        This propagates the result up the tree to all ancestor nodes.
        """
        self.visits += 1
         
        # store wins from the perspective of the bot
        if self.current_player != bot_symbol: # player who just moved here is your bot, not current player!
            if result == 1:
                self.wins += 1
            elif result == 0:
                self.wins += 0.5
            # bot lost (result == -1), no reward for our bot

        else: # it was the opponents turn
            if result == -1:
                self.wins += 1
            elif result == 0:
                self.wins += 0.5

        # propagate to parent
        if self.parent:
            self.parent.update_statistics(result, bot_symbol)


class MonteCarloTreeSearchBot(Bot):
    """A bot that uses Monte Carlo Tree Search to play tic-tac-toe."""
    
    def __init__(self, symbol, size=5, win_len=5, win_patterns=None, 
                 time_limit=5.0, simulations=None):
        """Initialize MCTS bot.
        
        Args:
            symbol: Bot's symbol ('X' or 'O')
            size: Board size
            win_len: Length needed to win
            win_patterns: Winning patterns for the board
            time_limit: Time limit in seconds (default: 5.0)
            simulations: Fixed number of simulations (overrides time_limit if set)
        """
        super().__init__(symbol, size, win_len, win_patterns)
        self.time_limit = time_limit
        self.simulations = simulations
        self.verbose = True
        
        if self.verbose:
            limit = f"{simulations} simulations" if simulations else f"{time_limit}s"
            print(f"MCTS Bot initialized with {limit}")
    
    def make_move(self, board):
        """Select the best move using Monte Carlo Tree Search."""
        start_time = time.time()
        
        # Handle trivial case
        available_moves = [i for i, cell in enumerate(board) if cell == ""]
        if len(available_moves) == 1:
            if self.verbose:
                print(f"MCTS: Only one move available: {available_moves[0]}")
            return available_moves[0]
        
        # Initialize search tree
        root = MCTSNode(board, self.symbol)
        simulations_run = 0
        
        # Main MCTS loop
        while self._should_continue_search(start_time, simulations_run):
            # Phase 1: Selection - traverse tree to find promising leaf
            node = self.selection(root)
            
            # Phase 2: Expansion - add new child if possible
            if not node.is_game_over(self.win_patterns) and node.untried_moves:
                move = random.choice(node.untried_moves)
                node = node.add_child(move)
            
            # Phase 3: Simulation - play random game from leaf
            result = node.simulate_random_game(self.win_patterns, self.symbol)
            
            # Phase 4: Backpropagation - update statistics up the tree
            node.update_statistics(result, self.symbol)
            
            simulations_run += 1
        
        # Select best move based on most visits (most reliable)
        best_move = self._select_best_move(root, simulations_run, start_time)
        return best_move
    
    def _should_continue_search(self, start_time, simulations_run):
        """Check if search should continue based on limits."""
        if self.simulations:
            return simulations_run < self.simulations
        else:
            return time.time() - start_time < self.time_limit
    
    def selection(self, root):
        """Traverse tree to find a promising leaf node."""
        node = root
        while not node.is_game_over(self.win_patterns) and node.is_fully_expanded():
            node = node.select_best_child()
        return node
    
    def _select_best_move(self, root, simulations_run, start_time):
        """Select the best move and print statistics if verbose."""
        if not root.children:
            # Fallback to random if no simulations completed
            available = [i for i, cell in enumerate(root.board) if cell == ""]
            return random.choice(available)
        
        # Choose child with most visits
        best_child = max(root.children, key=lambda child: child.visits)
        
        if self.verbose:
            self._print_statistics(root, best_child, simulations_run, start_time)
        
        return best_child.move
    
    def _print_statistics(self, root, best_child, simulations_run, start_time):
        """Print search statistics for debugging."""
        elapsed = time.time() - start_time
        win_rate = best_child.wins / max(best_child.visits, 1) * 100
        
        print(f"\nMCTS Statistics:")
        print(f"  Simulations: {simulations_run} in {elapsed:.3f}s")
        print(f"  Best move: {best_child.move} (visited {best_child.visits} times)")
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"\nTop 5 moves:")
        
        # Show top moves sorted by visit count
        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        for child in sorted_children[:5]:
            win_rate = child.wins / max(child.visits, 1) * 100
            print(f"  Move {child.move}: {child.visits} visits, {win_rate:.1f}% win rate")