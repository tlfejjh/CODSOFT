import math

# Function to initialize the board
def initialize_board():
    return [' ' for _ in range(9)]

# Function to print the board
def print_board(board):
    for row in [board[i*3:(i+1)*3] for i in range(3)]:
        print('| ' + ' | '.join(row) + ' |')

# Function to check for winner
def check_winner(board, player):
    win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),  # horizontal
                      (0, 3, 6), (1, 4, 7), (2, 5, 8),  # vertical
                      (0, 4, 8), (2, 4, 6)]             # diagonal
    for cond in win_conditions:
        if board[cond[0]] == board[cond[1]] == board[cond[2]] == player:
            return True
    return False

# Function to check for draw
def check_draw(board):
    return ' ' not in board

# Minimax algorithm with Alpha-Beta pruning
def minimax(board, depth, is_maximizing, alpha, beta):
    if check_winner(board, 'O'):
        return 1
    elif check_winner(board, 'X'):
        return -1
    elif check_draw(board):
        return 0

    if is_maximizing:
        max_eval = -math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                eval = minimax(board, depth + 1, False, alpha, beta)
                board[i] = ' '
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        return max_eval
    else:
        min_eval = math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                eval = minimax(board, depth + 1, True, alpha, beta)
                board[i] = ' '
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
        return min_eval

# Function to determine the AI move
def ai_move(board):
    best_score = -math.inf
    best_move = None
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            score = minimax(board, 0, False, -math.inf, math.inf)
            board[i] = ' '
            if score > best_score:
                best_score = score
                best_move = i
    return best_move

# Function to handle human move
def human_move(board):
    while True:
        move = input("Enter your move (1-9): ")
        if move.isdigit():
            move = int(move) - 1
            if 0 <= move < 9 and board[move] == ' ':
                return move
        print("Invalid move. Please try again.")

# Main function to run the game
def tic_tac_toe():
    board = initialize_board()
    print("Welcome to Tic-Tac-Toe!")
    print_board(board)
    
    while True:
        # Human move
        move = human_move(board)
        board[move] = 'X'
        print_board(board)
        
        if check_winner(board, 'X'):
            print("Congratulations! You win!")
            break
        if check_draw(board):
            print("It's a draw!")
            break
        
        # AI move
        print("AI is making a move...")
        move = ai_move(board)
        board[move] = 'O'
        print_board(board)
        
        if check_winner(board, 'O'):
            print("AI wins! Better luck next time.")
            break
        if check_draw(board):
            print("It's a draw!")
            break

if __name__ == "__main__":
    tic_tac_toe()
