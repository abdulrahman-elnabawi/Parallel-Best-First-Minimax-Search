#include <iostream>
#include <vector>
#include <climits>
#include <omp.h>
#include <SFML/Graphics.hpp>
#include <algorithm>
#include <atomic>

using namespace std;

#define BOARD_SIZE 8
#define TILE_SIZE 80
#define MAX_DEPTH 3

// Chess pieces enum
enum Piece {
    EMPTY,
    WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING,
    BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING
};

struct Move {
    int fromRow, fromCol;
    int toRow, toCol;
    int score;  // Added score field for parallel sorting
};

bool is_white(Piece p) {
    return p >= WHITE_PAWN && p <= WHITE_KING;
}

bool is_black(Piece p) {
    return p >= BLACK_PAWN && p <= BLACK_KING;
}

// Piece-square tables for improved evaluation
const int pawn_table[64] = {
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
};

const int knight_table[64] = {
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
};

const int bishop_table[64] = {
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
};

const int rook_table[64] = {
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
};

const int queen_table[64] = {
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
};

const int king_table[64] = {
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
};

const int king_endgame_table[64] = {
        -50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
};

int evaluate_board(const vector<vector<Piece>>& board) {
    int score = 0;
    bool is_endgame = false;
    int material_sum = 0;

    // First pass: count material to determine if we're in endgame
#pragma omp parallel for reduction(+:material_sum) collapse(2)
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            Piece piece = board[row][col];
            switch (piece) {
                case WHITE_QUEEN: case BLACK_QUEEN: material_sum += 900; break;
                case WHITE_ROOK: case BLACK_ROOK: material_sum += 500; break;
                case WHITE_BISHOP: case WHITE_KNIGHT:
                case BLACK_BISHOP: case BLACK_KNIGHT: material_sum += 300; break;
                default: break;
            }
        }
    }
    is_endgame = material_sum <= 3000;  // Threshold for endgame

#pragma omp parallel for reduction(+:score) collapse(2)
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            Piece piece = board[row][col];
            int square_idx = row * 8 + col;
            int reverse_idx = (7 - row) * 8 + col;

            int piece_value = 0;
            int position_value = 0;

            switch (piece) {
                case WHITE_PAWN:
                    piece_value = 100;
                    position_value = pawn_table[square_idx];
                    break;
                case BLACK_PAWN:
                    piece_value = -100;
                    position_value = -pawn_table[reverse_idx];
                    break;
                case WHITE_KNIGHT:
                    piece_value = 320;
                    position_value = knight_table[square_idx];
                    break;
                case BLACK_KNIGHT:
                    piece_value = -320;
                    position_value = -knight_table[reverse_idx];
                    break;
                case WHITE_BISHOP:
                    piece_value = 330;
                    position_value = bishop_table[square_idx];
                    break;
                case BLACK_BISHOP:
                    piece_value = -330;
                    position_value = -bishop_table[reverse_idx];
                    break;
                case WHITE_ROOK:
                    piece_value = 500;
                    position_value = rook_table[square_idx];
                    break;
                case BLACK_ROOK:
                    piece_value = -500;
                    position_value = -rook_table[reverse_idx];
                    break;
                case WHITE_QUEEN:
                    piece_value = 900;
                    position_value = queen_table[square_idx];
                    break;
                case BLACK_QUEEN:
                    piece_value = -900;
                    position_value = -queen_table[reverse_idx];
                    break;
                case WHITE_KING:
                    piece_value = 20000;
                    position_value = is_endgame ?
                                     king_endgame_table[square_idx] :
                                     king_table[square_idx];
                    break;
                case BLACK_KING:
                    piece_value = -20000;
                    position_value = is_endgame ?
                                     -king_endgame_table[reverse_idx] :
                                     -king_table[reverse_idx];
                    break;
                default:
                    break;
            }
            score += piece_value + position_value;
        }
    }
    return score;
}

bool is_valid_position(int row, int col) {
    return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

void add_move_if_valid(vector<Move>& moves, const vector<vector<Piece>>& board,
                       int fromRow, int fromCol, int toRow, int toCol, bool isWhiteTurn) {
    if (!is_valid_position(toRow, toCol)) return;

    Piece target = board[toRow][toCol];
    if (target == EMPTY || (isWhiteTurn && is_black(target)) || (!isWhiteTurn && is_white(target))) {
        moves.push_back({fromRow, fromCol, toRow, toCol, 0});
    }
}

void add_sliding_moves(vector<Move>& moves, const vector<vector<Piece>>& board,
                       int fromRow, int fromCol, bool isWhiteTurn,
                       const vector<pair<int, int>>& directions) {
    for (const auto& dir : directions) {
        int row = fromRow + dir.first;
        int col = fromCol + dir.second;

        while (is_valid_position(row, col)) {
            Piece target = board[row][col];
            if (target == EMPTY) {
                moves.push_back({fromRow, fromCol, row, col, 0});
            } else {
                if ((isWhiteTurn && is_black(target)) || (!isWhiteTurn && is_white(target))) {
                    moves.push_back({fromRow, fromCol, row, col, 0});
                }
                break;
            }
            row += dir.first;
            col += dir.second;
        }
    }
}

vector<Move> generate_moves(const vector<vector<Piece>>& board, bool isWhiteTurn) {
    vector<Move> moves;
#pragma omp parallel
    {
        vector<Move> local_moves;
#pragma omp for collapse(2) nowait
        for (int row = 0; row < BOARD_SIZE; ++row) {
            for (int col = 0; col < BOARD_SIZE; ++col) {
                Piece piece = board[row][col];
                if (piece == EMPTY) continue;

                if ((isWhiteTurn && !is_white(piece)) || (!isWhiteTurn && !is_black(piece))) continue;

                switch (piece) {
                    case WHITE_PAWN: case BLACK_PAWN: {
                        int direction = (piece == WHITE_PAWN) ? -1 : 1;
                        int startRow = (piece == WHITE_PAWN) ? 6 : 1;

                        // Forward move
                        if (is_valid_position(row + direction, col) &&
                            board[row + direction][col] == EMPTY) {
                            local_moves.push_back({row, col, row + direction, col, 0});

                            // Double move from starting position
                            if (row == startRow && board[row + 2 * direction][col] == EMPTY) {
                                local_moves.push_back({row, col, row + 2 * direction, col, 0});
                            }
                        }

                        // Captures
                        for (int dc : {-1, 1}) {
                            if (is_valid_position(row + direction, col + dc)) {
                                Piece target = board[row + direction][col + dc];
                                if (target != EMPTY &&
                                    ((isWhiteTurn && is_black(target)) ||
                                     (!isWhiteTurn && is_white(target)))) {
                                    local_moves.push_back({row, col, row + direction, col + dc, 0});
                                }
                            }
                        }
                        break;
                    }

                    case WHITE_KNIGHT: case BLACK_KNIGHT: {
                        vector<pair<int, int>> knight_moves = {
                                {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                                {1, -2}, {1, 2}, {2, -1}, {2, 1}
                        };
                        for (const auto& move : knight_moves) {
                            add_move_if_valid(local_moves, board, row, col,
                                              row + move.first, col + move.second, isWhiteTurn);
                        }
                        break;
                    }

                    case WHITE_BISHOP: case BLACK_BISHOP: {
                        vector<pair<int, int>> bishop_dirs = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
                        add_sliding_moves(local_moves, board, row, col, isWhiteTurn, bishop_dirs);
                        break;
                    }

                    case WHITE_ROOK: case BLACK_ROOK: {
                        vector<pair<int, int>> rook_dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
                        add_sliding_moves(local_moves, board, row, col, isWhiteTurn, rook_dirs);
                        break;
                    }

                    case WHITE_QUEEN: case BLACK_QUEEN: {
                        vector<pair<int, int>> queen_dirs = {
                                {-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                                {0, 1}, {1, -1}, {1, 0}, {1, 1}
                        };
                        add_sliding_moves(local_moves, board, row, col, isWhiteTurn, queen_dirs);
                        break;
                    }

                    case WHITE_KING: case BLACK_KING: {
                        vector<pair<int, int>> king_moves = {
                                {-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                                {0, 1}, {1, -1}, {1, 0}, {1, 1}
                        };
                        for (const auto& move : king_moves) {
                            add_move_if_valid(local_moves, board, row, col,
                                              row + move.first, col + move.second, isWhiteTurn);
                        }
                        break;
                    }
                }
            }
        }
#pragma omp critical
        {
            moves.insert(moves.end(), local_moves.begin(), local_moves.end());
        }
    }
    return moves;
}

int minimax(vector<vector<Piece>> board, int depth, bool isWhiteTurn, int alpha, int beta) {
    if (depth == 0) {
        return evaluate_board(board);
    }

    vector<Move> moves = generate_moves(board, isWhiteTurn);
    if (moves.empty()) return evaluate_board(board);

    if (isWhiteTurn) {
        int maxEval = INT_MIN;
#pragma omp parallel for reduction(max:maxEval) schedule(dynamic)
        for (int i = 0; i < moves.size(); ++i) {
            auto newBoard = board;
            const auto& move = moves[i];
            newBoard[move.toRow][move.toCol] = newBoard[move.fromRow][move.fromCol];
            newBoard[move.fromRow][move.fromCol] = EMPTY;
            int eval = minimax(newBoard, depth - 1, false, alpha, beta);
            maxEval = max(maxEval, eval);
        }
        return maxEval;
    } else {
        int minEval = INT_MAX;
#pragma omp parallel for reduction(min:minEval) schedule(dynamic)
        for (int i = 0; i < moves.size(); ++i) {
            auto newBoard = board;
            const auto& move = moves[i];
            newBoard[move.toRow][move.toCol] = newBoard[move.fromRow][move.fromCol];
            newBoard[move.fromRow][move.fromCol] = EMPTY;
            int eval = minimax(newBoard, depth - 1, true, alpha, beta);
            minEval = min(minEval, eval);
        }
        return minEval;
    }
}

Move best_move(vector<vector<Piece>>& board, int depth) {
    vector<Move> moves = generate_moves(board, true);
    atomic<int> bestScore{INT_MIN};
    Move bestMove{-1, -1, -1, -1, INT_MIN};

#pragma omp parallel
    {
        Move localBestMove{-1, -1, -1, -1, INT_MIN};
        int localBestScore = INT_MIN;

#pragma omp for schedule(dynamic)
        for (int i = 0; i < moves.size(); ++i) {
            auto newBoard = board;
            Move& move = moves[i];

            newBoard[move.toRow][move.toCol] = newBoard[move.fromRow][move.fromCol];
            newBoard[move.fromRow][move.fromCol] = EMPTY;

            int score = minimax(newBoard, depth - 1, false, INT_MIN, INT_MAX);
            move.score = score;

            if (score > localBestScore) {
                localBestScore = score;
                localBestMove = move;
            }
        }

#pragma omp critical
        {
            if (localBestScore > bestScore) {
                bestScore = localBestScore;
                bestMove = localBestMove;
            }
        }
    }
    return bestMove;
}

void draw_board(sf::RenderWindow& window, vector<vector<Piece>>& board, Move bestMove) {
    sf::RectangleShape square(sf::Vector2f(TILE_SIZE, TILE_SIZE));
    sf::Font font;
    font.loadFromFile("C:/Windows/Fonts/arial.ttf");

    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            square.setPosition(col * TILE_SIZE, row * TILE_SIZE);
            square.setFillColor((row + col) % 2 == 0 ? sf::Color::White : sf::Color(100, 100, 100));

            if (row == bestMove.fromRow && col == bestMove.fromCol) {
                square.setFillColor(sf::Color::Yellow);
            } else if (row == bestMove.toRow && col == bestMove.toCol) {
                square.setFillColor(sf::Color::Green);
            }
            window.draw(square);

            Piece p = board[row][col];
            if (p != EMPTY) {
                sf::Text text;
                text.setFont(font);
                text.setCharacterSize(24);
                text.setPosition(col * TILE_SIZE + 25, row * TILE_SIZE + 25);

                string symbol = "";
                switch (p) {
                    case WHITE_PAWN: symbol = "♙"; break;
                    case BLACK_PAWN: symbol = "♟"; break;
                    case WHITE_KNIGHT: symbol = "♘"; break;
                    case BLACK_KNIGHT: symbol = "♞"; break;
                    case WHITE_BISHOP: symbol = "♗"; break;
                    case BLACK_BISHOP: symbol = "♝"; break;
                    case WHITE_ROOK: symbol = "♖"; break;
                    case BLACK_ROOK: symbol = "♜"; break;
                    case WHITE_QUEEN: symbol = "♕"; break;
                    case BLACK_QUEEN: symbol = "♛"; break;
                    case WHITE_KING: symbol = "♔"; break;
                    case BLACK_KING: symbol = "♚"; break;
                    default: break;
                }
                text.setString(symbol);
                text.setFillColor(is_white(p) ? sf::Color::Blue : sf::Color::Black);
                window.draw(text);
            }
        }
    }
}

bool is_king_in_check(const vector<vector<Piece>>& board, bool isWhiteKing) {
    // Find king's position
    int kingRow = -1, kingCol = -1;
    Piece targetKing = isWhiteKing ? WHITE_KING : BLACK_KING;

    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            if (board[row][col] == targetKing) {
                kingRow = row;
                kingCol = col;
                break;
            }
        }
        if (kingRow != -1) break;
    }

    // Generate all opponent's moves and see if any can capture the king
    vector<Move> opponentMoves = generate_moves(board, !isWhiteKing);
    for (const auto& move : opponentMoves) {
        if (move.toRow == kingRow && move.toCol == kingCol) {
            return true;
        }
    }
    return false;
}

bool is_checkmate(const vector<vector<Piece>>& board, bool isWhiteTurn) {
    // If not in check, it's not checkmate
    if (!is_king_in_check(board, isWhiteTurn)) {
        return false;
    }

    // Try all possible moves to see if any can get out of check
    vector<Move> moves = generate_moves(board, isWhiteTurn);
    for (const auto& move : moves) {
        // Make move
        auto tempBoard = board;
        tempBoard[move.toRow][move.toCol] = tempBoard[move.fromRow][move.fromCol];
        tempBoard[move.fromRow][move.fromCol] = EMPTY;

        // If this move gets us out of check, it's not checkmate
        if (!is_king_in_check(tempBoard, isWhiteTurn)) {
            return false;
        }
    }
    return true;
}

bool is_stalemate(const vector<vector<Piece>>& board, bool isWhiteTurn) {
    // If in check, it's not stalemate
    if (is_king_in_check(board, isWhiteTurn)) {
        return false;
    }

    // If there are no legal moves, it's stalemate
    vector<Move> moves = generate_moves(board, isWhiteTurn);
    for (const auto& move : moves) {
        // Make move
        auto tempBoard = board;
        tempBoard[move.toRow][move.toCol] = tempBoard[move.fromRow][move.fromCol];
        tempBoard[move.fromRow][move.fromCol] = EMPTY;

        // If this move doesn't put us in check, it's a legal move
        if (!is_king_in_check(tempBoard, isWhiteTurn)) {
            return false;
        }
    }
    return true;
}

class ChessGame {
private:
    vector<vector<Piece>> board;
    sf::RenderWindow& window;
    bool isWhiteTurn;
    bool pieceSelected;
    int selectedRow, selectedCol;
    vector<Move> validMoves;
    sf::Font font;
    sf::Texture pieceTextures[13];  // One for each piece type including EMPTY
    sf::Sprite pieceSprites[13];
    bool isDragging;
    sf::Vector2i dragStart;
    Piece draggedPiece;
    bool gameOver;
    string gameOverMessage;

public:
    ChessGame(sf::RenderWindow& win) : window(win), isWhiteTurn(true), pieceSelected(false),
                                       isDragging(false), draggedPiece(EMPTY), gameOver(false) {
        // Initialize board
        board = vector<vector<Piece>>(BOARD_SIZE, vector<Piece>(BOARD_SIZE, EMPTY));

        // Initial board setup
        // Back rank pieces
        board[0][0] = BLACK_ROOK;
        board[0][1] = BLACK_KNIGHT;
        board[0][2] = BLACK_BISHOP;
        board[0][3] = BLACK_QUEEN;
        board[0][4] = BLACK_KING;
        board[0][5] = BLACK_BISHOP;
        board[0][6] = BLACK_KNIGHT;
        board[0][7] = BLACK_ROOK;

        board[7][0] = WHITE_ROOK;
        board[7][1] = WHITE_KNIGHT;
        board[7][2] = WHITE_BISHOP;
        board[7][3] = WHITE_QUEEN;
        board[7][4] = WHITE_KING;
        board[7][5] = WHITE_BISHOP;
        board[7][6] = WHITE_KNIGHT;
        board[7][7] = WHITE_ROOK;

        // Pawns
        for (int col = 0; col < BOARD_SIZE; ++col) {
            board[1][col] = BLACK_PAWN;
            board[6][col] = WHITE_PAWN;
        }

        // Load font
        font.loadFromFile("C:/Windows/Fonts/arial.ttf");
    }

    void handleInput(const sf::Event& event) {
        if (gameOver) return;  // Ignore input if game is over

        switch (event.type) {
            case sf::Event::MouseButtonPressed: {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    int row = event.mouseButton.y / TILE_SIZE;
                    int col = event.mouseButton.x / TILE_SIZE;

                    if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
                        if (!pieceSelected) {
                            Piece piece = board[row][col];
                            if ((isWhiteTurn && is_white(piece)) || (!isWhiteTurn && is_black(piece))) {
                                pieceSelected = true;
                                selectedRow = row;
                                selectedCol = col;
                                validMoves = generate_moves(board, isWhiteTurn);
                                isDragging = true;
                                dragStart = sf::Vector2i(event.mouseButton.x, event.mouseButton.y);
                                draggedPiece = board[row][col];
                            }
                        } else {
                            // Try to make a move
                            Move move = {selectedRow, selectedCol, row, col, 0};
                            if (isValidMove(move)) {
                                makeMove(move);
                                isWhiteTurn = !isWhiteTurn;

                                // AI's turn
                                if (!isWhiteTurn) {
                                    Move aiMove = best_move(board, MAX_DEPTH);
                                    makeMove(aiMove);
                                    isWhiteTurn = true;
                                }
                            }
                            pieceSelected = false;
                            isDragging = false;
                            draggedPiece = EMPTY;
                        }
                    }
                }
                break;
            }
            case sf::Event::MouseButtonReleased: {
                if (event.mouseButton.button == sf::Mouse::Left && isDragging) {
                    int row = event.mouseButton.y / TILE_SIZE;
                    int col = event.mouseButton.x / TILE_SIZE;

                    if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
                        Move move = {selectedRow, selectedCol, row, col, 0};
                        if (isValidMove(move)) {
                            makeMove(move);
                            isWhiteTurn = !isWhiteTurn;

                            // AI's turn
                            if (!isWhiteTurn) {
                                Move aiMove = best_move(board, MAX_DEPTH);
                                makeMove(aiMove);
                                isWhiteTurn = true;
                            }
                        }
                    }
                    pieceSelected = false;
                    isDragging = false;
                    draggedPiece = EMPTY;
                }
                break;
            }
            case sf::Event::MouseMoved: {
                if (isDragging) {
                    dragStart = sf::Vector2i(event.mouseMove.x, event.mouseMove.y);
                }
                break;
            }
            default:
                break;
        }
    }

    void makeMove(const Move& move) {
        board[move.toRow][move.toCol] = board[move.fromRow][move.fromCol];
        board[move.fromRow][move.fromCol] = EMPTY;

        // Check game state after move
        if (is_checkmate(board, !isWhiteTurn)) {
            gameOver = true;
            gameOverMessage = isWhiteTurn ? "White wins by checkmate!" : "Black wins by checkmate!";
        } else if (is_stalemate(board, !isWhiteTurn)) {
            gameOver = true;
            gameOverMessage = "Game drawn by stalemate!";
        }
    }

    bool isValidMove(const Move& move) {
        // First check if it's in the list of generated moves
        bool foundMove = false;
        for (const auto& validMove : validMoves) {
            if (validMove.fromRow == move.fromRow &&
                validMove.fromCol == move.fromCol &&
                validMove.toRow == move.toRow &&
                validMove.toCol == move.toCol) {
                foundMove = true;
                break;
            }
        }
        if (!foundMove) return false;

        // Then check if it would leave us in check
        auto tempBoard = board;
        tempBoard[move.toRow][move.toCol] = tempBoard[move.fromRow][move.fromCol];
        tempBoard[move.fromRow][move.fromCol] = EMPTY;

        return !is_king_in_check(tempBoard, isWhiteTurn);
    }

    void draw() {
        window.clear();

        // Draw board
        sf::RectangleShape square(sf::Vector2f(TILE_SIZE, TILE_SIZE));
        for (int row = 0; row < BOARD_SIZE; ++row) {
            for (int col = 0; col < BOARD_SIZE; ++col) {
                square.setPosition(col * TILE_SIZE, row * TILE_SIZE);

                // Normal square colors
                square.setFillColor((row + col) % 2 == 0 ? sf::Color(240, 217, 181) : sf::Color(181, 136, 99));

                // Highlight selected piece and valid moves
                if (pieceSelected) {
                    if (row == selectedRow && col == selectedCol) {
                        square.setFillColor(sf::Color(130, 151, 105));
                    } else {
                        // Highlight valid moves
                        for (const auto& move : validMoves) {
                            if (move.fromRow == selectedRow &&
                                move.fromCol == selectedCol &&
                                move.toRow == row &&
                                move.toCol == col) {
                                square.setFillColor(sf::Color(130, 151, 105, 200));
                                break;
                            }
                        }
                    }
                }

                window.draw(square);

                // Draw pieces
                Piece piece = board[row][col];
                if (piece != EMPTY && (!isDragging || row != selectedRow || col != selectedCol)) {
                    sf::Text text;
                    text.setFont(font);
                    text.setCharacterSize(40);
                    text.setPosition(col * TILE_SIZE + 20, row * TILE_SIZE + 15);

                    string symbol = "";
                    switch (piece) {
                        case WHITE_PAWN: symbol = "♙"; break;
                        case BLACK_PAWN: symbol = "♟"; break;
                        case WHITE_KNIGHT: symbol = "♘"; break;
                        case BLACK_KNIGHT: symbol = "♞"; break;
                        case WHITE_BISHOP: symbol = "♗"; break;
                        case BLACK_BISHOP: symbol = "♝"; break;
                        case WHITE_ROOK: symbol = "♖"; break;
                        case BLACK_ROOK: symbol = "♜"; break;
                        case WHITE_QUEEN: symbol = "♕"; break;
                        case BLACK_QUEEN: symbol = "♛"; break;
                        case WHITE_KING: symbol = "♔"; break;
                        case BLACK_KING: symbol = "♚"; break;
                        default: break;
                    }
                    text.setString(symbol);
                    text.setFillColor(is_white(piece) ? sf::Color::White : sf::Color::Black);
                    window.draw(text);
                }
            }
        }

        // Draw dragged piece
        if (isDragging && draggedPiece != EMPTY) {
            sf::Text text;
            text.setFont(font);
            text.setCharacterSize(40);
            text.setPosition(dragStart.x - TILE_SIZE/2, dragStart.y - TILE_SIZE/2);

            string symbol = "";
            switch (draggedPiece) {
                case WHITE_PAWN: symbol = "♙"; break;
                case BLACK_PAWN: symbol = "♟"; break;
                case WHITE_KNIGHT: symbol = "♘"; break;
                case BLACK_KNIGHT: symbol = "♞"; break;
                case WHITE_BISHOP: symbol = "♗"; break;
                case BLACK_BISHOP: symbol = "♝"; break;
                case WHITE_ROOK: symbol = "♖"; break;
                case BLACK_ROOK: symbol = "♜"; break;
                case WHITE_QUEEN: symbol = "♕"; break;
                case BLACK_QUEEN: symbol = "♛"; break;
                case WHITE_KING: symbol = "♔"; break;
                case BLACK_KING: symbol = "♚"; break;
                default: break;
            }
            text.setString(symbol);
            text.setFillColor(is_white(draggedPiece) ? sf::Color::White : sf::Color::Black);
            window.draw(text);
        }

        // Draw game over message if game is over
        if (gameOver) {
            sf::RectangleShape overlay(sf::Vector2f(BOARD_SIZE * TILE_SIZE, BOARD_SIZE * TILE_SIZE));
            overlay.setFillColor(sf::Color(0, 0, 0, 128));
            window.draw(overlay);

            sf::Text text;
            text.setFont(font);
            text.setCharacterSize(32);
            text.setFillColor(sf::Color::White);
            text.setString(gameOverMessage);

            // Center the text
            sf::FloatRect textRect = text.getLocalBounds();
            text.setOrigin(textRect.left + textRect.width/2.0f,
                           textRect.top + textRect.height/2.0f);
            text.setPosition(BOARD_SIZE * TILE_SIZE/2.0f, BOARD_SIZE * TILE_SIZE/2.0f);

            window.draw(text);
        }

        window.display();
    }

    const vector<vector<Piece>>& getBoard() const {
        return board;
    }
};

int main() {
    omp_set_num_threads(omp_get_max_threads());

    sf::RenderWindow window(sf::VideoMode(BOARD_SIZE * TILE_SIZE, BOARD_SIZE * TILE_SIZE),
                            "Chess AI", sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);

    ChessGame game(window);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();

            game.handleInput(event);
        }

        game.draw();
    }

    return 0;
}