import chess, chess.engine, pickle, sys, os, random

ENGINE_PATH = r"C:\Users\your_username\path_to_stockfish\stockfish.exe"
DEPTH = 8
LABELS = {
    chess.PAWN:"Pawn", chess.KNIGHT:"Knight",
    chess.BISHOP:"Bishop", chess.ROOK:"Rook",
    chess.QUEEN:"Queen"
}

# Use stockfish engine
def start_engine():
    e = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    e.configure({"Threads":1})
    return e

# Evaluate board position
def eval_board(engine, board):
    info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
    score = info["score"].white().score(mate_score=10000)
    return (score or 0)/100

# Generate legal random board position
def random_board():
    b = chess.Board()
    for _ in range(random.randint(10,20)):
        if b.is_game_over(): break
        move = random.choice(list(b.legal_moves))
        b.push(move)
    return b

# piece value estimation by removing piece and re-evaluating
def main(batch_id, n_positions):
    engine = start_engine()
    results = {pt:[] for pt in LABELS}
    for _ in range(n_positions):
        b = random_board()
        base = eval_board(engine,b)
        for sq,p in b.piece_map().items():
            if p.piece_type not in LABELS: continue
            tmp = b.copy(); tmp.remove_piece_at(sq)
            new = eval_board(engine,tmp)
            delta = base-new
            val = delta if p.color==chess.WHITE else -delta
            results[p.piece_type].append(val)
    engine.quit()
    with open(f"batch_{batch_id}.pkl","wb") as f: pickle.dump(results,f)

if __name__=="__main__":
    batch_id = int(sys.argv[1])
    n_positions = int(sys.argv[2])
    main(batch_id,n_positions)
