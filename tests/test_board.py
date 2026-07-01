from engine.board import Board


def test_board_has_legal_moves_initially() -> None:
    board = Board(size=9, win_length=5)
    state = board.initial_state()

    legal = board.legal_moves(state)
    assert len(legal) == 81


def test_horizontal_win_detected() -> None:
    board = Board(size=9, win_length=5)
    state = board.initial_state()

    moves = [0, 9, 1, 10, 2, 11, 3, 12, 4]
    for mv in moves:
        state = board.apply(state, mv)

    assert state.winner == 1
    assert board.terminal(state)
