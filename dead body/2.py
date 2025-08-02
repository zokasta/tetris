import pygame
import sys
import random
import numpy as np
import json
import os
import multiprocessing

    
GRID_WIDTH = 10
GRID_HEIGHT = 20
CELL_SIZE = 15
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 750


GAMES_PER_ROW = 3
GAME_AREA_WIDTH = GRID_WIDTH * CELL_SIZE + 40
GAME_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE + 40


BACKGROUND = (28, 33, 40)
GRID_COLOR = (50, 58, 69)
TEXT_COLOR = (230, 230, 230)
PIECE_COLORS = [
    (0, 255, 255),
    (255, 255, 0),
    (128, 0, 128),
    (0, 0, 255),
    (255, 165, 0),
    (0, 255, 0),
    (255, 0, 0),
]

SHAPES = [
    [[1, 1, 1, 1]],
    [[1, 1], [1, 1]],
    [[0, 1, 0], [1, 1, 1]],
    [[1, 0, 0], [1, 1, 1]],
    [[0, 0, 1], [1, 1, 1]],
    [[0, 1, 1], [1, 1, 0]],
    [[1, 1, 0], [0, 1, 1]],
]


POPULATION_SIZE = 6
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.2
NUM_INPUTS = 7
NUM_HIDDEN = 16


def relu(x):
    return np.maximum(0, x)


class NeuralNetwork:

    def __init__(self, weights=None):
        if weights is not None:
            self.W1 = np.array(weights[0])
            self.b1 = np.array(weights[1])
            self.W2 = np.array(weights[2])
            self.b2 = np.array(weights[3])
        else:

            self.W1 = np.random.uniform(-1, 1, (NUM_INPUTS, NUM_HIDDEN))
            self.b1 = np.random.uniform(-1, 1, (1, NUM_HIDDEN))
            self.W2 = np.random.uniform(-1, 1, (NUM_HIDDEN, 1))
            self.b2 = np.random.uniform(-1, 1, (1, 1))

    def predict(self, inputs):

        hidden = relu(np.dot(inputs, self.W1) + self.b1)
        output = np.dot(hidden, self.W2) + self.b2
        return output[0][0]

    def get_weights(self):
        return [self.W1.tolist(), self.b1.tolist(), self.W2.tolist(), self.b2.tolist()]


class Piece:

    def __init__(self, x, y, shape_index=None):
        self.shape_index = (
            shape_index
            if shape_index is not None
            else random.randint(0, len(SHAPES) - 1)
        )
        self.shape = SHAPES[self.shape_index]
        self.color = PIECE_COLORS[self.shape_index]
        self.x = x
        self.y = y

    def rotate(self):
        self.shape = list(zip(*self.shape[::-1]))


class Tetris:

    def __init__(self, x_offset, y_offset):
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.board_width = GRID_WIDTH * CELL_SIZE
        self.board_height = GRID_HEIGHT * CELL_SIZE
        self.reset()

    def reset(self):
        self.board = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        self.new_piece()

    def new_piece(self):
        self.current_piece = Piece(GRID_WIDTH // 2 - 1, 0)
        if not self.is_valid_position(
            self.current_piece.shape, (self.current_piece.x, self.current_piece.y)
        ):
            self.game_over = True

    def is_valid_position(self, shape, pos, board=None):
        board_to_check = board if board is not None else self.board
        px, py = pos
        for r, row_data in enumerate(shape):
            for c, cell in enumerate(row_data):
                if cell:
                    board_r, board_c = py + r, px + c
                    if not (
                        0 <= board_r < GRID_HEIGHT
                        and 0 <= board_c < GRID_WIDTH
                        and board_to_check[board_r][board_c] == 0
                    ):
                        return False
        return True

    def lock_piece(self, board, piece, pos):
        temp_board = [row[:] for row in board]
        for r, row_data in enumerate(piece.shape):
            for c, cell in enumerate(row_data):
                if cell:
                    temp_board[pos[1] + r][pos[0] + c] = piece.shape_index + 1
        return temp_board

    def clear_lines(self, board):
        lines_to_clear = [r for r, row in enumerate(board) if all(row)]
        if not lines_to_clear:
            return board, 0

        for r in sorted(lines_to_clear, reverse=True):
            del board[r]
            board.insert(0, [0] * GRID_WIDTH)
        return board, len(lines_to_clear)

    def get_board_metrics(self, board):
        heights = [0] * GRID_WIDTH
        for c in range(GRID_WIDTH):
            for r in range(GRID_HEIGHT):
                if board[r][c] != 0:
                    heights[c] = GRID_HEIGHT - r
                    break

        aggregate_height = sum(heights)
        max_height = max(heights)
        holes = 0
        wells = 0
        row_transitions = 0

        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH - 1):
                if (board[r][c] == 0 and board[r][c + 1] != 0) or (
                    board[r][c] != 0 and board[r][c + 1] == 0
                ):
                    row_transitions += 1
            if board[r][0] == 0:
                row_transitions += 1
            if board[r][-1] == 0:
                row_transitions += 1

        for c in range(GRID_WIDTH):
            found_block = False
            for r in range(GRID_HEIGHT):
                if board[r][c] != 0:
                    found_block = True
                elif found_block and board[r][c] == 0:
                    holes += 1

            for r in range(1, GRID_HEIGHT):
                left_wall = (c == 0) or (board[r][c - 1] != 0)
                right_wall = (c == GRID_WIDTH - 1) or (board[r][c + 1] != 0)
                if board[r][c] == 0 and left_wall and right_wall:
                    wells += 1

        bumpiness = sum(
            abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1)
        )

        return aggregate_height, max_height, holes, bumpiness, row_transitions, wells

    def step(self, move):
        if self.game_over:
            return
        rotations, x_pos = move
        for _ in range(rotations):
            self.current_piece.rotate()
        self.current_piece.x = x_pos
        while self.is_valid_position(
            self.current_piece.shape, (self.current_piece.x, self.current_piece.y + 1)
        ):
            self.current_piece.y += 1
        self.board = self.lock_piece(
            self.board, self.current_piece, (self.current_piece.x, self.current_piece.y)
        )
        new_board, lines_cleared = self.clear_lines(self.board)
        self.board = new_board
        self.lines_cleared += lines_cleared
        score_map = {1: 40, 2: 100, 3: 300, 4: 1200}
        self.score += score_map.get(lines_cleared, 0)
        self.pieces_placed += 1
        self.new_piece()

    def draw(self, screen, is_best=False):
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                rect = pygame.Rect(
                    self.x_offset + c * CELL_SIZE,
                    self.y_offset + r * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                )
                color = (
                    PIECE_COLORS[self.board[r][c] - 1]
                    if self.board[r][c]
                    else (40, 46, 56)
                )
                pygame.draw.rect(screen, color, rect)

        border_color = (255, 255, 0) if is_best else TEXT_COLOR
        pygame.draw.rect(
            screen,
            border_color,
            (self.x_offset, self.y_offset, self.board_width, self.board_height),
            2,
        )

        if not self.game_over:
            for r, row_data in enumerate(self.current_piece.shape):
                for c, cell in enumerate(row_data):
                    if cell:
                        rect = pygame.Rect(
                            self.x_offset + (self.current_piece.x + c) * CELL_SIZE,
                            self.y_offset + (self.current_piece.y + r) * CELL_SIZE,
                            CELL_SIZE,
                            CELL_SIZE,
                        )
                        pygame.draw.rect(screen, self.current_piece.color, rect)


def evaluate_move_worker(args):
    board, current_piece, move, nn_weights = args
    rotations, x_pos = move

    temp_game = Tetris(0, 0)
    temp_game.board = [row[:] for row in board]

    rotated_shape = current_piece.shape
    for _ in range(rotations):
        rotated_shape = list(zip(*rotated_shape[::-1]))

    temp_piece = Piece(0, 0, current_piece.shape_index)
    temp_piece.shape = rotated_shape

    y_pos = 0
    while temp_game.is_valid_position(rotated_shape, (x_pos, y_pos + 1)):
        y_pos += 1

    if not temp_game.is_valid_position(rotated_shape, (x_pos, y_pos)):
        return -float("inf"), move

    temp_board = temp_game.lock_piece(temp_game.board, temp_piece, (x_pos, y_pos))
    temp_board_after_clear, lines_cleared = temp_game.clear_lines(temp_board)

    agg_h, max_h, holes, bump, r_trans, wells = temp_game.get_board_metrics(
        temp_board_after_clear
    )

    inputs = np.array([lines_cleared, -agg_h, -max_h, -holes, -bump, -r_trans, -wells])

    nn = NeuralNetwork(nn_weights)
    score = nn.predict(inputs)

    return score, move


class Agent:

    def __init__(self, x_offset, y_offset, weights=None, pool=None):
        self.game = Tetris(x_offset, y_offset)
        self.nn = NeuralNetwork(weights)
        self.pool = pool

    def get_best_move(self):
        if self.pool is None:

            return self.get_best_move_single_threaded()

        possible_moves = []
        current_piece = self.game.current_piece

        for rotations in range(4):
            rotated_shape = current_piece.shape
            for _ in range(rotations):
                rotated_shape = list(zip(*rotated_shape[::-1]))

            for x_pos in range(GRID_WIDTH - len(rotated_shape[0]) + 1):
                possible_moves.append((rotations, x_pos))

        tasks = [
            (self.game.board, self.game.current_piece, move, self.nn.get_weights())
            for move in possible_moves
        ]

        results = self.pool.map(evaluate_move_worker, tasks)

        if not results:
            return (0, 0)

        best_score, best_move = max(results, key=lambda item: item[0])
        return best_move

    def update(self):
        if not self.game.game_over:
            move = self.get_best_move()
            self.game.step(move)

    def reset(self, weights=None):
        self.game.reset()
        if weights:
            self.nn = NeuralNetwork(weights)


def evolve(agents, pool):
    agents.sort(key=lambda agent: agent.game.score, reverse=True)

    parents = agents[: max(2, POPULATION_SIZE // 3)]
    parent_weights = [p.nn.get_weights() for p in parents]

    new_agents = []

    best_agent_idx = agents.index(parents[0])
    new_agents.append(
        Agent(
            agents[best_agent_idx].game.x_offset,
            agents[best_agent_idx].game.y_offset,
            parent_weights[0],
            pool,
        )
    )

    while len(new_agents) < POPULATION_SIZE:
        p1_weights, p2_weights = random.sample(parent_weights, 2)

        child_weights = []
        for w1, w2 in zip(p1_weights, p2_weights):
            w1, w2 = np.array(w1), np.array(w2)
            child_w = np.copy(w1)
            mask = np.random.rand(*w1.shape) > 0.5
            child_w[mask] = w2[mask]

            if random.random() < MUTATION_RATE:
                mutation = np.random.normal(0, MUTATION_STRENGTH, child_w.shape)
                child_w += mutation

            child_weights.append(child_w.tolist())

        agent_idx = len(new_agents)
        row = agent_idx // GAMES_PER_ROW
        col = agent_idx % GAMES_PER_ROW
        x = col * GAME_AREA_WIDTH + 50
        y = row * GAME_AREA_HEIGHT + 150
        new_agents.append(Agent(x, y, child_weights, pool))

    return new_agents


def draw_neural_network(screen, nn, x, y, width, height):
    font = pygame.font.SysFont("Consolas", 12)
    labels = ["Lines", "Agg H", "Max H", "Holes", "Bump", "R Trans", "Wells"]

    v_spacing = height / (NUM_INPUTS - 1)
    input_nodes = [(x, y + i * v_spacing) for i in range(NUM_INPUTS)]

    hidden_v_spacing = height / (NUM_HIDDEN - 1)
    hidden_nodes = [
        (x + width / 2, y + i * hidden_v_spacing) for i in range(NUM_HIDDEN)
    ]

    output_node = (x + width, y + height / 2)

    for i, (ix, iy) in enumerate(input_nodes):
        for j, (hx, hy) in enumerate(hidden_nodes):
            weight = nn.W1[i, j]
            color = (0, 255, 0, 50) if weight > 0 else (255, 0, 0, 50)
            thickness = min(3, int(abs(weight) * 2) + 1)
            pygame.draw.line(screen, color, (ix, iy), (hx, hy), thickness)

    for i, (hx, hy) in enumerate(hidden_nodes):
        weight = nn.W2[i, 0]
        color = (0, 255, 0, 100) if weight > 0 else (255, 0, 0, 100)
        thickness = min(4, int(abs(weight) * 3) + 1)
        pygame.draw.line(screen, color, (hx, hy), output_node, thickness)

    for i, (ix, iy) in enumerate(input_nodes):
        pygame.draw.circle(screen, (200, 200, 200), (ix, iy), 5)
        text = font.render(labels[i], True, TEXT_COLOR)
        screen.blit(text, (ix - 50, iy - 8))

    for hx, hy in hidden_nodes:
        pygame.draw.circle(screen, (150, 150, 250), (hx, hy), 4)

    pygame.draw.circle(screen, (250, 200, 150), output_node, 8)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Tetris AI - Parallel Processing")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("Consolas", 24, bold=True)
    font_small = pygame.font.SysFont("Consolas", 18)

    try:

        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    agents = []
    for i in range(POPULATION_SIZE):
        row = i // GAMES_PER_ROW
        col = i % GAMES_PER_ROW
        x = col * GAME_AREA_WIDTH + 50
        y = row * GAME_AREA_HEIGHT + 150
        agents.append(Agent(x, y, pool=pool))

    generation = 1
    running = True
    fast_mode = True
    generation_scores = []
    best_overall_score = 0
    best_nn_ever = agents[0].nn

    if os.path.exists("generation_data.json"):
        with open("generation_data.json", "r") as f:
            try:
                generation_scores = json.load(f)
                if generation_scores:
                    generation = generation_scores[-1]["generation"] + 1
                    best_overall_score = max(
                        item["high_score"] for item in generation_scores
                    )
            except (json.JSONDecodeError, IndexError):
                print("Could not load generation_data.json. Starting fresh.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    fast_mode = not fast_mode
                if event.key == pygame.K_q:
                    running = False

        games_running = any(not agent.game.game_over for agent in agents)

        if games_running:
            for agent in agents:
                agent.update()
        else:
            agents.sort(key=lambda agent: agent.game.score, reverse=True)
            best_of_gen = agents[0]
            high_score_gen = best_of_gen.game.score

            if high_score_gen > best_overall_score:
                best_overall_score = high_score_gen
                best_nn_ever = best_of_gen.nn

            print(
                f"Generation {generation} | High Score: {high_score_gen} | Overall Best: {best_overall_score}"
            )
            gen_data = {"generation": generation, "high_score": high_score_gen}
            generation_scores.append(gen_data)
            with open("generation_data.json", "w") as f:
                json.dump(generation_scores, f, indent=4)

            agents = evolve(agents, pool)
            generation += 1

        screen.fill(BACKGROUND)

        best_current_score = -1
        best_agent_idx = -1
        if games_running:
            scores = [a.game.score for a in agents]
            if scores:
                best_current_score = max(scores)
                best_agent_idx = scores.index(best_current_score)

        for i, agent in enumerate(agents):
            agent.game.draw(screen, is_best=(i == best_agent_idx))
            score_text = font_small.render(f"S: {agent.game.score}", True, TEXT_COLOR)
            screen.blit(
                score_text,
                (
                    agent.game.x_offset,
                    agent.game.y_offset + agent.game.board_height + 5,
                ),
            )
            if agent.game.game_over:
                s = pygame.Surface(
                    (agent.game.board_width, agent.game.board_height), pygame.SRCALPHA
                )
                s.fill((0, 0, 0, 150))
                screen.blit(s, (agent.game.x_offset, agent.game.y_offset))

        gen_text = font_large.render(f"Generation: {generation}", True, TEXT_COLOR)
        screen.blit(gen_text, (20, 20))
        best_score_text = font_large.render(
            f"All-Time Best Score: {best_overall_score}", True, (255, 215, 0)
        )
        screen.blit(best_score_text, (20, 60))
        speed_text = font_small.render(
            f"Speed: {'MAX' if fast_mode else 'VIEWABLE'} (Space to toggle)",
            True,
            TEXT_COLOR,
        )
        screen.blit(speed_text, (20, 100))

        nn_title = font_large.render("Best Brain (All Time)", True, TEXT_COLOR)
        screen.blit(nn_title, (WINDOW_WIDTH - 320, 20))
        draw_neural_network(screen, best_nn_ever, WINDOW_WIDTH - 350, 80, 300, 200)

        pygame.display.flip()

        if not fast_mode:
            clock.tick(10)

    pool.close()
    pool.join()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
