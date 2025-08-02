import pygame
import sys
import random
import numpy as np
import json
import os
import multiprocessing
import matplotlib


matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt


GRID_WIDTH = 10
GRID_HEIGHT = 20
CELL_SIZE = 15
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
SIDEBAR_WIDTH = 500


GAMES_PER_ROW = 3
GAME_AREA_WIDTH = GRID_WIDTH * CELL_SIZE + 40
GAME_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE + 40
UI_TOP_MARGIN = 180
GRAPH_HEIGHT = 220
BOTTOM_PADDING = 40


BACKGROUND = (28, 33, 40)
GRID_COLOR = (50, 58, 69)
TEXT_COLOR = (230, 230, 230)
SCROLLBAR_COLOR = (100, 100, 100)
SCROLLBAR_HANDLE_COLOR = (150, 150, 150)
INPUT_BOX_ACTIVE_COLOR = pygame.Color("dodgerblue2")
INPUT_BOX_INACTIVE_COLOR = pygame.Color("lightskyblue3")

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


class TextInputBox:
    """A class for a user-editable text input box."""

    def __init__(self, x, y, w, h, font, text="30"):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = INPUT_BOX_INACTIVE_COLOR
        self.text = text
        self.font = font
        self.txt_surface = self.font.render(text, True, self.color)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:

            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False

            self.color = (
                INPUT_BOX_ACTIVE_COLOR if self.active else INPUT_BOX_INACTIVE_COLOR
            )
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    self.active = False
                    self.color = INPUT_BOX_INACTIVE_COLOR
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:

                    if event.unicode.isnumeric():
                        self.text += event.unicode

                self.txt_surface = self.font.render(self.text, True, TEXT_COLOR)

    def draw(self, screen):

        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))

        pygame.draw.rect(screen, self.color, self.rect, 2)


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


class NeuralNetwork:
    """The AI's 'Brain' - with a hidden layer."""

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
    """Represents a single Tetris piece."""

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
    """A single Tetris game instance, controllable by an AI."""

    def __init__(self, x_offset, y_offset, cell_size=CELL_SIZE):
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.cell_size = cell_size
        self.board_width = GRID_WIDTH * self.cell_size
        self.board_height = GRID_HEIGHT * self.cell_size
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

    def step(self, move, sound=None):
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

        if lines_cleared > 0 and sound:
            sound.play()

        score_map = {1: 40, 2: 100, 3: 300, 4: 1200}
        self.score += score_map.get(lines_cleared, 0)
        self.pieces_placed += 1
        self.new_piece()

    def draw(self, screen, is_best=False, scroll_y=0):
        draw_y_offset = self.y_offset - scroll_y
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                rect = pygame.Rect(
                    self.x_offset + c * self.cell_size,
                    draw_y_offset + r * self.cell_size,
                    self.cell_size,
                    self.cell_size,
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
            (self.x_offset, draw_y_offset, self.board_width, self.board_height),
            2,
        )

        if not self.game_over:
            for r, row_data in enumerate(self.current_piece.shape):
                for c, cell in enumerate(row_data):
                    if cell:
                        rect = pygame.Rect(
                            self.x_offset + (self.current_piece.x + c) * self.cell_size,
                            draw_y_offset + (self.current_piece.y + r) * self.cell_size,
                            self.cell_size,
                            self.cell_size,
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
    """An AI agent that plays a game of Tetris."""

    def __init__(
        self, x_offset, y_offset, weights=None, pool=None, cell_size=CELL_SIZE
    ):
        self.game = Tetris(x_offset, y_offset, cell_size)
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

    def update(self, sound=None):
        if not self.game.game_over:
            move = self.get_best_move()
            self.game.step(move, sound=sound)

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

        x = col * GAME_AREA_WIDTH + 150
        y = row * GAME_AREA_HEIGHT + UI_TOP_MARGIN
        new_agents.append(Agent(x, y, child_weights, pool))
    return new_agents


def draw_neural_network(screen, nn, x, y, width, height, scroll_y=0):

    draw_y = y
    font = pygame.font.SysFont("Consolas", 12)
    labels = ["Lines", "Agg H", "Max H", "Holes", "Bump", "R Trans", "Wells"]
    v_spacing = height / (NUM_INPUTS - 1)
    input_nodes = [(x, draw_y + i * v_spacing) for i in range(NUM_INPUTS)]
    hidden_v_spacing = height / (NUM_HIDDEN - 1)
    hidden_nodes = [
        (x + width / 2, draw_y + i * hidden_v_spacing) for i in range(NUM_HIDDEN)
    ]
    output_node = (x + width, draw_y + height / 2)
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


def draw_score_graph(scores, width, height):
    if len(scores) < 2:
        return pygame.Surface((width, height), pygame.SRCALPHA)
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.gca()
    generations = [s["generation"] for s in scores]
    high_scores = [s["high_score"] for s in scores]
    fig.patch.set_facecolor((28 / 255, 33 / 255, 40 / 255))
    ax.set_facecolor((50 / 255, 58 / 255, 69 / 255))
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.plot(generations, high_scores, color="cyan", marker="o")
    ax.set_xlabel("Generation")
    ax.set_ylabel("High Score")
    ax.set_title("Score Progression")
    ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    plt.close(fig)
    return pygame.image.fromstring(raw_data, size, "RGB")


def make_sound(frequency, duration=0.1, sample_rate=44100):
    """Generates a pygame sound from a sine wave."""
    n_samples = int(duration * sample_rate)
    buf = np.zeros((n_samples, 2), dtype=np.int16)
    max_sample = 2**15 - 1
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    wave = np.sin(2 * np.pi * frequency * t)

    wave = wave * max_sample
    buf[:, 0] = wave.astype(np.int16)
    buf[:, 1] = wave.astype(np.int16)
    return pygame.sndarray.make_sound(buf)


def main():
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Tetris AI - Speed Control")
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("Consolas", 24, bold=True)
    font_small = pygame.font.SysFont("Consolas", 18)

    load_sound = make_sound(660, duration=0.2)
    line_clear_sound = make_sound(440, duration=0.1)

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    app_mode = "TRAINING"
    agents = []
    playback_agent = None

    generation = 1
    running = True
    fast_mode = True
    generation_scores = []
    best_overall_score = 0
    best_nn_ever = None
    graph_surface = None

    scroll_y = 0
    num_rows = (POPULATION_SIZE + GAMES_PER_ROW - 1) // GAMES_PER_ROW
    content_height = UI_TOP_MARGIN + (num_rows * GAME_AREA_HEIGHT) + BOTTOM_PADDING

    save_message_timer = 0

    speed_input_box = TextInputBox(380, 100, 60, 32, font_small)

    def initialize_training_agents():
        nonlocal agents, best_nn_ever
        agents = []
        for i in range(POPULATION_SIZE):
            row = i // GAMES_PER_ROW
            col = i % GAMES_PER_ROW

            x = col * GAME_AREA_WIDTH + 150
            y = row * GAME_AREA_HEIGHT + UI_TOP_MARGIN
            agents.append(Agent(x, y, pool=pool))
        if best_nn_ever is None:
            best_nn_ever = agents[0].nn

    initialize_training_agents()

    if os.path.exists("generation_data.json"):
        with open("generation_data.json", "r") as f:
            try:
                generation_scores = json.load(f)
                if generation_scores:
                    generation = generation_scores[-1]["generation"] + 1
                    best_overall_score = max(
                        item["high_score"] for item in generation_scores
                    )
                    graph_surface = draw_score_graph(
                        generation_scores, SIDEBAR_WIDTH - 100, 250
                    )
            except (json.JSONDecodeError, IndexError):
                print("Could not load generation_data.json. Starting fresh.")

    while running:
        current_window_height = screen.get_height()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            speed_input_box.handle_event(event)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_SPACE:
                    fast_mode = not fast_mode

                if event.key == pygame.K_s:
                    if best_nn_ever:
                        with open("best_brain.json", "w") as f:
                            json.dump(best_nn_ever.get_weights(), f, indent=4)
                        print("Best brain saved to best_brain.json")
                        save_message_timer = pygame.time.get_ticks()

                if event.key == pygame.K_l:
                    if os.path.exists("best_brain.json"):
                        with open("best_brain.json", "r") as f:
                            loaded_weights = json.load(f)
                        app_mode = "PLAYBACK"
                        playback_cell_size = 30
                        playback_x = (
                            WINDOW_WIDTH
                            - SIDEBAR_WIDTH
                            - (GRID_WIDTH * playback_cell_size)
                        ) // 2
                        playback_y = 100
                        playback_agent = Agent(
                            playback_x,
                            playback_y,
                            weights=loaded_weights,
                            pool=pool,
                            cell_size=playback_cell_size,
                        )
                        load_sound.play()
                        print("Switched to Playback Mode.")
                    else:
                        print("No saved brain (best_brain.json) found.")

                if event.key == pygame.K_t:
                    if app_mode == "PLAYBACK":
                        app_mode = "TRAINING"
                        initialize_training_agents()
                        print("Switched back to Training Mode.")

            if event.type == pygame.MOUSEWHEEL and app_mode == "TRAINING":
                scroll_y -= event.y * 20
                scroll_y = max(0, min(scroll_y, content_height - current_window_height))
            if event.type == pygame.VIDEORESIZE:
                scroll_y = max(0, min(scroll_y, content_height - event.h))

        screen.fill(BACKGROUND)

        if app_mode == "TRAINING":
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

                graph_surface = draw_score_graph(
                    generation_scores, SIDEBAR_WIDTH - 100, 250
                )
                agents = evolve(agents, pool)
                generation += 1

            best_current_score = -1
            best_agent_idx = -1
            if games_running:
                scores = [a.game.score for a in agents]
                if scores:
                    best_current_score = max(scores)
                    best_agent_idx = scores.index(best_current_score)

            for i, agent in enumerate(agents):
                agent.game.draw(
                    screen, is_best=(i == best_agent_idx), scroll_y=scroll_y
                )
                score_text = font_small.render(
                    f"S: {agent.game.score}", True, TEXT_COLOR
                )
                screen.blit(
                    score_text,
                    (
                        agent.game.x_offset,
                        agent.game.y_offset + agent.game.board_height + 5 - scroll_y,
                    ),
                )
                if agent.game.game_over:
                    s = pygame.Surface(
                        (agent.game.board_width, agent.game.board_height),
                        pygame.SRCALPHA,
                    )
                    s.fill((0, 0, 0, 150))
                    screen.blit(
                        s, (agent.game.x_offset, agent.game.y_offset - scroll_y)
                    )

            sidebar_x = WINDOW_WIDTH - SIDEBAR_WIDTH
            pygame.draw.line(
                screen,
                GRID_COLOR,
                (sidebar_x, 0),
                (sidebar_x, current_window_height),
                2,
            )

            gen_text = font_large.render(f"Generation: {generation}", True, TEXT_COLOR)
            screen.blit(gen_text, (20, 20))
            best_score_text = font_large.render(
                f"All-Time Best Score: {best_overall_score}", True, (255, 215, 0)
            )
            screen.blit(best_score_text, (20, 60))

            controls_text = "Speed: {} (Space) | Save: S | Load: L".format(
                "MAX" if fast_mode else "VIEWABLE"
            )
            speed_text = font_small.render(controls_text, True, TEXT_COLOR)
            screen.blit(speed_text, (20, 100))

            speed_input_box.draw(screen)

            nn_title = font_large.render("Best Brain (All Time)", True, TEXT_COLOR)
            screen.blit(nn_title, (sidebar_x + 20, 20))
            if best_nn_ever:
                draw_neural_network(
                    screen, best_nn_ever, sidebar_x + 50, 80, SIDEBAR_WIDTH - 100, 200
                )

            if graph_surface:
                screen.blit(graph_surface, (sidebar_x + 20, 300))

            if content_height > current_window_height:
                scrollbar_rect = pygame.Rect(
                    sidebar_x - 15, 0, 15, current_window_height
                )
                pygame.draw.rect(screen, SCROLLBAR_COLOR, scrollbar_rect)
                handle_height = current_window_height * (
                    current_window_height / content_height
                )
                handle_y = (scroll_y / (content_height - current_window_height)) * (
                    current_window_height - handle_height
                )
                handle_rect = pygame.Rect(sidebar_x - 15, handle_y, 15, handle_height)
                pygame.draw.rect(
                    screen, SCROLLBAR_HANDLE_COLOR, handle_rect, border_radius=5
                )

            if (
                save_message_timer
                and pygame.time.get_ticks() - save_message_timer < 2000
            ):
                save_text = font_large.render("Brain Saved!", True, (0, 255, 0))
                text_rect = save_text.get_rect(
                    center=((WINDOW_WIDTH - SIDEBAR_WIDTH) / 2, 30)
                )
                screen.blit(save_text, text_rect)
            else:
                save_message_timer = 0

        elif app_mode == "PLAYBACK":
            if playback_agent:
                playback_agent.update(sound=line_clear_sound)
                playback_agent.game.draw(screen)

                title_text = font_large.render("Playback Mode", True, (0, 255, 255))
                screen.blit(title_text, (20, 20))

                score_text = font_large.render(
                    f"Score: {playback_agent.game.score}", True, TEXT_COLOR
                )
                screen.blit(score_text, (20, 60))

                controls_text = font_small.render(
                    "Press 'T' to return to Training", True, TEXT_COLOR
                )
                screen.blit(controls_text, (20, current_window_height - 40))

                if playback_agent.game.game_over:
                    overlay = pygame.Surface(
                        (
                            playback_agent.game.board_width,
                            playback_agent.game.board_height,
                        ),
                        pygame.SRCALPHA,
                    )
                    overlay.fill((0, 0, 0, 180))
                    screen.blit(
                        overlay,
                        (playback_agent.game.x_offset, playback_agent.game.y_offset),
                    )

                    go_text = font_large.render("GAME OVER", True, (255, 0, 0))
                    go_rect = go_text.get_rect(
                        center=(
                            (WINDOW_WIDTH - SIDEBAR_WIDTH) / 2,
                            current_window_height / 2,
                        )
                    )
                    screen.blit(go_text, go_rect)

        pygame.display.flip()

        if not fast_mode:
            try:

                viewable_speed = int(speed_input_box.text)
                if viewable_speed <= 0:
                    viewable_speed = 30
            except ValueError:
                viewable_speed = 30
            clock.tick(viewable_speed)

    pool.close()
    pool.join()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
