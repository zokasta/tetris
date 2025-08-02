import pygame
import sys
import random


GRID_WIDTH = 10
GRID_HEIGHT = 20
CELL_SIZE = 30
WINDOW_WIDTH = 500
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE + 100 

BOARD_WIDTH = GRID_WIDTH * CELL_SIZE
BOARD_HEIGHT = GRID_HEIGHT * CELL_SIZE
BOARD_X = (WINDOW_WIDTH - BOARD_WIDTH) // 2
BOARD_Y = WINDOW_HEIGHT - BOARD_HEIGHT


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
    [[1, 1, 0], [0, 1, 1]]  
]

class Piece:
    def __init__(self, x, y):
        self.shape_index = random.randint(0, len(SHAPES) - 1)
        self.shape = SHAPES[self.shape_index]
        self.color = PIECE_COLORS[self.shape_index]
        self.x = x
        self.y = y
        self.rotation = 0

    def rotate(self):
        self.shape = list(zip(*self.shape[::-1]))

class Tetris:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 30, bold=True)
        self.reset_game()

    def reset_game(self):
        self.board = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.game_over = False
        self.fall_time = 0
        self.fall_speed = 500 
        self.new_piece()

    def new_piece(self):
        if not hasattr(self, 'next_piece'):
            self.next_piece = Piece(GRID_WIDTH // 2 - 1, 0)
        
        self.current_piece = self.next_piece
        self.next_piece = Piece(GRID_WIDTH // 2 - 1, 0)
        
        if not self.is_valid_position(self.current_piece.shape, (self.current_piece.x, self.current_piece.y)):
            self.game_over = True

    def run(self):
        while True:
            self.handle_events()
            
            if not self.game_over:
                self.update()

            self.draw()
            self.clock.tick(60)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if self.game_over:
                    if event.key == pygame.K_r:
                        self.reset_game()
                    return

                if event.key == pygame.K_LEFT:
                    self.move(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    self.move(1, 0)
                elif event.key == pygame.K_DOWN:
                    self.drop()
                elif event.key == pygame.K_UP:
                    self.rotate()
                elif event.key == pygame.K_SPACE:
                    self.hard_drop()

    def update(self):
        self.fall_time += self.clock.get_rawtime()
        if self.fall_time > self.fall_speed:
            self.fall_time = 0
            self.drop()

    def move(self, dx, dy):
        new_x = self.current_piece.x + dx
        new_y = self.current_piece.y + dy
        if self.is_valid_position(self.current_piece.shape, (new_x, new_y)):
            self.current_piece.x = new_x
            self.current_piece.y = new_y

    def drop(self):
        new_y = self.current_piece.y + 1
        if self.is_valid_position(self.current_piece.shape, (self.current_piece.x, new_y)):
            self.current_piece.y = new_y
        else:
            self.lock_piece()

    def hard_drop(self):
        while self.is_valid_position(self.current_piece.shape, (self.current_piece.x, self.current_piece.y + 1)):
            self.current_piece.y += 1
        self.lock_piece()

    def rotate(self):
        original_shape = self.current_piece.shape
        self.current_piece.rotate()
        if not self.is_valid_position(self.current_piece.shape, (self.current_piece.x, self.current_piece.y)):
            
            self.current_piece.shape = original_shape

    def is_valid_position(self, shape, pos):
        px, py = pos
        for r, row_data in enumerate(shape):
            for c, cell in enumerate(row_data):
                if cell:
                    board_r, board_c = py + r, px + c
                    if not (0 <= board_r < GRID_HEIGHT and 0 <= board_c < GRID_WIDTH and self.board[board_r][board_c] == 0):
                        return False
        return True

    def lock_piece(self):
        for r, row_data in enumerate(self.current_piece.shape):
            for c, cell in enumerate(row_data):
                if cell:
                    self.board[self.current_piece.y + r][self.current_piece.x + c] = self.current_piece.shape_index + 1
        self.clear_lines()
        self.new_piece()

    def clear_lines(self):
        lines_to_clear = [r for r, row in enumerate(self.board) if all(row)]
        
        if lines_to_clear:
            for r in lines_to_clear:
                del self.board[r]
                self.board.insert(0, [0] * GRID_WIDTH)
            
            num_cleared = len(lines_to_clear)
            self.lines_cleared += num_cleared
            
            
            score_map = {1: 40, 2: 100, 3: 300, 4: 1200}
            self.score += score_map.get(num_cleared, 0) * self.level
            
            
            if self.lines_cleared >= self.level * 10:
                self.level += 1
                self.fall_speed = max(100, self.fall_speed - 50)

    def draw(self):
        self.screen.fill(BACKGROUND)
        self.draw_board()
        self.draw_piece(self.current_piece)
        self.draw_ui()
        if self.game_over:
            self.draw_game_over()
        pygame.display.flip()

    def draw_board(self):
        
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                rect = pygame.Rect(BOARD_X + c * CELL_SIZE, BOARD_Y + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.board[r][c] == 0:
                    pygame.draw.rect(self.screen, (40, 46, 56), rect)
                else:
                    pygame.draw.rect(self.screen, PIECE_COLORS[self.board[r][c] - 1], rect)
                pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)
        
        
        pygame.draw.rect(self.screen, TEXT_COLOR, (BOARD_X, BOARD_Y, BOARD_WIDTH, BOARD_HEIGHT), 3)

    def draw_piece(self, piece):
        for r, row_data in enumerate(piece.shape):
            for c, cell in enumerate(row_data):
                if cell:
                    rect = pygame.Rect(BOARD_X + (piece.x + c) * CELL_SIZE, BOARD_Y + (piece.y + r) * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, piece.color, rect)
                    pygame.draw.rect(self.screen, BACKGROUND, rect, 1)

    def draw_ui(self):
        
        score_text = self.font.render(f"Score: {self.score}", True, TEXT_COLOR)
        self.screen.blit(score_text, (20, 20))

        
        level_text = self.font.render(f"Level: {self.level}", True, TEXT_COLOR)
        self.screen.blit(level_text, (20, 60))

        
        next_text = self.font.render("Next:", True, TEXT_COLOR)
        self.screen.blit(next_text, (WINDOW_WIDTH - 180, 20))
        
        preview_x = WINDOW_WIDTH - 170
        preview_y = 70
        for r, row_data in enumerate(self.next_piece.shape):
            for c, cell in enumerate(row_data):
                if cell:
                    rect = pygame.Rect(preview_x + c * CELL_SIZE, preview_y + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, self.next_piece.color, rect)

    def draw_game_over(self):
        overlay = pygame.Surface((BOARD_WIDTH, BOARD_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (BOARD_X, BOARD_Y))
        
        text1 = self.font.render("GAME OVER", True, (255, 0, 0))
        text2 = self.small_font.render("Press 'R' to Restart", True, TEXT_COLOR)
        
        text1_rect = text1.get_rect(center=(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2 - 30))
        text2_rect = text2.get_rect(center=(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2 + 20))
        
        self.screen.blit(text1, text1_rect)
        self.screen.blit(text2, text2_rect)

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Tetris")
    game = Tetris(screen)
    game.run()
