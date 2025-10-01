import math
import numpy as np
from .utils import *


class Snake:
    def __init__(
        self,
        fps=60,
        max_step=200,
        init_length=4,
        food_reward=10.0,          # thưởng ăn food mạnh hơn
        dist_reward=1.0,           # bật shaping theo khoảng cách
        living_bonus=-0.01,        # penalty nhỏ mỗi bước
        death_penalty=-50.0,       # phạt chết nặng hơn
        width=16,
        height=16,
        block_size=20,
        background_color=Color.black,
        food_color=Color.green,
        head_color=Color.grey,
        body_color=Color.white,
    ) -> None:

        self.episode = 0
        self.fps = fps
        self.max_step = max_step
        self.init_length = min(init_length, width//2)
        self.food_reward = food_reward
        self.dist_reward = dist_reward
        self.living_bonus = living_bonus
        self.death_penalty = death_penalty
        self.blocks_x = width
        self.blocks_y = height
        self.food_color = food_color
        self.head_color = head_color
        self.body_color = body_color
        self.background_color = background_color
        self.food = Food(self.blocks_x, self.blocks_y, food_color)
        Block.size = block_size

        self.screen = None
        self.clock = None
        self.human_playing = False

    def init(self):
        self.episode += 1
        self.score = 0
        self.direction = 3  # right
        self.current_step = 0
        self.head = Block(self.blocks_x//2, self.blocks_y//2, self.head_color)
        self.body = [self.head.copy(i, 0, self.body_color)
                     for i in range(-self.init_length, 0)]
        self.blocks = [self.food.block, self.head, *self.body]
        self.food.new_food(self.blocks)

        # Dynamic max_step
        self.max_step = 50 * len(self.body)

    def close(self):
        pygame.quit()
        pygame.display.quit()
        self.screen = None
        self.clock = None

    def render(self):
        if self.screen is None:
            self.screen, self.clock = game_start(
                self.blocks_x*Block.size, self.blocks_y*Block.size)
        self.clock.tick(self.fps)
        update_screen(self.screen, self)
        handle_input()

    def step(self, direction):
        if direction is None:
            direction = self.direction
        self.current_step += 1
        truncated = True if self.current_step >= self.max_step else False

        (x, y) = (self.head.x, self.head.y)
        step = Direction.step(direction)
        if (direction in [0, 1]) and (self.direction in [0, 1]):
            step = Direction.step(self.direction)
        elif (direction in [2, 3]) and (self.direction in [2, 3]):
            step = Direction.step(self.direction)
        else:
            self.direction = direction
        self.head.x += step[0]
        self.head.y += step[1]

        reward = self.living_bonus
        dead = False

        if self.head == self.food.block:
            self.score += 1
            self.grow(x, y)
            self.food.new_food(self.blocks)
            reward = self.food_reward
            # reset step count khi ăn được food
            self.current_step = 0
            self.max_step = 50 * len(self.body)
        else:
            # distance shaping
            old_dist = math.dist((x, y), (self.food.block.x, self.food.block.y))
            new_dist = math.dist((self.head.x, self.head.y),
                                 (self.food.block.x, self.food.block.y))
            if new_dist < old_dist:
                reward += 0.1
            else:
                reward -= 0.05

            self.move(x, y)
            for block in self.body:
                if self.head == block:
                    dead = True
            if (self.head.x >= self.blocks_x or self.head.x < 0 or
                self.head.y < 0 or self.head.y >= self.blocks_y):
                dead = True

        if dead:
            reward = self.death_penalty

        return self.observation(), reward, dead, truncated

    def observation(self):
        """Trả về vector 15 features (thay vì grid 3 kênh)."""
        head_x, head_y = self.head.x, self.head.y
        dir = self.direction

        # danger detection
        def collision(p):
            x, y = p
            if x < 0 or x >= self.blocks_x or y < 0 or y >= self.blocks_y:
                return True
            for b in self.body:
                if (b.x, b.y) == (x, y):
                    return True
            return False

        point_l = (head_x - 1, head_y)
        point_r = (head_x + 1, head_y)
        point_u = (head_x, head_y - 1)
        point_d = (head_x, head_y + 1)

        dir_l = dir == 2
        dir_r = dir == 3
        dir_u = dir == 0
        dir_d = dir == 1

        danger_straight = (dir_r and collision(point_r)) or \
                          (dir_l and collision(point_l)) or \
                          (dir_u and collision(point_u)) or \
                          (dir_d and collision(point_d))

        danger_right = (dir_u and collision(point_r)) or \
                       (dir_d and collision(point_l)) or \
                       (dir_l and collision(point_u)) or \
                       (dir_r and collision(point_d))

        danger_left = (dir_d and collision(point_r)) or \
                      (dir_u and collision(point_l)) or \
                      (dir_r and collision(point_u)) or \
                      (dir_l and collision(point_d))

        food_left = self.food.block.x < head_x
        food_right = self.food.block.x > head_x
        food_up = self.food.block.y < head_y
        food_down = self.food.block.y > head_y

        # extra normalized features
        dx = (self.food.block.x - head_x) / max(1, self.blocks_x - 1)
        dy = (self.food.block.y - head_y) / max(1, self.blocks_y - 1)
        snake_len = len(self.body) / float(self.blocks_x * self.blocks_y)
        head_norm_x = head_x / max(1, self.blocks_x - 1)
        head_norm_y = head_y / max(1, self.blocks_y - 1)

        state = [
            danger_straight, danger_right, danger_left,
            dir_l, dir_r, dir_u, dir_d,
            food_left, food_right, food_up, food_down,
            dx, dy, snake_len, head_norm_x, head_norm_y,
        ]
        return np.array(state, dtype=np.float32)

    def calc_reward(self):
        # đã xử lý trong step()
        return 0.0

    def grow(self, x, y):
        body = Block(x, y, self.body_color)
        self.blocks.append(body)
        self.body.append(body)

    def move(self, x, y):
        tail = self.body.pop(0)
        tail.move_to(x, y)
        self.body.append(tail)

    def info(self):
        return {
            'head': (self.head.x, self.head.y),
            'food': (self.food.block.x, self.food.block.y),
        }

    def play(self, fps=10, acceleration=True, step=1, frep=10):
        self.max_step = 99999
        self.fps = fps
        self.food_reward = 1
        self.living_bonus = 0
        self.dist_reward = 0
        self.death_penalty = 0
        self.human_playing = True
        self.init()
        screen, clock = game_start(
            self.blocks_x*Block.size, self.blocks_y*Block.size)
        total_r = 0

        while pygame.get_init():
            clock.tick(self.fps)
            _, r, d, _ = self.step(handle_input())
            total_r += r
            if acceleration and total_r == frep:
                self.fps += step
                total_r = 0
            if d:
                self.init()
                total_r = 0
                self.fps = fps
            update_screen(screen, self, True)
