import math
import numpy as np
from .utils import *


class Snake:
    def __init__(
        self,
        fps=60,
        max_step=200,
        init_length=4,
        food_reward=1.0,
        dist_reward=0.0,
        living_bonus=0.0,
        death_penalty=-10.0,
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
        self.direction = 3
        self.current_step = 0
        self.head = Block(self.blocks_x//2, self.blocks_y//2, self.head_color)
        self.body = [self.head.copy(i, 0, self.body_color)
                     for i in range(-self.init_length, 0)]
        self.blocks = [self.food.block, self.head, *self.body]
        self.food.new_food(self.blocks)

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
        truncated = True if self.current_step == self.max_step else False
        (x, y) = (self.head.x, self.head.y)
        step = Direction.step(direction)
        if (direction == 0 or direction == 1) and (self.direction == 0 or self.direction == 1):
            step = Direction.step(self.direction)
        elif (direction == 2 or direction == 3) and (self.direction == 2 or self.direction == 3):
            step = Direction.step(self.direction)
        else:
            self.direction = direction
        self.head.x += step[0]
        self.head.y += step[1]

        reward = self.living_bonus + self.calc_reward()
        dead = False

        if self.head == self.food.block:
            self.score += 1
            self.grow(x, y)
            self.food.new_food(self.blocks)
            reward = self.food_reward
        else:
            self.move(x, y)
            for block in self.body:
                if self.head == block:
                    dead = True
            if self.head.x >= self.blocks_x or self.head.x < 0 or self.head.y < 0 or self.head.y >= self.blocks_x:
                dead = True
        if dead:
            reward = self.death_penalty
        return self.observation(), reward, dead, truncated

    def observation(self):
        obs = np.zeros((self.blocks_x, self.blocks_y, 3), dtype=np.float32)
        if 0 <= self.head.x < self.blocks_x and 0 <= self.head.y < self.blocks_y:
            obs[self.head.x][self.head.y][0] = 1
        for block in self.blocks:
            if 0 <= block.x < self.blocks_x and 0 <= block.y < self.blocks_y:
                obs[block.x][block.y][1] = 1
        obs[self.food.block.x][self.food.block.y][2] = 1
        return obs

    def calc_reward(self):
        reward = 0.0

        # shaping reward theo khoảng cách (có sẵn)
        if self.dist_reward != 0.0:
            x = self.head.x - self.food.block.x
            y = self.head.y - self.food.block.y
            d = math.sqrt(x*x + y*y)
            reward += (self.dist_reward - d) / self.dist_reward

        # --- NEW: shaping reward theo hướng đi ---
        # vector từ đầu rắn tới food
        food_vec = np.array([self.food.block.x - self.head.x,
                            self.food.block.y - self.head.y], dtype=np.float32)
        # vector hướng di chuyển hiện tại
        dir_vec = np.array(Direction.step(self.direction), dtype=np.float32)

        if np.linalg.norm(food_vec) > 0:
            cos_theta = np.dot(dir_vec, food_vec) / (
                np.linalg.norm(dir_vec) * np.linalg.norm(food_vec)
            )
            reward += 0.1 * cos_theta   # hệ số 0.1 để không quá mạnh

        return reward


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
