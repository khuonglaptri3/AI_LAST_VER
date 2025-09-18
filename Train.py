import torch
import random
import numpy as np
from collections import deque
from environment import SnakeEnv
from model import Linear_QNet, QTrainer
from environment.utils import Direction

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self, state_size=11, hidden_size=256, action_size=3):
        self.n_games = 0
        self.epsilon = 1.0  # ban đầu full exploration
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        # Online model
        self.model = Linear_QNet(state_size, hidden_size, action_size)

        # Target model
        self.target_model = Linear_QNet(state_size, hidden_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())  # copy weight ban đầu

        self.trainer = QTrainer(
            self.model, lr=LR, gamma=self.gamma, target_model=self.target_model
        )

    def get_state(self, env):
        """Sinh state 11 chiều giống Patrick"""
        head_x, head_y = env.snake.head.x, env.snake.head.y
        dir = env.snake.direction

        # Các điểm liền kề
        point_l = (head_x - 1, head_y)
        point_r = (head_x + 1, head_y)
        point_u = (head_x, head_y - 1)
        point_d = (head_x, head_y + 1)

        dir_l = dir == 2
        dir_r = dir == 3
        dir_u = dir == 0
        dir_d = dir == 1

        def collision(p):
            x, y = p
            # check tường
            if x < 0 or x >= env.snake.blocks_x or y < 0 or y >= env.snake.blocks_y:
                return True
            # check thân
            for b in env.snake.body:
                if (b.x, b.y) == (x, y):
                    return True
            return False

        state = [
            # Danger straight
            (dir_r and collision(point_r))
            or (dir_l and collision(point_l))
            or (dir_u and collision(point_u))
            or (dir_d and collision(point_d)),

            # Danger right
            (dir_u and collision(point_r))
            or (dir_d and collision(point_l))
            or (dir_l and collision(point_u))
            or (dir_r and collision(point_d)),

            # Danger left
            (dir_d and collision(point_r))
            or (dir_u and collision(point_l))
            or (dir_r and collision(point_u))
            or (dir_l and collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            env.snake.food.block.x < env.snake.head.x,  # food left
            env.snake.food.block.x > env.snake.head.x,  # food right
            env.snake.food.block.y < env.snake.head.y,  # food up
            env.snake.food.block.y > env.snake.head.y,  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # def get_action(self, state):
    #     """epsilon-greedy chọn action"""
    #     self.epsilon = 80 - self.n_games
    #     final_move = [0, 0, 0]
    #     if random.randint(0, 200) < self.epsilon:
    #         move = random.randint(0, 2)
    #         final_move[move] = 1
    #     else:
    #         state0 = torch.tensor(state, dtype=torch.float)
    #         prediction = self.model(state0)
    #         move = torch.argmax(prediction).item()
    #         final_move[move] = 1
    #     return final_move
    def get_action(self, state):
        """epsilon-greedy chọn action với exponential decay"""
        epsilon_min = 0.01
        decay_rate = 0.995
        self.epsilon = max(epsilon_min, self.epsilon * decay_rate) if self.n_games > 0 else 1.0

        final_move = [0, 0, 0]
        if random.random() < self.epsilon:  # exploration
            move = random.randint(0, 2)
            final_move[move] = 1
        else:  # exploitation
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move



def action_to_direction(action, current_dir):
    """
    Map từ action 3 chiều (straight, right, left)
    sang 4 hướng (0: up, 1: down, 2: left, 3: right).
    """
    clock_wise = [0, 3, 1, 2]  # up, right, down, left
    idx = clock_wise.index(current_dir)

    if np.array_equal(action, [1, 0, 0]):  # straight
        new_dir = clock_wise[idx]
    elif np.array_equal(action, [0, 1, 0]):  # right turn
        new_dir = clock_wise[(idx + 1) % 4]
    else:  # left turn
        new_dir = clock_wise[(idx - 1) % 4]
    return new_dir


def train():
    options = {
        "fps": 0,
        "max_step": 200,
        "init_length": 3,
        "food_reward": 1.0,
        "dist_reward": 0.0,
        "living_bonus": 0.0,
        "death_penalty": -1.0,
        "width": 10,
        "height": 10,
        "block_size": 20,
    }
    env = SnakeEnv(render_mode=None, **options)
    agent = Agent()
    record = 0
    for episode in range(500):  # train 500 games
        env.reset()  # reset môi trường
        state_old = agent.get_state(env)
        done = False
        truncated = False
        total_reward = 0
        while not (done or truncated):
            final_move = agent.get_action(state_old)
            action_idx = action_to_direction(final_move, env.snake.direction)

            next_state, reward, done, truncated, _ = env.step(action_idx)
            state_new = agent.get_state(env)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            state_old = state_new
            total_reward += reward

        agent.n_games += 1
        agent.train_long_memory()
        agent.trainer.soft_update_target()

        if env.snake.score > record:
            record = env.snake.score
            agent.model.save("dqn_snake_3actions.pth")

        print(
            f"Game {agent.n_games}, Score: {env.snake.score}, Record: {record}, Reward: {total_reward}"
        )

    env.close()


if __name__ == "__main__":
    
    train()
    
