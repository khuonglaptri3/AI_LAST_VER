import torch
import numpy as np
from environment import SnakeEnv
from model import Linear_QNet
from environment.utils import Direction


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


def get_state(env):
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
        if x < 0 or x >= env.snake.blocks_x or y < 0 or y >= env.snake.blocks_y:
            return True
        for b in env.snake.body:
            if (b.x, b.y) == (x, y):
                return True
        return False

    state = [
        (dir_r and collision(point_r))
        or (dir_l and collision(point_l))
        or (dir_u and collision(point_u))
        or (dir_d and collision(point_d)),
        (dir_u and collision(point_r))
        or (dir_d and collision(point_l))
        or (dir_l and collision(point_u))
        or (dir_r and collision(point_d)),
        (dir_d and collision(point_r))
        or (dir_u and collision(point_l))
        or (dir_r and collision(point_u))
        or (dir_l and collision(point_d)),
        dir_l,
        dir_r,
        dir_u,
        dir_d,
        env.snake.food.block.x < env.snake.head.x,
        env.snake.food.block.x > env.snake.head.x,
        env.snake.food.block.y < env.snake.head.y,
        env.snake.food.block.y > env.snake.head.y,
    ]

    return np.array(state, dtype=int)


def play():
    # Khởi tạo môi trường với render_mode="human" để xem rắn chơi
    options = {
        "fps": 10,
        "max_step": 500,
        "init_length": 3,
        "food_reward": 1.0,
        "dist_reward": 0.0,
        "living_bonus": 0.0,
        "death_penalty": -1.0,
        "width": 10,
        "height": 10,
        "block_size": 20,
    }
    env = SnakeEnv(render_mode="human", **options)

    # Load model đã train
    model = Linear_QNet(11, 256, 3)
    model.load_state_dict(torch.load(r"D:\FINAL_FROJECT_AI\AI_CONVERT\model\dqn_snake_3actions.pth"))

    model.eval()

    for episode in range(5):  # chơi thử 5 games
        state, _ = env.reset()
        done, truncated = False, False
        total_reward = 0

        while not (done or truncated):
            state_arr = get_state(env)
            state_tensor = torch.tensor(state_arr, dtype=torch.float)
            prediction = model(state_tensor)
            move = torch.argmax(prediction).item()

            final_move = [0, 0, 0]
            final_move[move] = 1

            action_idx = action_to_direction(final_move, env.snake.direction)
            next_state, reward, done, truncated, _ = env.step(action_idx)
            total_reward += reward

        print(f"Episode {episode+1} finished with Score: {env.snake.score}, Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    play()
