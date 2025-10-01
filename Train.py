# Train.py (final, obs-driven)
import torch
import random
import numpy as np
from environment import SnakeEnv
from model import Linear_QNet, QTrainer

# =========================
# Config
# =========================
MAX_MEMORY = 100_000
BATCH_SIZE = 256
LR = 0.001
STATE_SIZE = 16   # CHANGED: đúng bằng số feature từ Snake.observation()
HIDDEN_SIZE = 256
ACTION_SIZE = 3
GAMMA = 0.95

# =========================
# Prioritized Replay Buffer
# =========================
class PrioritizedReplayBuffer:
    def __init__(self, capacity=MAX_MEMORY, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, transition, error=None):
        """transition = (state, action, reward, next_state, done)"""
        max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
        prio = abs(error) + 1e-6 if error is not None else max_prio

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = prio ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], None, None

        prios = self.priorities[: len(self.buffer)]
        probs = prios / prios.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-8)
        weights = torch.tensor(weights, dtype=torch.float32)

        return samples, indices, weights, probs

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            prio = abs(err) + 1e-6
            self.priorities[idx] = prio ** self.alpha

    def __len__(self):
        return len(self.buffer)

# =========================
# Agent
# =========================
class Agent:
    def __init__(self,
                 state_size=STATE_SIZE,
                 hidden_size=HIDDEN_SIZE,
                 action_size=ACTION_SIZE,
                 lr=LR,
                 gamma=GAMMA,
                 memory_capacity=MAX_MEMORY):
        self.n_games = 0
        self.gamma = gamma

        # epsilon control
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.step_count = 0

        # replay memory
        self.memory = PrioritizedReplayBuffer(capacity=memory_capacity, alpha=0.6)

        # models
        self.model = Linear_QNet(state_size, hidden_size, action_size)
        self.target_model = Linear_QNet(state_size, hidden_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())

        self.trainer = QTrainer(self.model, lr=lr, gamma=self.gamma,
                                target_model=self.target_model, tau=0.01)

        self.batch_size = BATCH_SIZE
        self.beta_start = 0.4
        self.beta_frames = 200000

    def remember(self, state, action, reward, next_state, done, error=None):
        self.memory.push((state, action, reward, next_state, done), error=error)

    def get_action(self, state):
        self.step_count += 1
        if self.step_count > 1:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return
        beta = min(1.0, self.beta_start +
                   (1.0 - self.beta_start) * (self.step_count / self.beta_frames))
        mini_sample, indices, weights, _ = self.memory.sample(self.batch_size, beta=beta)
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        loss, td_errors = self.trainer.train_step(states, actions, rewards,
                                                  next_states, dones, weights=weights)
        td_errs_np = td_errors.detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errs_np)
        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        loss, td_errors = self.trainer.train_step([state], [action], [reward],
                                                  [next_state], [done], weights=None)
        try:
            err = td_errors.detach().cpu().numpy()[0]
        except Exception:
            err = None
        self.remember(state, action, reward, next_state, done, error=err)
        return loss

# =========================
# Helpers
# =========================
def action_to_direction(action, current_dir):
    # clockwise order = [0,3,1,2] (up, right, down, left)
    clock_wise = [0, 3, 1, 2]
    idx = clock_wise.index(current_dir)
    if np.array_equal(action, [1, 0, 0]):  # straight
        new_dir = clock_wise[idx]
    elif np.array_equal(action, [0, 1, 0]):  # right
        new_dir = clock_wise[(idx + 1) % 4]
    else:  # left
        new_dir = clock_wise[(idx - 1) % 4]
    return new_dir

# =========================
# Training Loop
# =========================
def train(num_episodes=3000, phase=1, load_model=False, model_path="dqn_snake_per.pth"):
    if phase == 1:
        w, h = 10, 10
    elif phase == 2:
        w, h = 15, 15
    elif phase == 3:
        w, h = 25, 25
    else:
        raise ValueError("Phase chỉ nhận 1, 2 hoặc 3")

    options = {
        "fps": 0,
        "max_step": 600,
        "init_length": 3,
        "food_reward": 10.0,
        "dist_reward": 1.0,
        "living_bonus": -0.01,
        "death_penalty": -50.0,
        "width": w,
        "height": h,
        "block_size": 20,
    }

    env = SnakeEnv(render_mode=None, **options)
    agent = Agent(state_size=STATE_SIZE,
                  hidden_size=HIDDEN_SIZE,
                  action_size=ACTION_SIZE,
                  lr=LR,
                  gamma=GAMMA)

    device = torch.device("cuda" if (load_model and torch.cuda.is_available()) else "cpu")
    if load_model and model_path:
        try:
            print(f"Loading model from {model_path} ...")
            agent.model.load_state_dict(torch.load(model_path, map_location=device))
            agent.target_model.load_state_dict(agent.model.state_dict())
        except Exception as e:
            print(f"Could not load model: {e}. Starting fresh.")

    record = 0
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        state_old = np.array(obs, dtype=float)

        done, truncated = False, False
        total_reward = 0.0

        while not (done or truncated):
            final_move = agent.get_action(state_old)
            action_idx = action_to_direction(final_move, env.snake.direction)

            next_obs, reward, done, truncated, _ = env.step(action_idx)
            state_new = np.array(next_obs, dtype=float)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            state_old = state_new
            total_reward += reward

        agent.n_games += 1
        loss = agent.train_long_memory()
        agent.trainer.soft_update_target()

        if env.snake.score > record:
            record = env.snake.score
            agent.model.save(f"dqn_snake_phase{phase}.pth")

        print(
            f"Episode {episode}/{num_episodes} | Score: {env.snake.score} | Record: {record} | Reward: {total_reward:.3f} | Eps: {agent.epsilon:.4f} | Loss: {loss if loss is not None else 'N/A'}"
        )

    env.close()


if __name__ == "__main__":
    train(num_episodes=20000, phase=2,
          load_model=True,
          model_path="D:\\AL_FINAL_PROJECT_LAST_WEEK\\AI_LAST_VER\\model\\dqn_snake_per.pth")
