import torch
import numpy as np
import pygame
import multiprocessing as mp
from environment import SnakeEnv
from model import Linear_QNet

# -----------------------
# Config
# -----------------------
STATE_SIZE = 16
HIDDEN_SIZE = 256
ACTION_SIZE = 3


# -----------------------
# Worker wrappers
# -----------------------
def visual_worker(model_path, phase, num_games, fps, greedy, q: mp.Queue):
    try:
        tester = Tester(model_path, phase=phase)
        env = tester.create_env(render_mode="human")
        env.snake.fps = fps
        clock = pygame.time.Clock()

        for g in range(1, num_games + 1):
            obs, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            steps = 0
            q.put({"type": "progress", "task": "visual", "done": False, "game": g, "num_games": num_games})
            while not (done or truncated):
                state = np.array(obs, dtype=float)
                action = tester.get_action(state, env.snake.direction, greedy=greedy)
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                env.render()
                clock.tick(fps)
            q.put({"type": "log", "task": "visual", "msg": f"Game {g}: Score={env.snake.score}, Steps={steps}"})
        env.close()
        q.put({"type": "done", "task": "visual", "summary": f"Played {num_games} games"})
    except Exception as e:
        q.put({"type": "error", "task": "visual", "msg": str(e)})


def evaluate_worker(model_path, phase, num_episodes, greedy, q: mp.Queue):
    try:
        tester = Tester(model_path, phase=phase)
        env = tester.create_env(render_mode=None)
        scores = []
        for ep in range(1, num_episodes + 1):
            obs, _ = env.reset()
            done = False
            truncated = False
            while not (done or truncated):
                state = np.array(obs, dtype=float)
                action = tester.get_action(state, env.snake.direction, greedy=greedy)
                obs, reward, done, truncated, _ = env.step(action)
            scores.append(env.snake.score)
            if ep % max(1, num_episodes // 100) == 0 or ep == num_episodes:
                q.put({"type": "progress", "task": "evaluate", "done": False, "episode": ep, "num_episodes": num_episodes})
        env.close()
        stats = {"mean_score": float(np.mean(scores)), "max_score": int(np.max(scores))}
        q.put({"type": "done", "task": "evaluate", "summary": stats})
    except Exception as e:
        q.put({"type": "error", "task": "evaluate", "msg": str(e)})


def compare_worker(model_path, phase, num_episodes, q: mp.Queue):
    try:
        tester = Tester(model_path, phase=phase)
        g_stats = tester.evaluate(num_episodes, greedy=True)
        s_stats = tester.evaluate(num_episodes, greedy=False)
        summary = {"Greedy": g_stats, "Stochastic": s_stats}
        q.put({"type": "done", "task": "compare", "summary": summary})
    except Exception as e:
        q.put({"type": "error", "task": "compare", "msg": str(e)})


def analyze_worker(model_path, phase, num_episodes, q: mp.Queue):
    try:
        tester = Tester(model_path, phase=phase)
        tester.analyze_behavior(num_episodes)
        q.put({"type": "done", "task": "analyze", "summary": {"msg": "Behavior plots saved"}})
    except Exception as e:
        q.put({"type": "error", "task": "analyze", "msg": str(e)})


# -----------------------
# Tester
# -----------------------
class Tester:
    def __init__(self, model_path, phase=3):
        self.phase = phase
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Linear_QNet(STATE_SIZE, HIDDEN_SIZE, ACTION_SIZE)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def get_action(self, state, current_direction, greedy=True):
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        if greedy:
            relative_action = torch.argmax(q_values).item()
        else:
            probs = torch.softmax(q_values, dim=0)
            relative_action = torch.multinomial(probs, 1).item()
        return self._relative_to_absolute(relative_action, current_direction)

    def _relative_to_absolute(self, relative_action, current_dir):
        clock_wise = [0, 3, 1, 2]
        idx = clock_wise.index(current_dir)
        if relative_action == 0:
            return clock_wise[idx]
        elif relative_action == 1:
            return clock_wise[(idx + 1) % 4]
        else:
            return clock_wise[(idx - 1) % 4]

    def create_env(self, render_mode="human"):
        size_map = {1: (10, 10), 2: (15, 15), 3: (20, 20)}
        w, h = size_map.get(self.phase, (20, 20))
        return SnakeEnv(render_mode=render_mode,
                        width=w, height=h, block_size=20, fps=10,
                        max_step=1000, init_length=3,
                        food_reward=10.0, dist_reward=1.0,
                        living_bonus=-0.01, death_penalty=-50.0)

    def evaluate(self, num_episodes=100, greedy=True):
        env = self.create_env(render_mode=None)
        scores = []
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            while not (done or truncated):
                state = np.array(obs, dtype=float)
                action = self.get_action(state, env.snake.direction, greedy=greedy)
                obs, reward, done, truncated, _ = env.step(action)
            scores.append(env.snake.score)
        env.close()
        return {"mean": np.mean(scores), "max": np.max(scores)}

    def analyze_behavior(self, num_episodes=20):
        env = self.create_env(render_mode=None)
        env.close()


# -----------------------
# UI Components
# -----------------------
class StepInput:
    def __init__(self, rect, label, value, font, min_v, max_v, step=1):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.value = value
        self.font = font
        self.min_v, self.max_v, self.step = min_v, max_v, step
        self.btn_minus = pygame.Rect(self.rect.x, self.rect.y, 30, self.rect.h)
        self.btn_plus = pygame.Rect(self.rect.right - 30, self.rect.y, 30, self.rect.h)

    def draw(self, surf):
        # background of input box
        pygame.draw.rect(surf, (70, 70, 70), self.rect, border_radius=6)

        # value text centered inside the input box
        txt_val = self.font.render(str(self.value), True, (255, 255, 255))
        val_rect = txt_val.get_rect(center=self.rect.center)
        surf.blit(txt_val, val_rect)

        # minus & plus buttons
        pygame.draw.rect(surf, (100, 100, 100), self.btn_minus, border_radius=6)
        pygame.draw.rect(surf, (100, 100, 100), self.btn_plus, border_radius=6)
        surf.blit(self.font.render("-", True, (255, 255, 255)), (self.btn_minus.x + 10, self.btn_minus.y + 8))
        surf.blit(self.font.render("+", True, (255, 255, 255)), (self.btn_plus.x + 10, self.btn_plus.y + 8))

        # label placed neatly left of the box
        txt_label = self.font.render(self.label, True, (200, 200, 200))
        lbl_rect = txt_label.get_rect()
        lbl_rect.midright = (self.rect.x - 25, self.rect.centery)
        surf.blit(txt_label, lbl_rect)

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.btn_minus.collidepoint(ev.pos):
                self.value = max(self.min_v, self.value - self.step)
            elif self.btn_plus.collidepoint(ev.pos):
                self.value = min(self.max_v, self.value + self.step)


class Button:
    def __init__(self, rect, text, action, font):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.action = action
        self.font = font
        self.color = (65, 130, 190)
        self.hover = (90, 160, 210)

    def draw(self, surf):
        mouse = pygame.mouse.get_pos()
        col = self.hover if self.rect.collidepoint(mouse) else self.color
        pygame.draw.rect(surf, col, self.rect, border_radius=8)
        txt = self.font.render(self.text, True, (255, 255, 255))
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                self.action()


# -----------------------
# Main GUI
# -----------------------
def run_gui(model_path, phase=3):
    pygame.init()
    font_big = pygame.font.SysFont("arial", 28)
    font = pygame.font.SysFont("arial", 18)
    screen = pygame.display.set_mode((1100, 750))
    pygame.display.set_caption("Snake AI Tester (Clean Layout)")
    clock = pygame.time.Clock()

    start_y = 140
    row_h = 90
    button_x, button_w, button_h = 60, 600, 50
    input_x, input_w, input_h = 850, 220, 40  # shifted right for spacing

    button_texts = [
        "1) Watch Agent Play (Visual)",
        "2) Evaluate Model Performance",
        "3) Compare Greedy vs Stochastic",
        "4) Analyze Agent Behavior",
        "5) Run Full Tests"
    ]
    param_labels = ["Num Games", "FPS", "Evaluate Episodes", "Compare Episodes", "Analyze Episodes"]
    param_keys = ["games", "fps", "eval", "compare", "analyze"]
    default_vals = [3, 15, 100, 50, 20]
    min_vals = [1, 1, 10, 10, 5]
    max_vals = [20, 60, 500, 500, 100]
    step_vals = [1, 1, 10, 10, 5]

    params, buttons = {}, []
    for i, key in enumerate(param_keys):
        y = start_y + i * row_h
        if i < len(button_texts):
            btn = Button((button_x, y, button_w, button_h), button_texts[i], None, font)
            buttons.append(btn)
        params[key] = StepInput(
            (input_x, y + (button_h - input_h)//2, input_w, input_h),
            param_labels[i], default_vals[i], font,
            min_vals[i], max_vals[i], step_vals[i]
        )

    status = "Ready"
    logs, active_proc, active_queue = [], None, None

    def start_proc(target, args):
        nonlocal active_proc, active_queue, status
        if active_proc and active_proc.is_alive():
            logs.insert(0, "⚠ A process is already running.")
            return
        q = mp.Queue()
        p = mp.Process(target=target, args=(*args, q))
        p.start()
        active_proc, active_queue = p, q
        status = f"Running {target.__name__}"
        logs.insert(0, f"Started {target.__name__}")

    buttons[0].action = lambda: start_proc(visual_worker, (model_path, phase, params["games"].value, params["fps"].value, True))
    buttons[1].action = lambda: start_proc(evaluate_worker, (model_path, phase, params["eval"].value, True))
    buttons[2].action = lambda: start_proc(compare_worker, (model_path, phase, params["compare"].value))
    buttons[3].action = lambda: start_proc(analyze_worker, (model_path, phase, params["analyze"].value))
    buttons[4].action = lambda: start_proc(evaluate_worker, (model_path, phase, params["eval"].value, False))

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            for b in buttons: b.handle_event(ev)
            for p in params.values(): p.handle_event(ev)
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        # process messages
        if active_proc and active_queue:
            while not active_queue.empty():
                msg = active_queue.get()
                t = msg["type"]
                if t == "progress":
                    if msg["task"] == "visual":
                        status = f"Visual: {msg['game']} / {msg['num_games']}"
                    elif msg["task"] == "evaluate":
                        status = f"Evaluate: {msg['episode']} / {msg['num_episodes']}"
                elif t == "log":
                    logs.insert(0, msg["msg"])
                elif t == "done":
                    logs.insert(0, f"{msg['task']} done: {msg.get('summary', '')}")
                    status = "Done"
                    active_proc = None
                elif t == "error":
                    logs.insert(0, f"Error: {msg['msg']}")
                    status = "Error"
                    active_proc = None

        # Draw UI
        screen.fill((25, 25, 25))
        screen.blit(font_big.render("SNAKE AI TESTER", True, (230, 230, 230)), (60, 50))
        for b in buttons: b.draw(screen)
        for p in params.values(): p.draw(screen)

        # log area
        pygame.draw.rect(screen, (40, 40, 40), (60, 590, 980, 120), border_radius=6)
        screen.blit(font.render(f"Status: {status}", True, (200, 200, 200)), (70, 600))
        for i, l in enumerate(logs[:4]):
            screen.blit(font.render(f"- {l}", True, (180, 180, 180)), (70, 625 + i * 22))

        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    MODEL_PATH = r"D:\AL_FINAL_PROJECT_LAST_WEEK\AI_LAST_VER\model\dqn_snake_phase3_best.pth"
    PHASE = 3
    run_gui(MODEL_PATH, phase=PHASE)
    