import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from environment import SnakeEnv
from model import Linear_QNet

# =========================
# Config
# =========================
STATE_SIZE = 16
HIDDEN_SIZE = 256
ACTION_SIZE = 3  # Model đã train với 3 actions (straight/right/left)

class Tester:
    def __init__(self, model_path, phase=3):
        """
        model_path: đường dẫn tới file .pth
        phase: 1 (10x10), 2 (15x15), 3 (20x20)
        """
        self.phase = phase
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = Linear_QNet(STATE_SIZE, HIDDEN_SIZE, ACTION_SIZE)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # evaluation mode
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"📱 Device: {self.device}")
    
    def get_action(self, state, current_direction, greedy=True):
        """
        Lấy action từ model và convert sang absolute direction
        Model output: [straight, right, left]
        """
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        if greedy:
            relative_action = torch.argmax(q_values).item()
        else:
            probs = torch.softmax(q_values, dim=0)
            relative_action = torch.multinomial(probs, 1).item()
        
        # Convert relative action to absolute direction
        return self._relative_to_absolute(relative_action, current_direction)
    
    def _relative_to_absolute(self, relative_action, current_dir):
        """
        Convert relative action (0=straight, 1=right, 2=left) 
        to absolute direction (0=up, 1=down, 2=left, 3=right)
        """
        clock_wise = [0, 3, 1, 2]  # up, right, down, left
        idx = clock_wise.index(current_dir)
        
        if relative_action == 0:  # straight
            return clock_wise[idx]
        elif relative_action == 1:  # turn right
            return clock_wise[(idx + 1) % 4]
        else:  # turn left
            return clock_wise[(idx - 1) % 4]
    
    def create_env(self, render_mode="human"):
        """Tạo environment theo phase"""
        if self.phase == 1:
            w, h = 10, 10
        elif self.phase == 2:
            w, h = 15, 15
        elif self.phase == 3:
            w, h = 20, 20
        else:
            raise ValueError("Phase phải là 1, 2 hoặc 3")
        
        options = {
            "fps": 10,  # tốc độ hiển thị
            "max_step": 1000,
            "init_length": 3,
            "food_reward": 10.0,
            "dist_reward": 1.0,
            "living_bonus": -0.01,
            "death_penalty": -50.0,
            "width": w,
            "height": h,
            "block_size": 20,
        }
        
        return SnakeEnv(render_mode=render_mode, **options)
    
    # =========================
    # TEST 1: Chơi visual với pygame
    # =========================
    def play_visual(self, num_games=5, fps=10, greedy=True):
        """
        Xem agent chơi trực tiếp với pygame
        num_games: số game muốn chơi
        fps: tốc độ hiển thị
        greedy: True = chọn action tốt nhất, False = stochastic
        """
        env = self.create_env(render_mode="human")
        env.snake.fps = fps
        
        for game in range(1, num_games + 1):
            obs, _ = env.reset()
            done, truncated = False, False
            total_reward = 0
            steps = 0
            
            print(f"\n{'='*50}")
            print(f"🎮 Game {game}/{num_games}")
            print(f"{'='*50}")
            
            while not (done or truncated):
                state = np.array(obs, dtype=float)
                current_dir = env.snake.direction
                action = self.get_action(state, current_dir, greedy=greedy)
                
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # Hiển thị
                env.render()
                time.sleep(1.0 / fps)
            
            print(f" Kết quả:")
            print(f"   Score: {env.snake.score}")
            print(f"   Steps: {steps}")
            print(f"   Total Reward: {total_reward:.2f}")
            print(f"   Reason: {' Dead' if done else ' Timeout'}")
            
            if game < num_games:
                input("\n  Press Enter để chơi game tiếp theo...")
        
        env.close()
    
    # =========================
    # TEST 2: Đánh giá performance (không hiển thị)
    # =========================
    def evaluate(self, num_episodes=100, greedy=True):
        """
        Đánh giá performance trên nhiều episodes
        Trả về statistics
        """
        env = self.create_env(render_mode=None)
        
        scores = []
        steps_list = []
        rewards_list = []
        reasons = {"dead": 0, "timeout": 0}
        
        print(f"\n🔬 Đang đánh giá model trên {num_episodes} episodes...")
        
        for ep in range(1, num_episodes + 1):
            obs, _ = env.reset()
            done, truncated = False, False
            total_reward = 0
            steps = 0
            
            while not (done or truncated):
                state = np.array(obs, dtype=float)
                current_dir = env.snake.direction
                action = self.get_action(state, current_dir, greedy=greedy)
                obs, reward, done, truncated, _ = env.step(action)
                
                total_reward += reward
                steps += 1
            
            scores.append(env.snake.score)
            steps_list.append(steps)
            rewards_list.append(total_reward)
            
            if done:
                reasons["dead"] += 1
            else:
                reasons["timeout"] += 1
            
            if ep % 10 == 0:
                print(f"Progress: {ep}/{num_episodes} episodes")
        
        env.close()
        
        # Tính statistics
        stats = {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "max_score": np.max(scores),
            "min_score": np.min(scores),
            "mean_steps": np.mean(steps_list),
            "mean_reward": np.mean(rewards_list),
            "success_rate": (reasons["timeout"] / num_episodes) * 100,
            "death_rate": (reasons["dead"] / num_episodes) * 100,
        }
        
        # In kết quả
        print(f"\n{'='*60}")
        print(f" KẾT QUẢ ĐÁNH GIÁ ({num_episodes} episodes)")
        print(f"{'='*60}")
        print(f"Score Statistics:")
        print(f"   Mean ± Std: {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")
        print(f"   Max: {stats['max_score']}")
        print(f"   Min: {stats['min_score']}")
        print(f"\n  Steps/Rewards:")
        print(f"   Mean Steps: {stats['mean_steps']:.1f}")
        print(f"   Mean Reward: {stats['mean_reward']:.2f}")
        print(f"\n Death Analysis:")
        print(f"   Survival Rate: {stats['success_rate']:.1f}%")
        print(f"   Death Rate: {stats['death_rate']:.1f}%")
        print(f"{'='*60}\n")
        
        # Vẽ biểu đồ
        self._plot_evaluation(scores, steps_list, rewards_list)
        
        return stats, scores, steps_list, rewards_list
    
    def _plot_evaluation(self, scores, steps, rewards):
        """Vẽ biểu đồ kết quả evaluation"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Model Evaluation Results - Phase {self.phase}', fontsize=16)
        
        # Score histogram
        axes[0, 0].hist(scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(scores):.2f}')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Score over episodes
        axes[0, 1].plot(scores, alpha=0.6, linewidth=1)
        axes[0, 1].axhline(np.mean(scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(scores):.2f}')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Score over Episodes')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Steps histogram
        axes[1, 0].hist(steps, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(np.mean(steps), color='red', linestyle='--',
                          label=f'Mean: {np.mean(steps):.1f}')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Steps Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Reward histogram
        axes[1, 1].hist(rewards, bins=20, color='salmon', edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(np.mean(rewards), color='red', linestyle='--',
                          label=f'Mean: {np.mean(rewards):.2f}')
        axes[1, 1].set_xlabel('Total Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'evaluation_phase{self.phase}.png', dpi=150, bbox_inches='tight')
        print(f" Biểu đồ đã lưu: evaluation_phase{self.phase}.png")
        plt.show()
    
    # =========================
    # TEST 3: So sánh Greedy vs Stochastic
    # =========================
    def compare_strategies(self, num_episodes=50):
        """So sánh hiệu suất giữa greedy và stochastic policy"""
        print("\n So sánh Greedy vs Stochastic Policy...")
        
        print("\n1️ Testing Greedy Policy...")
        stats_greedy, scores_g, _, _ = self.evaluate(num_episodes, greedy=True)
        
        print("\n2️ Testing Stochastic Policy...")
        stats_stoch, scores_s, _, _ = self.evaluate(num_episodes, greedy=False)
        
        # So sánh
        print(f"\n{'='*60}")
        print(" COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Greedy':<15} {'Stochastic':<15} {'Winner'}")
        print(f"{'-'*60}")
        
        metrics = [
            ("Mean Score", stats_greedy['mean_score'], stats_stoch['mean_score']),
            ("Max Score", stats_greedy['max_score'], stats_stoch['max_score']),
            ("Mean Steps", stats_greedy['mean_steps'], stats_stoch['mean_steps']),
            ("Success Rate %", stats_greedy['success_rate'], stats_stoch['success_rate']),
        ]
        
        for name, g_val, s_val in metrics:
            winner = "Greedy" if g_val > s_val else "Stochastic" if s_val > g_val else "Tie"
            print(f"{name:<20} {g_val:<15.2f} {s_val:<15.2f} {winner}")
        
        print(f"{'='*60}\n")
        
        # Vẽ so sánh
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(scores_g, bins=15, alpha=0.6, label='Greedy', color='blue')
        axes[0].hist(scores_s, bins=15, alpha=0.6, label='Stochastic', color='orange')
        axes[0].set_xlabel('Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Score Distribution Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(scores_g, alpha=0.7, label='Greedy', linewidth=2)
        axes[1].plot(scores_s, alpha=0.7, label='Stochastic', linewidth=2)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Score over Episodes')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'comparison_phase{self.phase}.png', dpi=150)
        print(f"💾 So sánh đã lưu: comparison_phase{self.phase}.png")
        plt.show()
    
    # =========================
    # TEST 4: Phân tích hành vi
    # =========================
    def analyze_behavior(self, num_episodes=20):
        """Phân tích hành vi của agent"""
        env = self.create_env(render_mode=None)
        
        action_counts = defaultdict(int)
        collision_types = defaultdict(int)
        
        print(f"\n🔬 Phân tích hành vi trên {num_episodes} episodes...")
        
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done, truncated = False, False
            
            while not (done or truncated):
                state = np.array(obs, dtype=float)
                current_dir = env.snake.direction
                action = self.get_action(state, current_dir, greedy=True)
                
                action_counts[action] += 1
                obs, reward, done, truncated, _ = env.step(action)
            
            # Phân loại nguyên nhân chết
            if done:
                head = env.snake.head
                if head.x < 0 or head.x >= env.snake.blocks_x or \
                   head.y < 0 or head.y >= env.snake.blocks_y:
                    collision_types["wall"] += 1
                else:
                    collision_types["self"] += 1
            else:
                collision_types["timeout"] += 1
        
        env.close()
        
        # In kết quả
        print(f"\n{'='*60}")
        print("🎯 BEHAVIOR ANALYSIS")
        print(f"{'='*60}")
        
        total_actions = sum(action_counts.values())
        print("\n📍 Action Distribution:")
        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        for action in range(4):
            count = action_counts.get(action, 0)
            pct = (count / total_actions * 100) if total_actions > 0 else 0
            print(f"   {action_names[action]:<8}: {count:>6} ({pct:>5.1f}%)")
        
        print("\n💀 Death Causes:")
        total_deaths = sum(collision_types.values())
        for cause, count in collision_types.items():
            pct = (count / total_deaths * 100) if total_deaths > 0 else 0
            print(f"   {cause.capitalize():<10}: {count:>4} ({pct:>5.1f}%)")
        
        print(f"{'='*60}\n")
        
        # Vẽ biểu đồ
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Action distribution
        actions = [action_names[i] for i in range(4)]
        counts = [action_counts.get(i, 0) for i in range(4)]
        axes[0].bar(actions, counts, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        axes[0].set_ylabel('Count')
        axes[0].set_title('Action Distribution')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Death causes
        causes = list(collision_types.keys())
        cause_counts = list(collision_types.values())
        axes[1].pie(cause_counts, labels=causes, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Death Causes')
        
        plt.tight_layout()
        plt.savefig(f'behavior_phase{self.phase}.png', dpi=150)
        print(f" Phân tích đã lưu: behavior_phase{self.phase}.png")
        plt.show()


# =========================
# MAIN - Ví dụ sử dụng
# =========================
if __name__ == "__main__":
    # Đường dẫn tới model đã train
    MODEL_PATH = r"D:\AL_FINAL_PROJECT_LAST_WEEK\AI_LAST_VER\model\dqn_snake_phase3_best.pth"
    PHASE = 3
    
    # Khởi tạo tester
    tester = Tester(MODEL_PATH, phase=PHASE)
    
    # Chọn chế độ test:
    print("\n" + "="*60)
    print(" SNAKE AI MODEL TESTING")
    print("="*60)
    print("Chọn chế độ test:")
    print("1.  Xem agent chơi (visual với pygame)")
    print("2.  Đánh giá performance (100 episodes)")
    print("3.   So sánh Greedy vs Stochastic")
    print("4.  Phân tích hành vi")
    print("5.  Chạy tất cả tests")
    print("="*60)
    
    choice = input("\nNhập lựa chọn (1-5): ").strip()
    
    if choice == "1":
        fps = int(input("Tốc độ hiển thị (fps, recommended 10-30): ") or "10")
        num_games = int(input("Số games muốn xem (1-10): ") or "3")
        tester.play_visual(num_games=num_games, fps=fps, greedy=True)
    
    elif choice == "2":
        num_eps = int(input("Số episodes để đánh giá (recommended 100-500): ") or "100")
        tester.evaluate(num_episodes=num_eps, greedy=True)
    
    elif choice == "3":
        num_eps = int(input("Số episodes mỗi strategy (recommended 50-100): ") or "50")
        tester.compare_strategies(num_episodes=num_eps)
    
    elif choice == "4":
        num_eps = int(input("Số episodes để phân tích (recommended 20-50): ") or "20")
        tester.analyze_behavior(num_episodes=num_eps)
    
    elif choice == "5":
        print("\n Chạy tất cả tests...\n")
        print(" Test 1: Visual Play")
        tester.play_visual(num_games=3, fps=15, greedy=True)
        
        print("\n  Test 2: Performance Evaluation")
        tester.evaluate(num_episodes=100, greedy=True)
        
        print("\n  Test 3: Strategy Comparison")
        tester.compare_strategies(num_episodes=50)
        
        print("\n  Test 4: Behavior Analysis")
        tester.analyze_behavior(num_episodes=20)
        
        print("\n Tất cả tests hoàn thành!")
    
    else:
        print(" Lựa chọn không hợp lệ!")