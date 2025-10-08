import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, target_model=None, tau=0.01):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model  # target network
        self.tau = tau
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

    def train_step(self, state, action, reward, next_state, done, weights=None):

        device = next(self.model.parameters()).device
        state = torch.tensor(state, dtype=torch.float, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=device)
        action = torch.tensor(action, dtype=torch.long, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)
        done = torch.tensor(done, dtype=torch.float, device=device)

        # Q(s,a) hiện tại
        q_values = self.model(state)                       # [B, n_actions]
        q_pred = q_values.gather(1, action.unsqueeze(1)).squeeze(1)  # [B]

        # Q target
        with torch.no_grad():
            if self.target_model is not None:
                # Double DQN: chọn argmax bằng online, value bằng target
                next_q_online = self.model(next_state)
                next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)  # [B,1]
                next_q_target = self.target_model(next_state)
                next_q = next_q_target.gather(1, next_actions).squeeze(1)        # [B]
            else:
                next_q = self.model(next_state).max(1)[0]                        # [B]

            q_target = reward + (1 - done) * self.gamma * next_q

        td_errors = (q_target - q_pred).detach()

        # Huber loss per sample
        loss_per_sample = F.smooth_l1_loss(q_pred, q_target, reduction='none')

        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float, device=device)
            loss = (loss_per_sample * weights).mean()
        else:
            loss = loss_per_sample.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item(), td_errors.cpu().numpy()

    def soft_update_target(self):
        """Soft update target network"""
        if self.target_model is None:
            return
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
