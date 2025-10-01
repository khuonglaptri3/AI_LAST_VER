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
        self.criterion = nn.SmoothL1Loss(reduction="none")  # Huber Loss (no reduction)

    def train_step(self, state, action, reward, next_state, done, weights=None):
        """
        Nếu dùng PER:
        - weights: importance-sampling weights
        Trả về (loss_mean, td_errors)
        """
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)  # Q(s,·)
        target = pred.clone().detach()  # detach để không backprop vào đây

        td_errors = []
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                next_Q = self.target_model(next_state[idx]) if self.target_model else self.model(next_state[idx])
                Q_new = reward[idx] + self.gamma * torch.max(next_Q)

            a = torch.argmax(action[idx]).item()
            td_error = Q_new - pred[idx][a]
            td_errors.append(td_error.detach().cpu())

            target[idx][a] = Q_new

        td_errors = torch.stack(td_errors)

        # Loss cho từng sample
        losses = self.criterion(pred, target)

        # Nếu có importance-sampling weights → áp dụng
        if weights is not None:
            weights = weights.unsqueeze(1)  # broadcast
            losses = losses * weights

        loss = losses.mean()

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item(), td_errors  # trả về cho PER update priority

    def soft_update_target(self):
        """Soft update target network"""
        if self.target_model is None:
            return
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
