import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_sth._step_orchestrator import StepOrchestrator
from rl_sth._reset_orchestrator import ResetOrchestrator
from rl_sth._config import env_config, step_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(10, 64).to(device)
        self.fc2 = nn.Linear(64, 128).to(device)
        self.fc3 = nn.Linear(128, 320).to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = "/home/surglab/icra26_nh_system/rl_sth/model/model_97000.pth"

class RL:
    def __init__(self):
        self.q = Qnet().to(device)
        self.q.load_state_dict(torch.load(model, map_location=device))
        self.q.eval()

        self.step_count = 0
        self.prev_action_lst = []

        self.reset_orchestrator = ResetOrchestrator(env_config=env_config)
        self.step_orch = StepOrchestrator(
            env_config=env_config,
            step_config=step_config,
            reset=self.reset_orchestrator
        )
        self.goal_state: np.ndarray = self.reset_orchestrator.goal_state

    def infer(self, state):
        self.step_count += 1
        obs_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q(obs_tensor).cpu().numpy().squeeze()
        action = int(np.argmax(q_values))
        self.prev_action_lst.append(action)
        return action

    def check_done(self, state):
        done =False
        gp, go, gh = state[0], state[1], state[2]
        if gp == 1 and go == 0 and gh == 1:
            done = True
        return done

    def action_interpreter(self, action):
        pass

    def step(self, action, state, needle_pose_w):
        action = int(action)
        new_state, reward, done, new_needle_pose_w, joint_pos_1, joint_pos_2 = self.step_orch.step(
            action=action,
            state=state,
            Tw_no=needle_pose_w,
            num_step=self.step_count,
            prev_action_lst=self.prev_action_lst,
            goal_state=self.goal_state
        )
        return new_state, reward, done, new_needle_pose_w, joint_pos_1, joint_pos_2



    # def get_w2gp(self, state, last_robot):
    #     ###
    #     Tw2gp = np.eye(4)
    #     if last_robot is not None:
    #         return Tw2gp
    #     else:
    #         if last_robot == 1:
    #             hand = 'PSM1'
    #         else:
    #             hand = 'PSM2'
    #         return Tw2gp, hand

