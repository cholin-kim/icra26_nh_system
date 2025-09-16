import collections
import json
import random
import numpy as np
from time import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



from DQN_cam._config import env_config, step_config, train_cfg
from DQN_cam._env import NeedleHandoverEnv

np.set_printoptions(precision=5, suppress=True)

env = NeedleHandoverEnv(env_config=env_config, step_config=step_config)
print(f"Environment initialized successfully")
print(f"State dimension: {env.state_dim}")
print(f"Action space size: {env.num_action}")

# Hyperparameters
learning_rate = train_cfg['lr']
gamma = train_cfg['gamma']
buffer_limit = train_cfg['buffer_limit']
batch_size = train_cfg['batch_size']



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def to_tensor(np_array, dtype=None):
    return torch.tensor(np_array, dtype=dtype).to(device)


class ReplayBuffer():
    """
    최신 {buffer_limit}개의 data를 들고 있다가 필요할 때마다 batch_size만큼 데이터를 뽑아서 제공.
    """

    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        """
        상태전이(s_t, a_t, r_t, s_t+1, done_mask)를 buffer에 저장
        """
        self.buffer.append(transition)

    def sample(self, n):
        """
        buffer에서 random하게 mini_batch만큼 데이터를 뽑아서 mini_batch 구성
        """
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            # Action space validation
            assert a < env.num_action, f"Invalid action: {a}, max: {env.num_action - 1}"

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (to_tensor(np.array(s_lst), dtype=torch.float),
                to_tensor(np.array(a_lst)),
                to_tensor(np.array(r_lst)),
                to_tensor(np.array(s_prime_lst), dtype=torch.float),
                to_tensor(np.array(done_mask_lst)))

    def size(self):
        return len(self.buffer)



qnet_init_dim = 10

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        # 네트워크 구조: 환경의 state_dim과 num_action에 맞춤
        self.fc1 = nn.Linear(qnet_init_dim, 64).to(device)
        # self.fc2 = nn.Linear(64, 64).to(device)
        self.fc2 = nn.Linear(64, 128).to(device)
        self.fc3 = nn.Linear(128, env.num_action).to(device)

        # self.fc1 = nn.Linear(env.state_dim, 256)
        # self.fc2 = nn.Linear(256, 2048)
        # self.fc3 = nn.Linear(2048, int(env.num_action / 2))
        # self.fc4 = nn.Linear(int(env.num_action / 2), env.num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)  # Q-value이므로 마지막 층에 activation 없음
        return x


    def sample_action(self, obs, epsilon):
        """
        Epsilon-greedy 방식으로 action 선택 (invalid action 고려X, agent가 환경 자체를 학습할 수 없다고 판단됨.)
        """
        out = self.forward(obs)
        q_values = out.cpu().detach().numpy().squeeze()

        coin = random.random()
        if coin < epsilon:
            # Exploration: valid action만 선택
            valid_actions = [a for a in range(len(q_values))]
            action = random.choice(valid_actions)
        else:
            # Exploitation: 최대 Q-value의 action 선택
            action = np.argmax(q_values)

        return action



def train(q, q_target, memory, optimizer):
    """
    Q-network 학습 함수
    """
    loss_tmp = []
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        a = a.to(torch.int64)

        q_out = q(s)
        q_a = q_out.gather(1, a)  # 실제 선택된 action의 Q-value

        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask    # done인 경우, r 이후에 발생하는 보상이 0.0이 됨.

        loss = F.smooth_l1_loss(q_a, target)
        loss_tmp.append(loss.cpu().detach().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.average(loss_tmp)


'''
Config 화
'''
def qnet_to_config(model: nn.Module):
    config = {
        "type": "MLP",
        "layers": []
    }

    for layer in model.modules():
        if isinstance(layer, nn.Sequential) or layer == model:
            continue  # skip container or root itself

        layer_info = {}

        if isinstance(layer, nn.Linear):
            layer_info["in"] = layer.in_features
            layer_info["out"] = layer.out_features
            layer_info["activation"] = None  # 임시, 다음 루프에서 활성화 함수 찾음
            config["layers"].append(layer_info)

        elif isinstance(layer, nn.ReLU):
            # 바로 이전 layer에 activation 추가
            if config["layers"]:
                config["layers"][-1]["activation"] = "ReLU"
        elif isinstance(layer, nn.Sigmoid):
            if config["layers"]:
                config["layers"][-1]["activation"] = "Sigmoid"
        elif isinstance(layer, nn.Tanh):
            if config["layers"]:
                config["layers"][-1]["activation"] = "Tanh"
        # 필요 시 추가로 확장 가능

    return config

def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_ndarray(i) for i in obj]
    return obj


def main():
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()
    ti = time()



    # Q-networks 초기화
    q = Qnet().to(device)
    q_target = Qnet().to(device)
    # q_target.load_state_dict(q.state_dict())
    # 이어서 train
    # model = "data/model_28800.pth"
    # q.load_state_dict(torch.load(model, map_location=device))
    # q_target.load_state_dict(q.state_dict())

    memory = ReplayBuffer()



    # config 저장도 수정
    # save_config_to_dir(train_cfg, data_dir, fmt="json")

    model_cfg = qnet_to_config(q)
    full_cfg = {
        "env" : env_config,
        "step" : step_config,
        "train" : train_cfg,
        "model" : model_cfg
    }
    full_cfg = convert_ndarray(full_cfg)
    with open(os.path.join(data_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(full_cfg, f, indent=2, ensure_ascii=False)


    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    # 학습 파라미터
    iter = 200000
    episode_reward_lst = []
    loss_lst = []
    epsilon_lst = []
    best_loss = 1000

    print("Starting DQN_cam training...")

    # for n_epi in range(iter):
    for n_epi in range(iter):
        print(f"\n\n------------Episode {n_epi}------------")

        # Epsilon decay
        epsilon = max(0.01, 1.0 - 0.01 * (n_epi / 350))
        epsilon_lst.append(epsilon)

        # 환경 reset
        try:
            s = env.reset()
            if s is None:
                print(f"Reset failed at episode {n_epi}")
                continue

        except Exception as e:
            print(f"Reset error at episode {n_epi}: {e}")
            continue

        done = False
        episode_reward = 0
        step_count = 0
        # max_steps_per_episode = step_config['max_steps']  # 무한 루프 방지

        # while not done and step_count < max_steps_per_episode:
        while not done:
            try:
                # Action 선택
                a = q.sample_action(
                    torch.from_numpy(s.astype(np.float32)).to(device),
                    epsilon
                )


                # Step 실행
                s_prime, r, done = env.step(action=a)
                print(f"state_{step_count}:{s} --action:{a}, total_reward:{r}--> state_{step_count+1}:{s_prime}, done:{done}")
                print("step_count:", step_count, "---------->", step_count +1)
                print("\n")



                if s_prime is None:
                    print(f"Step returned None state at episode {n_epi}, step {step_count}")
                    break

                done_mask = 0.0 if done else 1.0
                memory.put((s, a, r, s_prime, done_mask))

                s = s_prime
                episode_reward += r
                step_count += 1

            except Exception as e:
                print(f"Step error at episode {n_epi}, step {step_count}: {e}")
                done = True  # 에러 시 에피소드 종료
                r = -10  # 페널티 부여
                episode_reward += r
                break

        episode_reward_lst.append(episode_reward)

        
        # TensorBoard 로깅
        writer.add_scalar("Reward", episode_reward, n_epi)
        writer.add_scalar("Epsilon", epsilon, n_epi)
        writer.add_scalar("Steps", step_count, n_epi)

        # 학습 실행
        if memory.size() > 3000:
            try:
                loss = train(q, q_target, memory, optimizer)
                loss_lst.append(loss)
                # print("loss:", loss)

                # 모델 저장 (best loss 기준)
                if loss < best_loss and n_epi > 14999:
                    print(f"Best loss: {loss:.4f} at episode: {n_epi}")
                    torch.save(q.state_dict(), data_dir + f'model_{n_epi}.pth')
                    best_loss = loss

            except Exception as e:
                print(f"Training error at episode {n_epi}: {e}")
                loss = np.nan
                loss_lst.append(loss)
        else:
            loss = np.nan
            loss_lst.append(loss)

        writer.add_scalar("Loss", loss, n_epi)

        # 주기적 저장 및 target network 업데이트
        if n_epi % 200 == 0 and n_epi != 0 and n_epi > 14999:
            q_target.load_state_dict(q.state_dict())
            print(f"Episode: {n_epi}, Reward: {episode_reward:.2f}, "
                  f"Time: {round((time() - ti) / 60, 1)} min")

            # 데이터 저장
            torch.save(q.state_dict(), data_dir + f'model_{n_epi}.pth')
            np.save(data_dir + "epsilon.npy", np.array(epsilon_lst))
            np.save(data_dir + "score.npy", np.array(episode_reward_lst))
            np.save(data_dir + "loss.npy", loss_lst)

    # 최종 저장
    time_required = round((time() - ti) / 60, 1)
    print(f'Training completed: {time_required} min for {iter} iterations')

    # 최종 모델 저장
    torch.save(q.state_dict(), data_dir + 'final_model.pth')
    np.save(data_dir + "final_epsilon.npy", np.array(epsilon_lst))
    np.save(data_dir + "final_score.npy", np.array(episode_reward_lst))
    np.save(data_dir + "final_loss.npy", loss_lst)


    log_file.close()


    writer.flush()
    writer.close()

    try:
        env.close()
    except:
        pass


if __name__ == '__main__':
    def get_data_dir():
        """환경 변수나 기본값으로 data_dir 설정"""
        base_dir = os.environ.get('DATA_DIR', 'data/')
        return base_dir


    data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)

    # 학습 결과 로그 저장
    log_file = open(data_dir + "training_log.txt", "w")
    import sys
    sys.stdout = sys.stderr = log_file
    main()
