import os
import numpy as np
import matplotlib.pyplot as plt

import json


def load_configs_from_data_dir(data_dir: str):
    """
    data_dir/config.json 파일을 읽어
    env_config, step_config, train_config 딕셔너리를 반환.
    """
    config_path = os.path.join(data_dir, "config.json")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No config.json found in {data_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        full_cfg = json.load(f)

    # JSON 구조에 맞춰 세 개의 설정 추출
    env_config   = full_cfg.get("env", {})
    step_config  = full_cfg.get("step", {})
    train_config = full_cfg.get("train", {})

    return env_config, step_config, train_config

def extract_number(f):
    # match = re.search(r'model_(\d+)\.pth', f)
    match = re.search(r'model_(\d+)\.pth', f)
    return int(match.group(1)) if match else -1


def visualize_training_metrics(data_dir: str, show: bool = True, save_path: str = None):
    """
    동일 디렉토리 내의 epsilon.npy, loss.npy, score.npy 파일을 로드하여 시각화.

    Args:
        data_dir (str): .npy 파일들이 저장된 디렉토리 경로
        show (bool): True면 plt.show()로 화면에 표시, False면 표시하지 않음
        save_path (str, optional): 지정하면 해당 경로에 PNG 파일로 저장

    Raises:
        FileNotFoundError: 필수 .npy 파일이 누락된 경우
    """
    # 파일 경로 구성
    eps_path   = os.path.join(data_dir, "epsilon.npy")
    loss_path  = os.path.join(data_dir, "loss.npy")
    score_path = os.path.join(data_dir, "score.npy")

    # 파일 존재 확인
    for path in (eps_path, loss_path, score_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"There is no file: {path}")

    # 데이터 로드
    epsilon = np.load(eps_path)
    loss    = np.load(loss_path)
    score   = np.load(score_path)

    # 시각화
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    episodes = np.arange(len(epsilon))

    axes[0].plot(episodes, epsilon, color='tab:blue')
    axes[0].set_title("Epsilon Decay")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Epsilon")

    axes[1].plot(episodes, loss, color='tab:orange')
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")

    axes[2].plot(episodes, score, color='tab:green')
    axes[2].set_title("Episode Reward (Score)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Reward")

    plt.tight_layout()

    # 저장 옵션
    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)





if __name__ == "__main__":
    #
    # # 1. infer for all saved models
    # data_dir = "data/"  # config.json이 있는 경로
    # env_cfg, step_cfg, train_cfg = load_configs_from_data_dir(data_dir)
    #
    # log_file = open(data_dir + "validation_log.txt", "w")
    # import sys
    # sys.stdout = sys.stderr = log_file
    #
    # print("=== env_config ===")
    # print(env_cfg)
    # print("\n=== step_config ===")
    # print(step_cfg)
    # print("\n=== train_config ===")
    # print(train_cfg)
    # # from _env_infer import NeedleHandoverEnv
    # from _env import NeedleHandoverEnv
    # import torch
    # import glob
    # import re
    # from _train import Qnet
    #
    # env = NeedleHandoverEnv(env_config=env_cfg, step_config=step_cfg)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    #
    # model_files = glob.glob(data_dir + "model_*.pth")
    # model_files = sorted(model_files, key=extract_number)
    #
    #
    # for model in model_files:
    #     q = Qnet().to(device)
    #     q.load_state_dict(torch.load(model, map_location=device))
    #     q.eval()
    #
    #     num_episodes = 300
    #     failure = 0
    #     fjfj0 = 0
    #     fjfj1 = 0
    #     fjfj2 = 0
    #     fjfj3 = 0
    #     max_steps_per_episode = step_cfg['max_steps']
    #
    #     for ep in range(num_episodes):
    #         obs = env.reset()
    #         done = False
    #         step_count = 0
    #         total_reward = 0.0
    #
    #         while not done and step_count < max_steps_per_episode:
    #             try:
    #                 with torch.no_grad():
    #                     obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
    #                     q_values = q(obs_tensor).cpu().numpy().squeeze()
    #                     action = int(np.argmax(q_values))
    #
    #                 obs, reward, done = env.step(action)
    #                 total_reward += reward
    #                 step_count += 1
    #
    #                 if step_count == 1:
    #                     fjfj0 += 1
    #                 elif step_count == 2:
    #                     fjfj1 += 1
    #                 elif step_count == 3:
    #                     fjfj2 += 1
    #                 elif step_count == 4:
    #                     fjfj3 += 1
    #             except:
    #                 pass
    #
    #         print(f"[Episode {ep:02d}] Steps: {step_count}, Total Reward: {total_reward:.2f}")
    #         if total_reward <= 0.0:
    #             failure += 1
    #
    #
    #     print("model", model, "failure:", failure, "among", num_episodes, "episodes")
    # # visualize_training_metrics(data_dir, show=True, save_path=None)

    import re

    file_path = "data/validation_log.txt"

    results = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"model (.+?) failure: (\d+) among (\d+) episodes", line)
            if match:
                model_path = match.group(1)
                failure = int(match.group(2))
                episodes = int(match.group(3))
                results.append((model_path, failure, episodes))

    # failure 기준으로 오름차순 정렬
    results.sort(key=lambda x: x[1])

    # 상위 20개만 출력
    top20 = results[:10]

    print("Top 10 models with lowest failure:")
    for model_path, failure, episodes in top20:
        print(f"{model_path} | failure: {failure} / {episodes}")

# Top 10 models with lowest failure:
# data/model_166600.pth | failure: 1 / 300
# data/model_84800.pth | failure: 2 / 300
# data/model_106800.pth | failure: 2 / 300
# data/model_156400.pth | failure: 2 / 300
# data/model_158800.pth | failure: 2 / 300
# data/model_159000.pth | failure: 2 / 300
# data/model_170200.pth | failure: 2 / 300
# data/model_172200.pth | failure: 2 / 300
# data/model_183600.pth | failure: 2 / 300
# data/model_102200.pth | failure: 3 / 300