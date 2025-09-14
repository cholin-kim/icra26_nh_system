import os
import csv
import numpy as np
from glob import glob
import re

# === Joint Limit 설정 ===
joint_range_lower_limit = np.deg2rad([-91, -53, 0, -260, -80, -80])
joint_range_upper_limit = np.deg2rad([91, 53, 240, 260, 80, 80])
joint_range_upper_limit[2] = 0.26  # q3 m 단위


def parse_joint_string(joint_str):
    """
    CSV joint 문자열을 numpy array로 변환
    예: "[ 0.88 -0.10 0.26 ...]" -> np.array([...])
    """
    if joint_str is None or joint_str == "":
        return None

    if isinstance(joint_str, (list, np.ndarray)):
        return np.array(joint_str, dtype=float)

    if isinstance(joint_str, str):
        try:
            # 문자열 전처리
            joint_str = joint_str.strip().replace("[", "").replace("]", "")
            if joint_str == "":
                return None
            values = [float(x) for x in joint_str.split()]
            return np.array(values, dtype=float)
        except Exception as e:
            print(f"[WARN] Failed to parse joint string: {joint_str} ({e})")
            return None

    return None


def check_joint_limit_each(joint_values):
    """joint별 limit 검사 결과 (True/False 리스트)"""
    margin = 0.05 * (joint_range_upper_limit - joint_range_lower_limit)
    lower_check = joint_values <= (joint_range_lower_limit + margin)
    upper_check = joint_values >= (joint_range_upper_limit - margin)
    return (lower_check | upper_check).astype(int)  # True=1, False=0


def compute_joint_margin_each(joint_values):
    """joint별 margin 계산"""
    mid = (joint_range_upper_limit + joint_range_lower_limit) / 2
    margin = 4 * ((joint_values - mid) ** 2) / ((joint_range_upper_limit - joint_range_lower_limit) ** 2)
    return 1 - margin  # joint별 margin 반환



def dqn_data_summary(log_dir="logs"):
    """logs/ 폴더의 *_handover_and_jaw_close_events.csv를 분석"""
    csv_files = sorted(glob(os.path.join(log_dir, "*_handover_and_jaw_close_events.csv")))

    summary_rows = []
    for csv_path in csv_files:
        idx = os.path.basename(csv_path).split("_")[0]
        with open(csv_path, "r") as f:
            reader = list(csv.reader(f))

        if len(reader) < 3:
            continue

        # 첫 번째 줄에서 handover_count
        try:
            handover_count = int(reader[0][1])
        except:
            continue

        # 이벤트 정보 파싱
        header = reader[1]
        joint_limits = []
        joint_margins = []

        for row in reader[2:]:
            row_dict = {header[i]: row[i] for i in range(len(header))}
            psm1_joint = parse_joint_string(row_dict.get("psm1_joint", ""))
            psm2_joint = parse_joint_string(row_dict.get("psm2_joint", ""))

            if psm1_joint is None or psm2_joint is None:
                continue

            joint_limits.append(check_joint_limit_each(psm1_joint))
            joint_limits.append(check_joint_limit_each(psm2_joint))
            joint_margins.append(compute_joint_margin_each(psm1_joint))
            joint_margins.append(compute_joint_margin_each(psm2_joint))

        if not joint_margins:
            continue

        joint_limits = np.array(joint_limits)
        joint_margins = np.array(joint_margins)

        limit_count = np.sum(joint_limits)
        margin_mean = np.mean(joint_margins, axis=0)
        margin_std = np.std(joint_margins, axis=0)

        row = {
            "index": idx,
            "handover_count": handover_count,
            "joint_limit_count": int(limit_count),
        }
        for j in range(6):
            row[f"joint_margin_q{j+1}_mean"] = margin_mean[j]
            row[f"joint_margin_q{j+1}_std"] = margin_std[j]
        summary_rows.append(row)

    # overall 계산
    all_counts = np.array([r["handover_count"] for r in summary_rows])
    all_limits = np.array([r["joint_limit_count"] for r in summary_rows])
    all_margins = np.array([[r[f"joint_margin_q{j+1}_mean"] for j in range(6)] for r in summary_rows])

    overall = {
        "index": "overall",
        "handover_count": f"{np.mean(all_counts):.2f} ± {np.std(all_counts):.2f}",
        "joint_limit_count": f"{np.mean(all_limits):.2f} ± {np.std(all_limits):.2f}",
    }
    for j in range(6):
        overall[f"joint_margin_q{j+1}_mean"] = f"{np.mean(all_margins[:, j]):.4f}"
        overall[f"joint_margin_q{j+1}_std"] = f"{np.std(all_margins[:, j]):.4f}"
    summary_rows.append(overall)

    # CSV 저장
    out_csv = os.path.join("dqn_analysis_summary.csv")
    fieldnames = list(summary_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"[INFO] Saved DQN summary to {out_csv}")


def parse_joint_line(line):
    """
    "(timestamp, (q1, q2, ...))" 형태의 문자열을 파싱해 numpy array 반환
    """
    match = re.search(r'\(([-0-9.]+),\s*\(([-0-9.,\s]+)\)\)', line.strip())
    if not match:
        return None, None
    ts = float(match.group(1))
    joints = np.array([float(x) for x in match.group(2).split(",")])
    return ts, joints

def read_joints(file_path):
    joints = []
    with open(file_path, "r") as f:
        for line in f:
            ts, q = parse_joint_line(line)
            if q is not None:
                joints.append(q)
    return np.array(joints)


def teleop_data_summary(data_dir="teleop_data", indices=range(20)):
    summary_rows = []

    for idx in indices:
        ts_file = glob(os.path.join(data_dir, f"{idx}_*_timestamps.txt"))
        blue_file = glob(os.path.join(data_dir, f"{idx}_*_blue_joint_positions.txt"))
        yellow_file = glob(os.path.join(data_dir, f"{idx}_*_yellow_joint_positions.txt"))

        if not ts_file or not blue_file or not yellow_file:
            print(f"[WARN] Missing files for index {idx}")
            continue

        ts_file, blue_file, yellow_file = ts_file[0], blue_file[0], yellow_file[0]

        # timestamps 읽기
        with open(ts_file, "r") as f:
            timestamps = [float(line.strip()) for line in f if line.strip()]
        handover_count = len(timestamps) - 1

        # joints 읽기
        blue_joints = read_joints(blue_file)
        yellow_joints = read_joints(yellow_file)
        if len(blue_joints) == 0 or len(yellow_joints) == 0:
            print(f"[WARN] No joints parsed for index {idx}")
            continue

        all_joints = np.concatenate([blue_joints, yellow_joints], axis=0)

        # joint limit count
        limit_count = np.sum([check_joint_limit_each(q) for q in all_joints])

        # joint margin per joint
        margins_all = np.array([compute_joint_margin_each(q) for q in all_joints])  # shape: (N,6)
        joint_margin_mean = np.mean(margins_all, axis=0)
        joint_margin_std = np.std(margins_all, axis=0)

        row = {
            "index": idx,
            "handover_count": handover_count,
            "joint_limit_count": int(limit_count),
        }
        for j in range(6):
            row[f"joint_margin_q{j+1}_mean"] = joint_margin_mean[j]
            row[f"joint_margin_q{j+1}_std"] = joint_margin_std[j]
        summary_rows.append(row)

    # 전체 평균
    all_counts = np.array([r["handover_count"] for r in summary_rows])
    all_limits = np.array([r["joint_limit_count"] for r in summary_rows])
    all_margins = np.array([[r[f"joint_margin_q{j+1}_mean"] for j in range(6)] for r in summary_rows])

    overall = {
        "index": "overall",
        "handover_count": f"{np.mean(all_counts):.2f} ± {np.std(all_counts):.2f}",
        "joint_limit_count": f"{np.mean(all_limits):.2f} ± {np.std(all_limits):.2f}",
    }
    for j in range(6):
        overall[f"joint_margin_q{j+1}_mean"] = f"{np.mean(all_margins[:, j]):.4f}"
        overall[f"joint_margin_q{j+1}_std"] = f"{np.std(all_margins[:, j]):.4f}"

    summary_rows.append(overall)

    # CSV 저장
    csv_file = "teleop_analysis_summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"[INFO] Saved summary to {csv_file}")



if __name__ == "__main__":
    # dqn_data_summary()
    # teleop_data_summary()

    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # # === 데이터 불러오기 ===
    # dqn_df = pd.read_csv("dqn_analysis_summary.csv")
    # teleop_df = pd.read_csv("teleop_analysis_summary.csv")
    #
    # # 마지막 행(overall) 제외
    # dqn_data = dqn_df.iloc[:-1]
    # teleop_data = teleop_df.iloc[:-1]
    #
    # # === Handover와 Joint Limit 값 추출 ===
    # dqn_handover = dqn_data["handover_count"].astype(float)
    # teleop_handover = teleop_data["handover_count"].astype(float)
    #
    # dqn_joint_limit = dqn_data["joint_limit_count"].astype(float)
    # teleop_joint_limit = teleop_data["joint_limit_count"].astype(float)
    #
    # # === Joint Margin 값 추출 ===
    # joint_labels = [col for col in dqn_df.columns if "joint_margin_q" in col and "_mean" in col]
    # dqn_margin_means = dqn_df.iloc[-1][joint_labels].astype(float).values
    # teleop_margin_means = teleop_df.iloc[-1][joint_labels].astype(float).values
    #
    # # === 시각화 ===
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    #
    # # 1. Handover & Joint Limit 비교
    # x = np.arange(2)
    # handover_means = [dqn_handover.mean(), teleop_handover.mean()]
    # limit_means = [dqn_joint_limit.mean(), teleop_joint_limit.mean()]
    #
    # axes[0].bar(x - 0.15, handover_means, width=0.3, label="Handover Count", color="skyblue")
    # axes[0].bar(x + 0.15, limit_means, width=0.3, label="Joint Limit Count", color="salmon")
    # axes[0].set_xticks(x)
    # axes[0].set_xticklabels(["DQN", "Teleop"])
    # axes[0].set_ylabel("Count")
    # axes[0].set_title("Handover & Joint Limit Count")
    # axes[0].legend()
    # axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    #
    # # 2. Joint Margin 비교
    # x = np.arange(len(joint_labels))
    # axes[1].bar(x - 0.15, dqn_margin_means, width=0.3, label="DQN", color="skyblue")
    # axes[1].bar(x + 0.15, teleop_margin_means, width=0.3, label="Teleop", color="salmon")
    # axes[1].set_xticks(x)
    # axes[1].set_xticklabels([f"q{i + 1}" for i in range(len(joint_labels))])
    # axes[1].set_ylabel("Joint Margin")
    # axes[1].set_title("Joint Margin per Joint")
    # axes[1].legend()
    # axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    #
    # plt.tight_layout()
    # plt.show()

    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # # === 데이터 불러오기 ===
    # dqn_df = pd.read_csv("dqn_analysis_summary.csv")
    # teleop_df = pd.read_csv("teleop_analysis_summary.csv")
    #
    # # 마지막 행(overall) 제외
    # dqn_data = dqn_df.iloc[:-1]
    # teleop_data = teleop_df.iloc[:-1]
    #
    # # === Handover & Joint Limit 값 ===
    # dqn_handover = dqn_data["handover_count"].astype(float)
    # teleop_handover = teleop_data["handover_count"].astype(float)
    #
    # dqn_joint_limit = dqn_data["joint_limit_count"].astype(float)
    # teleop_joint_limit = teleop_data["joint_limit_count"].astype(float)
    #
    # # === Handover, Joint Limit Mean & Std ===
    # handover_means = [dqn_handover.mean(), teleop_handover.mean()]
    # handover_stds = [dqn_handover.std(), teleop_handover.std()]
    #
    # limit_means = [dqn_joint_limit.mean(), teleop_joint_limit.mean()]
    # limit_stds = [dqn_joint_limit.std(), teleop_joint_limit.std()]
    #
    # # === Joint Margin 값 ===
    # joint_labels = [col for col in dqn_df.columns if "joint_margin_q" in col and "_mean" in col]
    # joint_labels_std = [col for col in dqn_df.columns if "joint_margin_q" in col and "_std" in col]
    #
    # dqn_margin_means = dqn_df.iloc[-1][joint_labels].astype(float).values
    # teleop_margin_means = teleop_df.iloc[-1][joint_labels].astype(float).values
    #
    # dqn_margin_stds = dqn_df.iloc[-1][joint_labels_std].astype(float).values
    # teleop_margin_stds = teleop_df.iloc[-1][joint_labels_std].astype(float).values
    #
    # # === 시각화 ===
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    #
    # # 1. Handover & Joint Limit Count (Error Bars)
    # x = np.arange(2)
    # width = 0.35
    #
    # axes[0].bar(x - width / 2, handover_means, yerr=handover_stds, width=width,
    #             label="Handover Count", color="skyblue", capsize=5)
    # axes[0].bar(x + width / 2, limit_means, yerr=limit_stds, width=width,
    #             label="Joint Limit Count", color="salmon", capsize=5)
    # axes[0].set_xticks(x)
    # axes[0].set_xticklabels(["DQN", "Teleop"])
    # axes[0].set_ylabel("Count")
    # axes[0].set_title("Handover & Joint Limit Count")
    # axes[0].legend()
    # axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    #
    # # 2. Joint Margin per Joint (Error Bars)
    # x = np.arange(len(joint_labels))
    # width = 0.35
    # axes[1].bar(x - width / 2, dqn_margin_means, yerr=dqn_margin_stds, width=width,
    #             label="DQN", color="skyblue", capsize=5)
    # axes[1].bar(x + width / 2, teleop_margin_means, yerr=teleop_margin_stds, width=width,
    #             label="Teleop", color="salmon", capsize=5)
    # axes[1].set_xticks(x)
    # axes[1].set_xticklabels([f"q{i + 1}" for i in range(len(joint_labels))])
    # axes[1].set_ylabel("Joint Margin")
    # axes[1].set_title("Joint Margin per Joint")
    # axes[1].legend()
    # axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    #
    # plt.tight_layout()
    # plt.show()
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib

    # === 논문용 스타일 설정 ===
    matplotlib.rcParams.update({
        "font.size": 14,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "legend.frameon": False
    })

    # === 데이터 불러오기 ===
    dqn_df = pd.read_csv("dqn_analysis_summary.csv")
    teleop_df = pd.read_csv("teleop_analysis_summary.csv")

    # 마지막 행(overall) 제외
    dqn_data = dqn_df.iloc[:-1]
    teleop_data = teleop_df.iloc[:-1]

    # === Handover & Joint Limit ===
    dqn_handover = dqn_data["handover_count"].astype(float)
    teleop_handover = teleop_data["handover_count"].astype(float)
    dqn_limit = dqn_data["joint_limit_count"].astype(float)
    teleop_limit = teleop_data["joint_limit_count"].astype(float)

    handover_means = [dqn_handover.mean(), teleop_handover.mean()]
    handover_stds = [dqn_handover.std(), teleop_handover.std()]
    limit_means = [dqn_limit.mean(), teleop_limit.mean()]
    limit_stds = [dqn_limit.std(), teleop_limit.std()]

    # === Joint Margin ===
    joint_labels = [c for c in dqn_df.columns if "joint_margin_q" in c and "_mean" in c]
    joint_labels_std = [c for c in dqn_df.columns if "joint_margin_q" in c and "_std" in c]

    dqn_margin_means = dqn_df.iloc[-1][joint_labels].astype(float).values
    teleop_margin_means = teleop_df.iloc[-1][joint_labels].astype(float).values
    dqn_margin_stds = dqn_df.iloc[-1][joint_labels_std].astype(float).values
    teleop_margin_stds = teleop_df.iloc[-1][joint_labels_std].astype(float).values

    # === 색상 ===
    colors = ["#4C72B0", "#DD8452"]  # Blue, Orange

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Handover & Joint Limit Count
    x = np.arange(2)
    width = 0.35
    axes[0].bar(x - width / 2, handover_means, yerr=handover_stds, width=width,
                label="Handover Count", color=colors[0], capsize=5, edgecolor="black")
    axes[0].bar(x + width / 2, limit_means, yerr=limit_stds, width=width,
                label="Joint Limit Count", color=colors[1], capsize=5, edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["DQN", "Teleop"])
    axes[0].set_ylabel("Count", fontweight="bold")
    axes[0].set_title("Handover & Joint Limit", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Joint Margin per Joint
    x = np.arange(len(joint_labels))
    axes[1].bar(x - width / 2, dqn_margin_means, yerr=dqn_margin_stds, width=width,
                label="DQN", color=colors[0], capsize=5, edgecolor="black")
    axes[1].bar(x + width / 2, teleop_margin_means, yerr=teleop_margin_stds, width=width,
                label="Teleop", color=colors[1], capsize=5, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"q{i + 1}" for i in range(len(joint_labels))])
    axes[1].set_ylabel("Joint Margin", fontweight="bold")
    axes[1].set_title("Joint Margin per Joint", fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()



