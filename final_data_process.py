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
    csv_file = "teleop_analysis_summary_sub_4.csv"
    fieldnames = list(summary_rows[0].keys())
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"[INFO] Saved summary to {csv_file}")



if __name__ == "__main__":
    # dqn_data_summary()
    # teleop_data_summary("teleop_data/sub_4")


    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import glob

    # === 파일 경로 ===
    dqn_file = "dqn_analysis_summary.csv"
    teleop_files = glob.glob("teleop_analysis_summary*.csv")

    # === DQN 파일 읽기 ===
    dqn_df = pd.read_csv(dqn_file)
    dqn_overall = dqn_df[dqn_df["index"] == "overall"].iloc[0]

    # === Teleop 파일 합치기 ===
    teleop_dfs = [pd.read_csv(f) for f in teleop_files]
    teleop_all = pd.concat(teleop_dfs, ignore_index=True)
    teleop_data = teleop_all[teleop_all["index"] != "overall"].copy()

    # === Teleop 전체 통계 계산 ===
    numeric_cols = [c for c in teleop_data.columns if c != "index"]
    teleop_numeric = teleop_data[numeric_cols].astype(float)

    teleop_overall = {
        "index": "overall"
    }
    for col in numeric_cols:
        teleop_overall[col] = f"{teleop_numeric[col].mean():.2f} ± {teleop_numeric[col].std():.2f}"
    teleop_overall = pd.Series(teleop_overall)


    # === 값 파싱 함수 ===
    def parse_mean_std(val):
        """'m ± s' 문자열에서 mean, std 분리"""
        if isinstance(val, str) and "±" in val:
            parts = val.split("±")
            return float(parts[0].strip()), float(parts[1].strip())
        return float(val), 0.0


    # === Handover Count & Joint Limit Count ===
    handover_mean_dqn, handover_std_dqn = parse_mean_std(dqn_overall["handover_count"])
    handover_mean_teleop, handover_std_teleop = parse_mean_std(teleop_overall["handover_count"])

    limit_mean_dqn, limit_std_dqn = parse_mean_std(dqn_overall["joint_limit_count"])
    limit_mean_teleop, limit_std_teleop = parse_mean_std(teleop_overall["joint_limit_count"])

    # === Joint Margin Mean & Std ===
    joints = [f"joint_margin_q{i + 1}_mean" for i in range(6)]
    joints_std = [f"joint_margin_q{i + 1}_std" for i in range(6)]

    joint_margin_dqn_mean = np.array([float(dqn_overall[j]) for j in joints])
    joint_margin_dqn_std = np.array([float(dqn_overall[s]) for s in joints_std])
    joint_margin_teleop_mean = np.array([float(teleop_overall[j].split()[0]) for j in joints])
    joint_margin_teleop_std = np.array([float(teleop_overall[s].split()[-1]) for s in joints_std])

    # === Plot 1: Handover Count & Joint Limit Count (Mean ± Std) ===
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["DQN", "Teleop"]

    handover_means = [handover_mean_dqn, handover_mean_teleop]
    handover_stds = [handover_std_dqn, handover_std_teleop]
    limit_means = [limit_mean_dqn, limit_mean_teleop]
    limit_stds = [limit_std_dqn, limit_std_teleop]

    # Handover Count
    ax[0].bar(labels, handover_means, yerr=handover_stds, capsize=5, color=["skyblue", "orange"])
    ax[0].set_title("Handover Count (Mean ± Std)")
    ax[0].set_ylabel("Count")
    for i, (mean, std) in enumerate(zip(handover_means, handover_stds)):
        ax[0].text(i, mean + std + 0.2, f"{mean:.1f}±{std:.1f}", ha='center', fontsize=10)

    # Joint Limit Count
    ax[1].bar(labels, limit_means, yerr=limit_stds, capsize=5, color=["skyblue", "orange"])
    ax[1].set_title("Joint Limit Count (Mean ± Std)")
    ax[1].set_ylabel("Count")
    for i, (mean, std) in enumerate(zip(limit_means, limit_stds)):
        ax[1].text(i, mean + std + 0.2, f"{mean:.1f}±{std:.1f}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()

    # === Plot 2: Joint Margin (Mean ± Std) ===
    x = np.arange(6)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, joint_margin_dqn_mean, width, yerr=joint_margin_dqn_std, capsize=5, label="DQN",
           color="skyblue")
    ax.bar(x + width / 2, joint_margin_teleop_mean, width, yerr=joint_margin_teleop_std, capsize=5, label="Teleop",
           color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Joint {i + 1}" for i in range(6)])
    ax.set_ylabel("Joint Margin")
    ax.set_title("Joint Margin Comparison (Mean ± Std)")
    ax.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import glob
    #
    # # === DQN 파일 로드 ===
    # dqn_df = pd.read_csv("dqn_analysis_summary.csv")
    # dqn_overall = dqn_df[dqn_df["index"] == "overall"].iloc[0]
    #
    # # DQN handover, joint limit
    # dqn_handover = float(dqn_overall["handover_count"].split("±")[0])
    # dqn_handover_std = float(dqn_overall["handover_count"].split("±")[1])
    # dqn_limit = float(dqn_overall["joint_limit_count"].split("±")[0])
    # dqn_limit_std = float(dqn_overall["joint_limit_count"].split("±")[1])
    #
    # # DQN joint margins
    # joints = [f"joint_margin_q{i + 1}_mean" for i in range(6)]
    # joints_std = [f"joint_margin_q{i + 1}_std" for i in range(6)]
    # dqn_joint_margin_mean = np.array([float(dqn_overall[j]) for j in joints])
    # dqn_joint_margin_std = np.array([float(dqn_overall[s]) for s in joints_std])
    #
    # # === Teleop 파일 로드 ===
    # teleop_files = sorted(glob.glob("teleop_analysis_summary_sub_*.csv"))
    # teleop_dfs = [pd.read_csv(f) for f in teleop_files]
    # people = ["DQN Algorithm"] + [f"Person {i + 1}" for i in range(len(teleop_dfs))]
    #
    # # === Teleop 통계 계산 ===
    # handover_means = [dqn_handover]
    # handover_stds = [dqn_handover_std]
    # limit_means = [dqn_limit]
    # limit_stds = [dqn_limit_std]
    # joint_margin_means = [dqn_joint_margin_mean]
    # joint_margin_stds = [dqn_joint_margin_std]
    #
    # for df in teleop_dfs:
    #     data = df[df["index"] != "overall"].copy()
    #     data["handover_count"] = data["handover_count"].astype(float)
    #     data["joint_limit_count"] = data["joint_limit_count"].astype(float)
    #
    #     # mean/std 계산
    #     handover_means.append(data["handover_count"].mean())
    #     handover_stds.append(data["handover_count"].std())
    #     limit_means.append(data["joint_limit_count"].mean())
    #     limit_stds.append(data["joint_limit_count"].std())
    #
    #     # joint margin
    #     margin_means = np.array([data[f"joint_margin_q{i + 1}_mean"].astype(float).mean() for i in range(6)])
    #     margin_stds = np.array([data[f"joint_margin_q{i + 1}_mean"].astype(float).std() for i in range(6)])
    #     joint_margin_means.append(margin_means)
    #     joint_margin_stds.append(margin_stds)
    #
    # # === Handover & Joint Limit 그래프 ===
    # x = np.arange(len(people))
    # fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    #
    # # Handover count
    # ax[0].bar(x, handover_means, yerr=handover_stds, capsize=5,
    #           color=["green"] + ["skyblue"] * (len(people) - 1), alpha=0.8)
    # ax[0].set_xticks(x)
    # ax[0].set_xticklabels(people, rotation=45)
    # ax[0].set_title("Handover Count (DQN vs Teleop)")
    # ax[0].set_ylabel("Mean ± Std")
    # ax[0].grid(axis='y', linestyle='--', alpha=0.7)
    #
    # # Joint limit count
    # ax[1].bar(x, limit_means, yerr=limit_stds, capsize=5,
    #           color=["green"] + ["orange"] * (len(people) - 1), alpha=0.8)
    # ax[1].set_xticks(x)
    # ax[1].set_xticklabels(people, rotation=45)
    # ax[1].set_title("Joint Limit Count (DQN vs Teleop)")
    # ax[1].set_ylabel("Mean ± Std")
    # ax[1].grid(axis='y', linestyle='--', alpha=0.7)
    #
    # plt.tight_layout()
    # plt.show()
    #
    # # === Joint Margin per Joint ===
    # joints = [f"q{i + 1}" for i in range(6)]
    # fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    # axes = axes.ravel()
    #
    # for j in range(6):
    #     joint_means = [m[j] for m in joint_margin_means]
    #     joint_stds = [s[j] for s in joint_margin_stds]
    #     axes[j].bar(x, joint_means, yerr=joint_stds, capsize=5,
    #                 color=["green"] + ["purple"] * (len(people) - 1), alpha=0.8)
    #     axes[j].set_xticks(x)
    #     axes[j].set_xticklabels(people, rotation=45)
    #     axes[j].set_title(f"Joint Margin {joints[j]} (DQN vs Teleop)")
    #     axes[j].set_ylabel("Margin (Mean ± Std)")
    #     axes[j].grid(axis='y', linestyle='--', alpha=0.7)
    #
    # plt.tight_layout()
    # plt.show()


    ### Heatmap
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import glob
    #
    # # === 파일 경로 ===
    # dqn_file = "dqn_analysis_summary.csv"
    # teleop_files = sorted(glob.glob("teleop_analysis_summary_sub_*.csv"))
    #
    # # === DQN 데이터 ===
    # dqn_df = pd.read_csv(dqn_file)
    # dqn_overall = dqn_df[dqn_df["index"] == "overall"]
    #
    # # Joint Margin 컬럼 추출
    # joint_margin_cols = [c for c in dqn_overall.columns if "joint_margin_q" in c and "_mean" in c]
    #
    # # DQN mean
    # dqn_joint_margin = [float(dqn_overall[c].iloc[0]) for c in joint_margin_cols]
    #
    # # === Teleop 데이터 (사람별 평균 계산) ===
    # teleop_person_means = []
    # for teleop_file in teleop_files:
    #     df = pd.read_csv(teleop_file)
    #     df = df[df["index"] != "overall"].copy()
    #     df[joint_margin_cols] = df[joint_margin_cols].astype(float)
    #     person_mean = df[joint_margin_cols].mean().values
    #     teleop_person_means.append(person_mean)
    #
    # teleop_person_means = np.array(teleop_person_means)
    #
    # # === 합치기 ===
    # joint_margin_matrix = np.vstack([dqn_joint_margin, teleop_person_means])
    # row_labels = ["DQN"] + [f"Person {i + 1}" for i in range(len(teleop_person_means))]
    #
    # # === Heatmap ===
    # fig, ax = plt.subplots(figsize=(10, 6))
    # im = ax.imshow(joint_margin_matrix, cmap="coolwarm", aspect="auto", vmin=0, vmax=1)
    #
    # # 라벨 설정
    # ax.set_xticks(range(len(joint_margin_cols)))
    # ax.set_xticklabels([f"Joint {i + 1}" for i in range(len(joint_margin_cols))])
    # ax.set_yticks(range(len(row_labels)))
    # ax.set_yticklabels(row_labels)
    #
    # ax.set_title("DQN vs Teleop (Person-wise Avg): Joint Margin Heatmap")
    # plt.colorbar(im, ax=ax, label="Margin")
    # plt.tight_layout()
    # plt.show()



