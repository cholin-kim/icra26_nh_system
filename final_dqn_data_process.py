import pickle
import numpy as np
import csv
import os





# === 2. Jaw 닫힘 유지 시작 시점 ===
JAW_CLOSED_TH = 0.01  # jaw 센서값 기준 (조정 가능)
JAW_OPEN_TH = np.deg2rad(55)  # 열림


# jaw 상태 분류 함수
def classify_jaw(jaw_values):
    labels = []
    for val in jaw_values:
        if np.isnan(val):
            labels.append("nan")
        elif val < JAW_CLOSED_TH:
            labels.append("closed")
        elif val > JAW_OPEN_TH:
            labels.append("open")
        else:
            labels.append("unknown")
    return labels


def simplify_labels(jaw_labels, timestamps):
    """
    jaw_labels: ["closed", "open", "unknown", ...]
    timestamps: 각 프레임 timestamp
    ------------------------
    같은 label이 연속되면 하나로 합쳐서 큰 흐름 반환
    [(label, start_time, end_time, start_idx, end_idx), ...]
    그리고 각 라벨별 start_idx 목록도 반환
    """
    simplified = []
    label_starts = {}  # 각 라벨별 start_idx 저장

    if not jaw_labels:
        return simplified, label_starts

    current_label = jaw_labels[0]
    start_idx = 0

    for i in range(1, len(jaw_labels)):
        if jaw_labels[i] != current_label:
            simplified.append((
                current_label,
                timestamps[start_idx],
                timestamps[i-1],
                start_idx,
                i-1
            ))

            # 라벨별 start_idx 저장
            label_starts.setdefault(current_label, []).append(start_idx)

            current_label = jaw_labels[i]
            start_idx = i

    # 마지막 구간 추가
    simplified.append((
        current_label,
        timestamps[start_idx],
        timestamps[-1],
        start_idx,
        len(jaw_labels)-1
    ))
    label_starts.setdefault(current_label, []).append(start_idx)

    return simplified, label_starts


def detect_open_close_events(jaw_labels, timestamps):
    """
    큰 흐름에서 open → unknown → closed 패턴 찾기
    unknown → closed로 내려가는 구간의 timestamp 반환
    """
    simplified, label_starts = simplify_labels(jaw_labels, timestamps)
    events = []

    # simplified: [(label, start_t, end_t, start_idx, end_idx), ...]
    # label_starts: {'open': [...], 'unknown': [...], 'closed': [...]}

    for i in range(2, len(simplified)):
        l1, s1_t, e1_t, s1_idx, e1_idx = simplified[i-2]
        l2, s2_t, e2_t, s2_idx, e2_idx = simplified[i-1]
        l3, s3_t, e3_t, s3_idx, e3_idx = simplified[i]

        # 패턴 매칭
        if l1 == "open" and l2 == "unknown" and l3 == "closed":
            events.append({
                "start_time": s2_t,  # unknown 시작 시점
                "end_time": s3_t,    # closed 시작 시점
                "start_idx": s2_idx,
                "end_idx": s3_idx
            })

    # 참고: label_starts 활용 예시
    # print("Open 구간 시작 인덱스:", label_starts.get("open", []))
    # print("Closed 구간 시작 인덱스:", label_starts.get("closed", []))

    return events


def label_data():
    # === file_idx 0~19 반복 ===
    for file_idx in range(20):
        pkl_path = f"logs/{file_idx}/experiment_log.pkl"
        if not os.path.exists(pkl_path):
            print(f"[WARN] {pkl_path} not found. Skipping...")
            continue

        print(f"\n[INFO] Processing file_idx={file_idx}")

        # === 로그 로드 ===
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        print(f"총 {len(data)} 프레임 기록됨")

        timestamps = np.array([frame["timestamp"] for frame in data])
        actions = [frame["action"] if frame["action"] is not None else -1 for frame in data]
        states = [frame["state"] if frame["state"] is not None else np.ones(10) * 100 for frame in data]
        states = np.array(states)

        blue_jaw = np.array([frame.get("blue_jaw", np.nan) for frame in data])
        yellow_jaw = np.array([frame.get("yellow_jaw", np.nan) for frame in data])
        psm1_js = [frame.get("yellow_joint", None) for frame in data]
        psm2_js = [frame.get("blue_joint", None) for frame in data]

        # === Jaw 이벤트 검출 ===
        blue_labels = classify_jaw(blue_jaw)
        yellow_labels = classify_jaw(yellow_jaw)
        blue_close_events = detect_open_close_events(blue_labels, timestamps)
        yellow_close_events = detect_open_close_events(yellow_labels, timestamps)

        # === 이벤트 정리 ===
        rows = []
        for idx, ev in enumerate(blue_close_events):
            rows.append({
                "event": f"blue_jaw_close_{idx}",
                "time": ev["end_time"],
                "psm1_joint": psm1_js[ev["end_idx"]],
                "psm2_joint": psm2_js[ev["end_idx"]],
                "closed_robot": "PSM2"
            })
        for idx, ev in enumerate(yellow_close_events):
            rows.append({
                "event": f"yellow_jaw_close_{idx}",
                "time": ev["end_time"],
                "psm1_joint": psm1_js[ev["end_idx"]],
                "psm2_joint": psm2_js[ev["end_idx"]],
                "closed_robot": "PSM1"
            })

        # === Handover count 계산 ===
        handover_count = 0
        prev_action = actions[0]
        for a in actions:
            if a != prev_action:
                handover_count += 1
            prev_action = a
        print(f"[INFO] Handover count: {handover_count}")

        # === CSV 저장 ===
        csv_file = f"logs/{file_idx}_handover_and_jaw_close_events.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["handover_count", handover_count])
            writer = csv.DictWriter(f, fieldnames=["event", "time", "psm1_joint", "psm2_joint", "closed_robot"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"[INFO] Saved {csv_file}")


# visualize = False
# if visualize:
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#
#     # jaw plotting용 데이터
#     blue_jaw = np.array([frame.get("blue_jaw", np.nan) for frame in data])
#     yellow_jaw = np.array([frame.get("yellow_jaw", np.nan) for frame in data])
#
#     plt.figure(figsize=(14, 6))
#     plt.plot(timestamps, blue_jaw, label="Blue Jaw", color='b', alpha=0.7)
#     plt.plot(timestamps, yellow_jaw, label="Yellow Jaw", color='orange', alpha=0.7)
#
#     # Blue jaw 이벤트 표시
#     for ev in blue_close_events:
#         plt.scatter(ev["end_time"], blue_jaw[ev["end_idx"]],
#                     color='blue', marker='x', s=100, label="Blue Jaw Close Start" if 'Blue Jaw Close Start' not in plt.gca().get_legend_handles_labels()[1] else "")
#
#     # Yellow jaw 이벤트 표시
#     for ev in yellow_close_events:
#         plt.scatter(ev["end_time"], yellow_jaw[ev["end_idx"]],
#                     color='orange', marker='x', s=100, label="Yellow Jaw Close Start" if 'Yellow Jaw Close Start' not in plt.gca().get_legend_handles_labels()[1] else "")
#
#     # 기준선
#     plt.axhline(JAW_CLOSED_TH, color='gray', linestyle='--', label='Closed Threshold')
#     plt.axhline(JAW_OPEN_TH, color='green', linestyle='--', label='Open Threshold')
#
#     plt.xlabel("Time (s)")
#     plt.ylabel("Jaw Opening (rad)")
#     plt.title("Jaw Opening with Close Start Events")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


if __name__ == "__main__":
    # label_data()
