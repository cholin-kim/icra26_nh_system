import rosbag
import cv2
import numpy as np

def record_timestamps(bag_path, image_topic="/dvrk/left/image_raw/compressed"):
    bag = rosbag.Bag(bag_path)
    messages = list(bag.read_messages(topics=[image_topic]))
    bag.close()

    print(f"[INFO] Loaded {len(messages)} frames from {bag_path}")
    print("Controls: 's' save timestamp, 'a' prev, 'd' next, 'q' quit")

    timestamps = []
    idx = 0
    cv2.namedWindow("Frame")


    while True:
        topic, msg, t = messages[idx]
        # === CompressedImage decode ===
        if msg._type == "sensor_msgs/CompressedImage":
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (640, 400))  # 속도 ↑
        else:
            print(f"⚠️ Unexpected message type: {msg._type}")
            idx = min(idx + 1, len(messages) - 1)
            continue

        # 화면에 현재 인덱스와 타임스탬프 표시
        disp_img = img.copy()
        ts = t.to_sec()
        cv2.putText(disp_img, f"Frame {idx}/{len(messages) - 1} | ts: {ts:.3f}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", disp_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamps.append(ts)
            print(f"[SAVED] Timestamp {ts:.6f}")
        elif key == ord('p'):
            idx = max(idx - 1, 0)
        elif key == ord('f'):
            step = 5  # 한 번에 5프레임씩
            idx = min(idx + step, len(messages) - 1)
            # idx = min(idx + 1, len(messages) - 1)
        else:
            # 아무 키 입력 없으면 그냥 다음 프레임
            idx = min(idx + 1, len(messages) - 1)

    cv2.destroyAllWindows()

    # 타임스탬프 저장
    ts_file = bag_path.replace(".bag", "_timestamps.txt")
    with open(ts_file, "w") as f:
        for ts in timestamps:
            f.write(f"{ts}\n")
    print(f"[INFO] {len(timestamps)} timestamps saved to {ts_file}")
    return timestamps


def extract_messages_at_timestamps(bag_path, timestamps, topic="/camera/image_raw", slop=0.005):
    """
    timestamps: 리스트 (초 단위)
    slop: 허용 시간 오차 (초)
    """
    bag = rosbag.Bag(bag_path)
    messages = []
    for topic_msg, msg, t in bag.read_messages(topics=[topic]):
        t_sec = t.to_sec()
        for ts in timestamps:
            if abs(t_sec - ts) <= slop:
                messages.append((ts, msg.position))
                # print(f"Matched message at {t_sec:.6f}")
    bag.close()
    return messages





if __name__ == "__main__":
    # bag_path = "teleop_data/sub_1/extra_2025-09-14-20-59-07.bag"
    # img_topic = "/dvrk/left/image_raw/compressed"
    # psm_blue_topic = "/PSM2/measured_js"
    # psm_yellow_topic = "/PSM1/measured_js"
    #
    #
    #
    # # with open("teleop_data/0_2025-09-14-20-23-31_timestamps.txt", "r") as f:
    # #     timestamps = [float(line.strip()) for line in f if line.strip()]
    #
    # timestamps = record_timestamps(bag_path, img_topic)
    # blue_msgs = extract_messages_at_timestamps(bag_path, timestamps, psm_blue_topic)
    # yellow_msgs = extract_messages_at_timestamps(bag_path, timestamps, psm_yellow_topic)
    #
    # blue_msg_file = bag_path.replace(".bag", "_blue_joint_positions.txt")
    # yellow_msg_file = bag_path.replace(".bag", "_yellow_joint_positions.txt")
    #
    # with open(blue_msg_file, "w") as f:
    #     for msg in blue_msgs:
    #         f.write(f"{msg}\n")
    # with open(yellow_msg_file, "w") as f:
    #     for msg in yellow_msgs:
    #         f.write(f"{msg}\n")

    import os
    bag_path = "teleop_data/sub_1/extra_2025-09-14-20-59-07.bag"
    # 저장할 폴더
    save_dir = "saved_images"
    os.makedirs(save_dir, exist_ok=True)


    left_topic = "/dvrk/left/image_raw/compressed"
    right_topic = "/dvrk/right/image_raw/compressed"

    # Bag 열기
    bag = rosbag.Bag(bag_path)

    # 메시지 읽기
    left_img = None
    right_img = None
    frame_id = 0

    cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right", cv2.WINDOW_NORMAL)

    for topic, msg, t in bag.read_messages(topics=[left_topic, right_topic]):
        # CompressedImage → OpenCV 이미지로 변환
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 이미지 저장
        if topic == left_topic:
            left_img = img.copy()
        elif topic == right_topic:
            right_img = img.copy()

        # 두 이미지가 모두 있으면 표시
        if left_img is not None and right_img is not None:
            stacked = np.hstack((left_img, right_img))
            cv2.imshow("Left", left_img)
            cv2.imshow("Right", right_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # 저장
                left_path = os.path.join(save_dir, f"{frame_id:04d}_left.jpg")
                right_path = os.path.join(save_dir, f"{frame_id:04d}_right.jpg")
                cv2.imwrite(left_path, left_img)
                cv2.imwrite(right_path, right_img)
                print(f"[INFO] Saved: {left_path}, {right_path}")
                frame_id += 1
            elif key == ord('q'):
                break

    cv2.destroyAllWindows()
    bag.close()