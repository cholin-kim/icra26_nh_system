import re
import numpy as np

def parse_step_lines(step_lines):
    """한 스텝의 라인들을 dict로 변환"""
    step_dict = {}
    key = None
    value_lines = []

    for line in step_lines:
        stripped = line.strip()
        if not stripped:
            continue

        # 한 줄에 key: value 있는 경우
        if ":" in stripped and not stripped.endswith(":"):
            k, v = stripped.split(":", 1)
            step_dict[k.strip()] = v.strip()

        # key: 만 있고 다음 줄부터 value 시작
        elif stripped.endswith(":"):
            if key is not None:
                step_dict[key] = "\n".join(value_lines).strip()
            key = stripped[:-1].strip()
            value_lines = []

        else:  # value만 있는 줄
            value_lines.append(stripped)

    # 마지막 key 저장
    if key is not None:
        step_dict[key] = "\n".join(value_lines).strip()

    return step_dict


def split_by_empty_lines(lines):
    """빈 줄 기준으로 리스트를 블록 단위로 분리"""
    blocks = []
    block = []
    for line in lines:
        if line.strip() == "":
            if block:
                blocks.append(block)
                block = []
        else:
            block.append(line)
    if block:  # 마지막 블록 추가
        blocks.append(block)
    return blocks


def extract_episode_dict(file_content, episode_num):
    """
    특정 episode_num의 데이터를 dict로 추출
    """
    lines = file_content.splitlines()
    if episode_num < 10:
        start_marker = f"[Episode 0{episode_num - 1}]"
        end_marker = f"[Episode 0{episode_num}]"
    elif episode_num == 10:
        start_marker = f"[Episode 0{episode_num - 1}]"
        end_marker = f"[Episode {episode_num}]"
    else:
        start_marker = f"[Episode {episode_num - 1}]"
        end_marker = f"[Episode {episode_num}]"

    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i + 1
        if end_marker in line and start_idx is not None:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        raise ValueError(f"Episode {episode_num-1} 또는 Episode {episode_num}를 찾을 수 없습니다.")

    episode_lines = lines[start_idx:end_idx]
    summary_line = lines[end_idx]

    # steps 수 추출
    match = re.search(r"Steps:\s*(\d+)", summary_line)
    if not match:
        raise ValueError("Steps 정보를 찾을 수 없습니다.")
    steps = int(match.group(1))

    episode_dict = {"steps": steps}

    # step 블록 나누기
    blocks = split_by_empty_lines(episode_lines)

    # steps 수와 블록 수 맞춰서 dict 구성
    for i, block in enumerate(blocks, start=1):
        episode_dict[str(i)] = parse_step_lines(block)

    return episode_dict


def episode_dict_to_numpy(step_dict):
    """
    step_dict: {key: value} 형태
    return: key는 그대로, value는 가능하면 np.array로 변환
    """
    result = {}
    for key, value in step_dict.items():
        if key == "reward_type":
            result[key] = value
            continue
        if key == "steps":
            result[key] = value
            continue

        try:
            if key == "action_unpacked":
                arr = np.array([float(x) for x in value.replace(",", " ").split()])
                result[key] = arr
            else:
                lines = value.strip().split("\n")
                arr = []
                for line in lines:
                    line = line.strip("[]")  # 양쪽 [] 제거
                    numbers = [float(x) for x in line.replace(",", " ").split()]
                    arr.append(numbers)
                result[key] = np.array(arr).reshape(4, 4)
        except Exception:
            result[key] = value  # 변환 실패 시 문자열 그대로

    return result


if __name__ == "__main__":
    file_path = "model_97000.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # episode_num [2, 99]

    episode_data_dict = extract_episode_dict(content, episode_num=2)

    # step 별 numpy 변환
    for step, step_dict in episode_data_dict.items():
        if step == "steps":
            continue
        print(f"\n=== Step {step} ===")
        episode_np_dict = episode_dict_to_numpy(step_dict)
        for k, v in episode_np_dict.items():
            print(k, ":\n", v, "\n", type(v))
