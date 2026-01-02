"""
ogbench 데이터 분석 스크립트
.npz 파일로 저장된 ogbench 데이터셋을 직접 로드하여 분석하고 시각화합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os


def load_npz_dataset(file_path):
    """
    .npz 파일을 로드합니다.
    
    Args:
        file_path: .npz 파일 경로
        
    Returns:
        로드된 데이터 (NpzFile 객체, 딕셔너리처럼 접근 가능)
    """
    print(f"파일 로드 중: {file_path}")
    try:
        npz_file = np.load(file_path, allow_pickle=True)
        print(f"✓ 파일 로드 완료")
        return npz_file
    except Exception as e:
        print(f"✗ 파일 로드 실패: {e}")
        raise


def analyze_npz_dataset(file_path):
    """
    .npz 파일로 저장된 ogbench 데이터셋을 분석합니다.
    
    Args:
        file_path: 분석할 .npz 파일 경로
    """
    print(f"\n{'='*60}")
    print(f"데이터셋 분석: {os.path.basename(file_path)}")
    print(f"{'='*60}\n")
    
    # 파일 로드
    try:
        npz_file = load_npz_dataset(file_path)
        print(f"데이터 타입: {type(npz_file)}\n")
    except Exception as e:
        print(f"✗ 데이터 로드 실패: {e}")
        return None
    
    # .npz 파일은 딕셔너리처럼 접근 가능
    # keys()로 저장된 배열 이름들을 확인
    try:
        keys = list(npz_file.keys())
        print(f"NPZ 파일 구조:")
        print(f"  - 저장된 배열 키: {keys}")
        print(f"  - 총 배열 개수: {len(keys)}\n")
        
        # 각 키의 정보 출력 및 데이터 로드
        data_dict = {}
        for key in keys:
            value = npz_file[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}:")
                print(f"    - Shape: {value.shape}")
                print(f"    - Dtype: {value.dtype}")
                print(f"    - Size: {value.size:,}")
                data_dict[key] = value
            else:
                print(f"  {key}: type={type(value)}")
                data_dict[key] = value
        
        print()
        
        # 데이터 분석
        results = analyze_npz_structure(data_dict, file_path)
        
        # 시각화
        if results:
            visualize_results(os.path.basename(file_path), results)
        
        return results
    finally:
        # npz_file 닫기
        npz_file.close()


def analyze_npz_structure(data_dict, file_path):
    """NPZ 파일의 데이터 구조를 분석하고 통계를 추출합니다."""
    results = {
        'file_path': file_path,
        'dataset_size': 0,
        'episode_count': 0,
        'episode_lengths': np.array([]),
        'episode_rewards': np.array([]),
        'episode_returns': np.array([]),
        'data_keys': [],
        'sample_info': {}
    }
    
    # NPZ 파일은 딕셔너리 형태
    if isinstance(data_dict, dict):
        data = data_dict
        results['data_keys'] = list(data.keys())
        print("딕셔너리 데이터 분석:")
        print("-" * 40)
        
        # 주요 키 확인
        observations = data.get('observations', None)
        actions = data.get('actions', None)
        rewards = data.get('rewards', None)
        dones = data.get('dones', None) or data.get('terminals', None)
        episode_starts = data.get('episode_starts', None) or data.get('is_first', None)
        
        if observations is not None:
            results['dataset_size'] = len(observations) if hasattr(observations, '__len__') else observations.shape[0]
            print(f"  총 샘플 수: {results['dataset_size']:,}")
            results['sample_info']['observations'] = {
                'shape': observations.shape if hasattr(observations, 'shape') else 'N/A',
                'dtype': str(observations.dtype) if hasattr(observations, 'dtype') else 'N/A'
            }
            print(f"  Observations: shape={observations.shape}, dtype={observations.dtype}")
        
        if actions is not None:
            results['sample_info']['actions'] = {
                'shape': actions.shape if hasattr(actions, 'shape') else 'N/A',
                'dtype': str(actions.dtype) if hasattr(actions, 'dtype') else 'N/A'
            }
            print(f"  Actions: shape={actions.shape}, dtype={actions.dtype}")
        
        if rewards is not None:
            rewards_array = np.array(rewards) if not isinstance(rewards, np.ndarray) else rewards
            print(f"  Rewards: shape={rewards_array.shape}, dtype={rewards_array.dtype}")
            print(f"    - 평균: {rewards_array.mean():.4f} ± {rewards_array.std():.4f}")
            print(f"    - 범위: [{rewards_array.min():.4f}, {rewards_array.max():.4f}]")
            results['sample_info']['rewards'] = {
                'mean': float(rewards_array.mean()),
                'std': float(rewards_array.std()),
                'min': float(rewards_array.min()),
                'max': float(rewards_array.max())
            }
        
        # 에피소드 분석
        if dones is not None:
            dones_array = np.array(dones) if not isinstance(dones, np.ndarray) else dones
            episode_ends = np.where(dones_array)[0]
            
            if len(episode_ends) > 0:
                # 에피소드 길이 계산
                episode_lengths = []
                episode_rewards_list = []
                
                start_idx = 0
                for end_idx in episode_ends:
                    length = end_idx - start_idx + 1
                    episode_lengths.append(length)
                    
                    if rewards is not None:
                        rewards_array = np.array(rewards) if not isinstance(rewards, np.ndarray) else rewards
                        episode_reward = rewards_array[start_idx:end_idx+1].sum()
                        episode_rewards_list.append(episode_reward)
                    
                    start_idx = end_idx + 1
                
                # 마지막 에피소드 처리
                if start_idx < results['dataset_size']:
                    length = results['dataset_size'] - start_idx
                    episode_lengths.append(length)
                    if rewards is not None:
                        rewards_array = np.array(rewards) if not isinstance(rewards, np.ndarray) else rewards
                        episode_reward = rewards_array[start_idx:].sum()
                        episode_rewards_list.append(episode_reward)
                
                results['episode_count'] = len(episode_lengths)
                results['episode_lengths'] = np.array(episode_lengths)
                results['episode_rewards'] = np.array(episode_rewards_list) if episode_rewards_list else np.array([])
                
                print(f"\n  총 에피소드 수: {results['episode_count']:,}")
                print(f"  평균 에피소드 길이: {results['episode_lengths'].mean():.2f} ± {results['episode_lengths'].std():.2f}")
                print(f"  최소/최대 에피소드 길이: {results['episode_lengths'].min()} / {results['episode_lengths'].max()}")
                
                if len(results['episode_rewards']) > 0:
                    print(f"  평균 에피소드 보상: {results['episode_rewards'].mean():.4f} ± {results['episode_rewards'].std():.4f}")
                    print(f"  최소/최대 에피소드 보상: {results['episode_rewards'].min():.4f} / {results['episode_rewards'].max():.4f}")
        
        # Goal 정보 확인 및 분석
        goals_info = analyze_goals(data, results)
        if goals_info:
            results['goals_info'] = goals_info
    
    else:
        print("  ⚠ 딕셔너리 형태의 데이터가 아닙니다.")
    
    print()
    return results


def analyze_goals(data, results):
    """Goal 정보를 분석합니다. qpos나 다른 필드에서 goal 정보를 추출합니다."""
    goals_info = {}
    
    # 직접 goal 키가 있는 경우
    if 'goals' in data or 'goal' in data:
        goal_key = 'goals' if 'goals' in data else 'goal'
        goals = data[goal_key]
        if isinstance(goals, np.ndarray):
            print(f"\n  Goal 정보 (직접):")
            print(f"    - Shape: {goals.shape}")
            print(f"    - Dtype: {goals.dtype}")
            goals_info['goals'] = goals
            goals_info['goal_type'] = 'direct'
            return goals_info
    
    # qpos에서 goal 정보 추출 (visual-antmaze 등의 경우)
    if 'qpos' in data:
        qpos = data['qpos']
        if isinstance(qpos, np.ndarray) and qpos.shape[1] >= 2:
            print(f"\n  Goal 분석 (qpos 기반):")
            print(f"    - Qpos shape: {qpos.shape}")
            
            # 에피소드별 goal 추출 (에피소드의 마지막 상태를 goal로 간주)
            episode_lengths = results.get('episode_lengths', np.array([]))
            episode_count = results.get('episode_count', 0)
            
            if len(episode_lengths) > 0:
                # 각 에피소드의 마지막 상태를 goal로 추정
                goals_list = []
                goal_positions_list = []
                
                start_idx = 0
                for episode_length in episode_lengths:
                    end_idx = start_idx + episode_length - 1
                    if end_idx < len(qpos):
                        # 마지막 상태의 qpos를 goal로 간주
                        goal_qpos = qpos[end_idx]
                        goals_list.append(goal_qpos)
                        
                        # 위치 정보 (일반적으로 qpos의 첫 2-3개 차원이 x, y, z)
                        if len(goal_qpos) >= 2:
                            goal_positions_list.append(goal_qpos[:2])  # x, y 좌표
                    
                    start_idx = end_idx + 1
                
                if len(goals_list) > 0:
                    goals_array = np.array(goals_list)
                    goal_positions = np.array(goal_positions_list)
                    
                    print(f"    - 추출된 Goal 수: {len(goals_list):,}")
                    print(f"    - Goal shape: {goals_array.shape}")
                    
                    # Goal 위치 분포 분석
                    if len(goal_positions) > 0:
                        print(f"\n    Goal 위치 분포 (x, y 좌표):")
                        print(f"      - X 범위: [{goal_positions[:, 0].min():.4f}, {goal_positions[:, 0].max():.4f}]")
                        print(f"      - Y 범위: [{goal_positions[:, 1].min():.4f}, {goal_positions[:, 1].max():.4f}]")
                        print(f"      - X 평균: {goal_positions[:, 0].mean():.4f} ± {goal_positions[:, 0].std():.4f}")
                        print(f"      - Y 평균: {goal_positions[:, 1].mean():.4f} ± {goal_positions[:, 1].std():.4f}")
                        
                        # 고유한 goal 개수 (반올림하여)
                        unique_goals_rounded = np.round(goal_positions, decimals=2)
                        unique_goals_count = len(np.unique(unique_goals_rounded, axis=0))
                        print(f"      - 고유한 Goal 개수 (소수점 2자리 기준): {unique_goals_count:,}")
                        
                        goals_info['goals'] = goals_array
                        goals_info['goal_positions'] = goal_positions
                        goals_info['goal_type'] = 'qpos_episode_end'
                        goals_info['unique_goals_count'] = unique_goals_count
                        
                        # 샘플 goal 위치 출력
                        print(f"\n    샘플 Goal 위치 (처음 10개):")
                        for i, pos in enumerate(goal_positions[:10]):
                            print(f"      Goal {i+1}: x={pos[0]:.4f}, y={pos[1]:.4f}")
                    
                    return goals_info
            else:
                # 에피소드 정보가 없는 경우, 전체 qpos의 분포만 분석
                print(f"    - Qpos 샘플 수: {len(qpos):,}")
                if qpos.shape[1] >= 2:
                    positions = qpos[:, :2]
                    print(f"    - 위치 분포 (x, y):")
                    print(f"      - X 범위: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}]")
                    print(f"      - Y 범위: [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}]")
                    goals_info['goal_positions'] = positions
                    goals_info['goal_type'] = 'qpos_all'
                    return goals_info
    
    print("  ⚠ Goal 정보를 찾을 수 없습니다.")
    return None


def visualize_results(dataset_name, results):
    """분석 결과를 시각화합니다."""
    print("시각화 생성 중...")
    
    # 출력 디렉토리 생성
    output_dir = 'ogbench_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 결과가 없는 경우
    if not results or len(results.get('episode_lengths', [])) == 0:
        print("  ⚠ 시각화할 데이터가 없습니다.")
        return
    
    # Goal 정보가 있으면 3x2로, 없으면 2x2로 subplot 생성
    goals_info = results.get('goals_info', None)
    if goals_info and 'goal_positions' in goals_info:
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Dataset Analysis: {dataset_name}', fontsize=16, fontweight='bold')
    
    # 에피소드 길이 분포
    ax = axes[0, 0]
    episode_lengths = results['episode_lengths']
    if len(episode_lengths) > 0:
        ax.hist(episode_lengths, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(episode_lengths.mean(), color='red', linestyle='--', 
                   label=f'Mean: {episode_lengths.mean():.2f}')
        ax.legend()
    ax.set_xlabel('Episode Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Episode Length Distribution')
    ax.grid(True, alpha=0.3)
    
    # 에피소드 보상 분포
    ax = axes[0, 1]
    episode_rewards = results['episode_rewards']
    if len(episode_rewards) > 0 and episode_rewards.sum() != 0:
        ax.hist(episode_rewards, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(episode_rewards.mean(), color='red', linestyle='--',
                   label=f'Mean: {episode_rewards.mean():.4f}')
        ax.legend()
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Episode Reward Distribution')
    ax.grid(True, alpha=0.3)
    
    # 데이터셋 통계
    ax = axes[1, 0]
    categories = ['Samples', 'Episodes']
    values = [results['dataset_size'], results['episode_count']]
    colors = ['skyblue', 'lightcoral']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Dataset Statistics')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:,}', ha='center', va='bottom')
    
    # 에피소드 길이 박스플롯
    ax = axes[1, 1]
    if len(episode_lengths) > 0:
        bp = ax.boxplot([episode_lengths], labels=['Episodes'], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Length Statistics')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Goal 위치 시각화 (있는 경우)
    if goals_info and 'goal_positions' in goals_info:
        goal_positions = goals_info['goal_positions']
        
        # Goal 위치 2D 산점도
        ax = axes[2, 0]
        ax.scatter(goal_positions[:, 0], goal_positions[:, 1], 
                  alpha=0.5, s=10, c='red', edgecolors='black', linewidths=0.5)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Goal Positions Distribution (Total: {len(goal_positions):,})')
        ax.grid(True, alpha=0.3)
        
        # Goal 위치 히스토그램 (2D)
        ax = axes[2, 1]
        if len(goal_positions) > 0:
            # 2D 히스토그램
            hb = ax.hexbin(goal_positions[:, 0], goal_positions[:, 1], 
                          gridsize=30, cmap='YlOrRd', mincnt=1)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title('Goal Positions Density')
            plt.colorbar(hb, ax=ax, label='Count')
    
    plt.tight_layout()
    
    # 파일 저장
    safe_name = dataset_name.replace('/', '_').replace('-', '_').replace('.npy', '')
    output_path = os.path.join(output_dir, f'{safe_name}_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 시각화 저장: {output_path}")
    
    plt.close()


def analyze_multiple_npz_files(file_paths):
    """여러 .npz 파일을 일괄 분석합니다."""
    all_results = {}
    
    for file_path in file_paths:
        try:
            results = analyze_npz_dataset(file_path)
            if results:
                all_results[os.path.basename(file_path)] = results
        except Exception as e:
            print(f"✗ {file_path} 분석 실패: {e}\n")
            continue
    
    # 요약 리포트 생성
    if all_results:
        print_summary_report(all_results)
    
    return all_results


def print_summary_report(all_results):
    """모든 데이터셋 분석 결과 요약 리포트를 출력합니다."""
    print(f"\n{'='*60}")
    print("요약 리포트")
    print(f"{'='*60}\n")
    
    print(f"{'Dataset':<50} {'Samples':<15} {'Episodes':<15}")
    print("-" * 80)
    
    for dataset_name, results in all_results.items():
        dataset_size = results.get('dataset_size', 0)
        episode_count = results.get('episode_count', 0)
        print(f"{dataset_name:<50} {dataset_size:<15,} {episode_count:<15,}")


if __name__ == '__main__':
    # 분석할 .npz 파일 경로
    file_path = '/ext2/skshyn/.ogbench/visual-antmaze-medium-navigate-v0.npz'
    
    # 단일 파일 분석
    analyze_npz_dataset(file_path)
    
    # 여러 파일 일괄 분석 (예시)
    # file_paths = [
    #     '/ext2/skshyn/.ogbench/visual-antmaze-medium-navigate-v0.npz',
    #     # 다른 파일들...
    # ]
    # analyze_multiple_npz_files(file_paths)
