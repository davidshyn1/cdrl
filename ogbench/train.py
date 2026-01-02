"""
ogbench 데이터로 Dreamer 학습을 위한 스크립트
"""

# MuJoCo headless 모드 설정 (DISPLAY 없이 동작)
# IMPORTANT: 다른 import 전에 반드시 설정해야 함
import os
import sys

# MuJoCo headless 모드 설정 (download_data.py와 완전히 동일하게)
# 여러 옵션 시도: 'egl', 'osmesa', 'glfw'
if 'DISPLAY' not in os.environ:
    # DISPLAY가 없으면 egl을 먼저 시도
    os.environ.setdefault('MUJOCO_GL', 'egl')
else:
    # DISPLAY가 있어도 egl 사용
    os.environ['MUJOCO_GL'] = 'egl'

# OpenGL 에러 체크 비활성화 (EGL 종료 에러 억제)
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# EGL 경고 억제를 위한 환경 변수 설정
os.environ['EGL_LOG_LEVEL'] = 'fatal'  # fatal 레벨만 출력
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

import argparse
import json
import pathlib
import warnings
from collections import OrderedDict

# EGL 종료 시 발생하는 에러 경고 억제
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*EGL.*')
warnings.filterwarnings('ignore', message='.*OpenGL.*')

# EGL 에러 필터링
class EGLErrorFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    
    def write(self, text):
        error_patterns = [
            'EGLError',
            'Renderer.__del__',
            'GLContext.__del__',
            'OpenGL.raw.EGL._errors.EGLError',
            'Exception ignored in: <function',
            'mujoco/renderer.py',
            'mujoco/egl/__init__.py',
            'libEGL',
            'libEGL warning',
            'egl:',
            'failed to create dri2 screen',
            'failed to open /dev/dri/card',
            'Permission denied',
            '/dev/dri/card'
        ]
        # 대소문자 구분 없이 체크
        text_lower = text.lower()
        if any(pattern.lower() in text_lower for pattern in error_patterns):
            return
        self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

original_stderr = sys.stderr
sys.stderr = EGLErrorFilter(original_stderr)


# ogbench import
import ogbench

import numpy as np
import ruamel.yaml as yaml
import torch
from torch import distributions as torchd

# 상위 디렉토리를 path에 추가
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import models
import tools

# dreamer.py가 MUJOCO_GL을 osmesa로 설정하므로, import 후 다시 egl로 설정
from dreamer import Dreamer
# dreamer import 후 환경 변수 다시 설정 (dreamer.py가 osmesa로 덮어씌웠을 수 있음)
os.environ['MUJOCO_GL'] = 'egl'



to_np = lambda x: x.detach().cpu().numpy()


def load_ogbench_datasets_from_tasks(dataset_name, base_dir, train_task_ids, eval_task_id):
    """
    ogbench.make_env_and_datasets를 사용하여 여러 task의 데이터를 로드합니다.
    
    Args:
        dataset_name: 데이터셋 이름 (예: 'visual-antmaze-medium-navigate-v0')
        base_dir: 기본 디렉토리 경로 (예: '/ext/skshyn/.ogbench')
        train_task_ids: training에 사용할 task ID 리스트 (예: [1, 2, 3, 4])
        eval_task_id: evaluation에 사용할 task ID (예: 5)
        
    Returns:
        (env, train_episodes, eval_episodes, obs_space, act_space)
    """
    train_episodes = OrderedDict()
    eval_episodes = OrderedDict()
    env = None
    obs_space = None
    act_space = None
    
    # Training tasks 로드
    print(f"Loading training data from tasks: {train_task_ids}")
    for task_id in train_task_ids:
        dataset_name = dataset_name.replace('-v0', f'-singletask-task{task_id}-v0')
        task_dir = os.path.join(base_dir, f"task{task_id}")
        print(f"  Loading task{task_id} from {task_dir}...")
        
        task_env, train_dataset, _ = ogbench.make_env_and_datasets(
            dataset_name,
            dataset_dir=task_dir,
            compact_dataset=False,
            add_info=False
        )
        # 첫 번째 task에서 observation/action space 추출
        if obs_space is None:
            obs_space = task_env.observation_space
            act_space = task_env.action_space
            env = task_env  # 참조만 유지
        
        # dataset을 episodes로 변환
        task_episodes = convert_ogbench_dataset_to_episodes(train_dataset, task_id)
        # task별로 그룹화
        train_episodes[f'task{task_id}'] = task_episodes
        
        print(f"    Loaded {len(task_episodes)} episodes from task{task_id}")
        
        # env는 한 번만 유지 (나중에 닫기 위해)
        # 하지만 space는 확실히 추출
        if obs_space is None or act_space is None:
            obs_space = task_env.observation_space
            act_space = task_env.action_space
            env = task_env
            print(f"    Extracted space from task{task_id}: obs={obs_space}, act={act_space}")
        
        if task_id != train_task_ids[0]:
            try:
                if hasattr(task_env, 'renderer') and task_env.renderer is not None:
                    try:
                        task_env.renderer.close()
                    except:
                        pass
                task_env.close()
            except:
                pass    
    # Evaluation task 로드
    print(f"\nLoading evaluation data from task: {eval_task_id}")
    eval_task_dir = os.path.join(base_dir, f"task{eval_task_id}")
    print(f"  Loading task{eval_task_id} from {eval_task_dir}...")
    
    try:
        eval_env, _, eval_dataset = ogbench.make_env_and_datasets(
            dataset_name,
            dataset_dir=eval_task_dir,
            compact_dataset=False,
        )
                
        # dataset을 episodes로 변환
        eval_episodes_dict = convert_ogbench_dataset_to_episodes(eval_dataset, eval_task_id)
        eval_episodes.update(eval_episodes_dict)
        
        print(f"    Loaded {len(eval_episodes_dict)} episodes from task{eval_task_id}")
        
        # eval_env는 space 추출용이 아니면 닫기 (env가 이미 설정되어 있다면)
        if env != eval_env:
            try:
                if hasattr(eval_env, 'renderer') and eval_env.renderer is not None:
                    try:
                        eval_env.renderer.close()
                    except:
                        pass
                eval_env.close()
            except:
                pass
            
    except Exception as e:
        print(f"    Warning: Failed to load eval task{eval_task_id}: {e}")
    
    # 총 episode 수 계산
    total_train_eps = sum(len(eps) for eps in train_episodes.values())
    total_eval_eps = sum(len(eps) for eps in eval_episodes.values())
    print(f"\nTotal training episodes: {total_train_eps} (across {len(train_episodes)} tasks)")
    print(f"Total evaluation episodes: {total_eval_eps} (across {len(eval_episodes)} tasks)")
    
    # 최종 확인
    if obs_space is None or act_space is None:
        raise ValueError(f"Failed to extract observation or action space. obs_space: {obs_space}, act_space: {act_space}")
    
    return eval_env, train_episodes, eval_episodes, obs_space, act_space


def convert_ogbench_dataset_to_episodes(dataset, task_id):
    """
    ogbench dataset을 Dreamer가 기대하는 episodes 형태로 변환합니다.
    
    ogbench의 dataset은 일반적으로 episode들의 리스트이거나,
    또는 각 episode가 dict 형태로 저장되어 있을 수 있습니다.
    
    Args:
        dataset: ogbench dataset (일반적으로 list of episodes 또는 iterable)
        task_id: task ID (naming용)
        
    Returns:
        OrderedDict of episodes
    """
    episodes = OrderedDict()
    
    # dataset이 dict이고 observation, action, terminal 등의 키를 가진 경우
    # terminal에 따라 episode로 나눔
    required_keys = ['observations', 'actions', 'terminals', 'next_observations', 'rewards']
    if all(key in dataset for key in required_keys):
        # terminal에 따라 episode로 나누기
        observations = dataset['observations']
        actions = dataset['actions']
        terminals = dataset['terminals']
        next_observations = dataset['next_observations']
        # rewards와 discount는 선택적
        rewards = dataset.get('rewards', None)
        # episode 시작 인덱스 추적
        ep_start = 0
        ep_idx = 0
        
        # terminal 값들을 순회하며 episode 구분
        for i, terminal in enumerate(terminals):
            # terminal이 True이거나 마지막 transition인 경우 episode 종료
            if terminal or i == len(terminals) - 1:
                # episode 생성 (i+1까지 포함)
                ep_end = i + 1
                
                # observations를 (H, W, C) -> (C, H, W)로 변환
                ep_obs = observations[ep_start:ep_end]
                ep_next_obs = next_observations[ep_start:ep_end]
                
                # # (T, H, W, C) 또는 (H, W, C) 형태를 (T, C, H, W) 또는 (C, H, W)로 변환
                # if ep_obs.ndim == 4:  # (T, H, W, C)
                #     ep_obs = np.transpose(ep_obs, (0, 3, 1, 2))  # (T, C, H, W)
                #     ep_next_obs = np.transpose(ep_next_obs, (0, 3, 1, 2))  # (T, C, H, W)
                # elif ep_obs.ndim == 3:  # (H, W, C)
                #     ep_obs = np.transpose(ep_obs, (2, 0, 1))  # (C, H, W)
                #     ep_next_obs = np.transpose(ep_next_obs, (2, 0, 1))  # (C, H, W)
                
                episode = {
                    'observations': ep_obs,
                    'actions': actions[ep_start:ep_end],
                    'terminals': terminals[ep_start:ep_end],
                    'next_observations': ep_next_obs,
                    'rewards': rewards[ep_start:ep_end],
                }
                episodes[f'episode_{ep_idx}'] = episode
                ep_start = ep_end
                ep_idx += 1
        
        print(f"    Converted {ep_idx} episodes from dataset (terminal-based splitting)")
    
    # # dataset이 list인 경우
    # elif isinstance(dataset, list):
    #     for ep_idx, episode in enumerate(dataset):
    #         if isinstance(episode, dict):
    #             # episode가 이미 dict 형태인 경우
    #             episodes[f'task{task_id}_episode_{ep_idx}'] = episode
    #         else:
    #             print(f"    Warning: Episode {ep_idx} is not a dict, type: {type(episode)}")
    
    # # dataset이 이미 OrderedDict인 경우
    # elif isinstance(dataset, OrderedDict):
    #     for key, episode in dataset.items():
    #         episodes[f'task{task_id}_{key}'] = episode
    
    # # dataset이 길이를 가지는 경우 (len() 사용 가능)
    # elif hasattr(dataset, '__len__'):
    #     try:
    #         length = len(dataset)
    #         print(f"    Dataset has length {length}, but structure is unclear. "
    #               f"Expected dict with 'observation', 'action', 'terminal' keys.")
    #     except Exception as e:
    #         print(f"    Warning: Could not get dataset length: {e}")
        
    return episodes

# def get_observation_space_from_data(data_dict, config):
#     """
#     데이터에서 observation space를 추론합니다.
#     """
#     import gym.spaces as spaces
    
#     observations = data_dict.get('observations', data_dict.get('observation', None))
#     if observations is None:
#         raise ValueError("observations not found in data")
    
#     observations = np.array(observations)
#     obs_dict = {}
    
#     # 이미지 shape 확인
#     if observations.ndim >= 3:  # (T, H, W, C) 또는 (H, W, C)
#         if observations.ndim == 3:
#             H, W, C = observations.shape
#         else:
#             # 첫 번째 샘플의 shape 사용
#             H, W, C = observations.shape[-3:]
#         obs_dict['observations'] = spaces.Box(0, 255, (H, W, C), dtype=np.uint8)
#     else:
#         # 1D 또는 2D observation
#         if observations.ndim == 1:
#             shape = (len(observations),)
#         else:
#             shape = observations.shape[1:]
#         obs_dict['observations'] = spaces.Box(0, 255, shape, dtype=np.uint8)
    
#     # is_first, is_last, is_terminal 추가
#     obs_dict['is_first'] = spaces.Box(0, 1, (1,), dtype=np.uint8)
#     obs_dict['is_last'] = spaces.Box(0, 1, (1,), dtype=np.uint8)
#     obs_dict['is_terminal'] = spaces.Box(0, 1, (1,), dtype=np.uint8)
    
#     return spaces.Dict(obs_dict)


# def get_action_space_from_data(data_dict):
#     """
#     데이터에서 action space를 추론합니다.
#     """
#     import gym.spaces as spaces
    
#     actions = data_dict.get('actions', None)
#     if actions is None:
#         raise ValueError("actions not found in data")
    
#     actions = np.array(actions)
    
#     if actions.ndim == 1:
#         # Discrete action space
#         num_actions = int(actions.max() + 1)
#         return spaces.Discrete(num_actions)
#     else:
#         # Continuous action space
#         action_shape = actions.shape[1:] if actions.ndim > 1 else (actions.shape[0],)
#         action_min = float(actions.min())
#         action_max = float(actions.max())
#         return spaces.Box(low=action_min, high=action_max, shape=action_shape, dtype=np.float32)


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    
    # logdir이 설정되지 않은 경우 기본값 사용
    logdir_path = getattr(config, 'logdir', None)
    if logdir_path is None:
        logdir_path = './logs/debug_train'
    logdir = pathlib.Path(logdir_path).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    
    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    
    # ogbench 데이터 로드 (make_env_and_datasets 사용)
    dataset_name = getattr(config, 'ogbench_dataset_name', 'visual-antmaze-medium-navigate-v0')
    base_dir = getattr(config, 'ogbench_base_dir', '/ext/skshyn/.ogbench')
    train_task_ids = getattr(config, 'ogbench_train_tasks', [2])
    eval_task_id = getattr(config, 'ogbench_eval_task', 3)
    
    # 디버깅: config 값 확인
    print(f"DEBUG: config.ogbench_dataset_name = {getattr(config, 'ogbench_dataset_name', 'NOT SET')}")
    print(f"DEBUG: dataset_name = {dataset_name}")
    
    print(f"Loading ogbench datasets...")
    print(f"  Dataset: {dataset_name}")
    print(f"  Base directory: {base_dir}")
    print(f"  Training tasks: {train_task_ids}")
    print(f"  Evaluation task: {eval_task_id}")
    
    eval_env, train_eps, eval_eps, obs_space, act_space = load_ogbench_datasets_from_tasks(
        dataset_name, base_dir, train_task_ids, eval_task_id
    )
    
    # env는 observation/action space 추출용으로만 사용했으므로 닫기
    if eval_env is not None:
        try:
            if hasattr(eval_env, 'renderer') and eval_env.renderer is not None:
                try:
                    eval_env.renderer.close()
                except:
                    pass
            eval_env.close()
        except:
            pass
    
    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")
    
    # action space에서 num_actions 설정
    if hasattr(act_space, 'n'):
        config.num_actions = act_space.n
    else:
        config.num_actions = act_space.shape[0] if hasattr(act_space, 'shape') else 1
    
    # logger 초기화 (wandb 사용)
    # config를 dict로 변환 (wandb에 전달하기 위해)
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    logger = tools.Logger(
        logdir, 
        0,
        project=getattr(config, 'wandb_project', 'dreamer-ogbench'),
        name=getattr(config, 'wandb_name', None),
        config=config_dict
    )
    
    # dataset 생성
    print("Creating dataset...")
    train_datasets = {}
    for key in train_eps.keys():
        train_datasets[key] = tools.from_generator(
            tools.sample_episodes(train_eps[key], config.batch_length),
            config.batch_size
        )
    eval_dataset = tools.from_generator(
        tools.sample_episodes(eval_eps, config.batch_length),
        config.batch_size
    )
    
    # train_datasets에서 task를 uniformly random으로 선택하는 generator
    def sample_from_tasks():
        task_keys = list(train_datasets.keys())
        np_random = np.random.RandomState(0)
        while True:
            # uniformly random으로 task 선택
            selected_task = np_random.choice(task_keys)
            yield next(train_datasets[selected_task])
    
    train_dataset = sample_from_tasks()
    
    # Dreamer agent 초기화
    print("Initializing Dreamer agent...")
    agent = Dreamer(
        obs_space,
        act_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    # agent.requires_grad_(requires_grad=False)
    
    # 체크포인트 로드
    if (logdir / "latest.pt").exists():
        print("Loading checkpoint...")
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
    
    # 학습 루프
    print("Starting training...")
    action_repeat = getattr(config, 'action_repeat', 1)
    config.steps = int(config.steps) // action_repeat
    config.eval_every = int(config.eval_every) // action_repeat
    config.log_every = int(config.log_every) // action_repeat
    
    # steps를 업데이트 수로 변환
    # train_ratio: 환경 스텝당 학습 업데이트 수
    batch_steps = config.batch_size * config.batch_length
    updates_per_env_step = batch_steps / config.train_ratio
    max_updates = int(config.steps * updates_per_env_step)
    
    print(f"Max updates: {max_updates}")
    print(f"Eval every {config.eval_every} env steps (~{int(config.eval_every * updates_per_env_step)} updates)")
    print(f"Log every {config.log_every} env steps (~{int(config.log_every * updates_per_env_step)} updates)")
    
    eval_update_freq = max(1, int(config.eval_every * updates_per_env_step)) if config.eval_every > 0 else max_updates + 1
    log_update_freq = max(1, int(config.log_every * updates_per_env_step))
    checkpoint_freq = max(1, int(config.eval_every * updates_per_env_step))
    
    while agent._update_count < max_updates:
        # Training step
        # train_dataset은 무한 generator이므로 StopIteration이 발생하지 않음
        data = next(train_dataset)
        ##TODO : Data가 뭔지 확인 (inputs로 들어가야 함.)
        agent._train(data)
        agent._update_count += updates_per_env_step
        
        # Logging (Dreamer의 내부 메커니즘 활용)
        if agent._update_count % log_update_freq == 0:
            # agent._metrics에서 로깅할 데이터 수집
            # for name, values in agent._metrics.items():
            for name, values in agent._metrics.items():
                if len(values) > 0:
                    logger.scalar(name, float(np.mean(values)))
                    agent._metrics[name] = []  # 로깅 후 클리어
            logger.write(fps=True)
        
        # Evaluation
        if config.eval_episode_num > 0 and agent._update_count % eval_update_freq == 0:
            print(f"[Update {agent._update_count}] Start evaluation (offline dataset only)...")
            if len(eval_eps) > 0:
                # Validation 데이터셋으로 평가 (loss 계산)
                eval_metrics = {}
                eval_dataset_iter = tools.from_generator(
                    tools.sample_episodes(eval_eps, config.batch_length),
                    config.batch_size
                )
                for _ in range(min(10, len(eval_eps) // config.batch_size + 1)):  # 몇 개 배치만 평가
                    try:
                        eval_data = next(eval_dataset_iter)
                        with torch.no_grad():
                            post, context, mets = agent._wm._train(eval_data, eval_mode=True)
                            for k, v in mets.items():
                                if k not in eval_metrics:
                                    eval_metrics[k] = []
                                eval_metrics[k].append(v)
                    except StopIteration:
                        break
                
                for k, v in eval_metrics.items():
                    logger.scalar(f"eval_{k}", float(np.mean(v)))
                logger.write()
        
        # 체크포인트 저장
        # if agent._update_count % checkpoint_freq == 0 and agent._update_count > 0:
        #     items_to_save = {
        #         "agent_state_dict": agent.state_dict(),
        #         "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        #     }
        #     torch.save(items_to_save, logdir / "latest.pt")
        #     print(f"Checkpoint saved at update {agent._update_count}")
    
    # 최종 체크포인트 저장
    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")
    
    # wandb 종료
    logger.finish()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()

    script_dir = pathlib.Path('/home/skshyn/crossdomain/cdrl/ogbench')
    
    # configs.yaml 경로 생성 및 검증
    conf_path = script_dir / "configs.yaml"
    conf_path = conf_path.resolve()  # 절대 경로로 변환
    
    print(f"DEBUG: script_dir = {script_dir}")
    print(f"DEBUG: conf_path = {conf_path}")
    print(f"DEBUG: conf_path.is_file() = {conf_path.is_file()}")
    print(f"DEBUG: conf_path.parent = {conf_path.parent}")
    print(f"DEBUG: conf_path.parent.is_dir() = {conf_path.parent.is_dir()}")
    
    if not conf_path.exists() or not conf_path.is_file():
        raise FileNotFoundError(
            f"Config file not found at: {conf_path}\n"
            f"Parent directory exists: {conf_path.parent.exists()}\n"
            f"Parent is directory: {conf_path.parent.is_dir()}"
        )
    
    # ruamel.yaml의 YAML 객체 사용 (load()는 deprecated됨)
    y = yaml.YAML(typ='safe', pure=True)
    with conf_path.open('r', encoding='utf-8') as f:
        configs = y.load(f)
    
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value
    
    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    
    # ogbench 특화 인자 추가
    parser.add_argument("--ogbench_dataset_name", type=str, default="visual-antmaze-medium-navigate-v0",
                       help="ogbench dataset name")
    parser.add_argument("--ogbench_base_dir", type=str, default="/ext/skshyn/.ogbench",
                       help="Base directory containing task1, task2, ... directories")
    parser.add_argument("--ogbench_train_tasks", type=int, nargs="+", default=[1], #[1, 2, 3, 4],
                       help="Task IDs to use for training (e.g., 1 2 3 4)")
    parser.add_argument("--ogbench_eval_task", type=int, default=5,
                       help="Task ID to use for evaluation (e.g., 5)")
    
    # wandb 관련 인자 추가
    parser.add_argument("--wandb_project", type=str, default="dreamer-ogbench",
                       help="wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                       help="wandb run name (default: logdir name)")
    
    main(parser.parse_args(remaining))

