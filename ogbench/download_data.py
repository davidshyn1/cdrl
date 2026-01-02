import os
import sys
import warnings
from contextlib import redirect_stderr
from io import StringIO

# MuJoCo headless 모드 설정 (DISPLAY 없이 동작)
# 여러 옵션 시도: 'egl', 'osmesa', 'glfw'
if 'DISPLAY' not in os.environ:
    # DISPLAY가 없으면 egl을 먼저 시도, 실패하면 환경 생성 시 에러 발생
    # 대안: Xvfb 사용 (xvfb-run -a python script.py)
    os.environ.setdefault('MUJOCO_GL', 'egl')

# EGL 종료 시 발생하는 에러 경고 억제
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*EGL.*')

# OpenGL 에러 체크 비활성화 (EGL 종료 에러 억제)
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# EGL 경고 억제를 위한 환경 변수 설정
os.environ['EGL_LOG_LEVEL'] = 'fatal'  # fatal 레벨만 출력
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

# stderr 필터링 클래스 (EGL 에러 메시지 숨기기)
class EGLErrorFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    
    def write(self, text):
        # EGL 관련 에러 메시지 필터링 (여러 패턴 체크)
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
        # 다른 속성은 원본 stderr로 위임
        return getattr(self.original_stderr, name)

# stderr 필터 적용
original_stderr = sys.stderr
sys.stderr = EGLErrorFilter(original_stderr)

import ogbench

# # Make an environment and load datasets.
# dataset_name = 'antmaze-large-navigate-v0'
# env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
#     dataset_name,  # Dataset name.
#     dataset_dir='~/.ogbench/data',  # Directory to save datasets (optional).
#     compact_dataset=False,  # Whether to use a compact dataset (optional; see below).
# )

'''
datasets = [
    'humanoidmaze-medium-navigate-v0','humanoidmaze-large-navigate-v0','humanoidmaze-giant-navigate-v0',
    'humanoidmaze-medium-stitch-v0', 'humanoidmaze-large-stitch-v0', 'humanoidmaze-giant-stitch-v0',
    'pointmaze-medium-navigate-v0', 'pointmaze-large-navigate-v0', 'pointmaze-giant-navigate-v0', 'pointmaze-teleport-navigate-v0',
    'pointmaze-medium-stitch-v0', 'pointmaze-large-stitch-v0', 'pointmaze-giant-stitch-v0', 'pointmaze-teleport-stitch-v0',
    'antmaze-medium-navigate-v0', 'antmaze-large-navigate-v0', 'antmaze-giant-navigate-v0', 'antmaze-teleport-navigate-v0',
    'antmaze-medium-stitch-v0', 'antmaze-large-stitch-v0', 'antmaze-giant-stitch-v0', 'antmaze-teleport-stitch-v0',
    'antmaze-medium-explore-v0', 'antmaze-large-explore-v0', 'antmaze-giant-explore-v0', 'antmaze-teleport-explore-v0',
    'antsoccer-arena-navigate-v0', 'antsoccer-medium-navigate-v0', 'antsoccer-arena-stitch-v0', 'antsoccer-medium-stitch-v0',
    'puzzle-3x3-play-v0, puzzle-4x4-play-v0, puzzle-4x5-play-v0, puzzle-4x6-play-v0,
    'puzzle-3x3-noisy-v0, puzzle-4x4-noisy-v0, puzzle-4x5-noisy-v0, puzzle-4x6-noisy-v0,
    'cube-single-play-v0, 'cube-double-play-v0, 'cube-triple-play-v0, 'cube-quadruple-play-v0,
    'cube-single-noisy-v0, 'cube-double-noisy-v0, 'cube-triple-noisy-v0, 'cube-quadruple-noisy-v0,
    'scene-play-v0', 'scene-noisy-v0',



    ]
visual_datasets=[
    'visual-antmaze-medium-navigate-v0', 'visual-antmaze-large-navigate-v0', 'visual-antmaze-giant-navigate-v0', 'visual-antmaze-teleport-navigate-v0',
    'visual-antmaze-medium-stitch-v0', 'visual-antmaze-large-stitch-v0', 'visual-antmaze-giant-stitch-v0', 'visual-antmaze-teleport-stitch-v0',
    'visual-antmaze-medium-explore-v0', 'visual-antmaze-large-explore-v0', 'visual-antmaze-giant-explore-v0', 'visual-antmaze-teleport-explore-v0',
    'visual-humanoidmaze-medium-navigate-v0', 'visual-humanoidmaze-large-navigate-v0', 'visual-humanoidmaze-giant-navigate-v0',
    'visual-humanoidmaze-medium-stitch-v0', 'visual-humanoidmaze-large-stitch-v0', 'visual-humanoidmaze-giant-stitch-v0',
    'visual-puzzle-3x3-play-v0', 'visual-puzzle-4x4-play-v0', 'visual-puzzle-4x5-play-v0', 'visual-puzzle-4x6-play-v0',
    'visual-puzzle-3x3-noisy-v0', 'visual-puzzle-4x4-noisy-v0', 'visual-puzzle-4x5-noisy-v0', 'visual-puzzle-4x6-noisy-v0',
    'visual-cube-single-play-v0', 'visual-cube-double-play-v0', 'visual-cube-triple-play-v0', 'visual-cube-quadruple-play-v0',
    'visual-cube-single-noisy-v0', 'visual-cube-double-noisy-v0', 'visual-cube-triple-noisy-v0', 'visual-cube-quadruple-noisy-v0',
    'visual-scene-play-v0', 'visual-scene-noisy-v0',
    'powderworld-easy-play-v0', 'powderworld-medium-play-v0', 'powderworld-hard-play-v0',
]
'''

# visual_datasets = ['visual-antmaze-medium-navigate-singletask-task2-v0', 'visual-antmaze-medium-navigate-singletask-task3-v0',
# 'visual-antmaze-medium-navigate-singletask-task4-v0','visual-antmaze-medium-navigate-singletask-task5-v0']
visual_datasets = ['visual-antmaze-medium-navigate-singletask-task5-v0']
    # 'visual-humanoidmaze-large-navigate-v0',
    # 'visual-humanoidmaze-giant-navigate-v0',
    # 'visual-humanoidmaze-medium-stitch-v0', 'visual-humanoidmaze-large-stitch-v0', 'visual-humanoidmaze-giant-stitch-v0',
    # 'visual-puzzle-4x4-play-v0', 'visual-puzzle-4x5-play-v0', 'visual-puzzle-4x6-play-v0',
    # 'visual-puzzle-3x3-noisy-v0', 'visual-puzzle-4x4-noisy-v0', 'visual-puzzle-4x5-noisy-v0', 'visual-puzzle-4x6-noisy-v0',
    # 'visual-cube-double-play-v0', 'visual-cube-triple-play-v0', 'visual-cube-quadruple-play-v0',
    # 'visual-cube-single-noisy-v0', 'visual-cube-double-noisy-v0', 'visual-cube-triple-noisy-v0', 'visual-cube-quadruple-noisy-v0',
    # 'visual-scene-noisy-v0',
    # 'powderworld-easy-play-v0', 'powderworld-medium-play-v0', 'powderworld-hard-play-v0',


for dataset_name in visual_datasets:

    for i in range(1, 6):
        dataset_dir = f'/ext/skshyn/.ogbench/task{i}'
        if f'task{i}' in dataset_name:
            try:
                env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
                    dataset_name,  # Dataset name.
                    dataset_dir=dataset_dir,  # Directory to save datasets (optional).
                    compact_dataset=False,  # Whether to use a compact dataset (optional; see below).
                    add_info=True
                )
                
                # 데이터셋 로드 후 환경 닫기 (렌더러 정리)
                print(f"{dataset_name} Task {i}: 데이터셋 로드 완료")
                print(f"  - Train dataset size: {len(train_dataset) if train_dataset else 'N/A'}")
                print(f"  - Val dataset size: {len(val_dataset) if val_dataset else 'N/A'}")
                
            finally:
                # 환경을 명시적으로 닫아서 렌더러 리소스 정리
                # EGL 에러는 이미 stderr 필터에 의해 필터링됨
                try:
                    if 'env' in locals():
                        try:
                            # 렌더러를 먼저 정리 시도
                            if hasattr(env, 'renderer') and env.renderer is not None:
                                try:
                                    env.renderer.close()
                                except:
                                    pass
                            env.close()
                        except:
                            pass
                except Exception:
                    # 모든 예외 무시 (이미 종료 중이므로)
                    pass