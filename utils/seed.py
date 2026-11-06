import os
import random

import numpy as np
import torch


def set_determinism(seed: int) -> None:
    # CUBLAS 환경변수는 첫 CUDA 호출 전에 설정해야 한다.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 결정적 커널이 없는 연산은 바로 에러를 내도록 함
    torch.use_deterministic_algorithms(True)
