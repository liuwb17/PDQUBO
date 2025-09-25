import sys
import time

sys.path.append('.')
from interface import FEM
import torch

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.set_printoptions(
    precision=6,    # 显示4位小数（默认是4）
    sci_mode=False  # 禁用科学计数法（可选）
)
num_trials = 1000
num_steps = 1000
dev = 'cuda'

# case_maxcut = FEM.from_file(
#         'maxcut', f'../instance/Random_regular/n10000d10s0.txt', index_start=1, discretization=True
#     )
case_maxcut = FEM.from_file(
        'maxcut', f'../instance/Gset/G70.txt', index_start=1, discretization=True
    )

case_maxcut.set_up_solver(
    num_trials, num_steps, manual_grad=True, betamin=0.001, betamax=0.5,
    learning_rate=0.01, optimizer='rmsprop', dev=dev
)
t = time.perf_counter()
config, result = case_maxcut.solve()

print(result.max())
