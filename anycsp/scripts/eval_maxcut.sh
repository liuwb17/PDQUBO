#!/bin/bash
# python evaluate_maxcut.py --data_path './data/test/GSET/G43.mtx' --model_dir models/MAXCUT --num_boost 20 --seed 0 --timeout 180 --verbose

python evaluate_maxcut.py --data_path './data/test/gset_txt/G22/G22.txt' --model_dir models/MAXCUT --num_boost 1 --seed 0 --timeout 10 --verbose


# python evaluate_maxcut.py --data_path './data/test/COLOR/anna.col' --model_dir models/MAXCUT --num_boost 20 --seed 0 --timeout 180 --verbose
# python evaluate_maxcut.py --data_path './data/test/COLOR/david.col' --model_dir models/MAXCUT --num_boost 20 --seed 0 --timeout 180 --verbose
# python evaluate_maxcut.py --data_path './data/test/COLOR/huck.col' --model_dir models/MAXCUT --num_boost 20 --seed 0 --timeout 180 --verbose