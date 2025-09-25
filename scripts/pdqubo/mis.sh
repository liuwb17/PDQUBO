for seed in {0..19}
do
python main.py --solver 'pdqubo' --task 'mis' --graph 'reg' --n 10000 --d 3 --seed $seed --timelimit 180 --batch 10 --lr_x 0.02 --lr_y 0.02 --dual_init 5
python main.py --solver 'pdqubo' --task 'mis' --graph 'reg' --n 10000 --d 100 --seed $seed --timelimit 180 --batch 10 --lr_x 0.02 --lr_y 0.02 --dual_init 5
python main.py --solver 'pdqubo' --task 'mis' --graph 'reg' --n 50000 --d 3 --seed $seed --timelimit 180 --batch 10 --lr_x 0.02 --lr_y 0.02 --dual_init 5
python main.py --solver 'pdqubo' --task 'mis' --graph 'reg' --n 50000 --d 100 --seed $seed --timelimit 180 --batch 10 --lr_x 0.02 --lr_y 0.02 --dual_init 5
done