for seed in {0..19}
do
python main.py --solver 'pignn' --task 'mis' --graph 'reg' --n 10000 --d 3 --seed $seed --timelimit 180
python main.py --solver 'pignn' --task 'mis' --graph 'reg' --n 10000 --d 100 --seed $seed --timelimit 180
python main.py --solver 'pignn' --task 'mis' --graph 'reg' --n 50000 --d 3 --seed $seed --timelimit 180
python main.py --solver 'pignn' --task 'mis' --graph 'reg' --n 50000 --d 100 --seed $seed --timelimit 180
done