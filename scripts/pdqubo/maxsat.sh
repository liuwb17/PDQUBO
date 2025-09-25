for id in {0..49}
do
for k in 3 4 5
do
python main.py --task 'maxsat' --solver 'pdqubo' --batch 10 --lr_x 0.01 --lr_y 0.005 --dual_init 2 --max_iters 8000 --cnf_id $id --cnf_k $k
done
done