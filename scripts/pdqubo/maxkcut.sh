for id in {1..66} 67 70 72 77 81
do
python ./main.py --solver 'pdqubo' --task 'maxkcut' --k 3 --graph 'Gset' --Gset_id $id --max_iters 5000 --lr_x 0.01 --lr_y 0.01 --dual_init 6 --batch 100
done