for id in 67 70 72 77 81
do
python ./main.py --solver 'pdqubo' --task 'mc' --graph 'Gset' --Gset_id $id --batch 100 --lr_x 0.025 --lr_y 0.025 --dual_init 6
done