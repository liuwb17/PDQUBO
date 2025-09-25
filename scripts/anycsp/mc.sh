for id in {1..67} 70 72 77 81
do
python ./main.py --solver 'anycsp' --task 'mc' --graph 'Gset' --Gset_id $id --timelimit 180 --num_boost 20
done