for id in {1..67} 70 72 77 81
do
python ./main.py --solver 'ros' --task 'maxkcut' --graph 'Gset' --Gset_id $id --k 3
done