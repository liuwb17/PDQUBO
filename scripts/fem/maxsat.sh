for k in 3 4 5
do
for id in {0..49}
do
python main_maxsat.py --solver 'fem' --cnf_k $k --cnf_id $id
done
done