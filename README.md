A Primal Dual Approach for Binary Optimization


## Primal optimizer

For `--solver pdqubo`, select `--optimizer gd`, `--optimizer rmsprop`, or
`--optimizer adam`. The default remains `rmsprop`; existing scripts retain their
original behavior. `gd` uses standard gradient descent without momentum.
`--lr_x` sets the primal learning rate. RMSProp and Adam retain their original
settings, and the dual update is unchanged.

```bash
python main.py --solver pdqubo --task mc --graph Gset --Gset_id 1 --optimizer gd
python main.py --solver pdqubo --task mc --graph Gset --Gset_id 1 --optimizer rmsprop
python main.py --solver pdqubo --task mc --graph Gset --Gset_id 1 --optimizer adam
```

The Python solver constructors also accept `optimizer_type='gd'`, `'rmsprop'`,
or `'adam'`.
