


import os
import numpy as np

n_batch = 128
n_hidden = 64
n_components = 8
n_runs = 100
transformer_type="spline"
inverse_flow = True

for rep in range(10):
    for backward in [False, True]:
        for n_grid_inversion in 2**np.arange(1,9):
            for dim_x in 2**np.arange(1,13,2):
                for inverse_algo in ["approx", "exact"]:
                    out = f"{rep}{'i' if inverse_flow else 'f'}{'b' if backward else 'f'}-g{n_grid_inversion}-d{dim_x}-{transformer_type[0]}-{inverse_algo[0]}"
                    command = f"""
    timings.py --n_batch {n_batch} --n_hidden {n_hidden} \\
     --dim_x {dim_x} --dim_y {dim_x} --n_components {n_components} \\
     --n_runs {n_runs} --inverse_flow={1 if inverse_flow else 0} --inverse_algo {inverse_algo} \\
     --n_grid_inversion {n_grid_inversion} --backward={1 if backward else 0}"""
                    print("python \\" + command)
                    #os.system("sbatch -o {out} -J {out}" +  command)
