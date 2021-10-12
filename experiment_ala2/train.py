#!/usr/bin/env python
# coding: utf-8
#SBATCH --time=100:00:00

### #SBATCH -p small --exclusive
#SBATCH -p gpu --gres=gpu:1 --mem=8GB

from argparse import ArgumentParser
import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from ala2flow import Ala2Data, Ala2Generator


def main():

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)#, gpus=0, min_epochs=5, max_epochs=20)
    parser = Ala2Generator.add_model_specific_args(parser)
    parser = Ala2Data.add_model_specific_args(parser)
    parser.add_argument("--n-samples", type=int, default=10000)
    args = parser.parse_args()
    
    # Data
    data = Ala2Data(**vars(args))
    data.prepare_data()
    data.setup()

    # Model
    gen = Ala2Generator(**vars(args))

    # Trainer
    logger = TensorBoardLogger("logs", name=f"a2_{data.slice}")
    #checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)#, callbacks=[checkpoint_callback])

    # Training
    trainer.fit(gen, data)


    # Sampling
    print("Sample best model...")
    loaded = Ala2Generator.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    with torch.no_grad():
        samples = []
        target_energies = []
        model_energies = []
        while len(samples) < args.n_samples:
            xyz = loaded.model.sample(min(256, args.n_samples-len(samples)))
            samples.extend(list(xyz.cpu().numpy()))
            tar = loaded.model._target.energy(xyz)
            target_energies.extend(list(tar.cpu().numpy()))
            mod = loaded.model.energy(xyz)
            model_energies.extend(list(mod.cpu().numpy()))
    np.savez(
        trainer.checkpoint_callback.best_model_path+".npz",
        samples=samples,
        u_target=target_energies,
        u_model=model_energies
    )
    print("Best model:", trainer.checkpoint_callback.best_model_path)
    import sys, traceback, threading
    thread_names = {t.ident: t.name for t in threading.enumerate()}
    for thread_id, frame in sys._current_frames().iteritems():
            print("Thread %s:" % thread_names.get(thread_id, thread_id))
            traceback.print_stack(frame)
            print()



if __name__ == "__main__":
    main()

