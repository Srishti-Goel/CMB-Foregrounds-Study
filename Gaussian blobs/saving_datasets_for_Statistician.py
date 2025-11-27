import numpy as np
import os
import torch 
import torch.nn as nn

def create_dataset(model, BATCHES_TO_SAVE, BATCH_SIZE, SAVING_DIR = 'temp_model'):
    # The numbers are read (LATENT_DIM)_(N_CONV_LATER_GEN)_(N_CONV_LATER_DISC)_(KERNEL_SIZE)_(PADDING)_(OUTPUT_PADDING)_(TRAINING_EPOCHS)
    # model.save_states(pathlib.Path("."), "Single_Gaussian_Blobs_" + str(LATENT_DIM) + '_5_5_4_1_0_' + str(N_EPOCHS))
    SAVING_DIR = 'datasets_for_Statistician/' + SAVING_DIR
    os.makedirs(SAVING_DIR, exist_ok=True)
    for n in range(BATCHES_TO_SAVE):
        try:
        
            z = model._get_latent_vector(BATCH_SIZE)
            # Generate a batch of images
            gen_imgs = model.generator(z).detach().cpu()
                
        except:
            z = torch.normal(mean = 0., std=1.0, size=(BATCH_SIZE, model.latent_dim)).to(model.device)
            # Generate a batch of images
            gen_imgs = model.decode(z, apply_sigmoid = True).detach().cpu()
                
        for i in range(BATCH_SIZE):
            frame = gen_imgs.numpy()[i, 0, :, :]

            np.save(SAVING_DIR + '/' + str((n * BATCH_SIZE) + i), arr = frame, allow_pickle=True)
            print(f'\r{n*BATCH_SIZE + i + 1} / {BATCHES_TO_SAVE * BATCH_SIZE}', end='')

    print('\n') 
