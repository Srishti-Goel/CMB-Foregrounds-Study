import os
import pathlib
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import use
use('TkAgg')
import torch
import gc
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


gc.collect()
torch.cuda.empty_cache()

# ---------- IMPORTANT CONSTANTS ----------
# Model Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 2

# Training parameters
N_EPOCHS = 200
N_CRITICS = 5
SAMPLE_INTERVALS = 1000
SAVE_INTERVAL = 50
DUMPING_INTERVAL = 1000
LEARNING_RATE = 2e-6
BETAS = (0.5, 0.9)
L_GP = 10.0     # Weight of the Gradient Policy cost
L_l1 = 0.0      # Weight of the L1 norm cost

# Data handling
TRAIN_DIR = '../../../data_map_cutouts/train3' #'../data__map_cutouts/train_v3'
SAVING_DIR = '../datasets_for_Statistician/single_Gaussian_Blobs_' + str(LATENT_DIM) + '_5_5_4_1_0_' + str(N_EPOCHS) + '/generated_data/' 
# os.makedirs(SAVING_DIR, exist_ok=True)
BATCH_SIZE = 16
BATCHES_TO_SAVE = 5

# ---------- DATA LINKING ----------
to_tensor = transforms.ToTensor()
random_flips_v = transforms.RandomVerticalFlip(0.5)
random_flips_h = transforms.RandomHorizontalFlip(0.5)
to_grayscale = transforms.Grayscale(1)

transform = transforms.Compose([
    to_grayscale,
    to_tensor,
    random_flips_h,
    random_flips_v
])

train_dir = pathlib.Path(TRAIN_DIR)  
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
dataLoader = DataLoader(dataset=train_data,
                     batch_size=BATCH_SIZE, 
                     shuffle=True,
                     drop_last=True)
# print("In dataset:", len(list(train_data)))   

# ---------- SOME IMPORTANT LAYERS ----------
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
        # print(shape)

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

class Generator_Basic(nn.Module):
    def __init__(self, ngpu = 1, latent_dim = 100) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(latent_dim, 3),
            nn.LeakyReLU(),
            nn.Linear(3, (4 *4**2)),
            nn.BatchNorm1d(num_features=(4* 4**2), eps = 0.01),
            # nn.ReLU(inplace=True),
            Reshape((4, 4, 4))
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels  = 1,
                out_channels = 1,
                kernel_size  = 3,
                stride       = 1,
                padding      = 1
            ),
            nn.LeakyReLU(0.01)
        )

        self.convT1 = nn.Sequential(
            
            nn.ConvTranspose2d(
                in_channels  = 4,
                out_channels = 4,
                kernel_size  = 4,
                stride       = 2,
                padding      = 1
            ),
            nn.BatchNorm2d(num_features=4, eps = 0.01),
            nn.Tanh()
        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels  = 4,
                out_channels = 4,
                kernel_size  = 4,
                stride       = 2,
                padding      = 1
            ),
            nn.BatchNorm2d(num_features=4, eps = 0.01),
            nn.Tanh(),
        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels  = 4,
                out_channels = 4,
                kernel_size  = 4,
                stride       = 2,
                padding      = 1
            ),
            nn.BatchNorm2d(num_features=4, eps = 0.01),
            nn.Tanh(),
        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels  = 4,
                out_channels = 2,
                kernel_size  = 4,
                stride       = 2,
                padding      = 1
            ),
            nn.BatchNorm2d(num_features=2, eps = 0.01),
            nn.Tanh(),
        )
        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels  = 2,
                out_channels = 1,
                kernel_size  = 4,
                stride       = 2,
                padding      = 1
            ),
            nn.Tanh()
        )

        self.model = nn.Sequential(
            self.layer1,
            self.convT1,
            self.convT2,
            self.convT3,
            self.convT4,
            self.convT5,
            # self.conv
        )
        self.ngpu = ngpu

    def forward(self, z):
        if z.is_cuda and self.ngpu > 0:
            ret = nn.parallel.data_parallel(self.model, z, range(self.ngpu))
        else:
            ret = self.model(z)
        return ret
    
gen = Generator_Basic(latent_dim=2)
print("gen created")

z = np.random.uniform(0, 1, (10, 2))
z = torch.FloatTensor(z)
gen_img = gen(z)
print("10 images output size:", gen_img.shape)

class Discriminator_Basic(nn.Module):
    def __init__(self, ngpu = 1, latent_dim = 100) -> None:
        super().__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
                              nn.Conv2d(
                                in_channels=1,
                                out_channels=2,
                                kernel_size=4,
                                stride=2,
                                padding=1
                              ),
                              nn.LeakyReLU(0.2),
                              nn.Conv2d(
                                in_channels=2,
                                out_channels=4,
                                kernel_size=4,
                                stride=2,
                                padding=1
                              ),
                              nn.LeakyReLU(0.2),
                              nn.Flatten(),
                              nn.Linear(4 * 32**2, 1)
                            )
    def forward(self, img):
        # print(img.shape)
        if img.is_cuda and self.ngpu > 0:
            ret = nn.parallel.data_parallel(self.model, img, range(self.ngpu))
        else:
            ret = self.model(img)
        return ret
    
class GAN_Basic(object):
    def __init__(self, identifier,
                 latent_dim,
                 cuda = False, ngpu = 1):
        self.cuda = cuda
        self.ngpu = 0 if not self.cuda else ngpu
        if torch.cuda.is_available() and not self.cuda:
            print("[WARNING] Probably better to run with your CUDA device? It'll be faster, I promise")
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        self.latent_dim = latent_dim
        self.latent_vector_sampler = self._get_default_latent_vector_sampler()

        self.identifier = identifier
        _root = os.path.join(os.path.dirname(os.path.abspath('./code')), '..')
        self.output_path = os.path.join(os.path.join(_root, "output"))
        self.experiment_path = os.path.join(self.output_path, identifier)

        self.generator = Generator_Basic(ngpu=ngpu, latent_dim=latent_dim)
        self.discriminator = Discriminator_Basic(ngpu=ngpu, latent_dim=latent_dim)

        self.generator.apply(self._weights_init_normal).to(device=self.device)
        self.discriminator.apply(self._weights_init_normal).to(device=self.device)

        self.l1_loss = torch.nn.L1Loss().to(device=self.device)

        # Defining a model parameters dictionary for convenience later on
        self.model_params = {
            "latent_dim": latent_dim, 
            "sampler": "normal"
        }

        # open('../output/training_single_simple_'+ str(self.latent_dim)+'/temp_g.pickle', 'x')
        # open('../output/training_single_simple_'+ str(self.latent_dim)+'/temp_d.pickle', 'x')

        # print(self.generator)
        # print(self.discriminator)

    def _eval_generator_loss(self, real_imgs, gen_imgs, l_l1):
        loss = -torch.mean(self.discriminator(gen_imgs))
        if l_l1 != 0.:
            real_ps = torch.var(real_imgs, dim=[-1, -2])
            gen_ps = torch.var(gen_imgs, dim=[-1, -2])
            loss = loss + l_l1 * self.l1_loss(real_ps, gen_ps)

        return loss
  
    def _eval_discriminator_loss(self, real_imgs, gen_imgs, l_gp):
        # determine the interpolation point 

        eps = self.Tensor(np.random.random((real_imgs.data.size(0), 1, 1, 1)))
        interp_data = (eps * real_imgs.data + ((1 - eps) * gen_imgs.data)).requires_grad_(True)
        disc_interp = self.discriminator(interp_data)
        storage = Variable(self.Tensor(real_imgs.data.shape[0], 1).fill_(1.0), requires_grad=False)
        # compute gradient w.r.t. interpolates

        gradients = autograd.grad(
            outputs=disc_interp,
            inputs=interp_data,
            grad_outputs=storage,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # real_vec = Reshape(128*128)(real_imgs)
        # geni_vec = Reshape(128*128)(gen_imgs)

        # print(real_vec.shape, geni_vec.shape)

        gradients = gradients.view(gradients.size(0), -1)
        GP = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        ret = -torch.mean(self.discriminator(real_imgs)) + torch.mean(self.discriminator(gen_imgs)) + l_gp* GP
        return ret

    def _get_default_latent_vector_sampler(self):
        np.random.randn
        return lambda x, y: np.random.rand(x, y)
    
    def _update_latent_vector_sampler(self, new_sampler):
        self.latent_vector_sampler = new_sampler

    def _get_latent_vector(self, nbatch, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return Variable(self.Tensor(self.latent_vector_sampler(nbatch, self.latent_dim)))

    def _get_optimizers(self, lr, betas):
        # lr, betas = kwargs['lr'], kwargs["betas"]
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        return opt_gen, opt_disc

    def _weights_init_normal(self, layer):
        classname = layer.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            nn.init.constant_(layer.bias.data, 0.0)


    def plot_and_save(self, g_losses, d_losses, place_to_save):
        fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,8), sharex=True)
        # fig.tight_layout(pad=2.0)

        temp = []
        # Reading the file
        if os.path.exists('../output/training_single_simple_'+str(LATENT_DIM)+'/temp_d.pickle'):
            with open('../output/training_single_simple_'+str(LATENT_DIM)+'/temp_d.pickle', 'rb') as handle:
                temp = pickle.load(handle)
            # Appending the read data
            d_losses = np.append(temp, d_losses)
                            

        d_l = []
        for i in range(0, d_losses.size, 5):
            d_l.append(d_losses[i])
        d_l = np.array(d_l)

        ax1.plot(d_l)

        temp = []
        # Reading the file
        if os.path.exists('../output/training_single_simple_'+str(LATENT_DIM)+'/temp_g.pickle'):
            with open('../output/training_single_simple_'+str(LATENT_DIM)+'/temp_g.pickle', 'rb') as handle:
                temp = pickle.load(handle)
            # Appending the read data
            g_losses = np.append(temp, g_losses)

        ax2.plot(g_losses)
        
        len_combined = min(g_losses.size, d_losses.size)
        comb_losses = g_losses[-len_combined:] + d_l[-len_combined:]

        ax3.plot(comb_losses)

        ax1.set_ylabel("Discriminator\nloss", fontsize=14)
        ax2.set_ylabel("Generator\nloss", fontsize=14)
        ax3.set_ylabel("Combined\nloss", fontsize=14)

        # ax1.set_xlabel("Iteration of losses")
        # ax2.set_xlabel("Iteration of losses")
        ax3.set_xlabel("Generator Update Iteration", fontsize=14)

        fig.suptitle('Variation of losses with iterations', fontsize=24)
        fig.savefig(place_to_save)   
        if os.path.exists('../output/training_single_simple_'+str(LATENT_DIM)+'/temp_d.pickle'):
            os.remove('../output/training_single_simple_'+str(LATENT_DIM)+'/temp_d.pickle')
        if os.path.exists('../output/training_single_simple_'+str(LATENT_DIM)+'/temp_g.pickle'):
            os.remove('../output/training_single_simple_'+str(LATENT_DIM)+'/temp_g.pickle') 

    def generate_samples(self, nbatch, seed=None):
        self.generator.eval()
        self.discriminator.eval()
        z = self._get_latent_vector(nbatch, seed)
        return self.generator(z).detach()
    
    def load_states(self, output_path, postfix=""):

        generator_state_file = os.path.join(output_path, "generator_{}.pt".format(postfix))
        discriminator_state_file = os.path.join(output_path, "discriminator_{}.pt".format(postfix))

        try:
            print("loading saved states", postfix)
            self.generator.load_state_dict(torch.load(generator_state_file, map_location=self.device))
            self.discriminator.load_state_dict(torch.load(discriminator_state_file, map_location=self.device))
            print("Loaded saved states")
        except Exception:
            print("FAILED to load saved states")

    def save_states(self, output_path, postfix=""):
        postfix = "" if postfix == "" else "_{}".format(str(postfix))
        print("saving states", postfix)
        generator_state_file = os.path.join(output_path, "generator{}.pt".format(postfix))
        discriminator_state_file = os.path.join(output_path, "discriminator{}.pt".format(postfix))
        saving_point_tracker_file = os.path.join(output_path, "saving_point.txt")
        with open(saving_point_tracker_file, "w") as handle:
            handle.write(postfix)
        torch.save(self.generator.state_dict(), generator_state_file)
        torch.save(self.discriminator.state_dict(), discriminator_state_file)
        print("Saved")

    def train(self, dataloader, 
              nepochs=200, ncritics=5, sample_interval=1000,
              save_interval=5, 
              load_states=True, save_states=True, 
              verbose=True,
              lr=0.0002, betas=(0.5, 0.999), 
              l_gp=10, l_l1=0.0, 
              DUMPING_INTERVAL = 1000,
              place_to_save = 'losses.png',
              **kwargs):
        
        kwargs.update({"nepochs": nepochs, "ncritics": ncritics})
        kwargs.update(self.model_params)

        # Base Setup
        run_id = "learning" 
        run_path = os.path.join(self.experiment_path, run_id)
        artifacts_path = os.path.join(run_path, "artifacts")
        model_path = os.path.join(run_path, "model")

        if not os.path.exists(artifacts_path):
            os.makedirs(artifacts_path)
            
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self.generator.train()
        self.discriminator.train()

        if load_states:
            self.load_states(model_path)

        self.save_states(model_path, 0)
        # Get Optimizers
        opt_gen, opt_disc = self._get_optimizers(lr, betas)
        batches_done = 0

        g_losses = np.array([])
        d_losses = np.array([])

        for epoch in range(nepochs):

            for i, sample in enumerate(dataloader):

                if g_losses.size > DUMPING_INTERVAL:
                    # If the generator losses become too long, append to the losses in the saved file, and save again
                    temp_g_losses = []
                    # Reading the file
                    if os.path.exists('../output/training_single_simple_'+str(LATENT_DIM)+'/temp_g.pickle'):
                        with open('../output/training_single_simple_'+str(LATENT_DIM)+'/temp_g.pickle', 'rb') as handle:
                            temp_g_losses = pickle.load(handle)
                    # Appending the data
                    temp_g_losses = np.append(temp_g_losses, g_losses, axis=0)
                    # Resaving
                    with open('../output/training_single_simple_'+str(LATENT_DIM)+'/temp_g.pickle', 'wb') as handle:
                        pickle.dump(temp_g_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    # Clearing the data from memory
                    g_losses = np.array([])

                if d_losses.size > DUMPING_INTERVAL:
                    # If the discriminator losses become too long, append to the losses in the saved file, and save again
                    temp_d_losses = []
                    # Reading the file
                    if os.path.exists('../output/training_single_simple_'+str(LATENT_DIM)+'/temp_d.pickle'):
                        with open('../output/training_single_simple_'+str(LATENT_DIM)+'/temp_d.pickle', 'rb') as handle:
                            temp_d_losses = pickle.load(handle)
                    # Appending the data
                    temp_d_losses = np.append(temp_d_losses, d_losses, axis=0)
                    # Resaving 
                    with open('../output/training_single_simple_'+str(LATENT_DIM)+'/temp_d.pickle', 'wb') as handle:
                        pickle.dump(temp_d_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    # Clearing the data from memory
                    d_losses = np.array([])

                imgs = sample[0]
                real_imgs = imgs.to(device)

                # Sample noise as generator input
                z = self._get_latent_vector(imgs.shape[0])

                # Generate a batch of images
                gen_imgs = self.generator(z).detach()

                # Adversarial loss 
                opt_disc.zero_grad()
                loss_D = self._eval_discriminator_loss(real_imgs, gen_imgs, l_l1)
                loss_D.backward()
                opt_disc.step()
                d_losses = np.append(d_losses, loss_D.item())

                if i % ncritics == 0:
                    opt_gen.zero_grad()

                    # Generate a batch of images
                    gen_imgs = self.generator(z)
                    # Adversarial loss
                    loss_G = self._eval_generator_loss(real_imgs, gen_imgs, l_gp)
                    loss_G.backward()
                    opt_gen.step()
                    if verbose:
                        print(f'\r[Epoch {epoch+1}/{nepochs}] [Batch {(batches_done+1) % len(dataloader)}/{len(dataloader)}] [D loss: {loss_D.item():.3f}] [G loss: {loss_G.item():.3f}] \t', end='')
                    g_losses = np.append(g_losses, loss_G.item())
                    
                    
                    if g_losses.size > 200 and d_losses.size > 200:
                        if(np.mean(abs(g_losses[-100:] - g_losses[-200:-100])) / np.mean(abs(g_losses[-200:-100])) < 0.01
                            ) and (np.mean(abs(d_losses[-100:] - d_losses[-200:-100])) / np.mean(abs(d_losses[-200:-100])) < 0.01):
                            if save_states:
                                self.save_states(model_path, nepochs)

                            self.plot_and_save(g_losses, d_losses, place_to_save)
                            return

                if batches_done % sample_interval == 0:
                    temp = torch.cat((real_imgs.data[:1], gen_imgs.data[:5]), 0)
                    temp = temp if gen_imgs.shape[-3] < 4 else torch.unsqueeze(torch.sum(temp, 1), 1)
                    save_image(temp, os.path.join(artifacts_path, "%d.png" % batches_done), normalize=True,
                               nrow=int(temp.shape[0] / 2.))
                batches_done += 1

            if int(epoch + 1) % save_interval == 0 and save_states:
                self.save_states(model_path, int(epoch + 1))
        if save_states:
            self.save_states(model_path, nepochs)

        
        print("Not converged")
        self.plot_and_save(g_losses, d_losses, place_to_save)

        

# ---------- THE MODEL ----------
for i in range(1):

    LATENT_DIM = 2 * 2**i
    print(LATENT_DIM)
    model = GAN_Basic(
              identifier       = "training_single_simple_" + str(LATENT_DIM), 
              latent_dim       = LATENT_DIM,
              cuda             = torch.cuda.is_available() 
            )
    # if not os.path.exists("./generator_Multiple_Gaussian_Blobs_"+str(LATENT_DIM)+"_5_5_4_1_0_1000.pt"):
    #     print("Old data not found")
    #     break
    # model.load_states(pathlib.Path("."), "Multiple_Gaussian_Blobs_"+str(LATENT_DIM)+"_5_5_4_1_0_1000")
    
    model.train(
        dataloader      = dataLoader,
        nepochs         = N_EPOCHS,
        ncritics        = N_CRITICS,
        sample_interval = SAMPLE_INTERVALS,
        save_interval   = SAVE_INTERVAL,
        load_states     = True,
        save_states     = True,
        verbose         = True,
        mlflow_run      = False,
        lr              = LEARNING_RATE,
        betas           = BETAS,
        l_gp       = L_GP,
        l_l1       = L_l1,
        
        DUMPING_INTERVAL= DUMPING_INTERVAL,
        place_to_save = '../output/training_single_simple_'+str(LATENT_DIM)+'/losses.png'
    )

    # The numbers are read (LATENT_DIM)_(N_CONV_LATER_GEN)_(N_CONV_LATER_DISC)_(KERNEL_SIZE)_(PADDING)_(OUTPUT_PADDING)_(TRAINING_EPOCHS)
    # model.save_states(pathlib.Path("."), "Single_Gaussian_Blobs_" + str(LATENT_DIM) + '_5_5_4_1_0_' + str(N_EPOCHS))
    SAVING_DIR = '../datasets_for_Statistician/single_blob_simple_' + str(LATENT_DIM) + '_5_5_4_1_0_' + str(N_EPOCHS) + '/generated_data/'
    os.makedirs(SAVING_DIR, exist_ok=True)

    for n in range(BATCHES_TO_SAVE):
        z = model._get_latent_vector(BATCH_SIZE)
        # Generate a batch of images
        gen_imgs = model.generator(z).detach().cpu()
        
        for i in range(BATCH_SIZE):
            frame = gen_imgs.numpy()[i, 0, :, :]
            frame = frame - np.min(frame)
            frame = frame * 255 / np.max(frame)
            frame = frame.astype(np.uint8)

            img = Image.fromarray(frame)# .convert('RGB')
            print(f'\r{n*BATCH_SIZE + i + 1} / {BATCHES_TO_SAVE * BATCH_SIZE}', end='')
            img.save(SAVING_DIR + str((n * BATCH_SIZE) + i) + '.jpg')
    
    print('\n')        