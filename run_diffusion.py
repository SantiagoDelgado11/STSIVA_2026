import os


batch_size = 170
image_size = 256
seed = 0
num_angles = 180
gpu_id = 0
save_image = "False"
plot = "False"
idx_list = [i for i in range(0, batch_size)]
wandb_id = "b879bf20f3c31bfcf13289e363f4d3394f7d7671"
use_wandb = "True"
device = "cuda"
weights = "weights/d_lodo_e_1000_bs_4_lr_0.0003_seed_2_img_256_schedule_cosine_gpu_1_c_1_si_100/checkpoints/latest.pth.tar"  # Diffusion weights
dncnn_weights = "weights/DNCNN/CT_SPECTRAL_NORM_lr_0.001_b_16_e_150_sd_2_im_256_ml_30/model/dcnn.pth"
use_spectral_norm = False
spectral_norm_power_iters = 1
spectral_norm_eps = 1e-12

sampling_ratio = 1 / 12
sampling_method = "uniform"
algo = "DDNM"


# DDNM parameters
ddnm_eta = 1.0

# DPS parameters
dps_scale = 0.0125


# CEDiff parameters (They may differ for uniform and non-uniform sampling)
CG_iter = 10
CE_iter = 10
mu = 0.5
rho = 0.5

# DiffPIR parameters
CG_iters_diffpir = 100
noise_level_img = 0.0

# PnP_FISTA parameters
fista_iter = 600
fista_step = 0.001
denoiser_strength = 0.00001


project_name = "NullDiff"  # f"{algo}_{sampling_method}_{180 * sampling_ratio // 1}"


for idx in idx_list:
    name = f"{idx}_{algo}_{sampling_method}_{180 * sampling_ratio // 1}"
    sn_args = ""
    if use_spectral_norm:
        sn_args = (
            f"--use_spectral_norm --spectral_norm_power_iters {spectral_norm_power_iters} "
            f"--spectral_norm_eps {spectral_norm_eps}"
        )
    os.system(
        f"python run_one_img.py --idx {idx} --weights {weights} --batch_size {batch_size} --image_size {image_size} --seed {seed} --num_angles {num_angles} --device {device} --gpu_id {gpu_id} --save_image {save_image} --plot {plot} --sampling_ratio {sampling_ratio} --sampling_method {sampling_method} --algo {algo} --dps_scale {dps_scale} --CG_iter {CG_iter} --CE_iter {CE_iter} --mu {mu} --rho {rho} --CG_iters_diffpir {CG_iters_diffpir} --noise_level_img {noise_level_img} --dncnn_weights {dncnn_weights} --fista_iter {fista_iter} --fista_step {fista_step} --denoiser_strength {denoiser_strength} --use_wandb {use_wandb} --wandb_id {wandb_id} --project_name {project_name} --name {name} {sn_args}"
    )
