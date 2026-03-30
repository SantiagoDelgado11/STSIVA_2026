import sys
import os

# Add parent directory to path early to allow importing from local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable

import wandb
from wandb.integration.sb3 import WandbCallback

class MetricsLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricsLoggerCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Extraer variables dinámicas del bucle de SB3
        infos = self.locals.get("infos", [{}])
        
        info = infos[0]
            
        # Retroalimentación en la consola
        if self.n_calls % 100 == 0:
            psnr = info.get("psnr", 0.0)
            ssim = info.get("ssim", 0.0)
            
            print(f"🔄 [Paso {self.n_calls}] PSNR Actual: {psnr:.2f} dB | SSIM Actual: {ssim:.4f}")
            
        return True

# Local imports
from algos.ddnm import DDNM
from algos.diffpir import DiffPIR
from algos.dps import DPS
from guided_diffusion.script_util import create_model
from utils.SPC_model import SPCModel
from rl_agent.envs.diffusion_mdp import DiffusionMDPEnv
from rl_agent.models.ppo_networks import DiffusionFeatureExtractor


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Decaimiento lineal de la tasa de aprendizaje.
    progress_remaining va de 1.0 (inicio) a 0.0 (fin).
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def main(opt):
    device = f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu"
    
    print("Loading Pretrained U-Net Model...")
    
    # Asegurar que la ruta a los pesos sea absoluta (relativa a la raíz del proyecto)
    weights_path = opt.weights
    if not os.path.isabs(weights_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_path = os.path.join(project_root, weights_path)
    
    # Check if the weights file actually exists, to safely initialize
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device, weights_only=False) # weights_only=False para evitar errores de dicts
        net = create_model(image_size=opt.image_size, num_channels=64, num_res_blocks=3, input_channels=3).to(device)
        net.load_state_dict(ckpt["model_state"])
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Weights file not found at {weights_path}. Initializing un-trained U-Net for structural test.")
        net = create_model(image_size=opt.image_size, num_channels=64, num_res_blocks=3, input_channels=3).to(device)
        
    net.eval()

    print("Initializing SPC Model and Algorithms...")
    inverse_model = SPCModel(im_size=opt.image_size, compression_ratio=opt.sampling_ratio).to(device)

    ddnm_params = DDNM(
        device=device,
        img_size=opt.image_size,
        noise_steps=opt.noise_steps,
        schedule_name="cosine",
        channels=3,
        eta=opt.ddnm_eta,
    )
    
    dps_params = DPS(
        device=device,
        img_size=opt.image_size,
        noise_steps=opt.noise_steps,
        schedule_name="cosine",
        channels=3,
        scale=opt.dps_scale,
        clip_denoised=False,
    )
    
    diffpir_params = DiffPIR(
        device=device,
        img_size=opt.image_size,
        noise_steps=opt.noise_steps,
        schedule_name="cosine",
        channels=3,
        cg_iters=opt.cg_iters_diffpir,
        noise_level_img=opt.noise_level_img,
        iter_num=opt.noise_steps,  # Force sequence to match steps exactly
        eta=0,
        zeta=1,
    )

    print("Setting up Environment...")
    env = DiffusionMDPEnv(
        unet_model=net,
        spc_model=inverse_model,
        ddnm_params=ddnm_params,
        diffpir_params=diffpir_params,
        dps_params=dps_params,
        device=device,
        noise_steps=opt.noise_steps,
        img_size=opt.image_size,
    )
    
    # Wrap in DummyVecEnv for SB3
    vec_env = DummyVecEnv([lambda: env])
    
    # Normalización de Recompensas para estabilizar el Crítico
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    print("Initializing PPO Agent...")
    policy_kwargs = dict(
        features_extractor_class=DiffusionFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        ortho_init=True, # Explícito: Inicialización ortogonal para mejor estabilidad temprana
    )
    
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(opt.lr), # Decaimiento lineal de LR
        n_steps=opt.n_steps,
        batch_size=opt.batch_size,
        ent_coef=0.05, # Entropía robusta a solicitud del usuario para exploración extendida
        normalize_advantage=True, # Explícito: Normalización de ventajas por batch
        tensorboard_log=f"{opt.save_dir}/tensorboard",
        verbose=1,
        device=device
    )

    print("Initializing WandB with Hardcoded API Key...")
    wandb.login(key="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")
    run = wandb.init(
        project="STSIVA AGENTS",
        name="ppo_meta_controller",
        config=vars(opt),
        sync_tensorboard=True,
        save_code=True,
    )

    print("Starting Training Loop...")
    model.learn(
        total_timesteps=opt.total_timesteps, 
        progress_bar=True, 
        callback=[MetricsLoggerCallback(), WandbCallback(model_save_path=opt.save_dir, verbose=2)]
    )
    
    run.finish()
    
    print("Saving Model and VecNormalize statistics...")
    os.makedirs(opt.save_dir, exist_ok=True)
    model.save(f"{opt.save_dir}/ppo_meta_controller")
    vec_env.save(f"{opt.save_dir}/vec_normalize.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/e_1000_bs_64_lr_0.0003_seed_2_img_32_schedule_cosine_gpu_0_c_3_si_100/checkpoints/latest.pth.tar")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--noise_steps", type=int, default=1000)
    
    # Env params
    parser.add_argument("--sampling_ratio", type=float, default=0.5)
    parser.add_argument("--ddnm_eta", type=float, default=1.0)
    parser.add_argument("--dps_scale", type=float, default=0.0125)
    parser.add_argument("--cg_iters_diffpir", type=int, default=5)
    parser.add_argument("--noise_level_img", type=float, default=0.0)
    
    # RL params
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_steps", type=int, default=1024, help="Steps before updating PPO")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--total_timesteps", type=int, default=1024000)
    parser.add_argument("--save_dir", type=str, default="rl_agent/checkpoints")
    
    opt = parser.parse_args()
    main(opt)
