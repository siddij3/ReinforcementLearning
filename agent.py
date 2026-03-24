import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import hf_token
hf_token.ensure_hf_environment()

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from fake_profiles.synthetic_data import SignalProcessor
from environment import CandidateEnv
from tb_callback import FraudTensorboardCallback

import torch.nn as nn


def make_env():
    # debug=False avoids per-reset log spam; set True when inspecting profiles.
    return Monitor(CandidateEnv(signal_processor=SignalProcessor(), debug=False))


if __name__ == "__main__":

    # ── Vectorised environments ────────────────────────────────────────────
    env      = make_vec_env(make_env, n_envs=8)
    eval_env = make_vec_env(make_env, n_envs=4)

    # ── Policy network architecture ────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 64], vf=[128, 64])],
        activation_fn=nn.Tanh,
    )

    # ── PPO model ──────────────────────────────────────────────────────────
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=512,        # steps per env per rollout; total = 512 * 8 = 4096
        batch_size=256,     # PPO mini-batch size
        n_epochs=10,        # gradient passes per rollout buffer
        gamma=0.99,         # discount — episodes are short, so this is fine
        gae_lambda=0.95,    # GAE smoothing
        clip_range=0.2,     # PPO ε clip
        ent_coef=0.01,      # entropy bonus — helps explore PROBE action
        vf_coef=0.5,        # value-loss weight
        verbose=1,
        tensorboard_log="./runs/fraud_detection/",
    )

    # ── Callbacks ──────────────────────────────────────────────────────────
    tb_callback = FraudTensorboardCallback(
        window=100,   # rolling window for precision/recall/action rates
        verbose=1,    # prints a summary line each rollout — set 0 to silence
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/",
        eval_freq=5_000,     # evaluate every 5 000 timesteps
        deterministic=True,  # greedy policy during eval
    )

    # ── Train ──────────────────────────────────────────────────────────────
    # CallbackList fans out each hook (on_step, on_rollout_end, …)
    # to both callbacks in order.
    model.learn(
        total_timesteps=500_000,
        callback=CallbackList([tb_callback, eval_callback]),
    )

    print("\nTraining complete.")
    print("Launch TensorBoard with:")
    print("  tensorboard --logdir ./runs/fraud_detection")