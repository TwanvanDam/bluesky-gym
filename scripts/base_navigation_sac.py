from stable_baselines3.common.callbacks import BaseCallback
import torch
from bluesky_gym.envs.base_navigation_env import BaseNavigationEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

DEVICE = "cpu" 
MODEL_PATH = "./scripts/common/results/models_backup/BaseNavigationEnv-v0/New_model_longer_trained"

class TensorboardCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            # Only log your custom termination statistics
            if 'termination_stats' in info:
                for stat_name, count in info['termination_stats'].items():
                    self.logger.record(f"termination/{stat_name}", count)

        return True

if __name__ == "__main__":
    train = False

    if train:
        env = Monitor(BaseNavigationEnv())
        model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log="./scripts/common/results/logs_backup/BaseNavigationEnv-v0", device=DEVICE)
        model.learn(total_timesteps=200_000, callback=TensorboardCallback())
        model.save(MODEL_PATH)
    else:
        env = BaseNavigationEnv(render_mode="human")
        model = SAC.load(MODEL_PATH, env=env, device=DEVICE)

        while True:
            obs, info = env.reset()
            done = False
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
