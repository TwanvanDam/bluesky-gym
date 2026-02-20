from stable_baselines3.common.callbacks import BaseCallback

from bluesky_gym.envs.base_navigation_env import BaseNavigationEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

DEVICE = "cuda"
MODEL_PATH = "./scripts/common/results/models_backup/BaseNavigationEnv-v0/New_faster_"

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = self.locals['infos'][0]['step_reward']
        self.logger.record('step_reward', value)
        return True

if __name__ == "__main__":
    train = False

    if train:
        env = Monitor(BaseNavigationEnv())
        model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/", device=DEVICE)
        model.learn(total_timesteps=50_000, callback=TensorboardCallback())
        model.save(MODEL_PATH)

    env = BaseNavigationEnv(render_mode="human")
    model = SAC.load(MODEL_PATH, env=env, device="cuda")

    while True:
        obs, info = env.reset()
        done = False
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
