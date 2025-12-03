from omegaconf import DictConfig, OmegaConf
from utils.utils import load_cfg
import hydra
from pathlib import Path
from utils.path_utils import convert_dict_to_path, save_config
@hydra.main(version_base=None, config_path="./configs", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # Create directories specified in the config if they do not exist
    cfg.paths = convert_dict_to_path(cfg.paths)
    # Set the current working directory to the cwd_dir specified in the config
    cfg.paths.cwd_dir = Path.cwd()
    cfg_temp = cfg.copy()
    cfg_temp.paths = convert_dict_to_string(cfg_temp.paths)
    OmegaConf.save(cfg_temp, cfg.paths.log_dir / "run_config.yaml")
    print(OmegaConf.to_yaml(cfg_temp, resolve=True))    
if __name__ == "__main__":
    my_app()