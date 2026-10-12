import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from wam_adapter.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json
from wam_adapter.utils.config_resolvers import register_default_resolvers
from wam_adapter.utils.logging_config import setup_logging

register_default_resolvers()


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    if cfg.data is None or cfg.data.get("train") is None:
        raise ValueError("`cfg.data.train` is required.")

    output_stats_path_cfg = cfg.get("output_stats_path")
    if output_stats_path_cfg in (None, "", "null"):
        output_stats_path = (Path(str(cfg.output_dir)).expanduser().resolve() / "dataset_stats.json")
    else:
        output_stats_path = Path(str(output_stats_path_cfg)).expanduser().resolve()
    output_stats_path.parent.mkdir(parents=True, exist_ok=True)

    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    processor_cfg = OmegaConf.create(train_cfg["processor"])

    train_cfg["processor"] = None
    train_cfg["pretrained_norm_stats"] = None
    train_cfg["use_latent_cache"] = False
    train_cfg["latent_cache_dir"] = None
    train_cfg["video_only"] = False
    train_cfg["is_training_set"] = True

    dataset = instantiate(train_cfg)
    processor = instantiate(processor_cfg).eval()
    dataset_stats = dataset.get_dataset_stats(processor)
    save_dataset_stats_to_json(dataset_stats, str(output_stats_path))

    print(f"[real-robot-stats] saved={output_stats_path}")
    print(f"[real-robot-stats] num_episodes={dataset_stats.get('num_episodes')}")
    print(f"[real-robot-stats] num_transition={dataset_stats.get('num_transition')}")


if __name__ == "__main__":
    main()
