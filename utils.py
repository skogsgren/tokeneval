from pathlib import Path


def create_mappings(cfg: dict):
    mappings = {}
    for lang in cfg["languages"]:
        datadir = Path(cfg["paths"]["goldfish_datadir"])
        traindir = datadir / "train"
        validdir = datadir / "valid"
        lang_filename = (lang + ".txt").lower()
        mappings[lang] = {
            "source": f"goldfish-models/{lang}_{cfg['models']['source_size']}".lower(),
            "target": f"goldfish-models/{lang}_{cfg['models']['target_size']}".lower(),
            "finetune_data": {
                "train": str(traindir / lang_filename),
                "valid": str(validdir / lang_filename),
            },
        }
    return mappings
