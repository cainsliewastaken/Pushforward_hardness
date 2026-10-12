import os

import yaml


def load_cfg(path, overrides=()):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for kv in overrides:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    cfg["out_root"] = os.path.expandvars(cfg["out_root"])
    return cfg
