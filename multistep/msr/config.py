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


def run_tag(dt, proto, init):
    return f"dt{dt:g}_{proto}" + ("" if proto == "onestep" else f"_{init}")


def run_dir(cfg, dt, proto, init):
    return os.path.join(cfg["out_root"], "runs", run_tag(dt, proto, init))
