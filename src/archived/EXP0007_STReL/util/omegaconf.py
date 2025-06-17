from omegaconf import OmegaConf


def register_custom_resolver():
    OmegaConf.register_new_resolver("len", lambda x: len(x))
