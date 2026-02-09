def read_yaml(fname: str):
    import munch
    import yaml 

    with open(fname) as fobj:
        config = yaml.safe_load(fobj)
        
    config = munch.munchify(config)

    return config