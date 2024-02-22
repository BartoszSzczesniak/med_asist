from configparser import ConfigParser

def load_config():

    config = ConfigParser()
    config.read("config.cfg")
    return config._sections

CONFIG = load_config()