import yaml

class Config(object):
    def __init__(self):
        conf = {
                   'newsgroup_path':'./resources/3_newsgroup',
                    'load_path':'./data/output'
        }
        self._config = conf # set it to conf

    def load(self,yaml_filename):
        yaml.load(yaml_filename)

    def get_property(self, property_name):
        if property_name not in self._config.keys(): # we don't want KeyError
            return None  # just return None if not found
        return self._config[property_name]