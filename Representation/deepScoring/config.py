import os
import shutil

class load_config:
  def __init__(self):
    # Must be in the same folder as three_dl
    # loads the configuration from config file
    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"config"), 'r')
    config = f.read()
    f.close()

    config = config.split('\n')
    
    for line in config:
      if line != '' and line[0] != '#':
        [name,var] = line.split('=')
        name, var = name.replace(' ', ''), var.replace(' ', '')
        self.__dict__[name] = var
   
    # flushing the tensorboard repository
    folder = self.TENSORBOARD_PATH
    for the_file in os.listdir(folder):
      file_path = os.path.join(folder, the_file)
      try:
        if os.path.isfile(file_path):
          os.unlink(file_path)
      except Exception as e:
        print(e)
