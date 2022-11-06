from lib.param_utils.load_params import load_params
from lib.param_utils.save_params import save_params
from glob import glob
import os

filenames = glob('woven-resonance*.dat')

for filename in filenames[1:]:
    param = load_params(filename)
    embedding = param['embedding']['embedding']
    new_name = filename.removesuffix('.dat') + '_embedding.dat'
    os.remove(filename)
    save_params(embedding, new_name)
