from PIL import Image
from joblib import Parallel, delayed
import multiprocessing

def make_square(im, min_size=224, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, int(((size - x) / 2), int((size - y) / 2)))
    return new_im