import os

from PIL import Image
from joblib import Parallel, delayed
import multiprocessing

#Resizing image and filling background black
def make_square(im, min_size=224, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    a, b = (int((size - x) / 2), int((size - y) / 2))
    new_im.paste(im, (a,b))
    return new_im

def resizeImageFillBlack(filename):
    toRezize = Image.open("../data/UTKCropped/%s"%filename).convert("RGB")
    make_square(toRezize).save("../data/UTKResizedBlack/%s" % filename)


num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(resizeImageFillBlack)(filename) for filename in os.listdir("../data/UTKCropped/"))