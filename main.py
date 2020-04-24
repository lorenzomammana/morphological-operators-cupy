from scipy.ndimage import grey_opening, grey_closing, grey_dilation, grey_erosion, black_tophat, white_tophat
from skimage import io
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from morphology_cupy import *


if __name__ == '__main__':
    image = io.imread('01.jpg')
    image = cp.array(image[:, :, 0]).astype(int)

    p = 121

    start = timer()
    out = grey_closing_cuda(image, p)
    end = timer()
    print(end - start)

    start = timer()
    out_cpu = grey_closing(cp.asnumpy(image), [p, p])
    end = timer()
    print(end - start)
    plt.imshow(cp.asnumpy(out), cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imshow(out_cpu, cmap='gray', vmin=0, vmax=255)
    plt.show()

    start = timer()
    [NWTH, NBTH] = grey_top_hat_cuda(image, p)
    end = timer()
    print(end - start)

    start = timer()
    NWTH_cpu = white_tophat(cp.asnumpy(image), structure=np.zeros([p, p]))
    NBTH_cpu = black_tophat(cp.asnumpy(image), structure=np.zeros([p, p]))
    end = timer()
    print(end - start)
