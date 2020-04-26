from scipy.ndimage import grey_opening, grey_closing, grey_dilation, grey_erosion, black_tophat, white_tophat
from skimage import io
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from morphology_cupy import *


if __name__ == '__main__':
    image = io.imread('Lena.png')
    image = cp.array(image).astype(int)

    p = 12

    start = timer()
    out = grey_erosion_cuda(image, p)
    end = timer()
    print("Erosion GPU: " + str(end - start))

    start = timer()
    out_cpu = grey_erosion(cp.asnumpy(image), [p, p])
    end = timer()
    print("Erosion CPU: " + str(end - start))

    print("Difference: " + str(cp.sum(cp.asnumpy(out) != out_cpu)))
    plt.imshow(cp.asnumpy(out), cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imshow(out_cpu, cmap='gray', vmin=0, vmax=255)
    plt.show()

    start = timer()
    out = grey_dilation_cuda(image, p)
    end = timer()
    print("Dilation GPU: " + str(end - start))

    start = timer()
    out_cpu = grey_dilation(cp.asnumpy(image), [p, p])
    end = timer()
    print("Dilation CPU: " + str(end - start))

    print("Difference: " + str(cp.sum(cp.asnumpy(out) != out_cpu)))

    start = timer()
    [NWTH, NBTH] = grey_top_hat_cuda(image, p)
    end = timer()
    print("Top-hat GPU: " + str(end - start))

    start = timer()
    NWTH_cpu = white_tophat(cp.asnumpy(image), structure=np.zeros([p, p]))
    NBTH_cpu = black_tophat(cp.asnumpy(image), structure=np.zeros([p, p]))
    end = timer()
    print("Top-hat CPU: " + str(end - start))
    print("Difference: " + str(cp.sum(cp.asnumpy(out) != out_cpu)))