from scipy.ndimage import grey_opening, grey_closing, grey_dilation
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from timeit import default_timer as timer

with open('tophat.cu', 'r') as f:
    code = f.read()

module = cp.RawModule(code=code)
dilation_horizontal = module.get_function('dilation_horizontal')


def square_closing(img, db, bb):
    return grey_closing(grey_opening(img, structure=db), structure=bb)


def square_opening(img, db, bb):
    return grey_opening(grey_closing(img, structure=db), structure=bb)


def top_hat(img, db, bb):
    NWTH = img - np.minimum(img, square_closing(img, db, bb))
    NBTH = np.maximum(img, square_opening(img, db, bb)) - img

    return [NWTH, NBTH]


def multiscale_top_hat(img, nw, nl, nm, ns, n):
    NWTH_out = np.zeros_like(img)
    NBTH_out = np.zeros_like(img)

    for i in range(n):
        bb = np.zeros([nl + ns * i, nl + ns * i])
        db = np.pad(np.zeros([nw + ns * i, nw + ns * i]), [nm, nm])

        single_scale_top_hat = top_hat(img, db, bb)
        NWTH_out = np.maximum(NWTH_out, single_scale_top_hat[0])
        NBTH_out = np.maximum(NBTH_out, single_scale_top_hat[1])

    return [NWTH_out, NBTH_out]


def grey_dilation_cuda(image, p):
    [img, window_size, reconstruction_shape, pad_size, n_window,
     out, required_blocks, thread_per_block] = prepare_morph(image, p)

    dilation_horizontal((required_blocks, ), (2, thread_per_block),
                        (img, out, p, window_size, n_window, img.shape[0]), shared_mem=2 * n_window * p * 4)

    out = out.reshape(reconstruction_shape)[:, pad_size:-pad_size].transpose()
    [out, window_size, reconstruction_shape, pad_size, n_window,
     out2, required_blocks, thread_per_block] = prepare_morph(out, p)

    dilation_horizontal((required_blocks, ), (2, thread_per_block),
                        (out, out2, p, window_size, n_window, out.shape[0]), shared_mem=2 * n_window * p * 4)
    out2 = out2.reshape(reconstruction_shape)[:, pad_size:-pad_size].transpose()

    return out2


def prepare_morph(img, p):
    window_size = 2 * p - 1

    pad_size = int((p - 1) / 2)
    img = cp.pad(img, ((0, 0), (pad_size, pad_size)))

    reconstruction_shape = (img.shape[0], img.shape[1])
    img = img.reshape(-1)
    n_window = int(np.floor(img.shape[0] / p))
    out = np.zeros_like(img)
    required_padding = (p - np.mod(img.shape[0], 2 * p - 1))

    if required_padding > 0:
        img = np.pad(img, (0, required_padding))

    required_blocks = int((n_window / 512) + 1)

    original_num_window = n_window
    if n_window > 512:
        thread_per_block = 512
        n_window = 512
    else:
        thread_per_block = n_window

    if 2 * n_window * p * 4 > dilation_horizontal.max_dynamic_shared_size_bytes:
        max_window = int(np.floor(dilation_horizontal.max_dynamic_shared_size_bytes / (2 * p * 4)))
        required_blocks = int((original_num_window / max_window) + 1)
        n_window = max_window
        thread_per_block = max_window

    return [img, window_size, reconstruction_shape, pad_size, n_window, out, required_blocks, thread_per_block]


if __name__ == '__main__':
    image = io.imread('01.jpg')
    image = cp.array(image[:, :, 0]).astype(int)

    p = 110

    start = timer()
    out = grey_dilation_cuda(image, p)
    end = timer()
    print(end - start)

    start = timer()
    out_cpu = grey_dilation(cp.asnumpy(image), [p, p])
    end = timer()
    print(end - start)
    plt.imshow(cp.asnumpy(out), cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imshow(out_cpu, cmap='gray', vmin=0, vmax=255)
    plt.show()

    # ax = plt.hist(image.ravel(), bins=256)
    # plt.show()
    #
    # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    # nW = 5
    # nL = 5
    # nM = 2
    # nS = 11
    # n = 9
    #
    # [NWTH, NBTH] = multiscale_top_hat(image, nW, nL, nM, nS, n)
    # out = image * 0.2 + 5 * NWTH - 3 * NBTH
    # out[out > 255] = 255
    # out[out < 0] = 0
    #
    # ax = plt.hist(out.ravel(), bins=256)
    # plt.show()
    # plt.imshow(out, cmap='gray', vmin=0, vmax=255)
    # plt.show()