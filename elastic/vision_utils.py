import math
import tempfile
import numpy as np
import ffmpeg
from PIL import Image


def save_grid(video, fname=None, nrow=None, fps=10):
    b, t, h, w, c = video.shape

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
        if b % nrow != 0:
            nrow = 8
    ncol = math.ceil(b / nrow)
    padding = 1
    new_h = (padding + h) * ncol + padding
    new_h += new_h % 2
    new_w = (padding + w) * nrow + padding
    new_w += new_w % 2
    video_grid = np.zeros((t, new_h, new_w, c), dtype='uint8')
    for i in range(b):
        r = i // nrow
        c = i % nrow

        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]

    return save(video_grid, fname=fname, fps=fps)

def save(video, fname=None, fps=10, crf=23):
    # video: THWC, uint8
    T, H, W, C = video.shape

    if T == 1:
        if fname is None:
            fname = tempfile.NamedTemporaryFile().name + '.jpg'
        image = Image.fromarray(video[0])
        image.save(fname)
    else:
        if fname is None:
            fname = tempfile.NamedTemporaryFile().name + '.mp4'
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{W}x{H}', r=fps)
            .output(fname, pix_fmt='yuv420p', vcodec='libx264', crf=crf, r=fps)
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

        for frame in video:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        process.wait()
    print('Saved video to', fname, video.shape)
    return fname
