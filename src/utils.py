import numpy as np


def SimT(s,a,b,tx,ty):

    assert s.ndim == 1

    n = s.shape[0] / 2
    s_proj = np.zeros(s.shape)
    for i in xrange(0,n):
        x = s[i]
        y = s[i + n]
        s_proj[i] = a * x - b * y + tx
        s_proj[i + n] = b * x + a * y + ty

    return s_proj


def CalcSimT(src,dst):

    assert src.ndim == 1
    assert dst.ndim == 1
    assert src.shape[0] == dst.shape[0]

    n = src.shape[0] / 2
    H = np.zeros((4,4))
    g = np.zeros((4,1))

    for i in xrange(0,n):

        H[0,0] = H[0,0] + src[i]**2 + src[i + n]**2
        H[0,2] = H[0,2] + src[i]
        H[0,3] = H[0,3] + src[i + n]

        g[0] = g[0] + src[i] * dst[i] + src[i + n] * dst[i + n]
        g[1] = g[1] + src[i] * dst[i + n] - src[i + n] * dst[i]
        g[2] = g[2] + dst[i]
        g[3] = g[3] + dst[i + n]

    H[1,1] = H[0,0]
    H[3,0] = H[0,3]
    H[1,2] = -H[0,3]
    H[2,1] = -H[0,3]
    H[1,3] = H[0,2]
    H[3,1] = H[0,2]
    H[2,0] = H[0,2]
    H[2,2] = n
    H[3,3] = n

    # p = H\g
    p = np.linalg.solve(H,g)
    a = p[0]
    b = p[1]
    tx = p[2]
    ty = p[3]

    return a,b,tx,ty


def draw_landmarks_red(img, lms):
    try:
        img = img.copy()
        img = np.repeat(img, 3, 2)

        for i in range(lms.shape[0]):
            lmx, lmy = int(lms[i, 0]), int(lms[i, 1])
            img[lmy - 1: lmy + 1, lmx - 1: lmx + 1, 0] = 255
    except:
        pass
    return img.astype(np.float32)


def draw_landmarks_green(img, lms):
    try:
        img = img.copy()
        img = np.repeat(img, 3, 2)

        for i in range(lms.shape[0]):
            lmx, lmy = int(lms[i, 0]), int(lms[i, 1])
            img[lmy - 1: lmy + 1, lmx - 1: lmx + 1, 1] = 255
    except:
        pass
    return img.astype(np.float32)


def draw_landmarks_blue(img, lms):
    # Bigger for blue since the contrast is bad otherwise
    try:
        img = img.copy()
        img = np.repeat(img, 3, 2)

        for i in range(lms.shape[0]):
            lmx, lmy = int(lms[i, 0]), int(lms[i, 1])
            img[lmy - 2: lmy + 2, lmx - 2: lmx + 2, 2] = 255
    except:
        pass
    return img.astype(np.float32)


def draw_landmarks(img, lms):
    try:
        img = img.copy()

        for i in range(lms.shape[0]):
            lmx, lmy = int(lms[i, 0]), int(lms[i, 1])
            img[lmy - 2:lmy + 2, lmx - 2:lmx + 2] = 0
    except:
        pass
    return img.astype(np.float32)


def batch_draw_landmarks(imgs, pred):
    return np.array([draw_landmarks(img, l) for img, l in zip(imgs, pred)])


def batch_draw_landmarks_green(imgs, pred):
    return np.array([draw_landmarks_green(img, l) for img, l in zip(imgs, pred)])


def batch_draw_landmarks_blue(imgs, pred):
    return np.array([draw_landmarks_blue(img, l) for img, l in zip(imgs, pred)])


def batch_draw_landmarks_red(imgs, pred):
    return np.array([draw_landmarks_red(img, l) for img, l in zip(imgs, pred)])
