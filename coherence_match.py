from itertools import product
from utils import to_1d
import numpy as np
from numpy.linalg import norm

def best_coherence_match(As, (A_h, A_w), BBp_feat, s, pixel, Bp_w, n_lg):
    
    assert(len(s) >= 1)
    row, col = pixel

    # construct iterables
    prs = []
    rs = []
    rows = np.arange(0, row+1, dtype=int)
    cols = np.arange(0, col+1, dtype=int)

    for r_coord in product(rows, cols):
        # discard anything after current pixel
        if to_1d(r_coord, Bp_w) < to_1d(pixel, Bp_w):
            # p_r = s(r) + (q - r)

            # pr is an index in a given image Ap
            pr = s[to_1d(r_coord, Bp_w)] + pixel - r_coord
            # i is a list of image nums for each pixel in Bp
            #img_nums = im[to_1d(r_coord, Bp_w)]

            # discard anything outside the bounds of A/Ap lg
            if 0 <= pr[0] < A_h and 0 <= pr[1] < A_w:
            	rs.append(np.array(r_coord))
                prs.append(to_1d(pr, A_w))


    if not rs:
        # no good coherence match
        return [-1, -1]

    rix = np.argmin(norm(As[np.array(prs)] - BBp_feat, ord=2, axis=1))
    # print rix, rs.shape
    r_star = rs[rix]
    # #i_star = ims[rix]
    # # s[r_star] + (q - r-star)
    return s[to_1d(r_star, Bp_w)] + pixel - r_star
    