from utils import px2idx, Ap_px2ix

def best_coherence_match(a_prime, (A_h, A_w), BBp_feat, s, im, pixel, Bp_w, n_lg):
    
    assert(len(s) >= 1)
    row, col = pixel

    # construct iterables
    rs = []
    ims = []
    prs = []
    rows = np.arange(np.max([0, row - np.floor(n_lg/2.)]), row + 1, dtype=int)
    cols = np.arange(np.max([0, col - np.floor(n_lg/2.)]), np.min([Bp_w, col + np.floor(n_lg/2.) + 1]), dtype=int)

    for r_coord in product(rows, cols):
        # discard anything after current pixel
        if px2ix(r_coord, Bp_w) < px2ix(px, Bp_w):
            # p_r = s(r) + (q - r)

            # pr is an index in a given image Ap_list[img_num]
            pr = s[px2ix(r_coord, Bp_w)] + px - r_coord

            # i is a list of image nums for each pixel in Bp
            img_nums = im[px2ix(r_coord, Bp_w)]

            # discard anything outside the bounds of A/Ap lg
            if 0 <= pr[0] < A_h and 0 <= pr[1] < A_w:
                rs.append(np.array(r_coord))
                ims.append(img_nums)
                prs.append(Ap_px2ix(pr, img_nums, A_h, A_w))


    if not rs:
        # no good coherence match
        return (-1, -1), 0, (0, 0)

    rix = np.argmin(norm(As[np.array(prs)] - BBp_feat, ord=2, axis=1))
    r_star = rs[rix]
    i_star = ims[rix]
    # s[r_star] + (q - r-star)
    return s[px2ix(r_star, Bp_w)] + px - r_star, i_star, r_star