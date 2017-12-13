from utils import px2ix, Ap_px2ix

def best_coherence_match(a_prime, (A_h, A_w), BBp_feat, s, im, pixel, Bp_w, c):
    
    row, col = pixel

    # construct iterables
    rs = []
    ims = []
    prs = []
    rows = np.arange(0, row + 1, dtype=int)
    cols = np.arange(0, col + 1, dtype=int)

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

