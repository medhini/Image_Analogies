def best_coherence_match(As, (A_h, A_w), BBp_feat, s, im, px, Bp_w, c):
    assert(len(s) >= 1)

    row, col = px

    # construct iterables
    rs = []
    ims = []
    prs = []
    rows = np.arange(np.max([0, row - c.pad_lg]), row + 1, dtype=int)
    cols = np.arange(np.max([0, col - c.pad_lg]), np.min([Bp_w, col + c.pad_lg + 1]), dtype=int)

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

def best_coherence_match(features, s, A_p, ):

	row, col = px

	rows = np.arange(0, row+1, dtype = int)
	cols = np.arange(0, col+1, dtype=int)

	for feature_level in features:

		for m in range(len(feature_level)):
			for n in range(len(feature_level[m])):
				r = feature_level[m][n]
				(feature_level[s[[m][n]] + (q - r)] - feature_level[q]) 
		"""F(s(r)+(q − r)) − F(q)"""