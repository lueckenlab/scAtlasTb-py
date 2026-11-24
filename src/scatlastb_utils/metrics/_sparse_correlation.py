import numba as nb
import numpy as np
import scipy.sparse as sp


@nb.njit(cache=True)
def _extract_row_nb(data, indices, indptr, i, cols):
    row_idx = indices[indptr[i] : indptr[i + 1]]
    row_data = data[indptr[i] : indptr[i + 1]]
    out = np.zeros(len(cols))
    for k in range(len(cols)):
        c = cols[k]
        pos = np.searchsorted(row_idx, c)
        if pos < len(row_idx) and row_idx[pos] == c:
            out[k] = row_data[pos]
    return out


@nb.njit(cache=True)
def _rankdata_nb(v):
    n = len(v)
    order = np.argsort(v)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1.0, n + 1.0)
    i = 0
    while i < n:
        j = i + 1
        while j < n and v[order[j]] == v[order[i]]:
            j += 1
        if j > i + 1:
            avg = (i + j + 1) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg
        i = j
    return ranks


@nb.njit(cache=True)
def _pearson_nb(x, y):
    n = len(x)
    mx = 0.0
    my = 0.0
    for k in range(n):
        mx += x[k]
        my += y[k]
    mx /= n
    my /= n
    num = 0.0
    sx = 0.0
    sy = 0.0
    for k in range(n):
        xc = x[k] - mx
        yc = y[k] - my
        num += xc * yc
        sx += xc * xc
        sy += yc * yc
    denom = (sx * sy) ** 0.5
    if denom == 0.0:
        return np.nan
    return num / denom


@nb.njit(parallel=True, cache=True)
def _correlation_kernel(
    m1_data,
    m1_indices,
    m1_indptr,
    m2_data,
    m2_indices,
    m2_indptr,
    mask_indices,
    mask_indptr,
    n_rows,
    use_spearman,  # bool — passed as a Numba literal for branch elimination
):
    results = np.full(n_rows, np.nan)
    for i in nb.prange(n_rows):
        cols = mask_indices[mask_indptr[i] : mask_indptr[i + 1]]
        n = len(cols)
        if n < 2:
            continue

        v1 = _extract_row_nb(m1_data, m1_indices, m1_indptr, i, cols)
        v2 = _extract_row_nb(m2_data, m2_indices, m2_indptr, i, cols)

        # Skip constant rows
        min1, max1 = v1[0], v1[0]
        min2, max2 = v2[0], v2[0]
        for k in range(1, n):
            if v1[k] < min1:
                min1 = v1[k]
            if v1[k] > max1:
                max1 = v1[k]
            if v2[k] < min2:
                min2 = v2[k]
            if v2[k] > max2:
                max2 = v2[k]
        if min1 == max1 or min2 == max2:
            continue

        if use_spearman:
            v1 = _rankdata_nb(v1)
            v2 = _rankdata_nb(v2)

        results[i] = _pearson_nb(v1, v2)

    return results


def sparse_correlation(
    m1: sp.spmatrix,
    m2: sp.spmatrix,
    mask: sp.spmatrix,
    method: str = "spearman",
) -> np.ndarray:
    """Calculate Pearson or Spearman correlation for each row within local neighborhoods.

    Neighborhoods are defined by the mask.

    For each row i, only columns j where mask[i, j] != 0 are considered.
    Values absent in m1 or m2 but present in mask are treated as 0.
    Rows with fewer than 2 neighbors or constant values return NaN.

    Spearman uses Pearson-on-ranks internally, which correctly handles ties
    (unlike the 1 - 6Σd²/n(n²-1) formula which breaks with ties).

    Parallelism is handled by Numba's prange — set the number of threads via:
        numba.set_num_threads(n)
    or the environment variable NUMBA_NUM_THREADS before importing numba.

    :param m1: First sparse matrix (scipy.sparse CSR format preferred).
    :param m2: Second sparse matrix (scipy.sparse CSR format preferred).
    :param mask: Sparse binary mask defining local neighborhoods.
    :param method: Either "spearman" (default) or "pearson".
    :return: 1D array of correlations, one per row. NaN where correlation is
        undefined (constant row or fewer than 2 neighbors).
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")

    assert m1.shape == m2.shape == mask.shape, f"Shape mismatch: {m1.shape}, {m2.shape}, {mask.shape}"

    m1, m2, mask = [m.tocsr() for m in (m1, m2, mask)]
    for m in (m1, m2, mask):
        m.sort_indices()

    m1_data = np.ascontiguousarray(m1.data, dtype=np.float64)
    m2_data = np.ascontiguousarray(m2.data, dtype=np.float64)

    return _correlation_kernel(
        m1_data,
        m1.indices,
        m1.indptr,
        m2_data,
        m2.indices,
        m2.indptr,
        mask.indices,
        mask.indptr,
        m1.shape[0],
        method == "spearman",
    )
