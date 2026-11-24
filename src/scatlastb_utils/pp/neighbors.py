"""Derive kNN graphs with smaller k from a precomputed kNN graph with a larger k.

The k nearest neighbours of a cell are the first k entries of its sorted K-nearest-neighbour list for any K >= k.
Computing a single kNN graph at the largest k of interest (e.g. with the heuristically optimised search of
:func:`scanpy.pp.neighbors` or :func:`rapids_singlecell.pp.neighbors`) is therefore sufficient to obtain the graphs for
all smaller k, without rerunning the search and without storing one adjacency matrix per k.

Only the directed distance matrix needs to be kept. Connectivities depend on k (both the UMAP fuzzy simplicial set and
the Gaussian kernel use k-dependent bandwidths and are symmetrised afterwards), so they are recomputed from the
truncated distances.
"""

from __future__ import annotations

from typing import Literal

import anndata as ad
import numpy as np
from scipy import sparse

_Method = Literal["umap", "gauss"]


def knn_from_distances(
    distances: sparse.spmatrix | sparse.sparray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the ``n_neighbors`` nearest neighbours per cell from a sparse kNN distance matrix.

    Follows the :mod:`scanpy` convention that ``n_neighbors`` counts the cell itself, i.e. the returned arrays have the
    cell itself as first column, followed by its ``n_neighbors - 1`` nearest other cells sorted by distance.

    Parameters
    ----------
    distances
        Sparse ``n_obs x n_obs`` matrix of directed kNN distances, as stored in ``adata.obsp['distances']`` by
        :func:`scanpy.pp.neighbors`. Self-edges are ignored if present. Explicitly stored zeros (duplicate cells) are
        treated as neighbours at distance 0. Note that for exact duplicates, scanpy may store the cell itself instead
        of its duplicate, leaving such rows with one neighbour fewer than the stored ``n_neighbors``.
    n_neighbors
        Number of neighbours to extract, including the cell itself.

    Returns
    -------
    indices
        ``n_obs x n_neighbors`` array of neighbour indices.
    dists
        ``n_obs x n_neighbors`` array of neighbour distances.

    Raises
    ------
    ValueError
        If ``n_neighbors < 2`` or any cell has fewer than ``n_neighbors - 1`` stored neighbours.
    """
    if n_neighbors < 2:
        raise ValueError(f"n_neighbors must be at least 2, got {n_neighbors}")

    D = sparse.csr_matrix(distances)
    n_obs = D.shape[0]
    rows = np.repeat(np.arange(n_obs), np.diff(D.indptr))

    # drop self-edges so that matrices with and without a stored self column are handled alike
    not_self = D.indices != rows
    rows, cols, data = rows[not_self], D.indices[not_self], D.data[not_self]

    n_stored = np.bincount(rows, minlength=n_obs)
    k_other = n_neighbors - 1
    if (too_few := n_stored < k_other).any():
        raise ValueError(
            f"Cannot extract n_neighbors={n_neighbors}: {too_few.sum()} cells have fewer than {k_other} stored "
            f"neighbours (minimum stored: {n_stored.min()}). Compute the kNN graph with a larger n_neighbors first."
        )

    # sort by row, then distance, then column (deterministic tie-breaking)
    order = np.lexsort((cols, data, rows))
    row_start = np.concatenate([[0], np.cumsum(n_stored)[:-1]])
    rank = np.arange(len(order)) - row_start[rows[order]]
    keep = order[rank < k_other]

    indices = np.empty((n_obs, n_neighbors), dtype=np.int64)
    dists = np.empty((n_obs, n_neighbors), dtype=D.dtype)
    indices[:, 0] = np.arange(n_obs)
    dists[:, 0] = 0
    indices[:, 1:] = cols[keep].reshape(n_obs, k_other)
    dists[:, 1:] = data[keep].reshape(n_obs, k_other)
    return indices, dists


def _distances_from_knn(indices: np.ndarray, dists: np.ndarray) -> sparse.csr_matrix:
    """Build a scanpy-style sparse distance matrix (self excluded, duplicates kept as explicit zeros)."""
    n_obs, n_neighbors = indices.shape
    indptr = np.arange(0, n_obs * (n_neighbors - 1) + 1, n_neighbors - 1)
    return sparse.csr_matrix(
        (dists[:, 1:].ravel().copy(), indices[:, 1:].ravel().copy(), indptr),
        shape=(n_obs, n_obs),
    )


def connectivities_from_knn(
    indices: np.ndarray,
    dists: np.ndarray,
    method: _Method = "umap",
) -> sparse.csr_matrix:
    """Compute symmetric connectivities from kNN arrays as :func:`scanpy.pp.neighbors` does.

    Parameters
    ----------
    indices
        ``n_obs x n_neighbors`` neighbour indices with the cell itself as first column,
        e.g. from :func:`knn_from_distances`.
    dists
        Matching ``n_obs x n_neighbors`` neighbour distances.
    method
        ``'umap'`` for the UMAP fuzzy simplicial set (scanpy default) or ``'gauss'`` for a Gaussian kernel.

    Returns
    -------
    Sparse ``n_obs x n_obs`` connectivity matrix.
    """
    # scanpy keeps these private; isolate the import so a future move only needs fixing here
    from scanpy.neighbors import _connectivity

    n_obs, n_neighbors = indices.shape
    if method == "umap":
        conn = _connectivity.umap(indices, dists, n_obs=n_obs, n_neighbors=n_neighbors)
    elif method == "gauss":
        conn = _connectivity.gauss(_distances_from_knn(indices, dists), n_neighbors, knn=True)
    else:
        raise ValueError(f"method must be 'umap' or 'gauss', got {method!r}")
    return sparse.csr_matrix(conn)


def _stored_n_neighbors(adata: ad.AnnData, neighbors_key: str, distances: sparse.csr_matrix) -> int:
    params = adata.uns[neighbors_key].get("params", {})
    if "n_neighbors" in params:
        return int(params["n_neighbors"])
    # infer from the matrix: stored non-self entries per row + the cell itself
    return int(np.diff(sparse.csr_matrix(distances).indptr).min()) + 1


def subset_neighbors(
    adata: ad.AnnData,
    n_neighbors: int,
    *,
    neighbors_key: str = "neighbors",
    key_added: str | None = None,
    method: _Method | None = None,
    inplace: bool = True,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix] | None:
    """Derive a kNN graph with ``n_neighbors`` from a precomputed kNN graph with more neighbours.

    Avoids rerunning the (approximate) nearest-neighbour search and storing one graph per k: compute the graph once at
    the largest k needed, e.g. ``sc.pp.neighbors(adata, n_neighbors=max_k)``, and derive smaller graphs on demand.

    For exact searches (scanpy with < 4096 cells, ``rapids_singlecell`` with ``algorithm='brute'``) the result is
    identical to running the search with ``n_neighbors`` directly. For approximate searches (pynndescent, CAGRA,
    IVF, NN-descent) the neighbours generally differ slightly from a direct run, but tend to be closer to the exact
    neighbours, since searching for more neighbours explores more candidates.

    Parameters
    ----------
    adata
        Annotated data matrix with a precomputed kNN graph.
    n_neighbors
        Number of neighbours of the derived graph, including the cell itself (as in :func:`scanpy.pp.neighbors`).
        Must not exceed the ``n_neighbors`` of the stored graph.
    neighbors_key
        Key in ``adata.uns`` of the precomputed graph. Its ``distances_key`` entry locates the distances in
        ``adata.obsp``.
    key_added
        Where to store the derived graph, following :func:`scanpy.pp.neighbors`: if ``None``, write to
        ``adata.uns['neighbors']``, ``adata.obsp['distances']`` and ``adata.obsp['connectivities']`` (replacing the
        precomputed graph if ``neighbors_key='neighbors'``); otherwise to ``adata.uns[key_added]``,
        ``adata.obsp[f'{key_added}_distances']`` and ``adata.obsp[f'{key_added}_connectivities']``.
    method
        Connectivity method, ``'umap'`` or ``'gauss'``. If ``None``, use the method stored in
        ``adata.uns[neighbors_key]['params']``, falling back to ``'umap'``.
    inplace
        Whether to write the derived graph to ``adata``. If ``False``, return it instead.

    Returns
    -------
    If ``inplace=False``, a tuple ``(distances, connectivities)`` of sparse matrices, otherwise ``None``.

    Examples
    --------
    >>> import scanpy as sc
    >>> import scatlastb_utils as sa
    >>> sc.pp.neighbors(adata, n_neighbors=50)
    >>> for k in [15, 30]:
    ...     sa.pp.subset_neighbors(adata, n_neighbors=k, key_added=f"neighbors_{k}")
    """
    if neighbors_key not in adata.uns:
        raise KeyError(f'neighbors_key "{neighbors_key}" not found in adata.uns')
    uns = adata.uns[neighbors_key]
    distances_key = uns.get("distances_key", "distances")
    if distances_key not in adata.obsp:
        raise KeyError(f'distances "{distances_key}" of neighbors_key "{neighbors_key}" not found in adata.obsp')
    distances = adata.obsp[distances_key]

    stored_k = _stored_n_neighbors(adata, neighbors_key, distances)
    if n_neighbors > stored_k:
        raise ValueError(
            f'n_neighbors={n_neighbors} exceeds n_neighbors={stored_k} of the precomputed graph "{neighbors_key}"'
        )

    params = dict(uns.get("params", {}))
    if method is None:
        method = params.get("method", "umap")

    stored_method = params.get("method", "umap")
    stored_conn_key = uns.get("connectivities_key", "connectivities")
    if n_neighbors == stored_k and method == stored_method and stored_conn_key in adata.obsp:
        # nothing to truncate; also sidesteps rows where scanpy stored the cell itself in place of a duplicate cell
        new_distances = sparse.csr_matrix(distances, copy=True)
        connectivities = sparse.csr_matrix(adata.obsp[stored_conn_key], copy=True)
    else:
        indices, dists = knn_from_distances(distances, n_neighbors)
        new_distances = _distances_from_knn(indices, dists)
        connectivities = connectivities_from_knn(indices, dists, method=method)

    if not inplace:
        return new_distances, connectivities

    if key_added is None:
        key_added = "neighbors"
        conn_key, dist_key = "connectivities", "distances"
    else:
        conn_key, dist_key = f"{key_added}_connectivities", f"{key_added}_distances"

    params |= {
        "n_neighbors": n_neighbors,
        "method": method,
        "subset_from_key": neighbors_key,
        "subset_from_n_neighbors": stored_k,
    }
    adata.obsp[dist_key] = new_distances
    adata.obsp[conn_key] = connectivities
    adata.uns[key_added] = {
        "connectivities_key": conn_key,
        "distances_key": dist_key,
        "params": params,
    }
    return None
