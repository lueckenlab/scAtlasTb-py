import logging
from collections.abc import Callable

import anndata as ad
import numpy as np
import scanpy as sc

logger = logging.getLogger(__name__)


def get_bootstrap_adata(
    adata: ad.AnnData,
    size: int | None = None,
    use_rep: str = "X_emb",
    seed: int | np.random.Generator | None = None,
) -> ad.AnnData:
    """Subsample cells without replacement and recompute the kNN graph.

    The kNN graph is recomputed for the subset, because graph-based metrics such as Moran's I depend on it.
    Neighbors are first recomputed from the precomputed distances in ``adata.obsp["distances"]``.
    If that fails (e.g. because cells lost too many neighbors in the subset), neighbors are computed from ``use_rep``.

    Parameters
    ----------
    adata
        Annotated data matrix with a kNN graph.
    size
        Number of cells to sample. Defaults to 90% of all cells.
    use_rep
        Representation in ``adata.obsm`` used as fallback to recompute neighbors.
    seed
        Random seed or generator for reproducible sampling.

    Returns
    -------
    Copy of the subsampled AnnData object with a recomputed kNN graph.
    """
    rng = np.random.default_rng(seed)
    if size is None:
        size = int(0.9 * adata.n_obs)

    logger.info("Subset to %d cells...", size)
    indices = np.sort(rng.choice(adata.n_obs, size=size, replace=False))
    adata_bootstrap = adata[indices].copy()

    n_neighbors = adata.uns.get("neighbors", {}).get("params", {}).get("n_neighbors", 15)
    n_neighbors = int(np.ravel(n_neighbors)[0])  # older AnnData files store it as an array
    try:
        logger.info("Recalculate neighbors from distances...")
        adata_bootstrap.obsm["distances"] = adata_bootstrap.obsp["distances"]
        sc.pp.neighbors(
            adata_bootstrap,
            use_rep="distances",
            metric="precomputed",
            transformer="sklearn",
            n_neighbors=n_neighbors,
        )
    except (KeyError, ValueError) as e:
        logger.info("Recalculation of neighbors from distances failed (%s), recalculate from %s...", e, use_rep)
        sc.pp.neighbors(adata_bootstrap, use_rep=use_rep, n_neighbors=n_neighbors)
    finally:
        adata_bootstrap.obsm.pop("distances", None)

    return adata_bootstrap


def bootstrap_metric(
    adata: ad.AnnData,
    metric_function: Callable,
    n_bootstraps: int = 5,
    size: int | None = None,
    use_rep: str = "X_emb",
    seed: int | np.random.Generator | None = None,
    **kwargs,
):
    """Compute a metric on multiple bootstrapped subsets of cells.

    Parameters
    ----------
    adata
        Annotated data matrix with a kNN graph.
    metric_function
        Function called as ``metric_function(adata, **kwargs)``.
    n_bootstraps
        Number of bootstraps. If ``n_bootstraps <= 1``, the metric is computed once on the full data.
    size
        Number of cells per bootstrap, see :func:`get_bootstrap_adata`.
    use_rep
        Fallback representation for recomputing neighbors, see :func:`get_bootstrap_adata`.
    seed
        Random seed or generator for reproducible sampling.
    **kwargs
        Passed to ``metric_function``.

    Returns
    -------
    Single score if ``n_bootstraps <= 1``, else a list of ``n_bootstraps`` scores.
    """
    if n_bootstraps <= 1:
        return metric_function(adata, **kwargs)

    rng = np.random.default_rng(seed)
    return [
        metric_function(get_bootstrap_adata(adata, size=size, use_rep=use_rep, seed=rng), **kwargs)
        for _ in range(n_bootstraps)
    ]
