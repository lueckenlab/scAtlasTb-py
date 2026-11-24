import itertools
import logging
import math
import re
from collections.abc import Iterable, Mapping

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from .bootstrap import bootstrap_metric

logger = logging.getLogger(__name__)


def _to_dense_or_csr(X):
    """Materialise dask arrays and make sure sparse input is CSR, as expected by scanpy's Moran's I."""
    if hasattr(X, "compute"):
        X = X.compute()
    if sparse.issparse(X):
        return sparse.csr_matrix(X)
    return np.asarray(X)


def _morans_i(
    adata: ad.AnnData,
    vals,
    graph_key: str = "connectivities",
    clip: bool = True,
) -> float | np.ndarray:
    """Compute Moran's I of values on the kNN graph in ``adata.obsp[graph_key]``.

    ``vals`` is a vector of length ``n_obs`` or a matrix of shape ``(n_features, n_obs)``.
    Negative scores are set to 0 if ``clip=True``.
    """
    if isinstance(vals, pd.Series):
        vals = vals.to_numpy(dtype=float)
    vals = _to_dense_or_csr(vals)
    try:
        score = sc.metrics.morans_i(adata.obsp[graph_key], vals=vals)
    except ZeroDivisionError:
        return np.nan
    if clip:
        score = np.maximum(score, 0)  # keeps NaN
    return float(score) if np.ndim(score) == 0 else np.asarray(score)


def _category_encodings(n_categories: int, num_mappings: int, rng: np.random.Generator) -> np.ndarray:
    """Integer encodings of categories: all orderings if there are at most ``num_mappings``, otherwise random ones."""
    if n_categories < 13 and math.factorial(n_categories) <= num_mappings:
        return np.array(list(itertools.permutations(range(1, n_categories + 1))))
    return np.array([rng.permutation(np.arange(1, n_categories + 1)) for _ in range(num_mappings)])


def morans_i_categorical(
    adata: ad.AnnData,
    covariate: str,
    num_mappings: int = 10,
    graph_key: str = "connectivities",
    clip: bool = True,
    seed: int | np.random.Generator | None = 0,
) -> float:
    """Moran's I of a categorical covariate.

    Categories have no natural order, so each category is mapped to an integer and Moran's I is averaged over
    ``num_mappings`` random orderings (or over all orderings, if there are no more than ``num_mappings``).
    Cells with a missing category result in a NaN score.

    Parameters
    ----------
    adata
        Annotated data matrix with a kNN graph.
    covariate
        Column in ``adata.obs``.
    num_mappings
        Number of random category orderings.
    graph_key
        Key of the graph in ``adata.obsp``.
    clip
        Set negative scores to 0.
    seed
        Random seed or generator for the category orderings.

    Returns
    -------
    Mean Moran's I over all category orderings.
    """
    rng = np.random.default_rng(seed)
    codes = pd.Categorical(adata.obs[covariate]).codes
    n_categories = codes.max() + 1
    if n_categories < 2:
        return np.nan

    encodings = _category_encodings(n_categories, num_mappings, rng).astype(float)
    vals = encodings[:, codes]  # (n_mappings, n_obs)
    vals[:, codes < 0] = np.nan
    scores = _morans_i(adata, vals, graph_key=graph_key, clip=clip)
    return float(np.mean(scores))


def morans_i(
    adata: ad.AnnData,
    covariate: str | Iterable[str],
    num_mappings: int = 10,
    graph_key: str = "connectivities",
    clip: bool = True,
    seed: int | np.random.Generator | None = 0,
) -> float | dict[str, float]:
    """Moran's I of one or more covariates in ``adata.obs`` on the kNN graph.

    Numeric covariates are scored directly; categorical covariates are scored with :func:`morans_i_categorical`.

    Parameters
    ----------
    adata
        Annotated data matrix with a kNN graph.
    covariate
        Column or list of columns in ``adata.obs``.
    num_mappings
        Number of category orderings for categorical covariates.
    graph_key
        Key of the graph in ``adata.obsp``.
    clip
        Set negative scores to 0.
    seed
        Random seed or generator for the category orderings.

    Returns
    -------
    A single score if ``covariate`` is a string, otherwise a dictionary of scores per covariate.
    """
    if isinstance(covariate, str):
        if pd.api.types.is_numeric_dtype(adata.obs[covariate]):
            return _morans_i(adata, adata.obs[covariate], graph_key=graph_key, clip=clip)
        return morans_i_categorical(
            adata, covariate, num_mappings=num_mappings, graph_key=graph_key, clip=clip, seed=seed
        )

    rng = np.random.default_rng(seed)
    return {
        cov: morans_i(adata, cov, num_mappings=num_mappings, graph_key=graph_key, clip=clip, seed=rng)
        for cov in covariate
    }


def morans_i_features(
    adata: ad.AnnData,
    use_rep: str = "X",
    features: Iterable[str] | None = None,
    graph_key: str = "connectivities",
    clip: bool = True,
    return_all: bool = False,
) -> float | np.ndarray:
    """Moran's I of each feature of a representation on the kNN graph.

    Parameters
    ----------
    adata
        Annotated data matrix with a kNN graph.
    use_rep
        ``"X"``, a key in ``adata.layers`` or a key in ``adata.obsm``.
    features
        Subset of ``adata.var_names`` to score. Only used for ``"X"`` and layers.
    graph_key
        Key of the graph in ``adata.obsp``.
    clip
        Set negative scores to 0.
    return_all
        Return the score of every feature instead of the mean.

    Returns
    -------
    Mean Moran's I over features (ignoring NaN, e.g. from constant features), or all scores if ``return_all=True``.
    """
    if use_rep == "X":
        X = adata.X
    elif use_rep in adata.layers:
        X = adata.layers[use_rep]
    elif use_rep in adata.obsm:
        X = adata.obsm[use_rep]
        if features is not None:
            raise ValueError("`features` can only be used with `use_rep='X'` or a layer")
    else:
        raise KeyError(f"Representation {use_rep!r} not found in X, layers or obsm")

    if features is not None:
        X = X[:, adata.var_names.isin(list(features))]

    scores = _morans_i(adata, _to_dense_or_csr(X).T, graph_key=graph_key, clip=clip)
    if return_all:
        return np.atleast_1d(scores)
    if np.all(np.isnan(scores)):
        return np.nan
    return float(np.nanmean(scores))


def match_genes(adata: ad.AnnData, genes: Iterable[str]) -> list[str]:
    """Match genes to ``adata.var_names``.

    Genes that are not found exactly are matched as substrings of ``var_names``
    (e.g. ``"CD8A"`` matches ``"ENSG00000153563_CD8A"``).

    Parameters
    ----------
    adata
        Annotated data matrix.
    genes
        Gene names.

    Returns
    -------
    Matching entries of ``adata.var_names``.
    """
    var_names = adata.var_names.astype(str)
    genes = [str(g) for g in genes]
    matched = [g for g in genes if g in var_names]
    missing = [g for g in genes if g not in var_names]
    if missing:
        mask = var_names.str.contains("|".join(re.escape(g) for g in missing))
        matched += [g for g in var_names[mask] if g not in matched]
    return matched


def morans_i_genes(
    adata: ad.AnnData,
    gene_sets: Mapping[str, Iterable[str]],
    layer: str = "X",
    graph_key: str = "connectivities",
    clip: bool = True,
) -> dict[str, float]:
    """Mean Moran's I of the expression of the genes in each gene set.

    Gene sets without any matching genes are skipped with a warning.

    Parameters
    ----------
    adata
        Annotated data matrix with a kNN graph.
    gene_sets
        Mapping of gene set names to genes, matched with :func:`match_genes`.
    layer
        ``"X"`` or a key in ``adata.layers`` with the expression values.
    graph_key
        Key of the graph in ``adata.obsp``.
    clip
        Set negative scores to 0.

    Returns
    -------
    Dictionary of scores per gene set.
    """
    scores = {}
    for set_name, genes in gene_sets.items():
        genes = match_genes(adata, genes)
        if len(genes) == 0:
            logger.warning("No genes found for gene set %s, skip", set_name)
            continue
        scores[set_name] = morans_i_features(adata, use_rep=layer, features=genes, graph_key=graph_key, clip=clip)
    return scores


def morans_i_genescore(
    adata: ad.AnnData,
    gene_sets: Iterable[str],
    key_prefix: str = "gene_score:",
    graph_key: str = "connectivities",
    clip: bool = True,
) -> dict[str, float]:
    """Moran's I of precomputed gene set scores in ``adata.obs``.

    Scores are expected in ``adata.obs[f"{key_prefix}{set_name}"]``, e.g. computed with :func:`scanpy.tl.score_genes`.
    Missing scores are skipped with a warning.

    Parameters
    ----------
    adata
        Annotated data matrix with a kNN graph.
    gene_sets
        Gene set names (a mapping of gene set names to genes also works).
    key_prefix
        Prefix of the score columns in ``adata.obs``.
    graph_key
        Key of the graph in ``adata.obsp``.
    clip
        Set negative scores to 0.

    Returns
    -------
    Dictionary of scores per gene set.
    """
    scores = {}
    for set_name in gene_sets:
        key = f"{key_prefix}{set_name}"
        if key not in adata.obs:
            logger.warning("Gene score %s not found in adata.obs, skip", key)
            continue
        scores[set_name] = _morans_i(adata, adata.obs[key], graph_key=graph_key, clip=clip)
    return scores


def morans_i_random(
    adata: ad.AnnData,
    n_bootstraps: int = 5,
    bootstrap_size: int | None = None,
    use_rep: str = "X_emb",
    graph_key: str = "connectivities",
    clip: bool = True,
    seed: int | np.random.Generator | None = 0,
) -> float | list[float]:
    """Moran's I of random normal values as a negative baseline.

    The baseline is computed on bootstrapped subsets of cells with a recomputed kNN graph,
    see :func:`bootstrap_metric`.

    Parameters
    ----------
    adata
        Annotated data matrix with a kNN graph.
    n_bootstraps
        Number of bootstraps. If ``n_bootstraps <= 1``, the full data is used.
    bootstrap_size
        Number of cells per bootstrap. Defaults to 90% of all cells.
    use_rep
        Fallback representation for recomputing neighbors.
    graph_key
        Key of the graph in ``adata.obsp``.
    clip
        Set negative scores to 0.
    seed
        Random seed or generator for sampling cells and random values.

    Returns
    -------
    Single score if ``n_bootstraps <= 1``, else a list of scores per bootstrap.
    """
    rng = np.random.default_rng(seed)

    def _random_score(_adata):
        return _morans_i(_adata, rng.normal(size=_adata.n_obs), graph_key=graph_key, clip=clip)

    return bootstrap_metric(
        adata,
        metric_function=_random_score,
        n_bootstraps=n_bootstraps,
        size=bootstrap_size,
        use_rep=use_rep,
        seed=rng,
    )
