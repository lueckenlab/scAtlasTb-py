import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import scatlastb_utils as sa


@pytest.fixture
def adata_scored(adata_real):
    """Add a structured and a random numeric covariate and a categorical covariate without structure."""
    rng = np.random.default_rng(0)
    adata_real.obs["pc1"] = adata_real.obsm["X_pca"][:, 0]
    adata_real.obs["noise"] = rng.normal(size=adata_real.n_obs)
    adata_real.obs["random_labels"] = pd.Categorical(rng.choice(list("abcd"), size=adata_real.n_obs))
    return adata_real


def test_morans_i_numeric(adata_scored):
    structured = sa.me.morans_i(adata_scored, "pc1")
    noise = sa.me.morans_i(adata_scored, "noise")
    assert isinstance(structured, float)
    assert structured > 0.5
    assert 0 <= noise < 0.1


def test_morans_i_categorical(adata_scored):
    structured = sa.me.morans_i(adata_scored, "bulk_labels")
    random = sa.me.morans_i(adata_scored, "random_labels")
    assert structured > 0.2
    assert 0 <= random < 0.1


def test_morans_i_categorical_reproducible(adata_scored):
    scores = [sa.me.morans_i_categorical(adata_scored, "bulk_labels", seed=1) for _ in range(2)]
    assert scores[0] == scores[1]


def test_morans_i_categorical_all_orderings(adata_scored):
    # 2 categories -> 2 orderings, which are mirror images and give the same score
    adata_scored.obs["two"] = pd.Categorical(np.where(adata_scored.obs["pc1"] > 0, "high", "low"))
    score_all = sa.me.morans_i_categorical(adata_scored, "two", num_mappings=10)
    score_single = sa.me.morans_i_categorical(adata_scored, "two", num_mappings=1)
    assert score_all == pytest.approx(score_single)


def test_morans_i_categorical_missing(adata_scored):
    adata_scored.obs["with_nan"] = adata_scored.obs["bulk_labels"].astype(object)
    adata_scored.obs.iloc[0, adata_scored.obs.columns.get_loc("with_nan")] = np.nan
    assert np.isnan(sa.me.morans_i_categorical(adata_scored, "with_nan"))


def test_morans_i_single_category(adata_scored):
    adata_scored.obs["constant"] = pd.Categorical(["a"] * adata_scored.n_obs)
    assert np.isnan(sa.me.morans_i(adata_scored, "constant"))


def test_morans_i_multiple_covariates(adata_scored):
    scores = sa.me.morans_i(adata_scored, ["pc1", "bulk_labels"])
    assert list(scores) == ["pc1", "bulk_labels"]
    assert scores["pc1"] == pytest.approx(sa.me.morans_i(adata_scored, "pc1"))


def test_morans_i_clip(adata_scored):
    # negative Moran's I is clipped to 0 by default
    unclipped = sa.me.morans_i(adata_scored, "noise", clip=False)
    assert sa.me.morans_i(adata_scored, "noise") == max(unclipped, 0)


def test_morans_i_graph_key(adata_scored):
    adata_scored.obsp["other_graph"] = adata_scored.obsp["connectivities"].copy()
    assert sa.me.morans_i(adata_scored, "pc1", graph_key="other_graph") == pytest.approx(
        sa.me.morans_i(adata_scored, "pc1")
    )


@pytest.mark.parametrize("use_rep", ["X", "X_pca"])
def test_morans_i_features(adata_real, use_rep):
    n_features = adata_real.n_vars if use_rep == "X" else adata_real.obsm[use_rep].shape[1]
    all_scores = sa.me.morans_i_features(adata_real, use_rep=use_rep, return_all=True)
    mean_score = sa.me.morans_i_features(adata_real, use_rep=use_rep)
    assert all_scores.shape == (n_features,)
    assert mean_score == pytest.approx(np.nanmean(all_scores))


def test_morans_i_features_layer_and_subset(adata_real):
    adata_real.layers["copy"] = adata_real.X.copy()
    genes = adata_real.var_names[:5]
    scores = sa.me.morans_i_features(adata_real, use_rep="copy", features=genes, return_all=True)
    assert scores.shape == (5,)


def test_morans_i_features_errors(adata_real):
    with pytest.raises(KeyError):
        sa.me.morans_i_features(adata_real, use_rep="missing")
    with pytest.raises(ValueError):
        sa.me.morans_i_features(adata_real, use_rep="X_pca", features=adata_real.var_names[:5])


def test_match_genes(adata_real):
    gene = adata_real.var_names[0]
    assert sa.me.match_genes(adata_real, [gene, "NOT_A_GENE"]) == [gene]
    # partial names are matched as substrings
    assert gene in sa.me.match_genes(adata_real, [gene[:-1]])


def test_morans_i_genes(adata_real):
    gene_sets = {
        "top_10": adata_real.var_names[:10],
        "missing": ["NOT_A_GENE"],
    }
    scores = sa.me.morans_i_genes(adata_real, gene_sets=gene_sets)
    assert list(scores) == ["top_10"]
    assert scores["top_10"] == pytest.approx(sa.me.morans_i_features(adata_real, features=adata_real.var_names[:10]))


def test_morans_i_genescore(adata_real):
    sc.tl.score_genes(adata_real, gene_list=adata_real.var_names[:20], score_name="gene_score:top_20")
    scores = sa.me.morans_i_genescore(adata_real, gene_sets=["top_20", "missing"])
    assert list(scores) == ["top_20"]
    assert scores["top_20"] > 0


@pytest.mark.parametrize("n_bootstraps", [1, 3])
def test_morans_i_random(adata_real, n_bootstraps):
    scores = sa.me.morans_i_random(adata_real, n_bootstraps=n_bootstraps, use_rep="X_pca")
    if n_bootstraps == 1:
        assert isinstance(scores, float)
        scores = [scores]
    assert len(scores) == n_bootstraps
    assert all(0 <= s < 0.1 for s in scores)


def test_morans_i_random_reproducible(adata_real):
    kwargs = {"n_bootstraps": 2, "use_rep": "X_pca", "seed": 42}
    assert sa.me.morans_i_random(adata_real, **kwargs) == sa.me.morans_i_random(adata_real, **kwargs)
