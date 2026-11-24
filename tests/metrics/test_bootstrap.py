import numpy as np
import pytest

import scatlastb_utils as sa


def test_get_bootstrap_adata(adata_real):
    ad_boot = sa.me.get_bootstrap_adata(adata_real, size=100, use_rep="X_pca", seed=0)
    assert ad_boot.n_obs == 100
    assert ad_boot.obsp["connectivities"].shape == (100, 100)
    assert ad_boot.obs_names.isin(adata_real.obs_names).all()
    assert ad_boot.obs_names.is_unique
    assert "distances" not in ad_boot.obsm


def test_get_bootstrap_adata_default_size(adata_real):
    ad_boot = sa.me.get_bootstrap_adata(adata_real, use_rep="X_pca", seed=0)
    assert ad_boot.n_obs == int(0.9 * adata_real.n_obs)


def test_get_bootstrap_adata_without_distances(adata_real):
    del adata_real.obsp["distances"]
    ad_boot = sa.me.get_bootstrap_adata(adata_real, size=100, use_rep="X_pca", seed=0)
    assert ad_boot.obsp["connectivities"].shape == (100, 100)


@pytest.mark.parametrize("n_bootstraps", [1, 3])
def test_bootstrap_metric(adata_real, n_bootstraps):
    scores = sa.me.bootstrap_metric(
        adata_real,
        metric_function=lambda ad, offset: ad.n_obs + offset,
        n_bootstraps=n_bootstraps,
        size=50,
        use_rep="X_pca",
        seed=0,
        offset=1,
    )
    if n_bootstraps == 1:
        assert scores == adata_real.n_obs + 1
    else:
        assert scores == [51] * n_bootstraps


def test_bootstrap_metric_reproducible(adata_real):
    def first_cell(ad):
        return ad.obs_names[0]

    kwargs = {"metric_function": first_cell, "n_bootstraps": 3, "size": 50, "use_rep": "X_pca"}
    assert sa.me.bootstrap_metric(adata_real, seed=1, **kwargs) == sa.me.bootstrap_metric(adata_real, seed=1, **kwargs)
    assert np.unique(sa.me.bootstrap_metric(adata_real, seed=1, **kwargs)).size > 1
