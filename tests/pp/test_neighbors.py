import anndata as ad
import numpy as np
import pytest
import scanpy as sc
from scipy import sparse

from scatlastb_utils.pp.neighbors import connectivities_from_knn, knn_from_distances, subset_neighbors

MAX_K = 50
N_OBS = 600  # < 4096, so scanpy uses an exact search and results must match a direct run exactly


def _make_adata(n_obs=N_OBS, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_obs, 10)) + rng.integers(0, 5, size=(n_obs, 1)) * 3
    return ad.AnnData(X=np.zeros((n_obs, 1), dtype=np.float32), obsm={"X_emb": X.astype(np.float32)})


def _direct(adata, k, method="umap"):
    direct = adata.copy()
    sc.pp.neighbors(direct, n_neighbors=k, use_rep="X_emb", method=method)
    return direct


@pytest.fixture(params=["umap", "gauss"])
def method(request):
    return request.param


@pytest.fixture
def adata_max_k(method):
    adata = _make_adata()
    sc.pp.neighbors(adata, n_neighbors=MAX_K, use_rep="X_emb", method=method)
    return adata


def _assert_sparse_close(a, b, atol=1e-6):
    assert a.shape == b.shape
    assert abs(sparse.csr_matrix(a) - sparse.csr_matrix(b)).max() <= atol


@pytest.mark.parametrize("k", [5, 15, 30, MAX_K])
def test_subset_matches_direct_run(adata_max_k, method, k):
    direct = _direct(adata_max_k, k, method=method)
    subset_neighbors(adata_max_k, n_neighbors=k, key_added=f"neighbors_{k}")

    _assert_sparse_close(adata_max_k.obsp[f"neighbors_{k}_distances"], direct.obsp["distances"])
    _assert_sparse_close(adata_max_k.obsp[f"neighbors_{k}_connectivities"], direct.obsp["connectivities"])


def test_knn_from_distances_sorted_with_self(adata_max_k):
    indices, dists = knn_from_distances(adata_max_k.obsp["distances"], 15)
    assert indices.shape == dists.shape == (N_OBS, 15)
    np.testing.assert_array_equal(indices[:, 0], np.arange(N_OBS))
    np.testing.assert_array_equal(dists[:, 0], 0)
    assert (np.diff(dists, axis=1) >= 0).all()


def test_key_added_and_uns(adata_max_k, method):
    subset_neighbors(adata_max_k, n_neighbors=10, key_added="knn10")
    uns = adata_max_k.uns["knn10"]
    assert uns["connectivities_key"] == "knn10_connectivities"
    assert uns["distances_key"] == "knn10_distances"
    assert uns["params"]["n_neighbors"] == 10
    assert uns["params"]["method"] == method
    assert uns["params"]["use_rep"] == "X_emb"
    assert uns["params"]["subset_from_key"] == "neighbors"
    assert uns["params"]["subset_from_n_neighbors"] == MAX_K
    # source graph untouched
    assert adata_max_k.uns["neighbors"]["params"]["n_neighbors"] == MAX_K
    assert np.unique(adata_max_k.obsp["distances"].getnnz(axis=1)).tolist() == [MAX_K - 1]


def test_key_added_none_replaces_default_graph(adata_max_k):
    subset_neighbors(adata_max_k, n_neighbors=10)
    assert adata_max_k.uns["neighbors"]["params"]["n_neighbors"] == 10
    assert np.unique(adata_max_k.obsp["distances"].getnnz(axis=1)).tolist() == [9]


def test_not_inplace(adata_max_k):
    before = adata_max_k.obsp["distances"].copy()
    distances, connectivities = subset_neighbors(adata_max_k, n_neighbors=10, inplace=False)
    assert set(adata_max_k.obsp.keys()) == {"distances", "connectivities"}
    _assert_sparse_close(adata_max_k.obsp["distances"], before, atol=0)
    assert np.unique(distances.getnnz(axis=1)).tolist() == [9]
    assert connectivities.shape == (N_OBS, N_OBS)


def test_custom_neighbors_key():
    adata = _make_adata()
    sc.pp.neighbors(adata, n_neighbors=MAX_K, use_rep="X_emb", key_added="big")
    direct = _direct(adata, 15)
    subset_neighbors(adata, n_neighbors=15, neighbors_key="big", key_added="small")
    _assert_sparse_close(adata.obsp["small_connectivities"], direct.obsp["connectivities"])


def test_n_neighbors_inferred_without_params(adata_max_k):
    del adata_max_k.uns["neighbors"]["params"]
    subset_neighbors(adata_max_k, n_neighbors=MAX_K, key_added="same")
    _assert_sparse_close(adata_max_k.obsp["same_distances"], adata_max_k.obsp["distances"])
    with pytest.raises(ValueError, match="exceeds"):
        subset_neighbors(adata_max_k, n_neighbors=MAX_K + 1)


def test_self_column_ignored(adata_max_k):
    D = adata_max_k.obsp["distances"].tolil()
    D.setdiag(1e-3)  # spurious self-edges, e.g. from search backends returning the cell itself
    expected = knn_from_distances(adata_max_k.obsp["distances"], 15)
    observed = knn_from_distances(D.tocsr(), 15)
    np.testing.assert_array_equal(observed[0], expected[0])
    np.testing.assert_allclose(observed[1], expected[1])


def test_non_constant_rows():
    # symmetrised distances have varying numbers of entries per row; must still pick the nearest per row
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5))
    dense = np.linalg.norm(X[:, None] - X[None], axis=-1)
    adata = ad.AnnData(obsm={"X_emb": X.astype(np.float32)}, X=np.zeros((200, 1), dtype=np.float32))
    sc.pp.neighbors(adata, n_neighbors=20, use_rep="X_emb")
    D = adata.obsp["distances"]
    D_sym = D.maximum(D.T).tocsr()
    assert len(np.unique(D_sym.getnnz(axis=1))) > 1

    indices, dists = knn_from_distances(D_sym, 10)
    np.fill_diagonal(dense, np.inf)
    expected = np.argsort(dense, axis=1)[:, :9]
    np.testing.assert_array_equal(np.sort(indices[:, 1:], axis=1), np.sort(expected, axis=1))


def test_explicit_zeros_are_neighbors():
    # duplicate cells stored as explicit zeros must count as nearest neighbours
    data = np.array([0.0, 1.0, 0.0, 2.0, 1.0, 2.0])
    cols = np.array([1, 2, 0, 2, 0, 1])
    D = sparse.csr_matrix((data, cols, [0, 2, 4, 6]), shape=(3, 3))
    assert D.nnz == 6
    indices, dists = knn_from_distances(D, 2)
    np.testing.assert_array_equal(indices[:, 1], [1, 0, 0])
    np.testing.assert_array_equal(dists[:, 1], [0, 0, 1])


def test_scanpy_duplicate_cells():
    # scanpy may store the cell itself instead of its exact duplicate
    adata = _make_adata(n_obs=300)
    adata.obsm["X_emb"][1] = adata.obsm["X_emb"][0]
    sc.pp.neighbors(adata, n_neighbors=20, use_rep="X_emb")

    subset_neighbors(adata, n_neighbors=20, key_added="same")  # same k: copied as is
    _assert_sparse_close(adata.obsp["same_connectivities"], adata.obsp["connectivities"], atol=0)

    direct = _direct(adata, 10)
    subset_neighbors(adata, n_neighbors=10, key_added="small")
    # rows not involving the duplicates match the direct run (which duplicate wins a tie is arbitrary)
    touches_dup = (direct.obsp["distances"][:, [0, 1]].getnnz(axis=1) > 0) | (
        adata.obsp["small_distances"][:, [0, 1]].getnnz(axis=1) > 0
    )
    mask = ~touches_dup
    mask[[0, 1]] = False
    assert mask.sum() > 250
    observed = adata.obsp["small_distances"][mask]
    expected = direct.obsp["distances"][mask]
    _assert_sparse_close(observed, expected)


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        pytest.param({"n_neighbors": MAX_K + 1}, ValueError, "exceeds", id="k too large"),
        pytest.param({"n_neighbors": 10, "neighbors_key": "missing"}, KeyError, "not found", id="missing key"),
        pytest.param({"n_neighbors": 1}, ValueError, "at least 2", id="k too small"),
        pytest.param({"n_neighbors": 10, "method": "foo"}, ValueError, "method", id="invalid method"),
    ],
)
def test_errors(adata_max_k, kwargs, error, match):
    with pytest.raises(error, match=match):
        subset_neighbors(adata_max_k, **kwargs)


def test_too_few_stored_neighbors():
    D = sparse.csr_matrix(np.array([[0, 1.0, 2.0], [1.0, 0, 0], [2.0, 0, 0]]))
    with pytest.raises(ValueError, match="fewer than"):
        knn_from_distances(D, 3)


def test_connectivities_invalid_method():
    with pytest.raises(ValueError, match="method"):
        connectivities_from_knn(np.zeros((2, 2), dtype=int), np.zeros((2, 2)), method="foo")
