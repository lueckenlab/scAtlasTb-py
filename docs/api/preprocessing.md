# Preprocessing: `pp`

```{eval-rst}
.. currentmodule:: scatlastb_utils.pp
```

```{eval-rst}
.. autosummary::
    :toctree: ../generated

    strip_barcodes
    find_library_obs
    pseudobulk
```

## kNN graphs

Compute a kNN graph once at the largest k and derive graphs for smaller k from it, instead of rerunning the neighbour search and storing one adjacency matrix per k:

```{eval-rst}
.. autosummary::
    :toctree: ../generated

    subset_neighbors
    knn_from_distances
    connectivities_from_knn
```
