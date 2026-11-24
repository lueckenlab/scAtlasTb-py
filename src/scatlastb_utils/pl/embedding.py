from __future__ import annotations

import logging
import traceback
from collections.abc import Iterable
from pathlib import Path
from pprint import pformat

import matplotlib as mpl
import matplotlib.patheffects
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype
from tqdm import tqdm

from scatlastb_utils.pp.sample import sample
from scatlastb_utils.utils import (
    _sanitize_default_file_name,
    dask_compute,
    handle_verbosity,
    parse_gene_names,
    remove_outliers,
)

logger = logging.getLogger(__name__)


def _get_legend_label_colors(legend):
    """Return legend colors keyed by the displayed legend text."""
    label_colors = {}
    for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=False):
        if not hasattr(handle, "get_facecolor"):
            continue
        facecolor = handle.get_facecolor()
        if len(facecolor) == 0:
            continue
        label_colors[text.get_text()] = mpl.colors.to_hex(facecolor[0], keep_alpha=False)
    return label_colors


def _format_legend_labels(legend, obs, color, annotate_legend, category_index_map, bold_labels):
    """
    Unified handler for legend styling.

    Appends counts, applies SCTK-style numbering, and bolds specific labels.
    """
    if bold_labels is None:
        bold_labels = []
    if category_index_map is None:
        category_index_map = {}

    # Get group sizes
    if annotate_legend:
        counts = obs[color].value_counts(dropna=False)
        category_counts_str = {str(k): int(v) for k, v in counts.items()}
    else:
        category_counts_str = {}

    annotated_labels = []
    for text in legend.get_texts():
        label = text.get_text()
        # category numbers (SCTK-style)
        if category_index_map and label in category_index_map:
            display = f"{category_index_map[label]}: {label}"
        else:
            display = label
        # append counts to legend labels
        count = category_counts_str.get(label)
        text.set_text(f"{display} (n={count})" if count is not None else display)
        # set font weight for bold labels
        if label in bold_labels:
            text.set_fontweight("bold")
        else:
            text.set_fontweight("normal")
        annotated_labels.append(text.get_text())

    return annotated_labels


def _plot_centroids_on_embedding(
    ax,
    adata,
    color,
    basis,
    legend,
    category_index_map,
    legend_fontsize=10,
    bold_labels=None,
    adjust_text_kwargs=None,
    luminance_threshold=0.4,
):
    """Plot category numbers at centroid positions on the embedding."""
    from adjustText import adjust_text

    # Only compute centroids for categories actually present in the current subset
    categories = [cat for cat in adata.obs[color].cat.categories if cat in adata.obs[color].unique()]

    # Compute centroids (median coordinates) for each category
    coords = adata.obsm[basis][:, :2]
    centroids = (
        pd.DataFrame(coords, index=adata.obs[color])
        .groupby(level=0, dropna=False, observed=True)
        .median()
        .reindex(categories)
    )

    # Extract colors from legend handles to match circles to categories
    color_map = {
        text.get_text(): handle.get_facecolor()[0]
        for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=False)
        if hasattr(handle, "get_facecolor")
    }

    if bold_labels is None:
        bold_labels = []

    created_texts = []
    arrow_colors = []
    for cat, row in centroids.iterrows():
        bg_color = color_map.get(cat, "white")

        # Automatic text contrast: determine if white or black text is more readable
        try:
            r, g, b = mpl.colors.to_rgba(bg_color)[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < luminance_threshold else "black"
        except (ValueError, TypeError):
            text_color = "white"

        # Determine label for centroid
        label = category_index_map.get(cat, cat)
        txt = ax.text(
            row.iloc[0],
            row.iloc[1],
            s=str(label),
            fontsize=legend_fontsize + 2 if cat in bold_labels else legend_fontsize,
            fontweight="bold" if cat in bold_labels else "normal",
            ha="center",
            va="center",
            color=text_color,
            bbox=dict(
                boxstyle="circle",
                facecolor="none",
                alpha=0.4,
                edgecolor="none",
            ),
        )
        txt.set_path_effects([mpl.patheffects.Stroke(linewidth=2, foreground=bg_color), mpl.patheffects.Normal()])
        created_texts.append(txt)
        arrow_colors.append(mpl.colors.to_hex(bg_color, keep_alpha=False))

    if adjust_text_kwargs is None:
        adjust_text_kwargs = dict(
            force_text=(0.5, 0.5),
            expand=(1.05, 1.05),
            autoalign=False,
        )

    arrowprops = adjust_text_kwargs.get("arrowprops") if isinstance(adjust_text_kwargs, dict) else None
    has_explicit_arrow_color = isinstance(arrowprops, dict) and any(
        key in arrowprops for key in ("color", "c", "ec", "edgecolor", "fc", "facecolor")
    )

    adjusted = adjust_text(created_texts, ax=ax, **adjust_text_kwargs)

    # If arrows are requested but no color is set, color each arrow by its category.
    if not has_explicit_arrow_color and isinstance(adjusted, tuple) and len(adjusted) > 1:
        arrows = adjusted[1]
        for arrow, arrow_color in zip(arrows, arrow_colors, strict=False):
            for setter in ("set_color", "set_edgecolor", "set_facecolor"):
                if hasattr(arrow, setter):
                    getattr(arrow, setter)(arrow_color)


def _estimate_legend_width(categories, fontsize):
    if len(categories) == 0:
        return 0

    # from scanpy hardcoded defaults
    ncol = 1 if len(categories) <= 14 else 2 if len(categories) <= 30 else 3

    longest_cat = max(categories, key=len)
    # estimate text width
    avg_char_width_pts = fontsize * 0.6
    max_text_width_pts = len(longest_cat) * avg_char_width_pts
    # add handle (marker) and internal padding (in points)
    # handlelength (2.0) + handletextpad (0.8) = 2.8
    column_width_pts = max_text_width_pts + (2.8 * fontsize)
    total_width_pts = (column_width_pts * ncol) + (2.0 * fontsize * (ncol - 1))

    # Convert points to inches (72 pts = 1 inch)
    return (total_width_pts / 72) + 0.1


@handle_verbosity
def _plot_color_axis(
    adata,
    color,
    basis,
    obs=None,
    annotate_legend=True,
    plot_centroids=False,
    centroid_label_bold=False,
    verbose=False,
    file_name=None,
    title="",
    output_dir=None,
    file_prefix="",
    file_type="png",
    max_label_length=10,
    dpi=200,
    figsize=(6, 6),
    outline_thickness=2,
    adjust_text_kwargs=None,
    bold_labels=None,
    luminance_threshold=0.4,
    return_annotated_labels=False,
    show=True,
    **kwargs,
):
    """Plot a single color (or list of gene colors) and optionally save to disk."""
    # verbosity handled by decorator; no local log-level changes required
    if obs is None:
        obs = adata.obs.copy()

    palette = None
    if file_name is None:
        file_name = _sanitize_default_file_name(color)
    colors = color if isinstance(color, list) else [color]
    if bold_labels is None:
        bold_labels = []

    fig_params = dict(
        frameon=False,
        vector_friendly=True,
        fontsize=9,
        figsize=figsize,
        dpi=dpi,
        dpi_save=dpi,
        format=file_type,
    )

    if len(colors) > 4:
        fig_params = dict(frameon=False)
    sc.set_figure_params(**fig_params)

    # Select palette according to the first matching obs column's type and cardinality
    color = colors[0] if colors else None
    palette = None
    categorical_legend = False
    continuous_legend = False

    if len(colors) == 1 and color in adata.obs.columns:
        color_vec = adata.obs[color]

        if is_numeric_dtype(color_vec):
            palette = "coolwarm" if color_vec.min() < 0 else "plasma"
            continuous_legend = True

        elif is_categorical_dtype(color_vec):
            categorical_legend = True
            ncat = color_vec.nunique()
            if ncat > 102:
                logger.warning("Color '%s' has more than 102 categories, using 'turbo' palette and no legend", color)
                palette = "turbo"
                categorical_legend = False
            elif ncat > 20:
                palette = sc.pl.palettes.godsnot_102

    legend_fontsize = kwargs.get("legend_fontsize", 10)
    colorbar_title_fontsize = kwargs.pop("colorbar_title_fontsize", legend_fontsize)
    colorbar_tick_fontsize = kwargs.pop("colorbar_tick_fontsize", max(colorbar_title_fontsize, 8))
    legend_width = 0
    if categorical_legend:
        legend_width = _estimate_legend_width(color_vec.cat.categories, legend_fontsize)

    if isinstance(plot_centroids, str):
        plot_centroids = [plot_centroids]

    total_width = figsize[0] + legend_width
    fig = plt.figure(figsize=(total_width, figsize[1]), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[figsize[0], legend_width], wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])
    annotated_labels = None

    try:
        sc.pl.embedding(
            adata,
            basis=basis,
            color=colors,
            show=False,
            ax=ax,
            palette=palette,
            legend_loc="right margin" if categorical_legend else None,
            colorbar_loc=None if continuous_legend else "right",
            **kwargs,
        )

        if categorical_legend:
            legend = ax.get_legend()
            category_index_map = None

            if plot_centroids:
                categories = [cat for cat in adata.obs[color].cat.categories if cat in adata.obs[color].unique()]
                category_index_map = {
                    cat: idx + 1 for idx, cat in enumerate(categories) if len(str(cat)) > max_label_length
                }
                _plot_centroids_on_embedding(
                    ax=ax,
                    adata=adata,
                    color=color,
                    basis=basis,
                    legend=legend,
                    category_index_map=category_index_map,
                    legend_fontsize=legend_fontsize,
                    bold_labels=bold_labels if centroid_label_bold else [],
                    adjust_text_kwargs=adjust_text_kwargs,
                    luminance_threshold=luminance_threshold,
                )

            annotated_labels = _format_legend_labels(
                legend=legend,
                obs=obs,
                color=color,
                annotate_legend=annotate_legend,
                category_index_map=category_index_map,
                bold_labels=bold_labels,
            )
        elif continuous_legend:
            mappable = next(c for c in ax.collections if hasattr(c, "get_array") and c.get_array() is not None)
            cbar = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label(color, fontsize=colorbar_title_fontsize, rotation=270, labelpad=15)
            cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
            ax.set_title("")

        # adjust figure layout to accommodate legend and title
        plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.1)
        ax.set_box_aspect(figsize[1] / figsize[0])
        if isinstance(title, str):
            fig.suptitle(f"{title}\nn={obs.shape[0]}", fontsize=12)

        logger.info(f'Plotting color "{file_name}" successful.')

    except (ValueError, RuntimeError) as e:
        traceback.print_exc()
        logger.error(f'Failed to plot "{file_name}": {e}')
        fig = plt.figure()
    except Exception:
        raise

    if output_dir is not None:
        out_path = Path(output_dir) / f"{file_prefix}{file_name}.{file_type}"
        try:
            fig.savefig(out_path, bbox_inches="tight")
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f'Failed to save plot "{file_name}" to {out_path}: {e}')
            traceback.print_exc()
        except Exception:
            raise
    if show:
        plt.show(fig)
    plt.close(fig)

    if return_annotated_labels:
        return file_name, {
            "labels": annotated_labels,
            "label_colors": _get_legend_label_colors(legend) if annotated_labels is not None else {},
        }


@handle_verbosity
def embedding(
    adata,
    basis: str = "X_umap",
    color: str | list | None = None,
    plot_centroids: list | bool | None = None,
    bold_labels: list | None = None,
    category_order: object = None,
    na_strings: str = ["NaN", "None", "", "nan", "unknown"],
    min_cells_per_category: float = 0,
    outlier_factor: float = 0,
    luminance_threshold: float = 0.4,
    gene_chunk_size: int = 10,
    output_dir: Path | str | None = None,
    file_prefix: str = "",
    file_type: str = "png",
    title: str | bool = "",
    annotate_legend: bool = True,
    dpi: int = 200,
    n_jobs: int = 1,
    figsize: tuple = (6, 6),
    downsample: float | int | None = None,
    adjust_text_kwargs: dict | None = None,
    verbose: bool = False,
    inplace: bool = False,
    return_annotated_labels: bool = False,
    show: bool = True,
    **kwargs,
):
    """Plot a scanpy embedding for one or more colors with preprocessing and post-processing.

    This wraps ``sc.pl.embedding`` with dataset cleaning, optional centroid
    placement and optional label-overlap resolution using :mod:`adjustText`.

    Parameters
    ----------
    adata
        Annotated data matrix.
    basis
        Key in ``adata.obsm`` for the embedding coordinates (e.g. ``"X_umap"``).
    color
        One or more ``adata.obs`` column names or gene names to color by.
    plot_centroids
        Subset of ``color`` values for which centroid labels are drawn on the
        embedding or bool if all colors should be considered. Numeric columns will be skipped.
    bold_labels
        List of category names to appear in bold in the legend.
    category_order
        Any iterable of strings for global order, or dict mapping {col_name: iterable of order}.
    na_strings
        For categorical or str columns, which strings to convert to NaN
    min_cells_per_category
        Minimum number of cells required per category.  Values in ``[0, 1)``
        are interpreted as a fraction of the total cell count.
    outlier_factor
        Cells whose embedding coordinate deviates more than this factor from
        the mean absolute value are removed.  Set to ``0`` to disable.
    gene_chunk_size
        Maximum number of genes rendered in a single gene-panel figure.
    output_dir
        Directory where output PNG files are written.  Set to ``None``
        (default) to display figures interactively instead of saving.
    title
        Prefix added to the figure suptitle (combined with the cell count).
    annotate_legend
        Whether to append category counts to legend labels (e.g. "CD4+ T (n=123)").
    dpi
        Resolution used for both rendering and saving figures.
    n_jobs
        Number of parallel threads for figure generation.
    figsize
        Tuple specifying figure size in inches, e.g. ``(8, 4)`` for wide aspect.
    downsample
        If float in (0, 1], randomly subsample that fraction of cells before plotting.
        If int > 1, randomly subsample up to that many cells. Default: None (no downsampling).
    adjust_text_kwargs : dict, optional
        If provided, these keyword arguments will be forwarded to
        ``adjust_text`` (from the ``adjustText`` package) to resolve overlapping
        centroid labels placed by this function (when ``plot_centroids=True``).
        Example keys include ``arrowprops`` (dict) and ``expand_text`` (tuple).
        Note: ``adjustText`` is a required dependency; import will fail with a
        clear error if it is not installed.
    warn
        If True, log warnings when colors are dropped due to invalidity or low category counts.
    inplace
        If True, remove slots from adata inplace. This can be useful for scripts where the
        adata is not used afterwards, and memory footprint should be minimised to only what
        is essential for plotting.
    return_annotated_labels
        If True, return a dictionary mapping each plotted categorical obs color to the
        annotated legend labels and their legend colors.
    **kwargs
        Additional keyword arguments forwarded to ``_plot_color_axis`` and
        ultimately to ``sc.pl.embedding`` (e.g. ``legend_fontsize``, ``ncols``).

    Examples
    --------
    - Automatic centroid placement + adjustment::

        embedding(
            adata,
            basis="X_umap",
            color="cluster_number",
            plot_centroids=True,
            adjust_text_kwargs={
                "arrowprops": {"arrowstyle": "->", "color": "black", "lw": 0.5},
                "expand_text": (1.05, 1.2),
            },
        )

    """
    original_adata = adata
    obs_columns = list(adata.obs.columns)

    # Parse color list and ensure centroid colors are included
    colors: list = color if isinstance(color, list) else ([color] if color is not None else [])

    # parse colors to include as centroids
    if plot_centroids:
        if plot_centroids is True:
            plot_centroids = colors
        elif isinstance(plot_centroids, str):
            plot_centroids = list(plot_centroids)
    else:
        plot_centroids = []

    # collect all colors
    colors += [c for c in plot_centroids if c not in colors]
    colors = list(dict.fromkeys(colors))
    logger.info(f"Configured colors:\n{pformat(colors)}")

    # Separate gene colors from obs colors
    gene_colors = [c for c in colors if c not in obs_columns]
    gene_colors = parse_gene_names(adata, gene_colors)
    gene_colors.sort()

    # Filter obs colors: must exist and have more than one unique value
    dropped_colors = [c for c in colors if c not in obs_columns or adata.obs[c].nunique() <= 1]
    if dropped_colors:
        logger.warning(
            f"The following colors were dropped because they are not in obs or have <=1 unique value: {dropped_colors}"
        )
    colors = [c for c in colors if c in obs_columns and adata.obs[c].nunique() > 1]
    logger.info(f"Colors from obs after filtering:\n{pformat(colors)}")

    obs = adata.obs[colors].copy()  # full annotation needed for legend
    if not inplace or adata.is_view:
        logger.info("Convert view to copy...")
        adata = adata.copy()

    # Resolve min_cells_per_category threshold
    if min_cells_per_category < 1:
        min_cells_per_category *= adata.n_obs

    # Prepare each obs color column
    for col in [col for col in colors if is_categorical_dtype(adata.obs[col]) or is_string_dtype(adata.obs[col])]:
        # set NaNs (optional)
        column = adata.obs[col].astype(object).replace(na_strings, float("nan")).astype("category")
        # remove rare categories
        value_counts = column.value_counts(dropna=False)
        rare = value_counts[value_counts <= min_cells_per_category].index
        if len(rare) > 0:
            logger.warning(f"In color '{col}', the following rare categories were dropped: {list(rare)}")
        adata.obs[col] = column.cat.remove_categories(rare)

        # handle category ordering if specified
        order = None
        if isinstance(category_order, dict):
            order = category_order.get(col)
        elif category_order is not None and not isinstance(category_order, dict):
            # Accept any non-string iterable
            if isinstance(category_order, Iterable) and not isinstance(category_order, (str, bytes)):
                order = list(category_order)
            else:
                order = None
        if order:
            # Filter order to only include categories present in the data to avoid errors
            existing_order = [c for c in order if c in adata.obs[col].cat.categories]
            # Add any categories present in data but missing from order to the end
            missing = [c for c in adata.obs[col].cat.categories if c not in existing_order]
            adata.obs[col] = adata.obs[col].cat.reorder_categories(existing_order + missing)

    if not colors:
        logger.info("No valid colors, skip...")
        colors = [None]

    # Remove embedding outliers
    logger.info("Remove outliers...")
    adata = remove_outliers(adata, "max", factor=outlier_factor, rep=basis, copy=False)
    adata = remove_outliers(adata, "min", factor=outlier_factor, rep=basis, copy=False)

    if downsample is not None:
        # Find a categorical color column for stratification
        stratify_col = None
        for col in colors:
            if col in adata.obs.columns and is_categorical_dtype(adata.obs[col]):
                stratify_col = col
                break
        if isinstance(downsample, float) and 0 < downsample < 1:
            adata = sample(adata, fraction=downsample, stratify=stratify_col, copy=False)
        elif isinstance(downsample, int) and downsample > 1 and adata.n_obs > downsample:
            adata = sample(adata, n=downsample, stratify=stratify_col, copy=False)
        # else: ignore if not valid

    if adata.is_view:
        logger.info("Convert view to copy...")
        adata = adata.copy()

    # Subset to requested genes, or drop X/var if no genes needed
    if gene_colors:
        logger.info(f"Subset to {len(gene_colors)} requested genes...")
        adata = adata[:, adata.var_names.isin(gene_colors)]
        adata = dask_compute(adata, layers="X")
        logger.info(str(adata.var))
    else:
        del adata.X
        del adata.var

    # Compute point size (clamp between default and 200)
    default_size = max(1, 200_000 / adata.n_obs)
    size = kwargs.pop("size", default_size) or default_size
    kwargs["size"] = min(200, max(default_size, size))

    logger.info("Parameters:\n" + pformat(kwargs))

    # Plot obs colors in parallel
    color_results = list(
        tqdm(
            Parallel(return_as="generator", backend="threading", n_jobs=n_jobs)(
                delayed(_plot_color_axis)(
                    adata=adata,
                    color=col,
                    basis=basis,
                    obs=obs,
                    annotate_legend=annotate_legend,
                    plot_centroids=col in plot_centroids,
                    luminance_threshold=luminance_threshold,
                    bold_labels=bold_labels,
                    title=title,
                    file_name=col,
                    dpi=dpi,
                    output_dir=output_dir,
                    file_prefix=file_prefix,
                    file_type=file_type,
                    figsize=figsize,
                    adjust_text_kwargs=adjust_text_kwargs,
                    verbose=verbose,
                    return_annotated_labels=return_annotated_labels,
                    show=show,
                    **kwargs,
                )
                for col in colors
            ),
            desc="Plotting colors",
            total=len(colors),
            miniters=1,
        )
    )

    # Propagate scanpy's computed categorical colors back onto the caller's adata,
    # since plotting above ran on a copy/subset that is discarded after this call.
    for col in colors:
        if col is None:
            continue
        uns_key = f"{col}_colors"
        if uns_key in adata.uns:
            original_adata.uns[uns_key] = adata.uns[uns_key]

    # Chunk genes and plot each group in parallel
    gene_groups = {
        f"genes_group={idx}": gene_colors[i : i + gene_chunk_size]
        for idx, i in enumerate(range(0, len(gene_colors), gene_chunk_size))
    }
    if gene_groups:
        kwargs.setdefault("ncols", 4)
        list(
            tqdm(
                Parallel(return_as="generator", backend="threading", n_jobs=n_jobs)(
                    delayed(_plot_color_axis)(
                        adata=adata,
                        color=group_color,
                        basis=basis,
                        obs=obs,
                        annotate_legend=annotate_legend,
                        verbose=False,
                        title=title,
                        file_name=group_title,
                        dpi=dpi,
                        output_dir=output_dir,
                        file_prefix=file_prefix,
                        file_type=file_type,
                        figsize=figsize,
                        return_annotated_labels=False,
                        show=show,
                        **kwargs,
                    )
                    for group_title, group_color in gene_groups.items()
                ),
                desc="Plotting genes",
                total=len(gene_groups),
                miniters=1,
            )
        )

    if return_annotated_labels:
        return {color: labels for color, labels in color_results if labels is not None}
