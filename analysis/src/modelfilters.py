import xarray as xr

def get_filter_param_options(filter_fn):
    """Return the discrete grid of parameter settings searched for a given filter function.

    Depth is 1-7 (the tree has 7 rows below the root); rank and value are both 1-9 (9 distinct
    treasure values). ``fit()`` grid-searches this list, so it also fixes the size of the
    ``filter_params`` dimension precomputed in ``BehaviorModel.pov_value_cache``.
    """
    defaults = {
        filter_depth: [1, 2, 3, 4, 5, 6, 7],
        filter_rank: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        filter_value: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    return defaults[filter_fn]

# Filter functions
def filter_value(pov_array: xr.DataArray, filter_params: dict= {"value": [1, 2, 3, 4, 5, 6, 7, 8, 9]}):
    """Keep only cells whose treasure value is large enough; zero out the rest.

    The parameter is inverted: setting ``i`` keeps cells with ``pov_array >= 10 - i``, i.e. the
    threshold is theta = 10 - i. So larger ``i`` means a LOWER threshold and MORE cells retained,
    matching depth/rank where larger parameter = more of the tree considered (i=1 keeps only 9s;
    i=9 keeps everything >= 1). Stacks one board per setting along a new ``filter_params`` dim.
    """
    assert "value" in filter_params.keys(), "Missing value in filter_params"
    filtered_array = xr.concat([xr.where(pov_array >= 10 - i, pov_array.copy(), 0) for i in filter_params["value"]], dim = "filter_params")
    filtered_array["filter_params"] = filter_params["value"]
    return filtered_array

def filter_rank(pov_array: xr.DataArray, filter_params: dict = {"rank": [1, 2, 3, 4, 5, 6, 7, 8, 9]}):
    """Keep the top ``rank`` DISTINCT SCORE LEVELS on the board; zero out the rest.

    This peels value levels, not top-k individual cells: rank 1 retains every cell tied at the
    board maximum (which may be many cells), rank 2 adds all cells at the next distinct value,
    and so on. Stacks one board per setting along a new ``filter_params`` dim.

    The peeling loop is hardcoded to 9 iterations (the 9 possible treasure values) regardless of
    ``filter_params["rank"]``: that list is assigned as the coordinate LABELS of those 9 slices
    and then selected from, so it must have exactly 9 entries or the assignment will not line up.
    """
    assert "rank" in filter_params.keys(), "Missing rank in filter_params"
    mask_pov_array = pov_array.copy()
    filtered_array = []

    # at each step, we mask the highest value in the mask_pov_array, so we iteratively "reveal" more values
    for _ in range(9):
        mask_pov_array = mask_pov_array.where(mask_pov_array < mask_pov_array.max(["rows", "cols"]), 0)
        filtered_array.append(pov_array - mask_pov_array)
    filtered_array = xr.concat(filtered_array, dim = "filter_params")

    filtered_array["filter_params"] = filter_params["rank"]
    filtered_array = filtered_array.sel(filter_params = filter_params["rank"])
    return filtered_array

def filter_depth(pov_array: xr.DataArray, filter_params: dict = {"depth": [1, 2, 3, 4, 5, 6, 7]}):
    """Keep only cells at row index <= ``depth``; zero out everything deeper (truncated search).

    Stacks one board per depth setting along a new ``filter_params`` dim, and transposes so the
    trailing dims are ``(trials, rows, cols)`` as the value functions expect.
    """
    assert "depth" in filter_params.keys(), "Missing depth in filter_params"
    
    filtered_array = xr.concat([xr.where(pov_array.rows <= i, pov_array.copy(), 0) for i in filter_params["depth"]], dim = "filter_params")
    filtered_array["filter_params"] = filter_params["depth"]
    return filtered_array.transpose(...,"trials", "rows", "cols")