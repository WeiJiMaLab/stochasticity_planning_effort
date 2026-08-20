import xarray as xr

# Value functions
def value_EV(pov_array: xr.DataArray, variant = None, value_params = {}):
    """Expected pointwise value contrast at the first planning split (row ``1``, cols ``0`` vs ``1``).

    Note on symbols: ``value_params`` is keyed by ``game.p``, the stochasticity level --
    the same quantity the paper and the figure legends call *q*. Inside this function
    ``q = 1 - p`` denotes the complement, the probability that nothing was perturbed:

        R: p = P(chest is randomised)   q = P(displayed value is the true draw)
        T: p = P(move is flipped)       q = P(intended move executes)

    **R:** At every visit, expected reward at the shown cell is ``q * V + (1 - q) * E`` with
    ``E = exp_value`` (default 5 = mean of discrete uniform treasure 1…9). Down-tree motion is
    deterministic; subtree backups are ``… + max(child, child)``.

    **T:** Shown board values are treated as true payoffs; ``q`` enters only on *transitions*.
    Backups are ``V + maxₐ  E[continuation | intend a]`` with the two children weighted by ``q``
    and ``1 - q``. The logit contrast returned for the first choice is therefore
    ``(2 * q - 1) * (T[1,0] - T[1,1])``, not the undiscounted subtree difference (see the final
    ``v_left`` / ``v_right`` block in the **T** branch).

    **V:** not supported (no ``value_EV`` in the fitted model space).
    """
    value_params_ = {"exp_value": 5, 0:0, 0.125:0.125, 0.25:0.25, 0.375:0.375, 0.5:0.5, 0.75:0.75, 1:1}
    value_params_.update(value_params)
    
    if variant == "R":
        v_diff = []

        for condition in pov_array.conditions:
            q = 1 - value_params_[condition.item()]
            E = value_params_["exp_value"]
            
            pov_array_condition = pov_array.sel(conditions = condition)
            values = pov_array_condition.copy()
            
            table = xr.zeros_like(values).astype(float)
            
            #base case: T[-1, col] = qV[-1, col] + (1 - q)E
            table[dict(rows = max(table.rows))] = q * values.isel(rows = max(table.rows)) + (1 - q) * E

            for row in pov_array.rows.values[:-1][::-1]:
                for col in range(row + 1): 
                    # T[row, col] = qV[row, col] + (1 - q)E + max(T[row + 1, col], T[row + 1, col + 1])
                    table[dict(rows = row, cols = col)] = q * values.isel(rows = row, cols = col) + (1 - q) * E + table.isel(rows = row + 1, cols = [col, col + 1]).max("cols")

            # First split: deterministic subtrees (reliability has no move flips).
            v_diff_ = table.isel(rows=1, cols=0) - table.isel(rows=1, cols=1)
            v_diff.append(v_diff_)
            
        v_diff = xr.concat(v_diff, dim = "conditions")
        v_diff["conditions"] = pov_array.conditions
        
        return v_diff
    
    if variant == "T":
        v_diff = []
        for condition in pov_array.conditions:
            q = 1 - value_params_[condition.item()]

            pov_array_condition = pov_array.sel(conditions = condition)
            values = pov_array_condition.copy()

            table = xr.zeros_like(values).astype(float)

            # Terminal: realized payoffs (no display noise for controllability in this codebase).
            table[dict(rows = max(table.rows))] = values.isel(rows = max(table.rows))

            for row in pov_array.rows.values[:-1][::-1]:
                for col in range(row + 1): 
                    # Immediate payoff + best intended move, where each intention hits the
                    # left/right child Markov transition with probabilities q and (1 - q).
                    v_left = (table.isel(rows = row + 1, cols = [col, col + 1]) * [q, (1 - q)]).sum("cols")
                    v_right = (table.isel(rows = row + 1, cols = [col, col + 1]) * [(1 - q), q]).sum("cols")
                    v_max = xr.concat([v_left, v_right], dim = "cols").max("cols")
                    
                    table[dict(rows = row, cols = col)] = values.isel(rows = row, cols = col) + v_max
                    
            # First stochastic move: intend-left vs intend-right contrasts are (2q - 1)(A - B)
            # for optimal subtree totals A, B at (row=1, col=0) and (row=1, col=1).
            v_left = (table.isel(rows = 1, cols = [0, 1]) * [q, (1 - q)]).sum("cols")
            v_right = (table.isel(rows = 1, cols = [0, 1]) * [(1 - q), q]).sum("cols")
            v_diff_ = v_left - v_right
            v_diff.append(v_diff_)
            
        v_diff = xr.concat(v_diff, dim = "conditions")
        v_diff["conditions"] = pov_array.conditions
        return v_diff

    raise ValueError(f"value_EV supports variants 'R' and 'T' only; got {variant!r}")

def value_path(pov_array: xr.DataArray, variant = None, value_params = None):
    """Best-path value: backward-induct ``T[row, col] = V[row, col] + max(children)`` from the
    leaves, i.e. the total reward of the single best root-to-leaf path through each subtree.

    Returns the left-minus-right contrast ``T[1, 0] - T[1, 1]``. This difference is the decision
    variable: it is scaled by the inverse temperature and fed to the softmax, so the return is a
    left-vs-right contrast rather than an absolute subtree value. Ignores stochasticity
    (``variant`` and ``value_params`` are unused; see ``value_EV`` for the variant-aware model).
    """
    values = pov_array.copy()
    
    table = xr.zeros_like(values).astype(float)
    table[dict(rows = max(table.rows))] = values.isel(rows = max(table.rows))

    for row in pov_array.rows.values[:-1][::-1]:
        for col in range(row + 1): 
            # T[row, col] = V[row, col] + max(T[row + 1, col], T[row + 1, col + 1])
            table[dict(rows = row, cols = col)] = values.isel(rows = row, cols = col) + table.isel(rows = row + 1, cols = [col, col + 1]).max("cols")

    #decision variable for each game, trial is T[1, 0] - T[1, 1]
    v_left = table.isel(rows = 1, cols = 0)
    v_right = table.isel(rows = 1, cols = 1)
    v_diff = v_left - v_right
    
    return v_diff

def value_levelmean(pov_array: xr.DataArray, variant = None, value_params = None):
    """Level-mean value: average the cells within each row of a subtree, then sum those row means
    over rows -- so every depth contributes equally regardless of how many cells it holds.

    Left and right subtrees are the below-diagonal cells of ``rows[1:]`` at ``cols[:-1]`` and
    ``cols[1:]``; off-subtree cells are masked to NaN and skipped by the mean. Returns the
    left-minus-right contrast, which is the decision variable scaled by the inverse temperature
    and fed to the softmax, not an absolute subtree value.
    """
    values = pov_array.copy()
    left = values.isel(rows = values.rows[1:], cols = values.cols[:-1])
    right = values.isel(rows = values.rows[1:], cols = values.cols[1:])
    
    #set the above-diagonal values to np.nan
    left_subtree = left.where(left.rows >= left.cols)
    right_subtree = right.where(right.rows >= right.cols)

    v_left = left_subtree.mean("cols").sum("rows")
    v_right = right_subtree.mean("cols").sum("rows")
    
    return v_left - v_right

def value_sum(pov_array: xr.DataArray, variant = None, value_params = None):
    """Sum value: total of every (unmasked) cell in a subtree, so deeper levels dominate simply by
    having more cells.

    Subtrees are extracted as in ``value_levelmean``, with off-subtree cells masked to NaN.
    Returns the left-minus-right contrast, which is the decision variable scaled by the inverse
    temperature and fed to the softmax, not an absolute subtree value.
    """
    values = pov_array.copy()
    left = values.isel(rows = values.rows[1:], cols = values.cols[:-1])
    right = values.isel(rows = values.rows[1:], cols = values.cols[1:])

    #set the above-diagonal values to np.nan
    left_subtree = left.where(left.rows >= left.cols)
    right_subtree = right.where(right.rows >= right.cols)
    
    v_left = left_subtree.sum(["rows", "cols"])
    v_right = right_subtree.sum(["rows", "cols"])
    
    return v_left - v_right

def value_max(pov_array: xr.DataArray, variant = None, value_params = None):
    """Max value: the single largest (unmasked) cell anywhere in a subtree, ignoring whether that
    cell is actually reachable along a legal path.

    Subtrees are extracted as in ``value_levelmean``, with off-subtree cells masked to NaN.
    Returns the left-minus-right contrast, which is the decision variable scaled by the inverse
    temperature and fed to the softmax, not an absolute subtree value.
    """
    values = pov_array.copy()
    left = values.isel(rows = values.rows[1:], cols = values.cols[:-1])
    right = values.isel(rows = values.rows[1:], cols = values.cols[1:])

    #set the above-diagonal values to np.nan
    left_subtree = left.where(left.rows >= left.cols)
    right_subtree = right.where(right.rows >= right.cols)
    
    v_left = left_subtree.max(["rows", "cols"])
    v_right = right_subtree.max(["rows", "cols"])
    
    return v_left - v_right

