
# Calculate score for single quantile prediction
def evaluate_quantile(sq_pred, tau, sy_obs):
    l_score = None
    if sq_pred > sy_obs:
        l_score = 2 * (1 - tau) * (sq_pred - sy_obs)
    else:
        l_score = 2 * tau * (sy_obs - sq_pred)
    return l_score


# Calculate score for all quantile predictions
def evaluate_horizon(quantilepreds, obs):
    taus = [0.025, 0.25, 0.5, 0.75, 0.975]
    if len(quantilepreds) != 5:
        raise ValueError("The quantiles must consist of five elements.")
    total_sum = 0
    for q, tau in zip(quantilepreds, taus):
        l_score = evaluate_quantile(q, tau, obs)
        total_sum += l_score
    return total_sum
