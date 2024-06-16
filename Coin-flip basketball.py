# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Coin-flip basketball
#
# Sorry for the filler subsection headings, I had to add them to make collapsing and expanding cells work.

# ## Main prose
#
# For each $n$, let $a(n)$ denote the largest integer $k$ such that
# $$
# \sum_{0 \leq i < k} \binom{n}{i} \leq 0.1 \cdot 2^n.
# $$
#
# In a game of coin-flip basketball with $n$ coins, the home team has a $\geq 90\%$ chance of winning at least $a(n)$ flips, and has a $< 90\%$ chance of winning at least $a(n) + 1$ flips.

# ### Computing $a$

# Looking at two consecutive rows of Pascal's triangle, we obtain the identity
# $$
# \sum_{0 \leq i < k} \binom{n+1}{i} =
# 2 \left( \sum_{0 \leq i < k} \binom{n}{i} \right) - \binom{n}{k}.
# $$
# This allows us to efficiently compute $a(n)$ for all $n \leq n_{\max}$, for a given $n_{\max}$, as the sums involved for a new value of $n$ can reuse work from smaller values of $n$.

# +
import math

def binom(n, k):
    if k < 0:
        return 0
    return math.comb(n, k)

def compute_a(n_max, *, threshold_frac):
    a = [0]
    n = 0
    threshold = threshold_frac
    k = 0
    s = 0

    while n < n_max:
        threshold *= 2
        s = 2 * s - binom(n, k-1)
        n += 1
        while True:
            d = binom(n, k)
            if s + d > threshold:
                break
            k += 1
            s += d
        a.append(k)

    return a


# -

# #### Sanity check

the_n_max = 101
the_threshold_frac = 0.1
a = compute_a(the_n_max, threshold_frac=the_threshold_frac)
assert len(a) == the_n_max + 1
for (n, k) in enumerate(a):
    left = sum(binom(n, j) for j in range(k))
    right = left + binom(n, k)
    threshold = the_threshold_frac * 2**n
    assert left <= threshold and right > threshold, (left, right, threshold)


# ## Back to prose
#
# If the home team has won $w$ coin flips and lost $\ell$ coin flips so far, then the home team has a $\geq 90\%$ chance of winning if and only if
# $$
# 51 - w \leq a(101 - w - \ell),
# $$
# or rearranging,
# $$
# w \geq 51 - a(101 - w - \ell).
# $$

# Since $a(n)$ is a non-decreasing function of $n$, we have that for any fixed value of $w$, there is some $\ell_{\text{NR}}(w)$ such that the above holds if and only if $0 \leq \ell < \ell_{\text{NR}}(w)$.  The "NR" stands for "no report," as for any $w, \ell < 51$, a game state with $w$ wins and $\ell$ losses will _not_ cause an ESPN report if and only if $\ell \geq \ell_{\text{NR}}(w)$.

# ### Computing $\ell_{\text{NR}}$

def compute_ℓ_nr(a):
    n_max = len(a) - 1
    n_maj = n_max // 2 + 1
    return [next(filter(lambda ℓ: w < n_maj - a[n_max - w - ℓ],
                        range(n_max - w + 1)),
                 n_maj)
            for w in range(n_maj + 1)]


a = compute_a(101, threshold_frac=0.1)
ℓ_nr = compute_ℓ_nr(a)

# ## Back to prose

# Now, we have
# $$
# \begin{aligned}
# \Pr\left[ \text{win} \ \middle|\ \text{ESPN report} \right]
# &=
# \frac{
#   \Pr\left[
#     \text{ESPN report \& win}
#   \right]
# }{
#   \Pr\left[ \text{ESPN report} \right]
# } \\[1em]
# &=
# \frac{
#   \Pr[\text{win}]
#   - \Pr\left[
#     \text{no ESPN report \& win}
#   \right]
# }{
#   1 - \Pr\left[ \text{no ESPN report \& win} \right] - \Pr\left[ \text{no ESPN report \& lose} \right]
# }.
# \end{aligned}
# $$
#
# (Here we are treating "ESPN report" as a shorthand for "there was at least one point before the end of the game at which the home team had a $\geq 90\%$ chance of winning."  That is, conditioned on a given sequence of outcomes, the ESPN report is not a random event; it happens if and only if there exists such a point.)
#
# Let $b(w, \ell)$ denote the number of possible sequences of outcomes for the first $w + \ell$ coin flip results such that:
# * The home team wins $w$ flips and loses $\ell$ flips
# * The sequence does not end in a loss (in other words, either the sequence ends in a win or $w = \ell = 0$)
# * The conditions for an ESPN report do not happen during the first $w + \ell$ flips.
#
# Then, we have
# $$
# \Pr\left[ \text{no ESPN report \& win} \right]
# =
# \sum_{0 \leq \ell < 51}
#   \frac{b(51, \ell)}{2^{51 + \ell}}
# $$
#
# and similarly,
# $$
# \Pr\left[ \text{no ESPN report \& lose} \right]
# =
# \sum_{\substack{0 \leq w < 51 \\
#                 \ell_{\text{NR}}(w) \leq \ell_1 < 51}}
#   \frac{b(w, \ell_1)}{2^{w + 51}}.
# $$
# In the latter equation, the summand is the probability that:
# * the home team loses overall, with $w < 51$ flips won at the end of the game;
# * there is no ESPN report;
# * the home team loses $\ell_1$ flips before their last win (or if $w=0$, then $\ell_1=0$), then loses the remaining $51 - \ell_1$ flips.

# ### Computing $b$

# There are exponentially many paths to count, but we can compute $b$ in quadratic time using the recurrence
# $$
# b(w + 1,\ \ell) = \sum_{j = \ell_{\text{NR}}(w)}^{j=\ell} b(w, j) \qquad (\ell \geq \ell_{\text{NR}}(w+1))
# $$
# which holds for each $w < 51$ and each $\ell \geq \ell_{\text{NR}}(w+1)$.  In prose, the recurrence says that the way to have $\ell$ losses (and no ESPN report) when you win your $(w+1)$st flip, is to have $j$ losses (and no ESPN report) when you win your $w$th flip for some $j \geq \ell_{\text{NR}}(w)$, then lose $\ell - j$ flips, then win your $(w+1)$st flip.

# +
import numpy as np
from sympy import Rational
R_zero = Rational(0)
R_one = Rational(1)
R_two = Rational(2)

def compute_b(ℓ_nr):
    n_maj = len(ℓ_nr) - 1
    b = np.tile(R_zero, (n_maj + 1, n_maj))
    b[0, 0] = R_one
    for w in range(1, n_maj + 1):
        b[w, :] = np.cumsum(b[w-1, :])
        # For w = n_maj, ESPN report never happens no matter how
        # many many losses there have been because the game is over
        if w != n_maj:
            b[w, :ℓ_nr[w]] = R_zero
    return b


# -

# ## Putting it all together

# +
import collections
Probs = collections.namedtuple("Probs", ["p_win_given_report",
                                         "p_no_report_and_win",
                                         "p_no_report_and_lose"])

def compute_probs(num_coins, *, threshold_frac):
    a = compute_a(num_coins, threshold_frac=threshold_frac)
    ℓ_nr = compute_ℓ_nr(a)
    b = compute_b(ℓ_nr)

    n_maj = len(ℓ_nr) - 1
    p_win = R_one / 2
    p_no_report_and_win = (np.sum(b[n_maj, :]
                           / R_two**(n_maj + np.arange(n_maj))))
    p_no_report_and_lose = (np.sum(b[:-1, :]
                            / np.reshape(R_two**(n_maj + np.arange(n_maj)),
                                         (-1, 1))))
    p_win_given_report = ((p_win - p_no_report_and_win)
                          / (1 - p_no_report_and_win - p_no_report_and_lose))

    return Probs(p_win_given_report=p_win_given_report,
                 p_no_report_and_win=p_no_report_and_win,
                 p_no_report_and_lose=p_no_report_and_lose)


# -

# ### The small version of the problem

compute_probs(5, threshold_frac=0.25)

# ### The bigger version

probs_big = compute_probs(101, threshold_frac=0.1)
probs_big

Probs(*map(float, probs_big))

# ### I guess there's a bug
#
# The above code agrees with the official solution for the small version of the problem but not the big one.  I guess there's a bug somewhere in here.
