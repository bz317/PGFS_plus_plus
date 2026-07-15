"""Reward functions for molecule design."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from rdkit import Chem, DataStructs
from rdkit.Chem import QED, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko_scaffold(smiles: str | None, *, generic: bool = False) -> str | None:
    """Return the Bemis-Murcko scaffold SMILES of ``smiles``.

    Returns ``None`` for invalid molecules and for acyclic molecules (whose
    Murcko scaffold is empty), so callers can skip scaffold bucketing in those
    cases. With ``generic=True`` atom/bond types are flattened (graph framework).
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        smi = Chem.MolToSmiles(scaffold)
    except Exception:
        return None
    return smi or None


_CACHE_MISS = object()


class _DiversityController:
    """Base class for episode-terminal diversity controllers.

    A controller inspects each *generated* (episode-terminal) molecule and may
    rewrite the episode reward to discourage over-represented chemotypes. The
    three concrete controllers (scaffold filter, soft exponential penalty, avoid
    list) are mutually exclusive — at most one is active per run, selected by
    :func:`build_diversity_controller`.

    Shared machinery: QED, Bemis-Murcko scaffold and Morgan-fingerprint lookups
    are memoised by input SMILES so the collapse case (the same magnet molecule
    recurring on nearly every episode) costs a dict lookup rather than fresh RDKit
    work. Caches are capped at ``cache_size`` entries; past the cap values are
    computed without being stored (correct, just uncached). All controller work
    runs once per *episode* (not per step).
    """

    name = "none"
    # Whether the controller shapes reward at *each reaction step* (via
    # :meth:`step_bonus`) rather than once at the episode terminal (via
    # :meth:`adjust_episode_reward`). Per-step controllers leave
    # ``adjust_episode_reward`` a no-op and vice versa.
    per_step = False

    def __init__(
        self,
        *,
        minscore: float = 0.0,
        generic_scaffold: bool = False,
        cache_size: int = 200_000,
    ) -> None:
        self.minscore = float(minscore)
        self.generic_scaffold = bool(generic_scaffold)
        self.cache_size = int(cache_size)
        self._qed_cache: dict[str, float] = {}
        self._scaffold_cache: dict[str, str | None] = {}
        self._fp_cache: dict[str, object] = {}

    def _qed_cached(self, smiles: str) -> float:
        cached = self._qed_cache.get(smiles, _CACHE_MISS)
        if cached is not _CACHE_MISS:
            return cached  # type: ignore[return-value]
        value = qed(smiles)
        if len(self._qed_cache) < self.cache_size:
            self._qed_cache[smiles] = value
        return value

    def _scaffold_cached(self, smiles: str) -> str | None:
        cached = self._scaffold_cache.get(smiles, _CACHE_MISS)
        if cached is not _CACHE_MISS:
            return cached  # type: ignore[return-value]
        value = murcko_scaffold(smiles, generic=self.generic_scaffold)
        if len(self._scaffold_cache) < self.cache_size:
            self._scaffold_cache[smiles] = value
        return value

    def _fp_cached(self, smiles: str, *, radius: int, fp_size: int):
        cached = self._fp_cache.get(smiles, _CACHE_MISS)
        if cached is not _CACHE_MISS:
            return cached
        value = _morgan_bitvect(smiles, radius=radius, fp_size=fp_size)
        if len(self._fp_cache) < self.cache_size:
            self._fp_cache[smiles] = value
        return value

    def adjust_episode_reward(
        self,
        smiles: str | None,
        ep_reward: float,
        *,
        start_smiles: str | None = None,
    ) -> tuple[float, bool]:
        """Return ``(new_episode_reward, modified)`` for a terminal molecule.

        ``smiles`` is the episode-terminal molecule; ``start_smiles`` is the
        molecule the episode *started* from (needed only by controllers that
        shape the reward by the input→output relationship — others ignore it).
        When ``modified`` is ``False`` the caller keeps the original reward.
        Subclasses override this.
        """
        return ep_reward, False

    def step_bonus(
        self,
        product_smiles: str | None,
        *,
        start_smiles: str | None = None,
        prev_smiles: str | None = None,
        step_delta_qed: float = 0.0,
    ) -> float:
        """Per-step reward bonus for a single successful reaction.

        Called by the trainer after each reaction that yields ``product_smiles``
        (transforming ``prev_smiles``) within an episode started from
        ``start_smiles``; ``step_delta_qed`` is that step's QED change. Only
        controllers with ``per_step == True`` return a non-zero bonus; the
        default is a no-op so episode-terminal controllers are unaffected.
        """
        return 0.0

    def metrics(self) -> dict[str, float]:
        """Scalar metrics for logging (keys are un-prefixed)."""
        return {}


class ScaffoldDiversityFilter(_DiversityController):
    """REINVENT-style scaffold memory that neutralises over-represented scaffolds.

    Each terminal molecule scoring at or above ``minscore`` is bucketed by its
    Bemis-Murcko scaffold. Once a scaffold bucket already holds ``bucket_size``
    molecules, any further molecule with that scaffold has its episode reward
    overridden to ``saturated_reward`` (a *hard* penalty), removing the incentive
    to keep re-generating the same chemotype.
    """

    name = "diversity_filter"

    def __init__(
        self,
        *,
        bucket_size: int = 25,
        minscore: float = 0.4,
        saturated_reward: float = 0.0,
        generic_scaffold: bool = False,
        cache_size: int = 200_000,
    ) -> None:
        super().__init__(
            minscore=minscore,
            generic_scaffold=generic_scaffold,
            cache_size=cache_size,
        )
        self.bucket_size = int(bucket_size)
        self.saturated_reward = float(saturated_reward)
        self._buckets: dict[str, int] = {}
        self.saturated_events = 0

    @property
    def num_scaffolds(self) -> int:
        return len(self._buckets)

    def adjust_episode_reward(
        self,
        smiles: str | None,
        ep_reward: float,
        *,
        start_smiles: str | None = None,
    ) -> tuple[float, bool]:
        if not smiles:
            return ep_reward, False
        if self._qed_cached(smiles) < self.minscore:
            return ep_reward, False
        scaffold = self._scaffold_cached(smiles)
        if scaffold is None:
            return ep_reward, False
        count = self._buckets.get(scaffold, 0)
        if count >= self.bucket_size:
            self.saturated_events += 1
            return self.saturated_reward, True
        self._buckets[scaffold] = count + 1
        return ep_reward, False

    def metrics(self) -> dict[str, float]:
        return {
            "df_num_scaffolds": float(self.num_scaffolds),
            "df_saturated_events_cum": float(self.saturated_events),
        }


class SoftExponentialPenalty(_DiversityController):
    """SyntheMol-style soft diversity penalty keyed on Bemis-Murcko scaffold.

    Instead of the hard cutoff of :class:`ScaffoldDiversityFilter`, the positive
    episode reward is scaled by ``exp(-count / decay)``, where ``count`` is how
    many *prior* terminal molecules shared this scaffold. The first occurrence of
    a scaffold keeps its full reward (``count == 0`` → factor ``1``); each repeat
    shrinks the reward smoothly (SyntheMol divides its MCTS score by the analogous
    ``exp((n - 1) / 100)`` building-block term). Non-positive rewards are left
    unchanged so the penalty only erodes *gains* from re-using a chemotype.
    """

    name = "soft_exponential_penalty"

    def __init__(
        self,
        *,
        decay: float = 25.0,
        minscore: float = 0.4,
        generic_scaffold: bool = False,
        cache_size: int = 200_000,
    ) -> None:
        super().__init__(
            minscore=minscore,
            generic_scaffold=generic_scaffold,
            cache_size=cache_size,
        )
        self.decay = max(1e-6, float(decay))
        self._buckets: dict[str, int] = {}
        self.penalised_events = 0

    @property
    def num_scaffolds(self) -> int:
        return len(self._buckets)

    def adjust_episode_reward(
        self,
        smiles: str | None,
        ep_reward: float,
        *,
        start_smiles: str | None = None,
    ) -> tuple[float, bool]:
        if not smiles:
            return ep_reward, False
        if self._qed_cached(smiles) < self.minscore:
            return ep_reward, False
        scaffold = self._scaffold_cached(smiles)
        if scaffold is None:
            return ep_reward, False
        count = self._buckets.get(scaffold, 0)
        self._buckets[scaffold] = count + 1
        factor = math.exp(-count / self.decay)
        if factor >= 1.0 or ep_reward <= 0.0:
            return ep_reward, False
        self.penalised_events += 1
        return ep_reward * factor, True

    def metrics(self) -> dict[str, float]:
        return {
            "soft_num_scaffolds": float(self.num_scaffolds),
            "soft_penalised_events_cum": float(self.penalised_events),
        }


class AvoidListPenalty(_DiversityController):
    """Penalise terminal molecules by similarity to a fixed, pre-defined avoid list.

    A reference set of SMILES is loaded once into Morgan fingerprints. There is no
    tolerance band: the episode reward is reduced in direct proportion to the
    *maximum* Tanimoto similarity between the terminal molecule and any avoided
    molecule::

        ``new = ep_reward - penalty * max_sim``

    so a molecule identical to an avoided one (sim ``1``) loses the full ``penalty``
    magnitude, a half-similar molecule loses ``penalty / 2``, and only a molecule
    with zero overlap escapes untouched. Raising ``power`` (>1) sharpens the
    penalty toward near-duplicates: ``penalty * max_sim ** power``.
    """

    name = "avoid_list"

    def __init__(
        self,
        *,
        avoid_file: str,
        penalty: float = 1.0,
        power: float = 1.0,
        minscore: float = 0.0,
        radius: int = 2,
        fp_size: int = 1024,
        cache_size: int = 200_000,
    ) -> None:
        super().__init__(minscore=minscore, cache_size=cache_size)
        self.penalty = float(penalty)
        self.power = float(power)
        self.radius = int(radius)
        self.fp_size = int(fp_size)
        self.avoid_file = str(avoid_file)
        smiles_list = load_avoid_smiles(self.avoid_file)
        self._avoid_fps = []
        for smi in smiles_list:
            fp = _morgan_bitvect(smi, radius=self.radius, fp_size=self.fp_size)
            if fp is not None:
                self._avoid_fps.append(fp)
        if not self._avoid_fps:
            raise ValueError(
                f"avoid_list file {self.avoid_file!r} produced no valid molecules"
            )
        self.hits = 0
        self._sim_sum = 0.0

    @property
    def list_size(self) -> int:
        return len(self._avoid_fps)

    def adjust_episode_reward(
        self,
        smiles: str | None,
        ep_reward: float,
        *,
        start_smiles: str | None = None,
    ) -> tuple[float, bool]:
        if not smiles:
            return ep_reward, False
        if self.minscore > 0.0 and self._qed_cached(smiles) < self.minscore:
            return ep_reward, False
        fp = self._fp_cached(smiles, radius=self.radius, fp_size=self.fp_size)
        if fp is None:
            return ep_reward, False
        sims = DataStructs.BulkTanimotoSimilarity(fp, self._avoid_fps)
        max_sim = max(sims) if sims else 0.0
        if max_sim <= 0.0:
            return ep_reward, False
        self.hits += 1
        self._sim_sum += max_sim
        weight = max_sim ** self.power if self.power != 1.0 else max_sim
        return ep_reward - self.penalty * weight, True

    def metrics(self) -> dict[str, float]:
        return {
            "avoid_list_size": float(self.list_size),
            "avoid_hits_cum": float(self.hits),
            "avoid_mean_max_sim": float(self._sim_sum / self.hits) if self.hits else 0.0,
        }


IN_OUT_SIM_MODES = ("per_episode_in_out", "per_step_to_start", "per_step_to_prev")


class InOutSimilarityReward(_DiversityController):
    """Reward molecules for staying *similar to a per-episode reference* molecule.

    Unlike :class:`AvoidListPenalty` (which compares every output to a single
    fixed external set, so it acts as a near-uniform tax that merely relocates
    the collapse magnet), this controller compares each output to a reference
    *specific to the trajectory*. The reference therefore differs for every
    episode, so it acts as a *per-start anchor* rather than a flat penalty.

    Mechanism / why it helps diversity: a high-QED "magnet" molecule ``M`` is
    cheap to reach from starts near ``M`` but expensive (low similarity) from
    starts far away. Rewarding the input→output relationship thus tethers each
    output to its own (diverse) input, so the diversity of the start set
    propagates to the outputs instead of all starts funnelling into ``M``. This
    is REINVENT4 Mol2Mol's "similar-but-improved" mechanism and a soft, tunable
    version of the short-``max_episode_len`` constraint that empirically produced
    high diversity.

    Three ``sim_mode`` variants select *what* is anchored and *when*:

    - ``per_episode_in_out`` (terminal): one bonus added at the episode end for
      ``sim(final, start)`` — rewards net displacement from the input, says
      nothing about the path. Cheapest, interferes least with QED.
    - ``per_step_to_start`` (per step): each reaction is rewarded for
      ``sim(product, start)`` — a dense signal that penalises cumulative drift
      from the origin. Stronger anchor, but charges the *trajectory length* in
      chemical space, so it fights QED harder.
    - ``per_step_to_prev`` (per step): each reaction is rewarded for
      ``sim(product, previous)`` — encourages *small local edits*. Smoothest, but
      permits a slow random walk away from the start (weaker for diversity).

    Shaping (a bonus *added* to the reward), with ``sim`` the relevant Tanimoto
    similarity::

        bonus = weight * (min(sim, sim_cap)) ** power           (sim >= sim_floor)

    For the per-step modes the bonus is additionally divided by
    ``max_episode_len`` when ``normalize_by_len`` is set, so the summed per-step
    bonus over a full episode is on the same scale as a single terminal bonus
    (otherwise a long chain would accumulate ~L times more shaping reward).
    ``sim_cap`` (< 1.0) stops over-rewarding a near-identical copy; the
    ``require_positive`` gate (only shape when QED improved — the *episode* gain
    for the terminal mode, the *step* gain for per-step modes) removes the
    degenerate "do nothing so sim == 1" exploit; ``sim_floor`` zeroes the bonus
    once a molecule has drifted too far to count as a local edit.

    ``outside_sim_floor_cap`` (opt-in, terminal mode only): a soft trust-region
    "wall". Below ``sim_floor`` the bonus is already zero, but the episode still
    keeps its full QED gain, so escaping the per-start ball toward a shared
    high-QED attractor is free. Setting this clamps the terminal reward of an
    outside-ball episode to ``min(ep_reward, cap)`` — removing only the excess of
    large attractor jumps while leaving small honest far edits untouched, so a
    local in-ball improvement can out-earn the distant magnet. ``None`` (default)
    disables it entirely, preserving prior behavior exactly.
    """

    name = "in_out_sim"

    def __init__(
        self,
        *,
        sim_mode: str = "per_episode_in_out",
        weight: float = 0.3,
        sim_cap: float = 1.0,
        sim_floor: float = 0.0,
        power: float = 1.0,
        require_positive: bool = True,
        normalize_by_len: bool = True,
        max_episode_len: int = 5,
        minscore: float = 0.0,
        radius: int = 2,
        fp_size: int = 1024,
        reaction_level_weights: "list[float] | tuple[float, ...] | None" = None,
        reaction_level_scale: "list[float] | tuple[float, ...] | None" = None,
        outside_sim_floor_cap: "float | None" = None,
        cache_size: int = 200_000,
    ) -> None:
        super().__init__(minscore=minscore, cache_size=cache_size)
        if sim_mode not in IN_OUT_SIM_MODES:
            raise ValueError(
                f"in_out_sim `sim_mode` must be one of {IN_OUT_SIM_MODES}, "
                f"got {sim_mode!r}"
            )
        self.sim_mode = str(sim_mode)
        self.per_step = self.sim_mode in ("per_step_to_start", "per_step_to_prev")
        self.weight = float(weight)
        self.sim_cap = float(sim_cap)
        self.sim_floor = float(sim_floor)
        self.power = float(power)
        self.require_positive = bool(require_positive)
        self.normalize_by_len = bool(normalize_by_len)
        self._step_norm = (
            1.0 / max(1, int(max_episode_len)) if self.normalize_by_len else 1.0
        )
        self.radius = int(radius)
        self.fp_size = int(fp_size)
        # Optional per-reaction-count coefficient schedule for the terminal bonus,
        # indexed by reaction count: ``schedule[n]`` is the coefficient for an
        # episode that ended with exactly ``n`` reactions. Index 0 is the
        # 0-reaction (no-op) case and should be 0.0 so a pure pass-through earns
        # nothing. Counts beyond the list length are clamped to the last entry.
        # Two MUTUALLY EXCLUSIVE forms are supported (set at most one), both for
        # sim_mode=per_episode_in_out only -- a per-step bonus cannot know the
        # final reaction count:
        #   - reaction_level_weights -> ADDITIVE bonus = weight(n) * sim ** power
        #   - reaction_level_scale   -> MULTIPLICATIVE bonus =
        #         max(0, delta_qed) * scale(n) * sim ** power
        #     so the bonus is proportional to the episode's QED gain; the
        #     diversity/length term only AMPLIFIES QED-improving episodes and
        #     vanishes when QED did not improve (a soft positivity gate).
        # When neither is set the single ``weight`` is used for any count >= 1.
        if reaction_level_weights is not None and reaction_level_scale is not None:
            raise ValueError(
                "in_out_sim: set only ONE of `reaction_level_weights` (additive) "
                "or `reaction_level_scale` (multiplicative), not both."
            )

        def _parse_levels(vals, key):
            if self.per_step:
                raise ValueError(
                    f"in_out_sim `{key}` is only valid for "
                    "sim_mode=per_episode_in_out (the per-step modes cannot know "
                    "the final reaction count when each step's bonus is paid)."
                )
            level = [float(w) for w in vals]
            if not level:
                raise ValueError(f"in_out_sim `{key}` must be a non-empty list.")
            return level

        self.reaction_level_weights: list[float] | None = (
            _parse_levels(reaction_level_weights, "reaction_level_weights")
            if reaction_level_weights is not None
            else None
        )
        self.reaction_level_scale: list[float] | None = (
            _parse_levels(reaction_level_scale, "reaction_level_scale")
            if reaction_level_scale is not None
            else None
        )
        # In multiplicative mode the terminal bonus is scaled by the episode's
        # (clamped) QED delta so it only ever rewards QED-improving episodes.
        self.scale_by_delta_qed = self.reaction_level_scale is not None
        # --- Optional trust-region "cap" outside the ball (opt-in; default off) -
        # Without this, an episode whose final molecule drifted OUTSIDE the ball
        # (``sim < sim_floor``) simply earns no bonus but keeps its full QED gain
        # -- so marching to a shared high-QED attractor is free, and the policy
        # can collapse onto it. When ``outside_sim_floor_cap`` is set, the
        # *terminal* reward of such an outside-ball episode is clamped to at most
        # the cap: ``ep_reward = min(ep_reward, cap)``. This removes only the
        # EXCESS of large jumps (the magnet's big dQED) while leaving small
        # honest far edits untouched, so a local in-ball improvement (whose dQED
        # may exceed the cap) can out-earn the distant attractor. ``min`` never
        # raises a reward, so QED-neutral/negative episodes are unaffected. Left
        # ``None`` (the default) the controller is byte-for-byte identical to
        # before, preserving every existing config. Only valid for the terminal
        # ``per_episode_in_out`` mode (the per-step modes do not touch the
        # terminal reward).
        if outside_sim_floor_cap is None:
            self.outside_sim_floor_cap: float | None = None
        else:
            if self.per_step:
                raise ValueError(
                    "in_out_sim `outside_sim_floor_cap` is only valid for "
                    "sim_mode=per_episode_in_out (the terminal mode); the "
                    "per-step modes do not adjust the terminal QED reward."
                )
            self.outside_sim_floor_cap = float(outside_sim_floor_cap)
        self.shaped_events = 0
        self._sim_sum = 0.0
        self._bonus_sum = 0.0
        # Cumulative diagnostics for the outside-ball cap.
        self._outside_capped_events = 0
        self._qed_removed_sum = 0.0

    def _level_weight(self, n_reactions: int) -> float:
        """Per-reaction-count coefficient for an episode of ``n_reactions``.

        Uses the active schedule -- ``reaction_level_scale`` (multiplicative mode)
        if set, else ``reaction_level_weights`` (additive mode) -- indexed directly
        by reaction count (``schedule[n]``, so index 0 is the 0-reaction case),
        clamped to the last entry for counts beyond the list. With no schedule:
        the constant ``weight`` for any count >= 1, 0.0 for a zero-reaction no-op.
        """
        n = int(n_reactions)
        active = (
            self.reaction_level_scale
            if self.reaction_level_scale is not None
            else self.reaction_level_weights
        )
        if active is None:
            return self.weight if n >= 1 else 0.0
        idx = min(max(n, 0), len(active) - 1)
        return active[idx]

    def _bonus_from_sim(
        self, sim: float, *, per_step: bool, weight: float | None = None
    ) -> float:
        if sim < self.sim_floor:
            return 0.0
        capped = min(sim, self.sim_cap)
        weighted = capped ** self.power if self.power != 1.0 else capped
        w = self.weight if weight is None else float(weight)
        bonus = w * weighted
        if per_step:
            bonus *= self._step_norm
        return bonus

    def _tanimoto(self, a: str | None, b: str | None) -> float | None:
        if not a or not b:
            return None
        fa = self._fp_cached(a, radius=self.radius, fp_size=self.fp_size)
        fb = self._fp_cached(b, radius=self.radius, fp_size=self.fp_size)
        if fa is None or fb is None:
            return None
        return float(DataStructs.TanimotoSimilarity(fa, fb))

    def adjust_episode_reward(
        self,
        smiles: str | None,
        ep_reward: float,
        *,
        start_smiles: str | None = None,
        n_reactions: int = 1,
    ) -> tuple[float, bool]:
        # Per-step modes do all their shaping in ``step_bonus``; leave the
        # terminal reward untouched to avoid double-counting.
        if self.sim_mode != "per_episode_in_out":
            return ep_reward, False
        if not smiles or not start_smiles:
            return ep_reward, False
        # Per-reaction-count weight (0 reactions => 0 => never pay for a no-op).
        level_w = self._level_weight(n_reactions)
        if level_w == 0.0:
            return ep_reward, False
        if self.require_positive and ep_reward <= 0.0:
            return ep_reward, False
        if self.minscore > 0.0 and self._qed_cached(smiles) < self.minscore:
            return ep_reward, False
        sim = self._tanimoto(smiles, start_smiles)
        if sim is None:
            return ep_reward, False
        # Trust-region cap: outside the ball, limit the QED reward to ``cap`` so a
        # big jump to a shared high-QED attractor cannot out-earn local in-ball
        # improvement. The bonus is zero below the floor anyway, so this fully
        # handles the outside-ball case. ``min`` only ever lowers the reward.
        if self.outside_sim_floor_cap is not None and sim < self.sim_floor:
            capped = min(ep_reward, self.outside_sim_floor_cap)
            if capped < ep_reward:
                self._outside_capped_events += 1
                self._qed_removed_sum += ep_reward - capped
                return capped, True
            return ep_reward, False
        bonus = self._bonus_from_sim(sim, per_step=False, weight=level_w)
        if self.scale_by_delta_qed:
            # Multiplicative mode: scale the (similarity * level) bonus by the
            # episode's QED gain so it only rewards QED-improving episodes and
            # amplifies them by the diversity/length factor. Clamp at 0 so a
            # QED-neutral/negative episode earns nothing (soft positivity gate).
            bonus *= max(0.0, float(ep_reward))
        if bonus == 0.0:
            return ep_reward, False
        self.shaped_events += 1
        self._sim_sum += sim
        self._bonus_sum += bonus
        return ep_reward + bonus, True

    def step_bonus(
        self,
        product_smiles: str | None,
        *,
        start_smiles: str | None = None,
        prev_smiles: str | None = None,
        step_delta_qed: float = 0.0,
    ) -> float:
        if not self.per_step or not product_smiles:
            return 0.0
        if self.require_positive and step_delta_qed <= 0.0:
            return 0.0
        if self.minscore > 0.0 and self._qed_cached(product_smiles) < self.minscore:
            return 0.0
        reference = (
            start_smiles if self.sim_mode == "per_step_to_start" else prev_smiles
        )
        sim = self._tanimoto(product_smiles, reference)
        if sim is None:
            return 0.0
        bonus = self._bonus_from_sim(sim, per_step=True)
        if bonus == 0.0:
            return 0.0
        self.shaped_events += 1
        self._sim_sum += sim
        self._bonus_sum += bonus
        return bonus

    def metrics(self) -> dict[str, float]:
        return {
            "inout_shaped_events_cum": float(self.shaped_events),
            "inout_mean_sim": (
                float(self._sim_sum / self.shaped_events) if self.shaped_events else 0.0
            ),
            "inout_mean_bonus": (
                float(self._bonus_sum / self.shaped_events)
                if self.shaped_events
                else 0.0
            ),
            "inout_outside_capped_events_cum": float(self._outside_capped_events),
            "inout_mean_qed_removed": (
                float(self._qed_removed_sum / self._outside_capped_events)
                if self._outside_capped_events
                else 0.0
            ),
        }


def _morgan_bitvect(smiles: str | None, *, radius: int = 2, fp_size: int = 1024):
    """Return an RDKit ExplicitBitVect Morgan fingerprint, or ``None`` if invalid."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size).GetFingerprint(mol)


def load_avoid_smiles(path: str) -> list[str]:
    """Load a list of SMILES from ``path`` (.smi/.txt/.csv/.pkl).

    For text formats the first whitespace/comma-delimited token of each non-empty,
    non-``#`` line is taken. For ``.pkl`` a list of SMILES or a dict (keys used as
    SMILES) is accepted.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"avoid_list file not found: {path}")
    suffix = p.suffix.lower()
    if suffix == ".pkl":
        import pickle

        with p.open("rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, dict):
            return [str(k) for k in obj.keys()]
        return [str(x) for x in obj]
    smiles: list[str] = []
    with p.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.replace(",", " ").split()[0]
            if token and token.lower() != "smiles":
                smiles.append(token)
    return smiles


def build_diversity_controller(
    config: dict,
    *,
    resolve_path_fn: Callable[[str], str] | None = None,
    max_episode_len: int = 5,
) -> tuple[_DiversityController | None, str, float]:
    """Build the single active diversity controller from ``config``.

    The four modes — ``diversity_filter``, ``soft_exponential_penalty``,
    ``avoid_list`` and ``in_out_sim`` — are mutually exclusive: at most one may
    have ``enabled: true``. Returns ``(controller_or_None, mode_name,
    after_mean_reactions)``. The ``after_mean_reactions`` gate is read from the
    active block so the controller only acts once the policy is reacting enough
    (see the trainer's gate).
    """
    blocks = {
        "diversity_filter": dict(config.get("diversity_filter") or {}),
        "soft_exponential_penalty": dict(config.get("soft_exponential_penalty") or {}),
        "avoid_list": dict(config.get("avoid_list") or {}),
        "in_out_sim": dict(config.get("in_out_sim") or {}),
    }
    enabled = [name for name, blk in blocks.items() if bool(blk.get("enabled", False))]
    if len(enabled) > 1:
        raise ValueError(
            "At most one diversity mode may be enabled, but multiple were: "
            f"{enabled}. Set `enabled: false` on all but one."
        )
    if not enabled:
        return None, "none", 0.0

    name = enabled[0]
    blk = blocks[name]
    after = float(blk.get("after_mean_reactions", 0.0))

    if name == "diversity_filter":
        controller: _DiversityController = ScaffoldDiversityFilter(
            bucket_size=int(blk.get("bucket_size", 25)),
            minscore=float(blk.get("minscore", 0.4)),
            saturated_reward=float(blk.get("penalty", 0.0)),
            generic_scaffold=bool(blk.get("generic_scaffold", False)),
        )
    elif name == "soft_exponential_penalty":
        controller = SoftExponentialPenalty(
            decay=float(blk.get("decay", 25.0)),
            minscore=float(blk.get("minscore", 0.4)),
            generic_scaffold=bool(blk.get("generic_scaffold", False)),
        )
    elif name == "avoid_list":
        avoid_file = blk.get("avoid_file")
        if not avoid_file:
            raise ValueError("avoid_list mode requires an `avoid_file` path")
        resolved = (
            resolve_path_fn(str(avoid_file))
            if resolve_path_fn is not None
            else str(avoid_file)
        )
        controller = AvoidListPenalty(
            avoid_file=resolved,
            penalty=float(blk.get("penalty", 1.0)),
            power=float(blk.get("power", 1.0)),
            minscore=float(blk.get("minscore", 0.0)),
            radius=int(blk.get("fp_radius", 2)),
            fp_size=int(blk.get("fp_size", 1024)),
        )
    else:  # in_out_sim
        controller = InOutSimilarityReward(
            sim_mode=str(blk.get("sim_mode", "per_episode_in_out")),
            weight=float(blk.get("weight", 0.3)),
            sim_cap=float(blk.get("sim_cap", 1.0)),
            sim_floor=float(blk.get("sim_floor", 0.0)),
            power=float(blk.get("power", 1.0)),
            require_positive=bool(blk.get("require_positive", True)),
            normalize_by_len=bool(blk.get("normalize_by_len", True)),
            max_episode_len=int(max_episode_len),
            minscore=float(blk.get("minscore", 0.0)),
            radius=int(blk.get("fp_radius", 2)),
            fp_size=int(blk.get("fp_size", 1024)),
            reaction_level_weights=blk.get("reaction_level_weights"),
            reaction_level_scale=blk.get("reaction_level_scale"),
            outside_sim_floor_cap=blk.get("outside_sim_floor_cap"),
        )
    return controller, name, after


def resolve_stop_penalty(
    reactions_done: int,
    *,
    stop_early_penalty: float,
    stop_penalty_until_step: int,
    stop_penalty_schedule: "list[float] | tuple[float, ...] | None" = None,
) -> float:
    """Penalty for choosing Stop after ``reactions_done`` reactions.

    ``reactions_done`` is the number of reactions already applied in the episode
    at the moment Stop is selected (0 means Stop is the very first action).

    Two mutually exclusive policies, in precedence order:

    1. **Per-step schedule** (``stop_penalty_schedule``, when non-empty): a list
       indexed by ``reactions_done``. ``schedule[reactions_done]`` is returned
       when in range, otherwise ``0.0``. This lets early stops be punished
       harder than later ones, e.g. ``[-0.5, -0.3, -0.1]`` charges -0.5 for
       stopping before any reaction, -0.3 after one, -0.1 after two, and nothing
       from the third reaction onward.

    2. **Legacy single threshold** (default, schedule unset): a flat
       ``stop_early_penalty`` whenever ``reactions_done < stop_penalty_until_step``
       (with ``stop_penalty_until_step`` < 0 disabling the penalty entirely).

    Keeping (2) as the fallback means every existing config that only sets
    ``stop_early_penalty`` + ``stop_penalty_until_step`` behaves exactly as before.
    """
    if stop_penalty_schedule:
        if 0 <= reactions_done < len(stop_penalty_schedule):
            return float(stop_penalty_schedule[reactions_done])
        return 0.0
    if stop_penalty_until_step >= 0 and reactions_done < stop_penalty_until_step:
        return float(stop_early_penalty)
    return 0.0


def qed(smiles: str | None) -> float:
    if not smiles:
        return 0.0
    mol = Chem.MolFromSmiles(smiles)
    return float(QED.qed(mol)) if mol is not None else 0.0


class RewardFunction:
    SUPPORTED = frozenset({"delta_qed", "final_qed", "final_seh", "delta_seh"})

    def __init__(
        self,
        reward_type: str = "delta_qed",
        invalid_penalty: float = -1.0,
        round_digits: int | None = None,
        qed_round_digits: int | None = None,
        seh_scorer=None,
    ):
        # ``qed`` is the canonical name for per-step absolute QED: every
        # reacting step is rewarded with QED(product), not just the terminal
        # step. ``final_qed`` is kept as a backward-compatible alias for the
        # same behaviour (the name is a misnomer — it was never terminal-only).
        # ``seh`` mirrors that for the sEH binding proxy (PGFS-style absolute
        # per-step reward).
        if reward_type == "qed":
            reward_type = "final_qed"
        if reward_type == "seh":
            reward_type = "final_seh"
        if reward_type not in self.SUPPORTED:
            raise ValueError(f"Unsupported reward type: {reward_type}")
        if reward_type in {"delta_seh", "final_seh"} and seh_scorer is None:
            raise ValueError(f"{reward_type} requires a configured SehScorer")
        self.reward_type = reward_type
        self.invalid_penalty = float(invalid_penalty)
        self.round_digits = round_digits
        self.qed_round_digits = qed_round_digits
        self.seh_scorer = seh_scorer

    def _maybe_round(self, value: float) -> float:
        if self.round_digits is None:
            return float(value)
        return float(round(value, int(self.round_digits)))

    def _qed(self, smiles: str | None) -> float:
        value = qed(smiles)
        if self.qed_round_digits is None:
            return value
        return float(round(value, int(self.qed_round_digits)))

    def step_reward(self, previous_smiles: str | None, current_smiles: str | None) -> float:
        if not current_smiles:
            return self.invalid_penalty
        if self.reward_type == "delta_seh":
            delta = self.seh_scorer.step_delta(previous_smiles, current_smiles)
            return self._maybe_round(delta)
        if self.reward_type == "final_seh":
            return self._maybe_round(self.seh_scorer.reward(current_smiles))
        current_qed = self._qed(current_smiles)
        if self.reward_type == "final_qed":
            return self._maybe_round(current_qed)
        return self._maybe_round(current_qed - self._qed(previous_smiles))

    def stop_reward(
        self,
        *,
        current_step: int,
        stop_early_penalty: float,
        stop_penalty_until_step: int,
        feasible_template_count: int = 1,
        stop_penalty_schedule: "list[float] | tuple[float, ...] | None" = None,
    ) -> float:
        # Conditional Stop penalty: only charge the early-stop penalty when the
        # agent actually had at least one feasible reaction template available
        # but chose Stop anyway. If no template is feasible at this state, Stop
        # is the only legal move and incurs no penalty. Default ``feasible_template_count=1``
        # preserves the legacy unconditional behavior for callers that don't
        # supply the flag.
        if feasible_template_count <= 0:
            return 0.0
        # ``current_step`` is 1-indexed (incremented before the Stop check), so
        # the number of reactions already applied is ``current_step - 1``.
        reactions_done = max(0, int(current_step) - 1)
        return resolve_stop_penalty(
            reactions_done,
            stop_early_penalty=stop_early_penalty,
            stop_penalty_until_step=stop_penalty_until_step,
            stop_penalty_schedule=stop_penalty_schedule,
        )
