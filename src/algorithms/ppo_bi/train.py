"""Hand-rolled PPO trainer for bi-reaction MultiDiscrete (T, R2) actions.

Example: python -m src.scripts.run_experiment --config configs/delta_qed_scale.yaml
"""

from __future__ import annotations

import pickle
import random
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import QED

import wandb

from src.algorithms.common import init_wandb, run_dir, set_seed
from src.algorithms.ppo_bi.policy import BiPolicy
from src.chem.fingerprints import morgan_fp_array
from src.chem.r2_valid_indices_store import try_load_r2_valid_indices_store
from src.chem.reaction_manager import BI_TYPE, UNI_TYPES, ReactionManager
from src.chem.representations import (
    MACCS_DIM,
    MORGAN_DIM,
    RLV2_DIM,
    make_representation,
)
from src.chem.qed_scores_store import try_load_qed_scores_store
from src.chem.seh_scorer import SehScorer
from src.config import resolve_path
from src.envs.rewards import (
    _morgan_bitvect,
    build_diversity_controller,
    resolve_stop_penalty,
)

STOP_NAME = "Stop"
R2_PAD = -1  # sentinel R2 index for STOP transitions


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _reactant_smiles(data: Any) -> list[str]:
    if isinstance(data, dict):
        return [str(k) for k in data.keys()]
    if isinstance(data, (list, tuple, set)):
        return [str(x) for x in data]
    raise ValueError("Reactant file must contain a dict or sequence of SMILES.")


def _qed(smiles: str, *, round_digits: int | None = None) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Match ``rewards.qed``: unparseable SMILES are treated as QED=0 so a
        # rare non-roundtrippable reaction product cannot poison GAE/PPO with NaN.
        return 0.0
    value = float(QED.qed(mol))
    return round(value, round_digits) if round_digits is not None else value


def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    var_y = float(np.var(returns))
    if var_y == 0.0:
        return float("nan")
    return float(1.0 - np.var(returns - values) / var_y)


@dataclass
class _Transition:
    smiles: str
    t_action: int
    r2_action: int  # R2_PAD when is_stop is True
    log_pi_old: float
    value: float
    reward: float
    done: bool
    is_stop: bool
    # Unimolecular reaction: a real reaction applied with no R2 partner. Like
    # STOP it carries no R2 action (r2_action == R2_PAD, r2_mask/r2_valid_idx
    # None) so the PPO update skips the R2 log-prob/entropy term, but UNLIKE
    # STOP it advances the molecule (is_stop is False, product is real).
    is_uni: bool = False
    template_mask: torch.Tensor | None = None  # bool [num_templates + 1], CPU
    r2_mask: torch.Tensor | None = None  # bool [num_reactants], CPU, None for STOP/UNI
    # Global pool indices of valid R2 candidates at rollout time. Used when
    # ``r2_scoring: sparse_r2_scoring`` instead of storing a full-length bool
    # mask and scoring the entire reactant pool.
    r2_valid_idx: torch.Tensor | None = None  # long [K], CPU


class _StartSampler:
    """Samples train/eval start molecules.

    Optionally applies a *QED-headroom curriculum*: early in training only the
    lowest-QED (highest-headroom) slice of the train pool is drawn from, and the
    eligible slice grows to the full pool as training progresses. This is fully
    opt-in via ``curriculum`` (``None`` / disabled preserves the original
    uniform-over-pool behaviour bit-for-bit).

    Schedule (when enabled): let ``p = clamp(progress / warmup_frac, 0, 1)`` and
    ``q = start_quantile + p * (end_quantile - start_quantile)``. Starts are then
    drawn uniformly from the ``q``-lowest-QED fraction of the train pool.
    """

    def __init__(
        self,
        train_smiles: list[str],
        test_smiles: list[str],
        seed: int,
        curriculum: dict | None = None,
        qed_scores: "np.ndarray | list[float] | None" = None,
        n_sampled_eval: int | None = None,
    ):
        if not train_smiles:
            raise ValueError("ppo_bi training requires at least one training molecule.")
        if not test_smiles:
            raise ValueError("ppo_bi evaluation requires at least one test molecule.")
        self.train_smiles = list(train_smiles)
        self.test_smiles = list(test_smiles)
        self.rng = random.Random(seed)
        if n_sampled_eval is not None:
            n_sampled_eval = int(n_sampled_eval)
            if n_sampled_eval <= 0:
                raise ValueError("n_sampled_eval must be a positive integer when set.")
        self.n_sampled_eval = n_sampled_eval

        self._progress = 0.0
        self._sorted_by_qed: list[str] | None = None
        self.curriculum_enabled = bool(curriculum and curriculum.get("enabled", False))
        if self.curriculum_enabled:
            self.start_quantile = float(curriculum.get("start_quantile", 0.3))
            self.end_quantile = float(curriculum.get("end_quantile", 1.0))
            self.warmup_frac = float(curriculum.get("warmup_frac", 0.5))
            if not (0.0 < self.start_quantile <= 1.0):
                raise ValueError("curriculum.start_quantile must be in (0, 1].")
            if not (self.start_quantile <= self.end_quantile <= 1.0):
                raise ValueError(
                    "curriculum.end_quantile must be in [start_quantile, 1]."
                )
            # QED scoring of the train pool (ascending: lowest QED = most
            # delta-QED headroom first). NaNs (unparseable) sort last. Use
            # precomputed scores (dataset.qed_scores_file) when available to
            # skip the slow ~857k single-threaded recompute; otherwise fall back
            # to computing on the fly.
            if qed_scores is not None and len(qed_scores) == len(self.train_smiles):
                qvals = [float(q) for q in qed_scores]
                print(
                    "[curriculum] using precomputed QED scores "
                    f"({len(qvals)} reactants)",
                    flush=True,
                )
            else:
                if qed_scores is not None:
                    print(
                        "[curriculum] precomputed QED length "
                        f"({len(qed_scores)}) != pool ({len(self.train_smiles)}); "
                        "computing QED on the fly.",
                        flush=True,
                    )
                qvals = [_qed(s) for s in self.train_smiles]
            scored = [
                (q if q == q else float("inf"), s)
                for q, s in zip(qvals, self.train_smiles)
            ]
            scored.sort(key=lambda item: item[0])
            self._sorted_by_qed = [s for _, s in scored]

    def set_progress(self, frac: float) -> None:
        """Update training progress in [0, 1] (no-op without curriculum)."""
        self._progress = max(0.0, min(1.0, float(frac)))

    def current_quantile(self) -> float:
        if not self.curriculum_enabled:
            return 1.0
        if self.warmup_frac <= 0.0:
            return self.end_quantile
        p = min(1.0, self._progress / self.warmup_frac)
        return self.start_quantile + p * (self.end_quantile - self.start_quantile)

    def sample_train(self) -> str:
        if not self.curriculum_enabled or self._sorted_by_qed is None:
            return self.rng.choice(self.train_smiles)
        q = self.current_quantile()
        k = max(1, int(round(q * len(self._sorted_by_qed))))
        return self._sorted_by_qed[self.rng.randrange(k)]

    def eval_starts(self) -> list[str]:
        if self.n_sampled_eval is None or self.n_sampled_eval >= len(self.test_smiles):
            return list(self.test_smiles)
        return self.rng.sample(self.test_smiles, self.n_sampled_eval)


class BiPPO:
    """PPO trainer for the bi-reaction MultiDiscrete([T+1, R2]) action space.

    Supports both ``hierarchical`` and ``multidiscrete`` policy architectures
    via ``config['ppo_bi']['policy_arch']`` (default: ``hierarchical``). The
    sampling, masking, and log-prob accounting differ between architectures
    but the PPO core (GAE, clipped objective, value clipping, target_kl early
    stop, explained variance, etc.) is shared.
    """

    def __init__(self, config: dict):
        if config.get("reaction_mode", "uni") != "bi":
            raise ValueError(
                "ppo_bi is the bi-reaction trainer and requires reaction_mode: bi. "
                "Use --algorithm ppo for uni mode (its behaviour is unchanged)."
            )
        self.config = config
        training_cfg = config.get("training", {})
        self.seed = int(config.get("seed", training_cfg.get("seed", 0)))
        set_seed(self.seed)

        dataset = config["dataset"]
        self.train_reactants = _load_pickle(resolve_path(dataset["training_file"]))
        self.test_reactants = _load_pickle(resolve_path(dataset["test_file"]))
        self.templates_raw = _load_pickle(resolve_path(dataset["templates_file"]))
        self.train_smiles = _reactant_smiles(self.train_reactants)
        self.test_smiles = _reactant_smiles(self.test_reactants)

        self.reaction_mode = "bi"
        self.masking = config.get("masking", "reaction_valid")
        if self.masking not in {"reaction_valid", "r2_available", "substructure"}:
            raise ValueError(
                f"Unsupported masking for ppo_bi: {self.masking!r}. "
                "Use 'reaction_valid' (zero-failure, slow) or 'substructure' "
                "(pattern-match, fast with rejection backstop)."
            )
        self.reward_name = config.get("reward", "delta_qed")
        if self.reward_name not in {"delta_qed", "delta_seh"}:
            raise ValueError(
                "ppo_bi currently supports reward: delta_qed or delta_seh"
            )
        self.seh_scorer = (
            SehScorer.from_config(config.get("seh"))
            if self.reward_name == "delta_seh"
            else None
        )

        env_cfg = config.get("env", {})
        self.max_episode_len = int(
            config.get("max_episode_len", env_cfg.get("max_episode_len", 5))
        )
        self.use_stop_action = bool(env_cfg.get("use_stop_action", True))
        self.qed_round_digits = env_cfg.get(
            "info_qed_round_digits", env_cfg.get("reward_round_digits")
        )
        self.invalid_reaction_penalty = float(
            env_cfg.get("invalid_reaction_penalty", -1.0)
        )
        self.stop_early_penalty = float(env_cfg.get("stop_early_penalty", 0.0))
        self.stop_penalty_until_step = int(env_cfg.get("stop_penalty_until_step", -1))
        _stop_schedule = env_cfg.get("stop_penalty_schedule")
        self.stop_penalty_schedule = (
            [float(x) for x in _stop_schedule] if _stop_schedule else None
        )

        # Optional diversity-encouragement controller. Exactly one of three
        # mutually exclusive modes may be enabled (or none):
        #   - diversity_filter:        REINVENT-style hard scaffold memory.
        #   - soft_exponential_penalty: SyntheMol-style soft scaffold decay.
        #   - avoid_list:               penalise similarity to a fixed SMILES set.
        # All neutralise/erode the episode reward of over-represented or
        # too-familiar chemotypes, breaking single-mode (global-optimum) collapse.
        # Disabled by default so legacy runs are unaffected.
        (
            self.diversity_controller,
            self.diversity_mode,
            self.diversity_after_mean_reactions,
        ) = build_diversity_controller(
            config,
            resolve_path_fn=resolve_path,
            max_episode_len=self.max_episode_len,
        )

        # Build the *training-pool* reaction manager. This is the
        # reaction_manager / reactant_keys / num_reactants that the rollout
        # loop and the PPO update use. The active-pool attributes are
        # initialised to this train pool below; ``evaluate()`` temporarily
        # swaps them to the eval-pool versions when r2_arch='encoder'.
        train_manager_source = (
            self.train_reactants
            if isinstance(self.train_reactants, dict)
            else {s: None for s in self.train_smiles}
        )
        self._train_reaction_manager = ReactionManager(
            self.templates_raw, train_manager_source
        )
        self._train_reaction_manager.templates = (
            self._train_reaction_manager.templates_for_mode("bi")
        )
        self._train_reaction_manager.template_keys = list(
            self._train_reaction_manager.templates.keys()
        )
        self._train_reaction_manager.template_mask_cache.clear()
        self._train_reaction_manager._bi_r2_valid_cache = {}

        store = try_load_r2_valid_indices_store(
            dataset.get("r2_valid_indices_file"),
            resolve_path_fn=resolve_path,
        )
        if store is not None:
            self._train_reaction_manager.attach_r2_valid_indices_store(store)

        self.num_templates = len(self._train_reaction_manager.templates)
        self.stop_index = self.num_templates
        # Template indices (== template-head positions == dict keys) for
        # UNIMOLECULAR reactions. These have no R2 partner by design, so the
        # hierarchical/multidiscrete samplers must apply them directly with
        # R2=None instead of routing them through R2 selection (an empty R2
        # set there was previously mis-handled as STOP — a hidden free exit).
        self._uni_template_idx = frozenset(
            k
            for k, tmpl in self._train_reaction_manager.templates.items()
            if tmpl.get("type") in UNI_TYPES
        )
        self._train_reactant_keys = list(self._train_reaction_manager.reactants.keys())
        self._train_num_reactants = len(self._train_reactant_keys)

        # Active-pool aliases. The rollout / update / sampling code paths
        # below read these (NOT the underscore-prefixed pool-specific
        # attributes), so pool swapping is a 3-attribute rebind in
        # ``evaluate()``.
        self.reaction_manager = self._train_reaction_manager
        self.reactant_keys = self._train_reactant_keys
        self.num_reactants = self._train_num_reactants

        method_cfg = self._method_cfg(config)
        self.device = torch.device(
            method_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy_arch = str(method_cfg.get("policy_arch", "hierarchical")).lower()
        if self.policy_arch not in {"hierarchical", "multidiscrete"}:
            raise ValueError(
                f"policy_arch must be 'hierarchical' or 'multidiscrete', got "
                f"{self.policy_arch!r}"
            )

        # R2 representation. ``lookup`` is the legacy fixed-pool
        # ``nn.Embedding`` (bit-identical to the original BiPolicy). ``encoder``
        # is a Morgan-FP MLP shared between train and eval. ``encoder_graph``
        # (in GraphTransBiPPO) is a Siamese R2 GraphTransformer + projection.
        #
        # Subclasses may extend ``_supported_r2_archs`` to add new values
        # (e.g. ``GraphTransBiPPO`` adds ``'encoder_graph'`` for the Siamese
        # GraphTransformer R2 encoder under Option 3). The validation is kept
        # here in the base so misspelled YAMLs fail loudly at construction
        # time, but the supported set is overridable.
        self.r2_arch = str(method_cfg.get("r2_arch", "lookup")).lower()
        supported_archs = self._supported_r2_archs()
        if self.r2_arch not in supported_archs:
            raise ValueError(
                f"r2_arch must be one of {sorted(supported_archs)}, "
                f"got {self.r2_arch!r}"
            )

        # ``eval_r2_pool`` chooses which R2 pool the policy draws from
        # during evaluate() — either the training pool ("train") or the
        # test pool ("test"). Compatibility matrix:
        #
        #   r2_arch=lookup            + eval_r2_pool=train → OK (gr7aa7z6 baseline).
        #   r2_arch=lookup            + eval_r2_pool=test  → ERROR (no test rows).
        #   r2_arch=encoder           + eval_r2_pool=train → OK (FP encoder on train pool).
        #   r2_arch=encoder           + eval_r2_pool=test  → OK (current encoder default).
        #   r2_arch=encoder_graph     + eval_r2_pool=train → OK (graph encoder on train pool).
        #   r2_arch=encoder_graph     + eval_r2_pool=test  → OK (current encoder_graph default).
        #   r2_arch=descriptor_fixed  + eval_r2_pool=train → OK (descriptor pool from train).
        #   r2_arch=descriptor_fixed  + eval_r2_pool=test  → OK (descriptors are computable
        #                                                       for any SMILES — no learned
        #                                                       per-pool state to mismatch).
        #
        # If unset, the default preserves prior implicit behaviour:
        # lookup → train, every other arch → test. Internally we map the
        # YAML "train"/"test" to the legacy role names "train"/"eval" so
        # ``_swap_active_pool`` and ``_compute_active_r2_keys`` keep their
        # existing call sites — only the binding changes.
        _default_eval_pool = "train" if self.r2_arch == "lookup" else "test"
        _eval_pool_yaml = str(
            method_cfg.get("eval_r2_pool", _default_eval_pool)
        ).lower()
        _yaml_to_internal = {"train": "train", "test": "eval", "eval": "eval"}
        if _eval_pool_yaml not in _yaml_to_internal:
            raise ValueError(
                "eval_r2_pool must be 'train' or 'test', got "
                f"{_eval_pool_yaml!r}"
            )
        self._eval_pool_role = _yaml_to_internal[_eval_pool_yaml]
        # Public, YAML-style spelling for logging / experiment tags.
        self.eval_r2_pool = "train" if self._eval_pool_role == "train" else "test"

        # lookup + test is structurally impossible — the embedding table is
        # sized to the training pool and has no rows for test reactants.
        # Fail loudly at construction time rather than silently index into
        # the wrong row at eval.
        if self.r2_arch == "lookup" and self._eval_pool_role == "eval":
            raise ValueError(
                "r2_arch='lookup' is incompatible with eval_r2_pool='test': "
                "the nn.Embedding(num_reactants, r2_embed_dim) table is sized "
                "to the TRAINING pool — there are no rows for test reactants. "
                "Use r2_arch='encoder', 'encoder_graph', or "
                "'encoder_graph_shared' to evaluate on test R2s, or set "
                "eval_r2_pool='train' to keep the legacy (gr7aa7z6) "
                "behaviour where eval draws R2 from the train pool."
            )

        # Derive the R2-axis mask source from the masking mode (the README
        # contract). reaction_valid → RDKit-validated set (zero -1 guarantee);
        # substructure / r2_available → pattern-match set (-1 surfaces when
        # RDKit fails despite the pattern match). Advanced users can override
        # via the explicit `ppo_bi.r2_mask_kind` key (rarely needed).
        masking_to_r2 = {
            "reaction_valid": "true_valid",
            "substructure": "pattern",
            "r2_available": "pattern",
        }
        self.r2_mask_kind = str(
            method_cfg.get("r2_mask_kind", masking_to_r2.get(self.masking, "pattern"))
        ).lower()
        if self.r2_mask_kind not in {"pattern", "true_valid"}:
            raise ValueError(
                f"r2_mask_kind must be 'pattern' or 'true_valid', got {self.r2_mask_kind!r}"
            )

        self.r2_scoring = str(
            method_cfg.get("r2_scoring", "dense_r2_scoring")
        ).lower()
        if self.r2_scoring not in {"dense_r2_scoring", "sparse_r2_scoring"}:
            raise ValueError(
                f"r2_scoring must be 'dense_r2_scoring' or 'sparse_r2_scoring', "
                f"got {self.r2_scoring!r}"
            )

        # reaction_valid is the only masking mode that promises zero -1
        # rewards. For hierarchical the per-(state, T) RDKit-validated mask
        # makes the promise hold by construction (no rejection needed). For
        # multidiscrete the joint (T, R2) is not guaranteed valid even with
        # per-axis true-valid masks, so we need rejection sampling to enforce
        # the promise. ``substructure`` / ``r2_available`` are pattern-only —
        # they intentionally allow apply_reaction failures to surface as -1.
        self._enforce_zero_invalid = self.masking == "reaction_valid"
        self._needs_joint_rejection = (
            self._enforce_zero_invalid and self.policy_arch == "multidiscrete"
        )

        # ``descriptor_fixed`` needs the molecular representation to be built
        # BEFORE the policy is constructed so we can size the R(2) query head
        # to match the descriptor dim (1024 for Morgan, 167 for MACCS, 35 for
        # RLV2). For other r2_arch values this is a no-op — those modes don't
        # consume a fixed descriptor anywhere on the R2 side.
        #
        # ``make_representation`` is cheap for Morgan / MACCS (RDKit only)
        # but for RLV2 it fits / loads the per-feature normaliser from
        # ``dataset.training_file``. We pass ``training_smiles=self.train_smiles``
        # so the normaliser can be (re)fit on the first run if the cache is
        # missing; subsequent runs hit the on-disk cache and the call is
        # near-instant.
        self._descriptor_representation = None
        self._descriptor_is_binary = None
        if self.r2_arch == "descriptor_fixed":
            repr_name = str(env_cfg.get("molecule_representation", "morgan")).lower()
            self._descriptor_representation = make_representation(
                repr_name,
                training_file=resolve_path(dataset["training_file"]),
                training_smiles=self.train_smiles,
            )
            self._descriptor_is_binary = bool(
                self._descriptor_representation.is_binary
            )

        self.policy = self._build_policy(method_cfg)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(method_cfg.get("learning_rate", 3e-4)),
            weight_decay=float(method_cfg.get("weight_decay", 0.0)),
            eps=float(method_cfg.get("adam_eps", 1e-5)),
        )

        # Build the eval-pool reaction manager. Source depends on the
        # ``eval_r2_pool`` knob (NOT on ``r2_arch``):
        #
        #   - ``_eval_pool_role == "train"``: eval shares the train pool.
        #     Aliases the eval-pool attributes at the train pool objects;
        #     ``_swap_active_pool('eval')`` is then a structural no-op. This
        #     is the gr7aa7z6 baseline under lookup, and also a legal
        #     "evaluate on the same pool you trained on" mode under encoder
        #     / encoder_graph for apples-to-apples comparison against
        #     lookup.
        #
        #   - ``_eval_pool_role == "eval"``: build a separate
        #     ``ReactionManager`` from ``data/reactants_test.pkl``. Only
        #     legal when ``r2_arch != 'lookup'`` (the lookup-table guard
        #     above already rules this combination out).
        if self._eval_pool_role == "train":
            self._eval_reaction_manager = self._train_reaction_manager
            self._eval_reactant_keys = self._train_reactant_keys
            self._eval_num_reactants = self._train_num_reactants
        else:
            test_manager_source = (
                self.test_reactants
                if isinstance(self.test_reactants, dict)
                else {s: None for s in self.test_smiles}
            )
            self._eval_reaction_manager = ReactionManager(
                self.templates_raw, test_manager_source
            )
            self._eval_reaction_manager.templates = (
                self._eval_reaction_manager.templates_for_mode("bi")
            )
            self._eval_reaction_manager.template_keys = list(
                self._eval_reaction_manager.templates.keys()
            )
            self._eval_reaction_manager.template_mask_cache.clear()
            self._eval_reaction_manager._bi_r2_valid_cache = {}
            self._eval_reactant_keys = list(
                self._eval_reaction_manager.reactants.keys()
            )
            self._eval_num_reactants = len(self._eval_reactant_keys)

        # Pre-compute R(2) pool keys for both pools as torch tensors. Two
        # ``r2_arch`` values consume this cache:
        #
        #   - ``encoder``: Morgan fingerprints, fed into the learned R2
        #     encoder MLP at every sampling / update call. Building them
        #     once at init avoids per-step morgan_fp overhead.
        #   - ``descriptor_fixed``: raw fixed descriptors (Morgan / MACCS /
        #     RLV2 — picked via ``env.molecule_representation``) used
        #     directly as R2 keys, no learned encoder. Binary descriptors
        #     are rescaled from {0, 1} to {-1, +1} here so the
        #     ``descriptor_fixed`` policy's tanh-ed query and the keys
        #     share a symmetric range — same convention as Bi-TD3's
        #     ``KNNWrapper`` / ``_to_r2_tensor`` paths.
        #
        # When ``_eval_pool_role == "train"`` the two pools are identical,
        # so the eval-side cache is just a view onto the train-side cache.
        # When ``_eval_pool_role == "eval"`` we build a separate test-pool
        # tensor. Subclasses (e.g. GraphTransBiPPO) extend this with
        # their own pool-specific caches via ``_init_extra_pool_data``.
        if self.r2_arch == "encoder":
            train_fp_np = np.stack(
                [morgan_fp_array(s) for s in self._train_reactant_keys], axis=0
            )
            self._train_r2_fps = torch.from_numpy(train_fp_np).float().to(self.device)
            if self._eval_pool_role == "train":
                self._eval_r2_fps = self._train_r2_fps
            else:
                test_fp_np = np.stack(
                    [morgan_fp_array(s) for s in self._eval_reactant_keys], axis=0
                )
                self._eval_r2_fps = torch.from_numpy(test_fp_np).float().to(self.device)
        elif self.r2_arch == "descriptor_fixed":
            repr_fn = self._descriptor_representation.fn
            is_binary = self._descriptor_is_binary
            train_desc_np = np.stack(
                [repr_fn(s) for s in self._train_reactant_keys], axis=0
            ).astype(np.float32)
            if is_binary:
                # Binary descriptors live in {0, 1}^d in the pickle; rescale
                # to {-1, +1}^d so the tanh-ed query (also in [-1, +1]^d)
                # and the keys share a symmetric range. Without this the
                # -L2 score is dominated by an all-ones offset and the
                # query's gradient signal is washed out.
                train_desc_np = 2.0 * train_desc_np - 1.0
            self._train_r2_fps = torch.from_numpy(train_desc_np).to(self.device)
            if self._eval_pool_role == "train":
                self._eval_r2_fps = self._train_r2_fps
            else:
                eval_desc_np = np.stack(
                    [repr_fn(s) for s in self._eval_reactant_keys], axis=0
                ).astype(np.float32)
                if is_binary:
                    eval_desc_np = 2.0 * eval_desc_np - 1.0
                self._eval_r2_fps = torch.from_numpy(eval_desc_np).to(self.device)
        else:
            self._train_r2_fps = None
            self._eval_r2_fps = None

        # Subclass hook for arch-specific pool caches (e.g. the R2 graph
        # Batches that ``GraphTransBiPPO`` needs for ``r2_arch='encoder_graph'``).
        # Default is a no-op so the base trainer's behaviour is unchanged.
        self._init_extra_pool_data()

        # Cached r2_keys for the active scope. Populated by
        # ``_compute_active_r2_keys`` at the start of each rollout / eval
        # sweep (no_grad) and recomputed per minibatch inside the PPO update
        # (with_grad), so MLP weight updates flow through into r2_keys.
        self._active_r2_keys: torch.Tensor | None = None

        # PPO knobs (defaults match graphtransppo/MaskablePPO).
        self.gamma = float(method_cfg.get("gamma", 0.99))
        self.gae_lambda = float(method_cfg.get("gae_lambda", 0.95))
        self.clip_range = float(method_cfg.get("clip_range", 0.2))
        clip_vf = method_cfg.get("clip_range_vf", None)
        self.clip_range_vf = float(clip_vf) if clip_vf is not None else None
        self.vf_coef = float(method_cfg.get("vf_coef", 0.5))
        self.ent_coef = float(method_cfg.get("ent_coef", 0.0))
        self.max_grad_norm = float(method_cfg.get("max_grad_norm", 0.5))
        self.target_kl = method_cfg.get("target_kl", None)
        if self.target_kl is not None:
            self.target_kl = float(self.target_kl)
        self.normalize_advantage = bool(method_cfg.get("normalize_advantage", True))

        self.n_steps = int(method_cfg.get("n_steps", training_cfg.get("n_steps", 2048)))
        self.minibatch = int(method_cfg.get("batch_size", training_cfg.get("batch_size", 64)))
        self.n_epochs = int(method_cfg.get("n_epochs", 10))

        # Joint-rejection cap used ONLY when self._needs_joint_rejection (i.e.
        # multidiscrete + reaction_valid). Each retry runs one RDKit
        # ``apply_reaction`` call; with reaction_valid masks the joint
        # rejection rate is moderate (most (T, R2) pairs in the union mask
        # are valid). substructure / r2_available do NOT use this; they take
        # the first sample and let -1 surface naturally.
        self.r2_resample_retries = int(method_cfg.get("r2_resample_retries", 16))

        # Optional QED-headroom curriculum (opt-in via top-level `curriculum`
        # or `dataset.curriculum`). Disabled by default → uniform sampling.
        curriculum_cfg = config.get("curriculum", dataset.get("curriculum"))
        # Load precomputed QED scores for the curriculum (skips the slow ~857k
        # on-the-fly QED sort). Only bother when the curriculum is actually on.
        qed_scores = None
        if curriculum_cfg and curriculum_cfg.get("enabled", False):
            qed_store = try_load_qed_scores_store(
                dataset.get("qed_scores_file"), resolve_path_fn=resolve_path
            )
            if qed_store is not None:
                try:
                    qed_store.validate_pool(self.train_smiles)
                    qed_scores = qed_store.qed
                except ValueError as exc:
                    print(
                        f"[curriculum] precomputed QED pool mismatch ({exc}); "
                        "computing QED on the fly.",
                        flush=True,
                    )
        self.sampler = _StartSampler(
            self.train_smiles,
            self.test_smiles,
            self.seed,
            curriculum=curriculum_cfg,
            qed_scores=qed_scores,
            n_sampled_eval=training_cfg.get("n_sampled_eval"),
        )
        if self.sampler.n_sampled_eval is not None:
            print(
                "[eval] subsampling "
                f"{self.sampler.n_sampled_eval}/{len(self.test_smiles)} "
                "test molecules per eval",
                flush=True,
            )
        if self.sampler.curriculum_enabled:
            print(
                "[curriculum] QED-headroom enabled: "
                f"start_quantile={self.sampler.start_quantile} "
                f"end_quantile={self.sampler.end_quantile} "
                f"warmup_frac={self.sampler.warmup_frac} "
                f"(pool size {len(self.train_smiles)})",
                flush=True,
            )

        self._current_smiles: str | None = None
        self._current_react_steps: int = 0
        # SMILES the current episode started from (the molecule sampled at episode
        # reset). Needed by input→output reward shaping (in_out_sim mode).
        self._episode_start_smiles: str | None = None

        self._ep_reward_window: deque[float] = deque(maxlen=100)
        self._ep_length_window: deque[int] = deque(maxlen=100)
        # Reactions per episode (excludes the terminal STOP action), used to gate
        # the diversity filter on whether the policy is actually reacting yet.
        self._ep_reactions_window: deque[int] = deque(maxlen=100)
        self._total_episodes: int = 0
        self._cumulative_reward: float = 0.0
        # invalid_reaction_count is the cumulative number of -1 transitions
        # recorded by the rollout. Must stay 0 under masking=reaction_valid
        # (enforced by `_sample_action_*` + the assert in the rollout). Under
        # masking=substructure / r2_available it grows naturally — the
        # README contract allows pattern-match leaks to surface as -1.
        self._invalid_reaction_count: int = 0
        self._stop_event_count: int = 0
        self._rejection_total: int = 0
        self._sample_calls: int = 0

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    # Subclasses (e.g. GraphTransBiPPO) override these three small methods to
    # swap the encoder while reusing the entire PPO core (rollout loop,
    # masking, hierarchical/multidiscrete sampling, rejection logic, GAE,
    # clipped surrogate, value clipping, target_kl early stop, etc.). The
    # default implementations below preserve the original BiPolicy +
    # Morgan-fingerprint trainer behaviour bit-for-bit.

    def _method_cfg(self, config: dict) -> dict:
        """Return the method-specific config block (overridable by subclasses).

        Default reads ``config['ppo_bi']`` and falls back to ``config['ppo']``
        for shared PPO knobs. Subclasses (e.g. GraphTransBiPPO) point this at
        their own config block but keep the PPO defaults compatible.
        """
        return config.get("ppo_bi", config.get("ppo", {}))

    def _supported_r2_archs(self) -> set[str]:
        """Return the set of legal ``r2_arch`` values for this trainer.

        The base BiPPO trainer supports the Morgan-FP-only set
        ``{'lookup', 'encoder', 'descriptor_fixed'}``. ``descriptor_fixed``
        is the PGFS-style soft-KNN ablation (raw fixed descriptors as
        keys, no learned R2 encoder). Subclasses can extend it to
        include encoder variants their policy understands (e.g.
        GraphTransBiPPO adds ``'encoder_graph'`` / ``'encoder_graph_shared'``).
        Overriding this is preferred over re-implementing the YAML
        validation in each subclass.
        """
        return {"lookup", "encoder", "descriptor_fixed"}

    def _init_extra_pool_data(self) -> None:
        """Hook for arch-specific pool-data caches built at init time.

        Default is a no-op (the base trainer only needs the Morgan-FP
        tensors built unconditionally above). Subclasses override this
        to allocate their own caches; for instance
        ``GraphTransBiPPO._init_extra_pool_data`` builds
        ``_train_r2_graphs`` and ``_eval_r2_graphs`` (torch_geometric
        ``Batch`` objects) when ``r2_arch='encoder_graph'``.
        """
        return None

    def _build_policy(self, method_cfg: dict) -> torch.nn.Module:
        """Construct the policy module (overridable by subclasses).

        Default builds ``BiPolicy`` (Morgan-FP MLP trunk + BiPolicy heads) and
        moves it to ``self.device``. Subclasses can return any module that
        exposes ``forward_trunk``, ``template_logits``, ``value``, and
        ``r2_logits`` with the same signatures so the rollout loop and PPO
        update are encoder-agnostic.
        """
        # Under ``descriptor_fixed`` the policy's R(2) query head must
        # output into the descriptor dim (so query and keys live in the
        # same R^d for the -L2 score). Plumb that dim from the
        # representation we built earlier in ``__init__`` (``None`` for
        # every other r2_arch — the BiPolicy default of obs_dim=1024 then
        # applies, which is the legacy behaviour).
        descriptor_fp_dim = (
            int(self._descriptor_representation.dim)
            if self.r2_arch == "descriptor_fixed"
            else None
        )
        return BiPolicy(
            num_templates=self.num_templates,
            num_reactants=self.num_reactants,
            conditional_r2=(self.policy_arch == "hierarchical"),
            obs_dim=1024,
            trunk_hidden=int(method_cfg.get("trunk_hidden", 256)),
            template_embed_dim=int(method_cfg.get("template_embed_dim", 64)),
            r2_embed_dim=int(method_cfg.get("r2_embed_dim", 64)),
            r2_arch=self.r2_arch,
            r2_encoder_hidden=method_cfg.get("r2_encoder_hidden"),
            # Option 1: residual MLP for the R2 encoder. Defaults preserve
            # the legacy plain 2-layer MLP behaviour so older YAMLs reproduce
            # bit-for-bit; new ppo_bi YAMLs flip ``r2_encoder_residual: true``
            # to opt into the deeper variant.
            r2_encoder_n_layers=int(method_cfg.get("r2_encoder_n_layers", 2)),
            r2_encoder_residual=bool(method_cfg.get("r2_encoder_residual", False)),
            r2_encoder_n_res_blocks=int(method_cfg.get("r2_encoder_n_res_blocks", 2)),
            r2_fp_dim=descriptor_fp_dim,
            descriptor_query_tanh=bool(
                method_cfg.get("descriptor_query_tanh", True)
            ),
            descriptor_temperature=method_cfg.get("descriptor_temperature"),
        ).to(self.device)

    def _encode_smiles(self, smiles_list: list[str]) -> torch.Tensor:
        """Map a list of SMILES to trunk features ``z(R1) ∈ R^{B x trunk_dim}``.

        Default builds a Morgan-fingerprint batch and runs ``policy.
        forward_trunk(fps)``. Subclasses override this to plug in any other
        encoder (e.g. ``GraphTransBiPPO`` runs the GraphTransformer over the
        molecular graphs) without changing the rollout loop or PPO update.
        """
        fps = np.stack([morgan_fp_array(s) for s in smiles_list], axis=0)
        return self.policy.forward_trunk(torch.from_numpy(fps).to(self.device))

    # ------------------------------------------------------------------
    # Active-pool helpers (r2_arch='encoder' swaps train ↔ test at eval)
    # ------------------------------------------------------------------

    def _swap_active_pool(self, pool: str) -> None:
        """Point ``self.reaction_manager`` / ``reactant_keys`` / ``num_reactants``
        at the named pool.

        Only ``r2_arch='encoder'`` mode has distinct train and eval pools; in
        ``lookup`` mode both attributes alias the train pool and this call is
        a structural no-op. Callers MUST restore the previous pool by calling
        ``_swap_active_pool('train')`` after the eval section (see
        :meth:`evaluate`); we keep the swap explicit (rather than a context
        manager) so the sampling functions don't have to thread a pool
        argument through the rollout loop's hot path.
        """
        if pool == "train":
            self.reaction_manager = self._train_reaction_manager
            self.reactant_keys = self._train_reactant_keys
            self.num_reactants = self._train_num_reactants
        elif pool == "eval":
            self.reaction_manager = self._eval_reaction_manager
            self.reactant_keys = self._eval_reactant_keys
            self.num_reactants = self._eval_num_reactants
        else:
            raise ValueError(f"pool must be 'train' or 'eval', got {pool!r}")

    def _r2_pool_data_for(self, pool: str):
        """Return the input passed to ``policy.encode_r2_pool`` for ``pool``.

        Default returns the pre-computed Morgan-FP tensor for the named
        pool — this is what ``r2_arch='encoder'`` consumes. Subclasses
        override this to return arch-specific data; e.g.
        ``GraphTransBiPPO._r2_pool_data_for`` returns a torch_geometric
        ``Batch`` when ``r2_arch='encoder_graph'`` so the policy's
        Siamese R2 GraphTransformer can encode the pool directly from
        graphs. Called by :meth:`_compute_active_r2_keys` only when
        ``r2_arch != 'lookup'``.
        """
        if pool == "train":
            return self._train_r2_fps
        if pool == "eval":
            return self._eval_r2_fps
        raise ValueError(f"pool must be 'train' or 'eval', got {pool!r}")

    def _sparse_r2_graph_encode(self) -> bool:
        """When True, score / encode only mask-valid R2 pool indices.

        Controlled by ``ppo_bi.r2_scoring: sparse_r2_scoring``. Subclasses may
        override to force sparse behaviour for expensive encoders.
        """
        return self.r2_scoring == "sparse_r2_scoring"

    def _active_r2_pool_role(self) -> str:
        """``'train'`` or ``'eval'`` for the reaction manager currently in use."""
        if self.reaction_manager is self._eval_reaction_manager:
            return "eval"
        return "train"

    def _r2_keys_for_valid_indices(
        self,
        valid_idx: torch.Tensor,
        *,
        pool: str,
        with_grad: bool,
    ) -> torch.Tensor:
        """Return ``(K, D)`` R2 keys for global pool indices ``valid_idx``."""
        if valid_idx.numel() == 0:
            raise ValueError("valid_idx must be non-empty")
        valid_idx = valid_idx.to(self.device, dtype=torch.long)
        if self.r2_arch == "lookup":
            return self.policy.r2_embed.weight.index_select(0, valid_idx)
        if self.r2_arch in {"encoder_graph", "encoder_graph_shared"}:
            raise NotImplementedError(
                f"sparse_r2_scoring is not implemented for r2_arch={self.r2_arch!r}; "
                "use lookup or encoder, or dense_r2_scoring."
            )
        pool_data = self._r2_pool_data_for(pool)
        if pool_data is None:
            raise RuntimeError(
                f"r2_arch={self.r2_arch!r} requires pool data for sparse R2 scoring"
            )
        subset = pool_data.index_select(0, valid_idx)
        if with_grad:
            return self.policy.encode_r2_pool(subset)
        with torch.no_grad():
            return self.policy.encode_r2_pool(subset)

    def _global_r2_to_local(self, valid_idx: torch.Tensor, global_r2: int) -> torch.Tensor:
        """Map a global pool index to its position in ``valid_idx``."""
        matches = (valid_idx == int(global_r2)).nonzero(as_tuple=True)[0]
        if matches.numel() != 1:
            raise RuntimeError(
                f"global R2 index {global_r2} not found in sparse valid_idx "
                f"(len={valid_idx.numel()})"
            )
        return matches[0]

    def _apply_eval_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale action logits by the eval sampling temperature.

        Defaults to 1.0 (no-op) so training rollouts are unaffected. Set
        ``trainer.eval_sampling_temperature`` from the eval driver to flatten
        (>1) or sharpen (<1) the categorical at inference time. Masked
        entries (``-1e9``) stay effectively ``-inf`` after division, so the
        action mask is preserved, and ``argmax`` (greedy) is invariant to any
        positive scaling.
        """
        temp = float(getattr(self, "eval_sampling_temperature", 1.0))
        if temp == 1.0 or temp <= 0.0:
            return logits
        return logits / temp

    def _build_r2_categorical(
        self,
        trunk: torch.Tensor,
        r2_mask: torch.Tensor,
        t_idx: int | None,
    ) -> tuple[torch.distributions.Categorical, torch.Tensor | None]:
        """Build the masked R2 distribution for one state.

        Returns ``(dist, valid_idx)``. ``valid_idx`` is a long ``[K]`` tensor
        of global pool indices when :meth:`_sparse_r2_graph_encode` is True;
        otherwise ``None`` and ``dist`` is over the full pool with invalid
        entries masked to ``-1e9``.
        """
        if self._sparse_r2_graph_encode():
            raise ValueError(
                "sparse R2 scoring requires valid_idx; call _build_r2_categorical_sparse "
                "directly from the sampling path."
            )
        if self.policy_arch == "hierarchical":
            if t_idx is None:
                raise ValueError("hierarchical R2 sampling requires t_idx")
            t_tensor = torch.tensor([t_idx], device=self.device, dtype=torch.long)
            r2_logits_all = self.policy.r2_logits(
                trunk, t_tensor, r2_keys=self._active_r2_keys
            )[0]
        else:
            r2_logits_all = self.policy.r2_logits(
                trunk, None, r2_keys=self._active_r2_keys
            )[0]
        masked_r2_logits = r2_logits_all.masked_fill(~r2_mask, -1e9)
        masked_r2_logits = self._apply_eval_temperature(masked_r2_logits)
        return torch.distributions.Categorical(logits=masked_r2_logits), None

    def _build_r2_categorical_sparse(
        self,
        trunk: torch.Tensor,
        valid_idx: torch.Tensor,
        t_idx: int | None,
    ) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        """Build an R2 distribution over the provided global pool indices."""
        if valid_idx.numel() == 0:
            raise ValueError("valid_idx must be non-empty")
        pool = self._active_r2_pool_role()
        r2_keys = self._r2_keys_for_valid_indices(
            valid_idx, pool=pool, with_grad=False
        )
        if self.policy_arch == "hierarchical":
            if t_idx is None:
                raise ValueError("hierarchical R2 sampling requires t_idx")
            t_tensor = torch.tensor([t_idx], device=self.device, dtype=torch.long)
            r2_logits = self.policy.r2_logits(trunk, t_tensor, r2_keys=r2_keys)[0]
        else:
            r2_logits = self.policy.r2_logits(trunk, None, r2_keys=r2_keys)[0]
        r2_logits = self._apply_eval_temperature(r2_logits)
        return torch.distributions.Categorical(logits=r2_logits), valid_idx

    def _sample_r2_from_categorical(
        self,
        r2_dist: torch.distributions.Categorical,
        valid_idx: torch.Tensor | None,
        *,
        deterministic: bool,
    ) -> tuple[int, float]:
        """Sample a global R2 pool index and its log-prob under ``r2_dist``."""
        logits = r2_dist.logits
        r2_t = (
            torch.argmax(logits)
            if deterministic
            else r2_dist.sample()
        )
        if valid_idx is not None:
            r2_idx = int(valid_idx[r2_t].item())
        else:
            r2_idx = int(r2_t.item())
        log_pi_r2 = float(r2_dist.log_prob(r2_t).item())
        return r2_idx, log_pi_r2

    def _compute_active_r2_keys(self, *, pool: str, with_grad: bool) -> torch.Tensor:
        """Return ``r2_keys`` for the named pool, honouring the grad context.

        In ``r2_arch='lookup'`` mode this returns ``self.policy.r2_embed.weight``
        regardless of pool (the embedding is fixed to the train pool by design).
        Otherwise the policy's R2 encoder is applied to the pool input data
        returned by :meth:`_r2_pool_data_for` — Morgan-FP tensor under
        ``r2_arch='encoder'``, torch_geometric ``Batch`` under
        ``r2_arch='encoder_graph'`` in subclasses. ``with_grad=False`` is used
        for rollouts and evaluation (one forward pass per sweep, no autograd);
        ``with_grad=True`` is used inside the PPO update so gradients flow
        into the encoder each minibatch.
        """
        if self.r2_arch == "lookup":
            return self.policy.r2_embed.weight
        pool_data = self._r2_pool_data_for(pool)
        if pool_data is None:
            raise RuntimeError(
                f"r2_arch={self.r2_arch!r} but {pool} pool data is not initialised."
            )
        if with_grad:
            return self.policy.encode_r2_pool(pool_data)
        with torch.no_grad():
            return self.policy.encode_r2_pool(pool_data)

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------

    def _template_mask(self, smiles: str, *, force_stop: bool = False) -> torch.Tensor:
        mask = torch.zeros(self.num_templates + 1, dtype=torch.bool, device=self.device)
        if not force_stop:
            template_mask = (
                self.reaction_manager.get_mask(smiles, kind=self.masking).to(self.device)
                > 0.5
            )
            mask[: self.num_templates] = template_mask
        if self.use_stop_action:
            mask[self.stop_index] = True
        return mask

    def _r2_valid_idx_for_template(self, smiles: str, t_idx: int) -> torch.Tensor:
        """Global pool indices of valid R2 partners for hierarchical sampling."""
        if self.r2_mask_kind == "true_valid":
            idx_np = self.reaction_manager.bi_r2_valid_indices(smiles, t_idx)
        else:
            idx_np = self.reaction_manager.get_valid_reactant_indices(t_idx)
        return torch.as_tensor(idx_np, device=self.device, dtype=torch.long)

    def _r2_mask_per_template(self, smiles: str, t_idx: int) -> torch.Tensor:
        """Dense per-(state, T) R2 mask (dense scoring path only)."""
        if self.r2_mask_kind == "true_valid":
            mask_np = self.reaction_manager.bi_r2_valid_mask(smiles, t_idx)
        else:
            mask_np = self.reaction_manager.r2_mask(t_idx)
        return torch.from_numpy(mask_np.astype(np.bool_)).to(self.device)

    def _r2_valid_idx_union_for_state(
        self, smiles: str, valid_t_indices: list[int]
    ) -> torch.Tensor:
        """Union of valid global R2 indices over valid templates (multidiscrete)."""
        parts: list[np.ndarray] = []
        for t in valid_t_indices:
            if t < 0 or t >= self.num_templates:
                continue
            if self.r2_mask_kind == "true_valid":
                parts.append(self.reaction_manager.bi_r2_valid_indices(smiles, t))
            else:
                parts.append(self.reaction_manager.get_valid_reactant_indices(t))
        if not parts:
            return torch.zeros(0, device=self.device, dtype=torch.long)
        union = np.unique(np.concatenate(parts))
        return torch.as_tensor(union, device=self.device, dtype=torch.long)

    def _r2_mask_per_state(
        self, smiles: str, valid_t_indices: list[int]
    ) -> torch.Tensor:
        """Dense per-state R2 mask (dense scoring path only)."""
        union = np.zeros(self.num_reactants, dtype=np.bool_)
        valid_idx = self._r2_valid_idx_union_for_state(smiles, valid_t_indices)
        if valid_idx.numel():
            union[valid_idx.detach().cpu().numpy()] = True
        return torch.from_numpy(union).to(self.device)

    # ------------------------------------------------------------------
    # Single-state forward + sampling
    # ------------------------------------------------------------------

    def _fp(self, smiles: str) -> torch.Tensor:
        """Backward-compatible single-SMILES Morgan-fingerprint helper.

        Kept for callers outside this module that still expect the
        fingerprint-tensor API. The trainer itself now goes through
        ``_encode_smiles`` so a graph-aware subclass needs no extra hooks.
        """
        return torch.from_numpy(morgan_fp_array(smiles)).to(self.device)

    def _sample_action(
        self, smiles: str, *, force_stop: bool = False, deterministic: bool = False
    ) -> tuple[
        int,
        int,
        float,
        float,
        torch.Tensor,
        torch.Tensor | None,
        str | None,
        torch.Tensor | None,
    ]:
        """Dispatch to the per-architecture sampler."""
        self._sample_calls += 1
        if self.policy_arch == "hierarchical":
            return self._sample_action_hierarchical(
                smiles, force_stop=force_stop, deterministic=deterministic
            )
        return self._sample_action_multidiscrete(
            smiles, force_stop=force_stop, deterministic=deterministic
        )

    # ------------------------------------------------------------------
    # Hierarchical sampling: T then R2|T
    # ------------------------------------------------------------------

    def _sample_action_hierarchical(
        self, smiles: str, *, force_stop: bool, deterministic: bool
    ) -> tuple[
        int,
        int,
        float,
        float,
        torch.Tensor,
        torch.Tensor | None,
        str | None,
        torch.Tensor | None,
    ]:
        """Sample (T, R2) autoregressively.

        With ``masking=reaction_valid`` the per-(state, T) R2 mask is
        RDKit-validated, so ``apply_reaction`` is *guaranteed* to succeed —
        we still run it once to get the product SMILES. With ``substructure``
        / ``r2_available`` the R2 mask is pattern-only; ``apply_reaction``
        may fail and the trainer returns ``product=None``, which the rollout
        loop records as an ``invalid_reaction_penalty`` transition.
        """
        trunk = self._encode_smiles([smiles])
        tmpl_logits = self.policy.template_logits(trunk)
        value = float(self.policy.value(trunk).item())

        tmpl_mask = self._template_mask(smiles, force_stop=force_stop)
        if not bool(tmpl_mask.any()):
            return self._stop_return(value, tmpl_mask)

        masked_tmpl_logits = tmpl_logits[0].masked_fill(~tmpl_mask, -1e9)
        masked_tmpl_logits = self._apply_eval_temperature(masked_tmpl_logits)
        tmpl_dist = torch.distributions.Categorical(logits=masked_tmpl_logits)
        t_t = (
            torch.argmax(masked_tmpl_logits)
            if deterministic
            else tmpl_dist.sample()
        )
        t_idx = int(t_t.item())
        log_pi_t = float(tmpl_dist.log_prob(t_t).item())

        if t_idx == self.stop_index:
            return self._stop_return(value, tmpl_mask, log_pi_t=log_pi_t, t_idx=t_idx)

        if t_idx in self._uni_template_idx:
            # Unimolecular reaction: apply directly with no R2, never enter the
            # R2-selection branch (whose empty R2 set would mis-fire as STOP).
            return self._uni_action_return(smiles, t_idx, log_pi_t, value, tmpl_mask)

        if self._sparse_r2_graph_encode():
            valid_idx = self._r2_valid_idx_for_template(smiles, t_idx)
            if valid_idx.numel() == 0:
                return self._stop_return(value, tmpl_mask, log_pi_t=log_pi_t)
            r2_dist, r2_valid_idx = self._build_r2_categorical_sparse(
                trunk, valid_idx, t_idx
            )
        else:
            r2_mask = self._r2_mask_per_template(smiles, t_idx)
            if not bool(r2_mask.any()):
                return self._stop_return(value, tmpl_mask, log_pi_t=log_pi_t)
            r2_dist, r2_valid_idx = self._build_r2_categorical(trunk, r2_mask, t_idx)
        r2_idx, log_pi_r2 = self._sample_r2_from_categorical(
            r2_dist, r2_valid_idx, deterministic=deterministic
        )

        template = self.reaction_manager.templates[t_idx]
        product = self.reaction_manager.apply_reaction(
            smiles, template, self.reactant_keys[r2_idx]
        )
        if product is None and self._enforce_zero_invalid:
            # reaction_valid + hierarchical: the mask is exact, so this is a
            # logic bug or an RDKit pathology. Bump a counter for visibility
            # and fall back to STOP rather than emit a -1 the contract forbids.
            self._invalid_reaction_count += 1
            return self._stop_return(value, tmpl_mask, log_pi_t=log_pi_t)

        # product may be None here only when masking ∈ {substructure,
        # r2_available}; the rollout loop will record an invalid-penalty
        # transition. We still return the action and log_pi so the policy
        # learns from the failure signal.
        return (
            t_idx,
            r2_idx,
            log_pi_t + log_pi_r2,
            value,
            tmpl_mask.detach().to("cpu"),
            None if r2_valid_idx is not None else r2_mask.detach().to("cpu"),
            product,
            r2_valid_idx.detach().to("cpu") if r2_valid_idx is not None else None,
        )

    # ------------------------------------------------------------------
    # Multidiscrete sampling: independent T and R2 with joint rejection
    # ------------------------------------------------------------------

    def _sample_action_multidiscrete(
        self, smiles: str, *, force_stop: bool, deterministic: bool
    ) -> tuple[
        int,
        int,
        float,
        float,
        torch.Tensor,
        torch.Tensor | None,
        str | None,
        torch.Tensor | None,
    ]:
        """Sample (T, R2) independently from the per-axis masked distributions.

        With ``masking=reaction_valid`` the per-state R2 mask is the union of
        the per-(state, T) RDKit-validated sets. The independent-sampling
        assumption means the joint ``(T_sampled, R2_sampled)`` might pair an
        R2 with a template it doesn't actually work for; ``_needs_joint_
        rejection`` is True in this case and the loop below retries the joint
        until a valid pair is found (or budget exhausted → fall through to
        STOP). With ``substructure`` / ``r2_available`` no rejection happens:
        we take the first sample and let ``invalid_reaction_penalty`` surface
        if ``apply_reaction`` fails (this is the README contract for those
        masking modes).
        """
        trunk = self._encode_smiles([smiles])
        tmpl_logits = self.policy.template_logits(trunk)
        value = float(self.policy.value(trunk).item())

        tmpl_mask = self._template_mask(smiles, force_stop=force_stop)
        if not bool(tmpl_mask.any()):
            return self._stop_return(value, tmpl_mask)

        valid_t = [
            int(i)
            for i in torch.where(tmpl_mask[: self.num_templates])[0].tolist()
        ]
        masked_tmpl_logits = tmpl_logits[0].masked_fill(~tmpl_mask, -1e9)
        masked_tmpl_logits = self._apply_eval_temperature(masked_tmpl_logits)
        tmpl_dist = torch.distributions.Categorical(logits=masked_tmpl_logits)

        if not valid_t:
            # Only STOP is available.
            t_t = torch.tensor(self.stop_index, device=self.device, dtype=torch.long)
            return (
                self.stop_index,
                R2_PAD,
                float(tmpl_dist.log_prob(t_t).item()),
                value,
                tmpl_mask.detach().to("cpu"),
                None,
                None,
                None,
            )

        if self._sparse_r2_graph_encode():
            state_valid_idx = self._r2_valid_idx_union_for_state(smiles, valid_t)
            if state_valid_idx.numel() == 0:
                return self._stop_return(value, tmpl_mask)
            r2_dist, r2_valid_idx = self._build_r2_categorical_sparse(
                trunk, state_valid_idx, None
            )
            state_r2_mask = None
        else:
            state_r2_mask = self._r2_mask_per_state(smiles, valid_t)
            if not bool(state_r2_mask.any()):
                return self._stop_return(value, tmpl_mask)
            r2_dist, r2_valid_idx = self._build_r2_categorical(
                trunk, state_r2_mask, None
            )

        # One-shot fast path for substructure / r2_available: take a single
        # sample, run apply_reaction, return whatever it produces (product
        # may be None → rollout records the -1 transition).
        if not self._needs_joint_rejection:
            t_t = (
                torch.argmax(masked_tmpl_logits)
                if deterministic
                else tmpl_dist.sample()
            )
            t_idx = int(t_t.item())
            log_pi_t = float(tmpl_dist.log_prob(t_t).item())
            if t_idx == self.stop_index:
                return (
                    self.stop_index,
                    R2_PAD,
                    log_pi_t,
                    value,
                    tmpl_mask.detach().to("cpu"),
                    None,
                    None,
                    None,
                )
            if t_idx in self._uni_template_idx:
                return self._uni_action_return(
                    smiles, t_idx, log_pi_t, value, tmpl_mask
                )

            r2_idx, log_pi_r2 = self._sample_r2_from_categorical(
                r2_dist, r2_valid_idx, deterministic=deterministic
            )
            template = self.reaction_manager.templates[t_idx]
            product = self.reaction_manager.apply_reaction(
                smiles, template, self.reactant_keys[r2_idx]
            )
            return (
                t_idx,
                r2_idx,
                log_pi_t + log_pi_r2,
                value,
                tmpl_mask.detach().to("cpu"),
                None if r2_valid_idx is not None else state_r2_mask.detach().to("cpu"),
                product,
                r2_valid_idx.detach().to("cpu") if r2_valid_idx is not None else None,
            )

        # reaction_valid path: rejection-sample the joint until valid or budget
        # exhausted. The recorded log_pi is taken against the original per-axis
        # distributions so the PPO importance ratio still corresponds to the
        # policy's parameterisation (the implicit truncation by the rejection
        # set has bounded mass and is treated as a small ignorable bias).
        tried_pairs: set[tuple[int, int]] = set()
        retries = 0
        for _ in range(max(1, self.r2_resample_retries) + 1):
            t_t = (
                torch.argmax(masked_tmpl_logits)
                if deterministic
                else tmpl_dist.sample()
            )
            t_idx = int(t_t.item())
            log_pi_t = float(tmpl_dist.log_prob(t_t).item())
            if t_idx == self.stop_index:
                return (
                    self.stop_index,
                    R2_PAD,
                    log_pi_t,
                    value,
                    tmpl_mask.detach().to("cpu"),
                    None,
                    None,
                    None,
                )
            if t_idx in self._uni_template_idx:
                return self._uni_action_return(
                    smiles, t_idx, log_pi_t, value, tmpl_mask
                )

            r2_idx, log_pi_r2 = self._sample_r2_from_categorical(
                r2_dist, r2_valid_idx, deterministic=deterministic
            )

            pair = (t_idx, r2_idx)
            if pair in tried_pairs:
                if deterministic:
                    break
                retries += 1
                continue
            tried_pairs.add(pair)

            template = self.reaction_manager.templates[t_idx]
            product = self.reaction_manager.apply_reaction(
                smiles, template, self.reactant_keys[r2_idx]
            )
            if product is not None:
                self._rejection_total += retries
                return (
                    t_idx,
                    r2_idx,
                    log_pi_t + log_pi_r2,
                    value,
                    tmpl_mask.detach().to("cpu"),
                    None if r2_valid_idx is not None else state_r2_mask.detach().to("cpu"),
                    product,
                    r2_valid_idx.detach().to("cpu") if r2_valid_idx is not None else None,
                )
            retries += 1

        self._rejection_total += retries
        # Could not find a valid joint within budget → fall through to STOP
        # rather than violate the reaction_valid contract.
        return self._stop_return(value, tmpl_mask)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stop_return(
        self,
        value: float,
        tmpl_mask: torch.Tensor,
        *,
        log_pi_t: float = 0.0,
        t_idx: int | None = None,
    ) -> tuple[
        int,
        int,
        float,
        float,
        torch.Tensor,
        torch.Tensor | None,
        str | None,
        torch.Tensor | None,
    ]:
        """Build the canonical 'fall back to STOP' return tuple."""
        return (
            t_idx if t_idx is not None else self.stop_index,
            R2_PAD,
            log_pi_t,
            value,
            tmpl_mask.detach().to("cpu"),
            None,
            None,
            None,
        )

    def _uni_action_return(
        self,
        smiles: str,
        t_idx: int,
        log_pi_t: float,
        value: float,
        tmpl_mask: torch.Tensor,
    ) -> tuple[
        int,
        int,
        float,
        float,
        torch.Tensor,
        torch.Tensor | None,
        str | None,
        torch.Tensor | None,
    ]:
        """Apply a UNIMOLECULAR template directly (no R2) and return its action.

        Unimolecular templates have no R2 partner in the pool, so they must NOT
        be routed through R2 selection (where an empty R2 set was previously
        mis-read as STOP). We apply the reaction with ``reactant=None`` and
        return the action with no R2 component (``r2_idx = R2_PAD``,
        ``r2_mask = r2_valid_idx = None``); the log-prob is the template term
        only. ``product`` may be None for non-validated masks
        (substructure / r2_available) → the rollout records the standard
        ``invalid_reaction_penalty`` transition, exactly as for bi reactions.
        With ``reaction_valid`` we fall back to STOP rather than emit an
        invalid action the contract forbids.
        """
        template = self.reaction_manager.templates[t_idx]
        product = self.reaction_manager.apply_reaction(smiles, template, None)
        if product is None and self._enforce_zero_invalid:
            self._invalid_reaction_count += 1
            return self._stop_return(value, tmpl_mask, log_pi_t=log_pi_t)
        return (
            t_idx,
            R2_PAD,
            log_pi_t,
            value,
            tmpl_mask.detach().to("cpu"),
            None,
            product,
            None,
        )

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(
        self,
        n_steps: int,
        *,
        base_step: int = 0,
        log_episodes: bool = True,
    ) -> tuple[list[_Transition], float]:
        self.policy.eval()
        transitions: list[_Transition] = []
        if self._current_smiles is None:
            self._current_smiles = self.sampler.sample_train()
            self._current_react_steps = 0
            self._episode_start_smiles = self._current_smiles

        ep_reward = 0.0
        ep_length = 0
        steps_taken = 0

        with torch.no_grad():
            # Cache r2_keys for the entire rollout. Policy weights are frozen
            # during rollout (PPO updates happen after), so r2_keys is constant
            # and computing it once amortises the encoder forward over n_steps.
            # In lookup mode this is just a view onto ``r2_embed.weight``.
            # Sparse graph encoders skip the full-pool precompute and encode
            # only mask-valid indices per step instead.
            if not self._sparse_r2_graph_encode():
                self._active_r2_keys = self._compute_active_r2_keys(
                    pool="train", with_grad=False
                )
            else:
                self._active_r2_keys = None
            while steps_taken < n_steps:
                current = self._current_smiles
                react_steps = self._current_react_steps

                at_max = react_steps >= self.max_episode_len
                if at_max and not self.use_stop_action:
                    if transitions:
                        transitions[-1].done = True
                    self._end_episode(
                        ep_reward, ep_length, base_step + steps_taken, log_episodes
                    )
                    ep_reward, ep_length = 0.0, 0
                    continue

                (
                    t_idx,
                    r2_idx,
                    log_pi,
                    value,
                    tmpl_mask,
                    r2_mask,
                    product,
                    r2_valid_idx,
                ) = self._sample_action(current, force_stop=at_max)

                if t_idx < 0:
                    if transitions:
                        transitions[-1].done = True
                    self._end_episode(
                        ep_reward, ep_length, base_step + steps_taken, log_episodes
                    )
                    ep_reward, ep_length = 0.0, 0
                    continue

                done = False
                is_stop = t_idx == self.stop_index
                is_uni = (not is_stop) and (t_idx in self._uni_template_idx)
                reward = 0.0
                if is_stop:
                    self._stop_event_count += 1
                    done = True
                    # ``react_steps`` is the number of reactions already applied,
                    # matching ``reactions_done`` in resolve_stop_penalty.
                    reward = resolve_stop_penalty(
                        react_steps,
                        stop_early_penalty=self.stop_early_penalty,
                        stop_penalty_until_step=self.stop_penalty_until_step,
                        stop_penalty_schedule=self.stop_penalty_schedule,
                    )
                elif product is None:
                    # apply_reaction failed despite the mask passing the
                    # candidate. With masking ∈ {substructure, r2_available}
                    # this is the README contract (pattern-only masks are
                    # allowed to leak): record the transition with
                    # invalid_reaction_penalty so the policy learns from the
                    # failure. With masking=reaction_valid this branch is
                    # unreachable because `_sample_action_*` already routes
                    # any pathological non-product case through STOP at the
                    # source. The assert below makes that contract explicit.
                    assert not self._enforce_zero_invalid, (
                        "reaction_valid produced an invalid action — this "
                        "should be unreachable; the per-arch sampler must "
                        "fall through to STOP rather than route here."
                    )
                    self._invalid_reaction_count += 1
                    reward = self.invalid_reaction_penalty
                    done = True
                else:
                    if self.reward_name == "delta_seh":
                        reward = float(
                            self.seh_scorer.step_delta(current, product)
                        )
                    else:
                        prev_qed = _qed(current, round_digits=self.qed_round_digits)
                        next_qed = _qed(product, round_digits=self.qed_round_digits)
                        reward = float(next_qed - prev_qed)
                    # Per-step input->output similarity shaping (in_out_sim modes
                    # per_step_to_start / per_step_to_prev). Terminal-only modes
                    # leave per_step False and are handled at episode end.
                    if (
                        self._diversity_active()
                        and self.diversity_controller.per_step
                    ):
                        reward += self.diversity_controller.step_bonus(
                            product,
                            start_smiles=self._episode_start_smiles,
                            prev_smiles=current,
                            step_delta_qed=reward,
                        )
                    self._current_smiles = product
                    self._current_react_steps = react_steps + 1
                    if self._current_react_steps >= self.max_episode_len:
                        done = True

                transitions.append(
                    _Transition(
                        smiles=current,
                        t_action=t_idx,
                        r2_action=r2_idx if not (is_stop or is_uni) else R2_PAD,
                        log_pi_old=log_pi,
                        value=value,
                        reward=reward,
                        done=done,
                        is_stop=is_stop,
                        is_uni=is_uni,
                        template_mask=tmpl_mask,
                        r2_mask=r2_mask,
                        r2_valid_idx=r2_valid_idx,
                    )
                )
                steps_taken += 1
                ep_reward += reward
                ep_length += 1

                if done:
                    # Diversity controller: rewrite this episode's reward if it
                    # generated an over-represented / too-familiar chemotype. STOP
                    # keeps the current molecule; a successful reaction yields
                    # ``product``. Invalid reactions (product is None, not STOP)
                    # are left alone — they already carry the invalid penalty.
                    if self._diversity_active() and not (
                        not is_stop and product is None
                    ):
                        term_smiles = current if is_stop else product
                        new_ep_reward, penalised = (
                            self.diversity_controller.adjust_episode_reward(
                                term_smiles,
                                ep_reward,
                                start_smiles=self._episode_start_smiles,
                                n_reactions=self._current_react_steps,
                            )
                        )
                        if penalised and transitions:
                            transitions[-1].reward += new_ep_reward - ep_reward
                            ep_reward = new_ep_reward
                    self._end_episode(
                        ep_reward, ep_length, base_step + steps_taken, log_episodes
                    )
                    ep_reward, ep_length = 0.0, 0

        last_value = 0.0
        if transitions and not transitions[-1].done:
            with torch.no_grad():
                trunk = self._encode_smiles([self._current_smiles])
                last_value = float(self.policy.value(trunk).item())
        # Drop the cached keys so a stale tensor doesn't accidentally survive
        # into the next phase (PPO update recomputes per-minibatch, eval
        # recomputes once at the start of evaluate()).
        self._active_r2_keys = None
        return transitions, last_value

    def _end_episode(
        self,
        ep_reward: float,
        ep_length: int,
        step: int,
        log: bool,
    ) -> None:
        if ep_length > 0:
            self._ep_reward_window.append(float(ep_reward))
            self._ep_length_window.append(int(ep_length))
            # ``_current_react_steps`` is the number of successful reactions in
            # this episode (STOP does not increment it), recorded before the
            # reset below.
            self._ep_reactions_window.append(int(self._current_react_steps))
            self._total_episodes += 1
            self._cumulative_reward += float(ep_reward)
            if log and wandb.run is not None:
                wandb.log(
                    {
                        "train/global_step": int(step),
                        "train/episode_reward": float(ep_reward),
                        "train/episode_length": int(ep_length),
                        "train/mean_reward": float(np.mean(self._ep_reward_window)),
                        "train/mean_ep_length": float(np.mean(self._ep_length_window)),
                        "train/total_episodes": float(self._total_episodes),
                        "train/invalid_reaction_count": float(self._invalid_reaction_count),
                        "train/stop_event_count": float(self._stop_event_count),
                        "train/rejection_total": float(self._rejection_total),
                        "cumulative_reward": float(self._cumulative_reward),
                    },
                    step=int(step),
                )
        self._current_smiles = self.sampler.sample_train()
        self._current_react_steps = 0
        self._episode_start_smiles = self._current_smiles

    def _mean_recent_reactions(self) -> float:
        """Running mean reactions per episode (excludes STOP); 0.0 if no data."""
        if not self._ep_reactions_window:
            return 0.0
        return float(np.mean(self._ep_reactions_window))

    def _diversity_active(self) -> bool:
        """Whether the diversity controller should act now.

        True only when a controller is configured and the policy is reacting
        enough (recent mean reactions/episode >= the mode's gate). The gate keeps
        the controller from kneecapping the productive reward signal while the
        policy is still in STOP-collapse. Gate 0.0 => always active.
        """
        return (
            self.diversity_controller is not None
            and self._mean_recent_reactions() >= self.diversity_after_mean_reactions
        )

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        rollout: list[_Transition],
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(rollout)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            non_terminal = 0.0 if rollout[t].done else 1.0
            next_value = (
                last_value if t == n - 1 else (0.0 if rollout[t].done else rollout[t + 1].value)
            )
            delta = rollout[t].reward + self.gamma * next_value * non_terminal - rollout[t].value
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        values = np.array([tr.value for tr in rollout], dtype=np.float32)
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _r2_keys_for_update(self) -> torch.Tensor:
        """Hook: return ``r2_keys`` for the current PPO minibatch.

        Default recomputes from scratch *with grad* each call — fine for
        cheap encoders (``lookup`` is a free view onto ``r2_embed.weight``;
        ``encoder`` is a single MLP forward over ~116k FPs, milliseconds
        on GPU). Subclasses with expensive pool-encoding paths can
        override this to amortise: e.g. ``GraphTransBiPPO`` under
        ``r2_arch='encoder_graph'`` refreshes the Siamese R2
        GraphTransformer's keys only every ``r2_keys_refresh_minibatches``
        and reuses a detached cache in between. See
        :meth:`_begin_update_cycle` for the per-update reset hook.
        """
        return self._compute_active_r2_keys(pool="train", with_grad=True)

    def _begin_update_cycle(self) -> None:
        """Hook called once at the top of every :meth:`ppo_update`.

        Default is a no-op. Subclasses use this to reset any per-update
        state — e.g. ``GraphTransBiPPO`` invalidates its
        ``_cached_r2_keys`` here so the next ``_r2_keys_for_update`` call
        triggers a fresh, gradient-attached pool encoding.
        """
        return None

    def _evaluate_minibatch(
        self,
        smiles_batch: list[str],
        t_actions: torch.Tensor,
        r2_actions: torch.Tensor,
        is_stop: torch.Tensor,
        tmpl_masks: torch.Tensor,
        r2_masks: torch.Tensor | None,
        r2_valid_indices: list[torch.Tensor | None] | None = None,
        is_uni: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_pi_new, entropy_per_sample, values)`` for the minibatch.

        Uses the same architecture (conditional vs unconditional R2 head) as
        the trainer was configured with. The masks recorded at rollout time
        are reused verbatim so the update sees the same distribution that the
        old log-probs were taken under. Rows that carry no R2 action — STOP
        (``is_stop``) and UNIMOLECULAR reactions (``is_uni``) — contribute only
        the template log-prob/entropy; their R2 term is zeroed.
        """
        if is_uni is None:
            is_uni = torch.zeros_like(is_stop)
        if self._sparse_r2_graph_encode():
            if r2_valid_indices is None:
                raise ValueError(
                    "sparse R2 encoding requires r2_valid_indices per sample"
                )
            return self._evaluate_minibatch_sparse(
                smiles_batch,
                t_actions,
                r2_actions,
                is_stop,
                tmpl_masks,
                r2_valid_indices,
                is_uni=is_uni,
            )

        # Rows with no R2 action: STOP or unimolecular reaction.
        no_r2 = is_stop | is_uni

        trunk = self._encode_smiles(smiles_batch)
        tmpl_logits = self.policy.template_logits(trunk)
        values = self.policy.value(trunk)

        tmpl_logits = tmpl_logits.masked_fill(~tmpl_masks.to(self.device), -1e9)
        tmpl_dist = torch.distributions.Categorical(logits=tmpl_logits)
        log_pi_t = tmpl_dist.log_prob(t_actions)

        # R2 component: zero contribution for STOP rows, otherwise log π(R2|...).
        # The PPO update always runs on transitions collected from the train
        # pool. ``_r2_keys_for_update`` is the hook subclasses use to amortise
        # expensive encoders; the base trainer just delegates straight to
        # ``_compute_active_r2_keys(pool='train', with_grad=True)``.
        r2_keys = self._r2_keys_for_update()
        if self.policy_arch == "hierarchical":
            safe_t = torch.where(is_stop, torch.zeros_like(t_actions), t_actions)
            r2_logits = self.policy.r2_logits(trunk, safe_t, r2_keys=r2_keys)
        else:
            r2_logits = self.policy.r2_logits(trunk, None, r2_keys=r2_keys)
        if r2_masks is None:
            raise ValueError("dense R2 evaluation requires r2_masks")
        r2_logits = r2_logits.masked_fill(~r2_masks.to(self.device), -1e9)
        r2_dist = torch.distributions.Categorical(logits=r2_logits)
        safe_r2 = torch.where(r2_actions < 0, torch.zeros_like(r2_actions), r2_actions)
        log_pi_r2_raw = r2_dist.log_prob(safe_r2)
        log_pi_r2 = torch.where(no_r2, torch.zeros_like(log_pi_r2_raw), log_pi_r2_raw)

        log_pi = log_pi_t + log_pi_r2

        ent_t = tmpl_dist.entropy()
        ent_r2_raw = r2_dist.entropy()
        ent_r2 = torch.where(no_r2, torch.zeros_like(ent_r2_raw), ent_r2_raw)
        entropy = ent_t + ent_r2
        return log_pi, entropy, values

    def _evaluate_minibatch_sparse(
        self,
        smiles_batch: list[str],
        t_actions: torch.Tensor,
        r2_actions: torch.Tensor,
        is_stop: torch.Tensor,
        tmpl_masks: torch.Tensor,
        r2_valid_indices: list[torch.Tensor | None],
        is_uni: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO minibatch eval with per-sample sparse R2 scoring."""
        if is_uni is None:
            is_uni = torch.zeros_like(is_stop)
        trunk = self._encode_smiles(smiles_batch)
        tmpl_logits = self.policy.template_logits(trunk)
        values = self.policy.value(trunk)

        tmpl_logits = tmpl_logits.masked_fill(~tmpl_masks.to(self.device), -1e9)
        tmpl_dist = torch.distributions.Categorical(logits=tmpl_logits)
        log_pi_t = tmpl_dist.log_prob(t_actions)

        log_pi_r2_parts: list[torch.Tensor] = []
        ent_r2_parts: list[torch.Tensor] = []
        for i, valid_cpu in enumerate(r2_valid_indices):
            if bool(is_stop[i].item()) or bool(is_uni[i].item()) or valid_cpu is None:
                log_pi_r2_parts.append(
                    torch.zeros((), device=self.device, dtype=log_pi_t.dtype)
                )
                ent_r2_parts.append(
                    torch.zeros((), device=self.device, dtype=log_pi_t.dtype)
                )
                continue
            valid_idx = valid_cpu.to(self.device, dtype=torch.long)
            r2_keys = self._r2_keys_for_valid_indices(
                valid_idx, pool="train", with_grad=True
            )
            if self.policy_arch == "hierarchical":
                t_i = t_actions[i].unsqueeze(0)
                r2_logits = self.policy.r2_logits(
                    trunk[i : i + 1], t_i, r2_keys=r2_keys
                )[0]
            else:
                r2_logits = self.policy.r2_logits(
                    trunk[i : i + 1], None, r2_keys=r2_keys
                )[0]
            r2_dist = torch.distributions.Categorical(logits=r2_logits)
            local_r2 = self._global_r2_to_local(valid_idx, int(r2_actions[i].item()))
            log_pi_r2_parts.append(r2_dist.log_prob(local_r2))
            ent_r2_parts.append(r2_dist.entropy())

        log_pi_r2 = torch.stack(log_pi_r2_parts)
        log_pi = log_pi_t + log_pi_r2

        ent_t = tmpl_dist.entropy()
        ent_r2 = torch.stack(ent_r2_parts)
        entropy = ent_t + ent_r2
        return log_pi, entropy, values

    def ppo_update(
        self,
        rollout: list[_Transition],
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, float]:
        self.policy.train()
        # Subclass hook: invalidate any per-update caches (e.g. the
        # encoder_graph r2_keys cache in GraphTransBiPPO) so the next
        # minibatch starts from a fresh, gradient-attached encoding.
        self._begin_update_cycle()
        n = len(rollout)
        old_values = np.array([tr.value for tr in rollout], dtype=np.float32)
        adv_norm = advantages.copy()
        if self.normalize_advantage and n > 1:
            adv_norm = (adv_norm - adv_norm.mean()) / (adv_norm.std() + 1e-8)

        smiles_all = [tr.smiles for tr in rollout]
        t_actions_all = np.array([tr.t_action for tr in rollout], dtype=np.int64)
        r2_actions_all = np.array([tr.r2_action for tr in rollout], dtype=np.int64)
        is_stop_all = np.array([tr.is_stop for tr in rollout], dtype=np.bool_)
        is_uni_all = np.array([tr.is_uni for tr in rollout], dtype=np.bool_)
        log_pi_old_all = np.array([tr.log_pi_old for tr in rollout], dtype=np.float32)
        template_masks_all = torch.stack([tr.template_mask for tr in rollout]).bool()
        sparse_r2 = self._sparse_r2_graph_encode()
        if sparse_r2:
            r2_valid_indices_all = [tr.r2_valid_idx for tr in rollout]
            r2_masks_all = None
        else:
            r2_valid_indices_all = None
            zero_r2 = torch.zeros(self.num_reactants, dtype=torch.bool)
            r2_masks_all = torch.stack(
                [(tr.r2_mask if tr.r2_mask is not None else zero_r2) for tr in rollout]
            ).bool()

        idx = np.arange(n)
        loss_acc, pg_acc, v_acc, ent_acc, clip_frac_acc, kl_acc = [], [], [], [], [], []
        epochs_done = 0
        last_kl = 0.0
        early_stopped = False
        for epoch in range(self.n_epochs):
            np.random.shuffle(idx)
            kl_epoch: list[float] = []
            for start in range(0, n, self.minibatch):
                mb = idx[start : start + self.minibatch]
                if len(mb) == 0:
                    continue
                mb_smiles = [smiles_all[i] for i in mb]
                mb_t = torch.as_tensor(t_actions_all[mb], device=self.device)
                mb_r2 = torch.as_tensor(r2_actions_all[mb], device=self.device)
                mb_is_stop = torch.as_tensor(is_stop_all[mb], device=self.device)
                mb_is_uni = torch.as_tensor(is_uni_all[mb], device=self.device)
                mb_log_pi_old = torch.as_tensor(log_pi_old_all[mb], device=self.device)
                mb_adv = torch.as_tensor(adv_norm[mb], device=self.device)
                mb_ret = torch.as_tensor(returns[mb], device=self.device)
                mb_old_v = torch.as_tensor(old_values[mb], device=self.device)
                mb_tmpl_mask = template_masks_all[mb]
                if sparse_r2:
                    mb_r2_mask = None
                    mb_r2_valid = [r2_valid_indices_all[i] for i in mb]
                else:
                    mb_r2_mask = r2_masks_all[mb]
                    mb_r2_valid = None

                log_pi_new, entropy_per_sample, values = self._evaluate_minibatch(
                    mb_smiles,
                    mb_t,
                    mb_r2,
                    mb_is_stop,
                    mb_tmpl_mask,
                    mb_r2_mask,
                    r2_valid_indices=mb_r2_valid,
                    is_uni=mb_is_uni,
                )

                ratio = torch.exp(log_pi_new - mb_log_pi_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                if self.clip_range_vf is None:
                    v_loss = F.mse_loss(values, mb_ret)
                else:
                    v_clipped = mb_old_v + torch.clamp(
                        values - mb_old_v, -self.clip_range_vf, self.clip_range_vf
                    )
                    v_loss = torch.max((values - mb_ret).pow(2), (v_clipped - mb_ret).pow(2)).mean()

                entropy = entropy_per_sample.mean()
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = log_pi_new - mb_log_pi_old
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio).item()
                    clip_frac = float((torch.abs(ratio - 1.0) > self.clip_range).float().mean().item())

                kl_epoch.append(approx_kl)
                loss_acc.append(float(loss.detach().cpu().item()))
                pg_acc.append(float(pg_loss.detach().cpu().item()))
                v_acc.append(float(v_loss.detach().cpu().item()))
                ent_acc.append(float(entropy.detach().cpu().item()))
                clip_frac_acc.append(clip_frac)
                kl_acc.append(approx_kl)

            epochs_done = epoch + 1
            mean_kl_epoch = float(np.mean(kl_epoch)) if kl_epoch else 0.0
            last_kl = mean_kl_epoch
            if self.target_kl is not None and mean_kl_epoch > 1.5 * self.target_kl:
                early_stopped = True
                break

        ev = _explained_variance(old_values, returns.astype(np.float32))
        return {
            "train/loss": float(np.mean(loss_acc)) if loss_acc else 0.0,
            "train/policy_loss": float(np.mean(pg_acc)) if pg_acc else 0.0,
            "train/value_loss": float(np.mean(v_acc)) if v_acc else 0.0,
            "train/entropy": float(np.mean(ent_acc)) if ent_acc else 0.0,
            "train/approx_kl": float(np.mean(kl_acc)) if kl_acc else 0.0,
            "train/clip_fraction": float(np.mean(clip_frac_acc)) if clip_frac_acc else 0.0,
            "train/epochs_done": float(epochs_done),
            "train/early_stop_kl": float(last_kl),
            "train/early_stopped": float(1.0 if early_stopped else 0.0),
            "train/explained_variance": float(ev),
            "train/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

    # ------------------------------------------------------------------
    # Greedy evaluation
    # ------------------------------------------------------------------

    def _objective_score(self, smiles: str | None) -> float:
        """Scalar objective for the configured training reward."""
        if not smiles:
            return 0.0
        if self.reward_name == "delta_seh":
            assert self.seh_scorer is not None
            return float(self.seh_scorer.reward(smiles))
        return float(_qed(smiles, round_digits=self.qed_round_digits))

    def _greedy_trajectory(
        self, start_smiles: str
    ) -> tuple[float, float, int, float, str, float, float]:
        self.policy.eval()
        current = str(start_smiles)
        start_qed = _qed(current, round_digits=self.qed_round_digits)
        start_objective = self._objective_score(current)
        max_qed = start_qed
        max_objective = start_objective
        react_steps = 0
        with torch.no_grad():
            for _ in range(self.max_episode_len + int(self.use_stop_action)):
                at_max = react_steps >= self.max_episode_len
                if at_max and not self.use_stop_action:
                    break
                (
                    t_idx,
                    _r2_idx,
                    _,
                    _,
                    _tmpl_mask,
                    _,
                    product,
                    _,
                ) = self._sample_action(current, force_stop=at_max, deterministic=True)
                if t_idx < 0 or t_idx == self.stop_index:
                    break
                if product is None:
                    break
                next_qed = _qed(product, round_digits=self.qed_round_digits)
                next_objective = self._objective_score(product)
                current = product
                react_steps += 1
                max_qed = max(max_qed, next_qed)
                max_objective = max(max_objective, next_objective)
        final_qed = _qed(current, round_digits=self.qed_round_digits)
        final_objective = self._objective_score(current)
        final_objective_delta = final_objective - start_objective
        best_objective_delta = max_objective - start_objective
        final_qed_delta = final_qed - start_qed
        best_qed_delta = max_qed - start_qed
        return (
            final_objective_delta,
            best_objective_delta,
            react_steps,
            max_qed,
            current,
            final_qed_delta,
            best_qed_delta,
        )

    def evaluate(self) -> dict[str, float]:
        self.policy.eval()
        objective_final_deltas: list[float] = []
        objective_best_deltas: list[float] = []
        qed_final_deltas: list[float] = []
        qed_best_deltas: list[float] = []
        lengths: list[int] = []
        max_qeds: list[float] = []
        final_smiles: list[str] = []
        # Swap the active pool to the eval pool (no-op in lookup mode — same
        # train pool — but a real swap to the disjoint test pool in encoder
        # mode). The cached r2_keys is computed once for the eval sweep; like
        # the rollout cache, this amortises the encoder forward over all test
        # starts. The try/finally guarantees we restore the train pool even
        # if a trajectory raises.
        self._swap_active_pool(self._eval_pool_role)
        prev_active_keys = self._active_r2_keys
        try:
            if not self._sparse_r2_graph_encode():
                with torch.no_grad():
                    self._active_r2_keys = self._compute_active_r2_keys(
                        pool=self._eval_pool_role, with_grad=False
                    )
            else:
                self._active_r2_keys = None
            for s in self.sampler.eval_starts():
                obj_final, obj_best, l, mq, fs, qed_final, qed_best = (
                    self._greedy_trajectory(s)
                )
                objective_final_deltas.append(obj_final)
                objective_best_deltas.append(obj_best)
                qed_final_deltas.append(qed_final)
                qed_best_deltas.append(qed_best)
                lengths.append(l)
                max_qeds.append(mq)
                final_smiles.append(fs)
        finally:
            self._swap_active_pool("train")
            self._active_r2_keys = prev_active_keys
        if not objective_final_deltas:
            empty = {
                "eval/mean_reward": 0.0,
                "eval/avg_delta_qed": 0.0,
                "eval/mean_final_delta_qed": 0.0,
                "eval/mean_best_delta_qed": 0.0,
                "eval/mean_ep_length": 0.0,
                "eval/max_qed": float("nan"),
                "eval/n_molecules": 0,
                "eval/n_test_molecules": float(len(self.test_smiles)),
                "eval/diversity": float("nan"),
                "eval/top_output_share": float("nan"),
                "eval/unique_finals": 0,
            }
            if self.reward_name == "delta_seh":
                empty["eval/mean_final_delta_seh"] = 0.0
                empty["eval/mean_best_delta_seh"] = 0.0
            return empty
        diversity, top_share, n_unique = self._final_diversity_metrics(final_smiles)
        mean_final_objective = float(np.mean(objective_final_deltas))
        mean_best_objective = float(np.mean(objective_best_deltas))
        mean_final_qed = float(np.mean(qed_final_deltas))
        mean_best_qed = float(np.mean(qed_best_deltas))
        # With the stop action OFF the episode is forced to run to the cap, so the
        # final molecule need not be the agent's best. Score the headline reward by
        # the best molecule reached along the trajectory in that case; with stop ON
        # the agent chooses where to halt, so the final molecule is the answer.
        headline = (
            mean_best_objective if not self.use_stop_action else mean_final_objective
        )
        metrics: dict[str, float] = {
            "eval/mean_reward": headline,
            "eval/mean_final_delta_qed": mean_final_qed,
            "eval/mean_best_delta_qed": mean_best_qed,
            "eval/avg_delta_qed": mean_final_qed,
            "eval/mean_ep_length": float(np.mean(lengths)),
            "eval/max_qed": float(np.max(max_qeds)),
            "eval/n_molecules": float(len(objective_final_deltas)),
            "eval/n_test_molecules": float(len(self.test_smiles)),
            "eval/diversity": diversity,
            "eval/top_output_share": top_share,
            "eval/unique_finals": float(n_unique),
        }
        if self.reward_name == "delta_qed":
            metrics["eval/avg_delta_qed"] = headline
        elif self.reward_name == "delta_seh":
            metrics["eval/mean_final_delta_seh"] = mean_final_objective
            metrics["eval/mean_best_delta_seh"] = mean_best_objective
        return metrics

    @staticmethod
    def _final_diversity_metrics(
        final_smiles: list[str],
    ) -> tuple[float, float, int]:
        """Structural diversity + top-output share over greedy-eval finals.

        - ``diversity`` = 1 - mean pairwise Tanimoto over Morgan FPs (radius 2,
          1024 bits), the project's internal-diversity metric. Computed over the
          full set of valid final molecules (one per eval start).
        - ``top_output_share`` = count of the single most frequent final SMILES
          divided by the number of finals (collapse indicator; 1.0 => every
          start maps to the same molecule).
        - ``n_unique`` = number of distinct final SMILES.
        """
        n = len(final_smiles)
        if n == 0:
            return float("nan"), float("nan"), 0
        counts = Counter(final_smiles)
        n_unique = len(counts)
        top_share = counts.most_common(1)[0][1] / n
        fps = [
            fp
            for fp in (_morgan_bitvect(s, radius=2, fp_size=1024) for s in final_smiles)
            if fp is not None
        ]
        m = len(fps)
        if m < 2:
            return float("nan"), float(top_share), n_unique
        sim_sum = 0.0
        n_pairs = 0
        for i in range(m - 1):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
            sim_sum += float(sum(sims))
            n_pairs += len(sims)
        diversity = 1.0 - sim_sum / n_pairs if n_pairs else float("nan")
        return float(diversity), float(top_share), n_unique

    def save(
        self,
        path: Path,
        *,
        global_step: int | None = None,
        best_eval: float | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Persist the *full* training state, not just policy weights, so a run
        # resumed from this checkpoint continues as if it had never stopped
        # (modulo env/process nondeterminism) instead of restarting Adam's
        # momentum from zero. This matters most for the slow GTPPO-Bi runs that
        # are resumed across SLURM job boundaries: a cold optimizer effectively
        # throws away accumulated curvature/variance estimates and perturbs the
        # learning dynamics for many updates after every resume.
        checkpoint = {
            "policy": self.policy.state_dict(),
            "config": self.config,
            "policy_arch": self.policy_arch,
            "optimizer": self.optimizer.state_dict(),
            "global_step": int(global_step) if global_step is not None else None,
            "best_eval": float(best_eval) if best_eval is not None else None,
            "invalid_reaction_count": int(self._invalid_reaction_count),
            "rejection_total": int(self._rejection_total),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }
        torch.save(checkpoint, path)

    def load(self, path: Path) -> dict:
        """Restore policy + full training state; returns the raw checkpoint.

        Optimizer momentum, cumulative counters and RNG state are restored only
        when present, so legacy weight-only checkpoints still load (just without
        exact-resume state). The returned dict lets the training loop recover
        ``global_step`` / ``best_eval`` without re-parsing the filename.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy"])
        self.policy.to(self.device)

        opt_state = checkpoint.get("optimizer")
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)
        if checkpoint.get("invalid_reaction_count") is not None:
            self._invalid_reaction_count = int(checkpoint["invalid_reaction_count"])
        if checkpoint.get("rejection_total") is not None:
            self._rejection_total = int(checkpoint["rejection_total"])
        rng = checkpoint.get("rng_state")
        if rng is not None:
            self._restore_rng_state(rng)
        return checkpoint

    @staticmethod
    def _restore_rng_state(rng: dict) -> None:
        """Restore python/numpy/torch RNG streams saved by :meth:`save`.

        Tensors may have been mapped onto the trainer device by ``torch.load``;
        ``set_rng_state`` requires CPU uint8 ByteTensors, hence the coercion.
        """

        def _cpu_byte(t):
            if isinstance(t, torch.Tensor):
                return t.detach().to("cpu", dtype=torch.uint8)
            return t

        if rng.get("python") is not None:
            random.setstate(rng["python"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("torch") is not None:
            torch.set_rng_state(_cpu_byte(rng["torch"]))
        cuda_states = rng.get("torch_cuda")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([_cpu_byte(s) for s in cuda_states])


def global_step_from_checkpoint(
    path: Path | str, checkpoint: dict | None = None
) -> int:
    """Infer training step from ``model_step_<N>.pt`` name or embedded checkpoint metadata."""
    if checkpoint is not None and checkpoint.get("global_step") is not None:
        return int(checkpoint["global_step"])

    stem = Path(path).stem
    match = re.fullmatch(r"model_step_(\d+)", stem)
    if match:
        return int(match.group(1))

    if checkpoint is None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("global_step") is not None:
        return int(checkpoint["global_step"])

    raise ValueError(
        f"Cannot infer global_step from checkpoint name {Path(path).name!r}; "
        "expected model_step_<N>.pt, final_model.pt/best_model.pt with global_step "
        "in the file, or training.resume_global_step in config."
    )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def run_training_loop(trainer: BiPPO, run, config: dict, experiment_name: str) -> None:
    """Shared PPO training loop used by both BiPPO and its graph-aware subclass.

    The loop is encoder-agnostic; ``trainer`` only needs to expose
    ``collect_rollout``, ``compute_gae``, ``ppo_update``, ``evaluate``, and
    ``save`` plus the cumulative-counter attributes referenced below. The
    extracted helper is so ``GraphTransBiPPO`` can reuse the exact same
    rollout / eval / checkpoint cadence as the fingerprint trainer without
    having to copy-paste the body.
    """
    training_cfg = config.get("training", {})
    total_timesteps = int(training_cfg.get("total_timesteps", 1_000_000))
    eval_freq = int(training_cfg.get("eval_freq", 10_000))
    save_freq = int(training_cfg.get("save_freq", 100_000))
    n_steps = trainer.n_steps
    run_id_override = training_cfg.get("run_id")
    resume_checkpoint = training_cfg.get("resume_checkpoint")

    out_dir = run_dir(run_id_override or (run.id if run is not None else experiment_name))
    best_eval = -float("inf")

    global_step = 0
    if resume_checkpoint:
        resume_path = Path(resolve_path(str(resume_checkpoint)))
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume_checkpoint not found: {resume_path}")
        resume_ckpt = trainer.load(resume_path)
        # global_step precedence: explicit config override > value embedded in
        # the checkpoint by save() > parsed from the model_step_<N>.pt filename
        # (legacy weight-only checkpoints).
        if training_cfg.get("resume_global_step") is not None:
            global_step = int(training_cfg["resume_global_step"])
        elif resume_ckpt.get("global_step") is not None:
            global_step = int(resume_ckpt["global_step"])
        else:
            global_step = global_step_from_checkpoint(resume_path)
        if resume_ckpt.get("best_eval") is not None:
            best_eval = float(resume_ckpt["best_eval"])
        has_opt = resume_ckpt.get("optimizer") is not None
        print(
            f"[resume] loaded {resume_path} at global_step={global_step}; "
            f"optimizer_state={'restored' if has_opt else 'MISSING (cold start)'}; "
            f"best_eval={best_eval}; "
            f"continuing until total_timesteps={total_timesteps}",
            flush=True,
        )

    last_eval_bucket = (global_step // eval_freq) if eval_freq > 0 else -1
    last_save_bucket = (global_step // save_freq) if save_freq > 0 else -1
    while global_step < total_timesteps:
        if total_timesteps > 0:
            trainer.sampler.set_progress(global_step / total_timesteps)
        rollout, last_value = trainer.collect_rollout(
            n_steps, base_step=global_step, log_episodes=True
        )
        advantages, returns = trainer.compute_gae(rollout, last_value)
        update_metrics = trainer.ppo_update(rollout, advantages, returns)
        global_step += len(rollout)

        rewards = [tr.reward for tr in rollout]
        stop_frac = float(np.mean([tr.is_stop for tr in rollout])) if rollout else 0.0
        rollout_metrics = {
            "train/global_step": global_step,
            "train/rollout_steps": float(len(rollout)),
            "train/rollout_mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "train/rollout_stop_fraction": stop_frac,
            "train/rollout_invalid_count_cum": float(trainer._invalid_reaction_count),
            "train/rejection_count_cum": float(trainer._rejection_total),
        }
        rollout_metrics["train/mean_reactions"] = trainer._mean_recent_reactions()
        if trainer.diversity_controller is not None:
            rollout_metrics["train/diversity_active"] = float(
                trainer._diversity_active()
            )
            for key, value in trainer.diversity_controller.metrics().items():
                rollout_metrics[f"train/{key}"] = float(value)
        if trainer.sampler.curriculum_enabled:
            rollout_metrics["train/curriculum_quantile"] = float(
                trainer.sampler.current_quantile()
            )
        wandb.log({**rollout_metrics, **update_metrics}, step=global_step)

        bucket = global_step // eval_freq
        if eval_freq > 0 and bucket > last_eval_bucket:
            last_eval_bucket = bucket
            eval_metrics = trainer.evaluate()
            eval_metrics["train/global_step"] = global_step
            wandb.log(eval_metrics, step=global_step)
            if eval_metrics["eval/mean_reward"] > best_eval:
                best_eval = eval_metrics["eval/mean_reward"]
                trainer.save(
                    out_dir / "best_model.pt",
                    global_step=global_step,
                    best_eval=best_eval,
                )

        sbucket = global_step // save_freq
        if save_freq > 0 and sbucket > last_save_bucket:
            last_save_bucket = sbucket
            trainer.save(
                out_dir / f"model_step_{global_step}.pt",
                global_step=global_step,
                best_eval=best_eval,
            )

    final_eval = trainer.evaluate()
    final_eval["train/global_step"] = global_step
    wandb.log(final_eval, step=global_step)
    trainer.save(out_dir / "final_model.pt", global_step=global_step, best_eval=best_eval)
    if run is not None:
        run.finish()


def train(config: dict, experiment_name: str) -> None:
    trainer = BiPPO(config)
    run = init_wandb(config, "ppo_bi", experiment_name)
    run_training_loop(trainer, run, config, experiment_name)


__all__ = ["BiPPO", "train", "run_training_loop", "global_step_from_checkpoint"]
