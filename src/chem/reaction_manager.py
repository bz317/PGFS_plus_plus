"""Shared RDKit reaction manager for PGFS++."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from src.chem.product_selection import best_qed_product_smiles
from src.chem.r2_valid_indices_store import R2ValidIndicesStore
from src.chem.rdkit_logging import suppress_rdkit_logs

logger = logging.getLogger(__name__)

UNI_TYPES = {"unimolecular", "unimolecular_explicit_reagent"}
BI_TYPE = "bimolecular"


class ReactionManager:
    """Applies templates and provides masks for template/R2 selection."""

    def __init__(self, templates: dict, reactants: dict, *, r2_mask_mode: str = "new"):
        self.templates = self._normalize_templates(templates)
        self.reactants = reactants
        if r2_mask_mode not in {"new", "legacy"}:
            raise ValueError(f"r2_mask_mode must be 'new' or 'legacy', got {r2_mask_mode!r}")
        self.r2_mask_mode = r2_mask_mode
        self.template_mask_cache: dict[tuple[str | None, str], torch.Tensor] = {}
        self.valid_reactants_cache: dict[int, list[str]] = {}
        self._r2_valid_indices_store: R2ValidIndicesStore | None = None
        self.template_types = self._template_types_tensor()
        self.template_keys = list(self.templates.keys())

    def attach_r2_valid_indices_store(self, store: R2ValidIndicesStore) -> None:
        """Attach mmap-backed pattern-valid R2 indices (train pool only)."""
        pool_keys = (
            list(self.reactants.keys())
            if isinstance(self.reactants, dict)
            else list(self.reactants)
        )
        store.validate_pool(pool_keys)
        if store.num_templates != len(self.templates):
            raise ValueError(
                f"R2 index store has {store.num_templates} templates but "
                f"ReactionManager has {len(self.templates)}"
            )
        self._r2_valid_indices_store = store

    @staticmethod
    def _normalize_templates(templates: dict) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        # Preserve pickle insertion order so action indices match the legacy PPO/A2C/TD3 code.
        for idx, (_, template) in enumerate(templates.items()):
            if not isinstance(template, dict):
                raise ValueError(f"Template {idx} must be a dict.")
            out[idx] = dict(template)
        return out

    def templates_for_mode(self, reaction_mode: str) -> dict[int, dict[str, Any]]:
        """``uni``: only UNI_TYPES templates. ``bi``: full pool (uni + bi templates together)."""
        if reaction_mode == "bi":
            return dict(self.templates)
        if reaction_mode != "uni":
            raise ValueError(f"Unsupported reaction_mode: {reaction_mode}")
        selected = [t for t in self.templates.values() if t.get("type") in UNI_TYPES]
        if not selected:
            raise ValueError("Uni mode requested but no unimolecular templates were found.")
        return {i: dict(t) for i, t in enumerate(selected)}

    def _template_types_tensor(self) -> torch.Tensor:
        mapping = {"unimolecular": 0, "unimolecular_explicit_reagent": 0, "bimolecular": 1}
        return torch.tensor(
            [mapping.get(t.get("type", "unimolecular"), 0) for t in self.templates.values()],
            dtype=torch.long,
        )

    @staticmethod
    def _template_smarts(template: dict | str) -> str:
        return template["smarts"] if isinstance(template, dict) else str(template)

    @staticmethod
    def _template_fixed_reagents(template: dict | str) -> list[str]:
        if isinstance(template, dict):
            return list(template.get("_explicit_reagents", []))
        return []

    @staticmethod
    def _mol_from_reagent_smarts(smarts: str):
        mol = Chem.MolFromSmiles(smarts)
        if mol is not None:
            return mol
        return Chem.MolFromSmarts(smarts)

    def apply_reaction(self, state: str | None, template: dict | str, reactant: str | None = None) -> str | None:
        if not state:
            return None
        try:
            with suppress_rdkit_logs():
                state_mol = Chem.MolFromSmiles(state)
                if state_mol is None:
                    return None
                reaction = AllChem.ReactionFromSmarts(self._template_smarts(template))
                product_sets = self._run_reaction(
                    state_mol,
                    reaction,
                    reactant,
                    self._template_fixed_reagents(template),
                )
                return best_qed_product_smiles(product_sets)
        except Exception as exc:
            logger.debug("Reaction failed for %s: %s", state, exc)
            return None

    @staticmethod
    def _mol_matches_reactant_template(mol, pattern) -> bool:
        return mol.HasSubstructMatch(pattern, useChirality=True)

    def _run_reaction(self, state_mol, reaction, reactant: str | None, fixed_reagents: Iterable[str]):
        num_reactants = reaction.GetNumReactantTemplates()
        fixed = list(fixed_reagents or [])
        if fixed:
            reagent_mols = [self._mol_from_reagent_smarts(s) for s in fixed]
            if any(m is None for m in reagent_mols):
                return []
            # Uni-explicit and bi templates with all non-R1 slots fixed: R1 + fixed only.
            if len(reagent_mols) == num_reactants - 1:
                return reaction.RunReactants((state_mol, *reagent_mols))
            # Multi-component bi templates: R1 (state) + pool R2 + trailing fixed
            # co-reagents. Expanded SMARTS place the two variable slots first
            # (0=R1, 1=R2) and the fixed reagents last. Rigid PGFS convention:
            # R1 = slot 0, pool R2 = slot 1. The flipped R1/R2 assignment ships
            # as a separate ``_R2`` template, so we never try the reversed order.
            if reactant and len(reagent_mols) == num_reactants - 2:
                reactant_mol = Chem.MolFromSmiles(reactant)
                if reactant_mol is None:
                    return []
                if num_reactants - len(reagent_mols) == 2:
                    return reaction.RunReactants((state_mol, reactant_mol, *reagent_mols))
                return []
            return []
        if num_reactants == 1:
            return reaction.RunReactants((state_mol,))
        if num_reactants == 2 and reactant:
            # Rigid PGFS convention: R1 (state) = slot 0, pool R2 = slot 1.
            # Flipped directions are separate ``_R2`` templates.
            reactant_mol = Chem.MolFromSmiles(reactant)
            if reactant_mol is not None:
                return reaction.RunReactants((state_mol, reactant_mol))
        return []

    @classmethod
    def _r2_reactant_template_indices(cls, reaction, template: dict | str) -> list[int]:
        """Rigid PGFS R2 slot: the pool second reactant is always SMARTS slot 1.

        Applies to plain 2-slot bimolecular templates (``num == 2``) and to
        multi-component bi templates that expand to two leading variable slots
        (R1 = slot 0, R2 = slot 1) followed by ``num - 2`` trailing fixed
        reagents. Flipped R1/R2 directions ship as separate ``_R2`` templates,
        so slot 0 is never exposed as an R2 candidate slot.
        """
        num = reaction.GetNumReactantTemplates()
        n_fixed = len(cls._template_fixed_reagents(template))
        if isinstance(template, dict) and template.get("type") != BI_TYPE:
            return []
        if n_fixed == 0 and num == 2:
            return [1]
        if n_fixed == num - 2:
            return [1]
        return []

    @classmethod
    def _r2_reactant_template_indices_legacy(cls, reaction, template: dict | str) -> list[int]:
        """Pre-fix R2 pattern slots: slot 1 only for 2-slot reactions, empty otherwise."""
        if isinstance(template, dict) and template.get("type") != BI_TYPE:
            return []
        if reaction.GetNumReactantTemplates() == 2:
            return [1]
        return []

    @classmethod
    def r2_pattern_queries(cls, template: dict | str, *, mode: str = "new") -> list:
        """RDKit query mols for ``match_template(...)[\"second\"]`` / precomputed R2 indices."""
        if isinstance(template, dict) and template.get("type") != BI_TYPE:
            return []
        try:
            with suppress_rdkit_logs():
                reaction = AllChem.ReactionFromSmarts(cls._template_smarts(template))
                if mode == "legacy":
                    idxs = cls._r2_reactant_template_indices_legacy(reaction, template)
                else:
                    idxs = cls._r2_reactant_template_indices(reaction, template)
                queries: list = []
                for idx in idxs:
                    pattern = reaction.GetReactantTemplate(idx)
                    query = Chem.MolFromSmarts(Chem.MolToSmarts(pattern))
                    if query is not None:
                        queries.append(query)
                return queries
        except Exception:
            return []

    def match_template(self, reactant: str | None, template: dict | str) -> dict[str, bool]:
        try:
            with suppress_rdkit_logs():
                if not reactant:
                    return {"first": False, "second": False}
                reaction = AllChem.ReactionFromSmarts(self._template_smarts(template))
                mol = Chem.MolFromSmiles(reactant)
                if mol is None:
                    return {"first": False, "second": False}
                if self.r2_mask_mode == "legacy":
                    matches = {"first": False, "second": False}
                    matches["first"] = mol.HasSubstructMatch(
                        reaction.GetReactantTemplate(0), useChirality=True
                    )
                    if reaction.GetNumReactantTemplates() == 2:
                        matches["second"] = mol.HasSubstructMatch(
                            reaction.GetReactantTemplate(1), useChirality=True
                        )
                    return matches
                # Rigid PGFS slot convention: R1 = SMARTS slot 0, pool R2 =
                # slot 1 (see ``_r2_reactant_template_indices``). Flipped
                # directions are separate ``_R2`` templates, so each molecule is
                # matched against a single designated slot per role rather than
                # OR-ed across both slots.
                r2_idxs = self._r2_reactant_template_indices(reaction, template)
                matches = {
                    "first": self._mol_matches_reactant_template(
                        mol, reaction.GetReactantTemplate(0)
                    ),
                    "second": self._mol_matches_reactant_template(
                        mol, reaction.GetReactantTemplate(r2_idxs[0])
                    )
                    if r2_idxs
                    else False,
                }
                return matches
        except Exception:
            return {"first": False, "second": False}

    def template_substructure_mask(self, reactant: str | None) -> torch.Tensor:
        """Pure pattern-only template mask: R1 first-reactant substructure match.

        Bit-identical to the legacy behaviour for both uni and bi templates.
        Per the README contract: this does not run the reaction and does not
        inspect R2 availability. A template can pass this mask and still fail
        ``apply_reaction`` later (RDKit kekulisation/sanitisation quirks,
        missing R2 for bi templates) — those failures are intended to surface
        as ``invalid_reaction_penalty`` (-1) at env-step time.

        ``r2_available`` is the masking type that *does* add the bi-R2 pattern
        check; ``reaction_valid`` is the masking type that guarantees no -1
        via a full apply_reaction validation.
        """
        key = (reactant, "substructure")
        if key not in self.template_mask_cache:
            values = [int(self.match_template(reactant, t)["first"]) for t in self.templates.values()]
            self.template_mask_cache[key] = torch.tensor(values, dtype=torch.float32)
        return self.template_mask_cache[key].clone()

    def template_r2_available_mask(self, reactant: str | None) -> torch.Tensor:
        """Validate R1 match and, for bimolecular templates, availability of any R2.

        Unimolecular templates do not run RDKit product generation here; they only need
        a first-reactant substructure match. Bimolecular templates additionally need at
        least one pool molecule that matches the template's second reactant pattern.
        """
        key = (reactant, "r2_available")
        if key not in self.template_mask_cache:
            out = torch.zeros(len(self.templates), dtype=torch.float32)
            for idx, template in self.templates.items():
                if not self.match_template(reactant, template)["first"]:
                    continue
                ttype = template.get("type", "unimolecular")
                if ttype in UNI_TYPES:
                    out[idx] = 1.0
                elif ttype == BI_TYPE and self.has_pattern_valid_r2(idx):
                    out[idx] = 1.0
            self.template_mask_cache[key] = out
        return self.template_mask_cache[key].clone()

    def template_reaction_valid_mask(self, reactant: str | None) -> torch.Tensor:
        """Validate by R1 match plus successful RDKit product generation.

        Uni-type templates (``unimolecular`` and ``unimolecular_explicit_reagent``)
        are validated with ``apply_reaction(state, t, None)`` — R2 is never used.
        This preserves the exact legacy uni behaviour: in ``reaction_mode: uni``
        the manager only holds uni-type templates, so the bi branch below is
        unreachable and the mask is bit-identical to the pre-fix implementation.

        Bi-type templates (``bimolecular``) require finding an R2: a template is
        valid iff R1 matches AND there is some ``r2`` in ``self.reactants`` such
        that ``apply_reaction(state, t, r2)`` returns a sanitised product. The
        loop also tries an initial ``None``-R2 call first so bi templates that
        bake fixed reagents into ``_explicit_reagents`` are handled by the same
        code path without paying the R2-pool scan.
        """
        key = (reactant, "reaction_valid")
        if key not in self.template_mask_cache:
            out = torch.zeros(len(self.templates), dtype=torch.float32)
            for idx, template in self.templates.items():
                if not self.match_template(reactant, template)["first"]:
                    continue
                ttype = template.get("type", "unimolecular")
                if ttype in UNI_TYPES:
                    if self.apply_reaction(reactant, template, None):
                        out[idx] = 1.0
                elif ttype == BI_TYPE:
                    if self.apply_reaction(reactant, template, None):
                        out[idx] = 1.0
                        continue
                    for r2 in self.get_valid_reactants(idx):
                        if self.apply_reaction(reactant, template, r2):
                            out[idx] = 1.0
                            break
            self.template_mask_cache[key] = out
        return self.template_mask_cache[key].clone()

    def get_mask(self, reactant: str | None, *, kind: str = "substructure") -> torch.Tensor:
        aliases = {
            "current": "substructure",
            "executable": "r2_available",
            "ppo_original": "reaction_valid",
        }
        kind = aliases.get(kind, kind)
        if kind == "none":
            return torch.ones(len(self.templates), dtype=torch.float32)
        if kind == "reaction_valid":
            return self.template_reaction_valid_mask(reactant)
        if kind == "r2_available":
            return self.template_r2_available_mask(reactant)
        if kind != "substructure":
            raise ValueError(f"Unsupported mask kind: {kind}")
        return self.template_substructure_mask(reactant)

    def get_feasible_mask(self, reactant: str | None) -> torch.Tensor:
        """Compatibility alias used by the existing PGFS TD3 agent."""
        return self.template_r2_available_mask(reactant)

    def feasible_first_reactant_templates(self, reactant: str | None, *, kind: str = "substructure") -> list[int]:
        mask = self.get_mask(reactant, kind=kind)
        return [int(i) for i in torch.where(mask > 0.5)[0]]

    def has_pattern_valid_r2(self, template_index: int) -> bool:
        store = self._r2_valid_indices_store
        if store is not None:
            return store.has_any(int(template_index))
        return bool(self.get_valid_reactants(template_index))

    def get_valid_reactant_indices(self, template_index: int) -> np.ndarray:
        """Global pool indices of pattern-valid R2 partners for ``template_index``."""
        store = self._r2_valid_indices_store
        if store is not None:
            return store.indices_for_template(int(template_index))
        template = self.templates[int(template_index)]
        reactant_keys = (
            list(self.reactants.keys())
            if isinstance(self.reactants, dict)
            else list(self.reactants)
        )
        return np.asarray(
            [
                i
                for i, smiles in enumerate(reactant_keys)
                if self.match_template(smiles, template)["second"]
            ],
            dtype=np.int32,
        )

    def get_valid_reactants(self, template_index: int) -> list[str]:
        if template_index not in self.valid_reactants_cache:
            reactant_keys = (
                list(self.reactants.keys())
                if isinstance(self.reactants, dict)
                else list(self.reactants)
            )
            idxs = self.get_valid_reactant_indices(template_index)
            self.valid_reactants_cache[template_index] = [
                reactant_keys[int(i)] for i in idxs
            ]
        return list(self.valid_reactants_cache[template_index])

    def r2_mask(self, template_index: int) -> np.ndarray:
        """Dense mask (legacy). Prefer :meth:`get_valid_reactant_indices` in hot paths."""
        mask = np.zeros(len(self.reactants), dtype=np.int8)
        idxs = self.get_valid_reactant_indices(template_index)
        if idxs.size:
            mask[idxs] = 1
        return mask

    def bi_r2_valid_indices(self, state: str | None, template_index: int) -> np.ndarray:
        """Global pool indices where ``apply_reaction(state, T, R2)`` succeeds."""
        template = self.templates.get(int(template_index))
        if state is None or template is None or template.get("type") != BI_TYPE:
            return np.zeros(0, dtype=np.int32)
        reactant_keys = (
            list(self.reactants.keys())
            if isinstance(self.reactants, dict)
            else list(self.reactants)
        )
        pattern_idx = self.get_valid_reactant_indices(template_index)
        if pattern_idx.size == 0:
            return pattern_idx
        hits: list[int] = []
        for i in pattern_idx:
            r2 = reactant_keys[int(i)]
            if self.apply_reaction(state, template, r2) is not None:
                hits.append(int(i))
        return np.asarray(hits, dtype=np.int32)

    def bi_r2_valid_mask(self, state: str | None, template_index: int) -> np.ndarray:
        """True-validity R2 mask for hierarchical Bi-PPO sampling.

        Returns a ``(len(reactants),)`` int8 mask where index ``i`` is 1 iff
        ``apply_reaction(state, templates[template_index], reactants[i])`` returns
        a sanitised product. The pattern-match set from ``get_valid_reactants`` is
        a strict superset; we filter it through RDKit so the policy can never
        sample a (T, R2) pair that the env would reject. Cached per (state, t).

        For uni-type templates or null states the result is a zero mask; callers
        should not invoke this for uni templates because the action space is
        single-Discrete in uni mode.
        """
        cache = getattr(self, "_bi_r2_valid_cache", None)
        if cache is None:
            cache = {}
            self._bi_r2_valid_cache = cache
        key = (state, int(template_index))
        cached = cache.get(key)
        if cached is not None:
            return cached.copy()
        mask = np.zeros(len(self.reactants), dtype=np.int8)
        template = self.templates.get(int(template_index))
        if state is None or template is None or template.get("type") != BI_TYPE:
            cache[key] = mask
            return mask.copy()
        valid_idx = self.bi_r2_valid_indices(state, int(template_index))
        if valid_idx.size:
            mask[valid_idx] = 1
        cache[key] = mask
        return mask.copy()
