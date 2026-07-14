"""Product selection policies for RDKit reaction outcomes."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import QED

from src.chem.rdkit_logging import suppress_rdkit_logs


def best_qed_product_smiles(product_sets) -> str | None:
    """Select the first product from each product set, sanitized, maximizing QED."""
    valid_products = []
    fallback_smiles: list[str] = []
    with suppress_rdkit_logs():
        for product_set in product_sets or []:
            if not product_set:
                continue
            product = product_set[0]
            try:
                if Chem.SanitizeMol(product, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE:
                    valid_products.append(product)
                    continue
            except Exception:
                pass
            # Some multi-component templates yield kekulization warnings but still
            # round-trip through MolToSmiles (e.g. triazole cyclizations).
            try:
                smi = Chem.MolToSmiles(product)
                if smi:
                    roundtrip = Chem.MolFromSmiles(smi)
                    if roundtrip is not None:
                        valid_products.append(roundtrip)
                    else:
                        fallback_smiles.append(smi)
            except Exception:
                continue
        if valid_products:
            try:
                return Chem.MolToSmiles(max(valid_products, key=QED.qed))
            except Exception:
                return None
        if fallback_smiles:
            return fallback_smiles[0]
        return None
