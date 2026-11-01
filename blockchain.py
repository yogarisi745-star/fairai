"""
blockchain.py - Cryptographic Immutable Ledger for Legal Verdicts

Ensures that every judicial verdict, AI risk assessment, SHAP explanation,
and judge's rationale is immutably timestamped and cryptographically linked.
Any retroactive tampering breaks the SHA-256 chain.
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple


class Block:
    """Represents an immutable block in the judicial ledger."""

    def __init__(
        self,
        index: int,
        timestamp: float,
        case_data: Dict[str, Any],
        previous_hash: str,
        nonce: int = 0,
    ):
        self.index = index
        self.timestamp = timestamp
        self.case_data = case_data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculates SHA-256 hash of block contents."""
        block_string = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "case_data": self.case_data,
                "previous_hash": self.previous_hash,
                "nonce": self.nonce,
            },
            sort_keys=True,
        )
        return hashlib.sha256(block_string.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)
            ),
            "case_data": self.case_data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash,
        }


class JudicialBlockchain:
    """
    Cryptographic ledger storing judge determinations, AI scores, and rationale.
    Guarantees immutability and non-repudiation of judicial decisions.
    """

    def __init__(self, difficulty: int = 2):
        self.chain: List[Block] = []
        self.difficulty = difficulty
        # Create genesis block
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        genesis_data = {
            "case_id": "GENESIS-000",
            "message": "Judicial Integrity & AI Bias Mitigation Ledger Initialized",
            "jurisdiction": "High Court Legal AI Oversight Authority",
        }
        genesis_block = Block(
            index=0,
            timestamp=1700000000.0,
            case_data=genesis_data,
            previous_hash="0" * 64,
            nonce=0,
        )
        self.chain.append(genesis_block)

    @property
    def latest_block(self) -> Block:
        return self.chain[-1]

    def record_verdict(
        self,
        case_id: str,
        accused_id: str,
        demographics: Dict[str, Any],
        merits: Dict[str, Any],
        ai_original_score: float,
        ai_debiased_score: float,
        bias_gap: float,
        shap_summary: Dict[str, float],
        ai_recommended_verdict: str,
        judge_id: str,
        judge_action: str,  # "ACCEPTED" or "OVERRIDDEN"
        final_verdict: str,
        judicial_reasoning: str,
    ) -> Block:
        """Records a completed judicial case verdict onto the blockchain."""
        case_payload = {
            "case_id": case_id,
            "accused_id": accused_id,
            "demographics": demographics,
            "merits": merits,
            "ai_original_score": round(ai_original_score, 2),
            "ai_debiased_score": round(ai_debiased_score, 2),
            "bias_gap": round(bias_gap, 2),
            "shap_summary": {k: round(v, 2) for k, v in shap_summary.items()},
            "ai_recommended_verdict": ai_recommended_verdict,
            "judge_id": judge_id,
            "judge_action": judge_action,
            "final_verdict": final_verdict,
            "judicial_reasoning": judicial_reasoning,
            "recorded_at": time.time(),
        }

        new_index = len(self.chain)
        timestamp = time.time()
        prev_hash = self.latest_block.hash

        # Lightweight proof of work to prevent spam/tampering
        nonce = 0
        prefix = "0" * self.difficulty
        while True:
            candidate = Block(new_index, timestamp, case_payload, prev_hash, nonce)
            if candidate.hash.startswith(prefix):
                break
            nonce += 1

        self.chain.append(candidate)
        return candidate

    def verify_chain_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Validates cryptographic hashes and links across the entire blockchain.
        Returns (True, None) if valid, or (False, reason) if corrupted.
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prev = self.chain[i - 1]

            # 1. Check current block hash matches recalculated hash
            if current.hash != current.calculate_hash():
                return False, f"Block #{current.index} hash mismatch! Data has been tampered with."

            # 2. Check block points to previous block's hash
            if current.previous_hash != prev.hash:
                return (
                    False,
                    f"Block #{current.index} previous_hash does not match Block #{prev.index} hash!",
                )

            # 3. Check proof of work
            if not current.hash.startswith("0" * self.difficulty):
                return False, f"Block #{current.index} does not satisfy proof-of-work difficulty."

        return True, "Chain integrity verified. All records are cryptographically authentic."

    def simulate_tampering(self, block_index: int, new_verdict: str) -> None:
        """Demonstrates what happens if a malicious entity tries to alter past records."""
        if block_index < 1 or block_index >= len(self.chain):
            raise IndexError("Invalid block index to tamper")
        self.chain[block_index].case_data["final_verdict"] = new_verdict
        # Note: hash is intentionally not updated to simulate unauthorized database edit
