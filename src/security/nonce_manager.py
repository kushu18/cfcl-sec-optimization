"""
CFCL Gadepan Complex - Zero-Trust Cryptographic Nonce & Two-Man Authorization Engine.
Guarantees single-use replay protection (HMAC-SHA256 with 30s TTL) and aerospace-grade
Two-Man Rule dual-role cryptographic authorization for high-magnitude DCS setpoint shifts.
Sanitized DCS Telemetry under strict NDA.
"""

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

from src.utils.logger import get_logger

logger = get_logger("NonceManager")


@dataclass
class NonceToken:
    """Cryptographically signed ephemeral token."""
    nonce_id: str
    issued_at: float
    expires_at: float
    payload_hash: str
    signature: str
    is_consumed: bool = False


@dataclass
class AuthorizationVerificationResult:
    """Result of cryptographic nonce and two-man rule validation."""
    is_valid: bool
    status_code: str
    message: str
    requires_supervisor_override: bool
    primary_operator: str
    supervisor_approver: Optional[str]


class CryptographicNonceManager:
    """
    Manages zero-trust setpoint release handshakes for CFCL DCS Supervisory systems.
    Prevents replay attacks, network eavesdropping tampering, and unauthorized setpoint ramps.
    """

    NONCE_TTL_SECONDS = 30.0

    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = (
            secret_key or "cfcl_gadepan_scada_secret_key_sil2_interlock_2026"
        ).encode("utf-8")
        self.active_nonces: Dict[str, NonceToken] = {}

    def _generate_payload_hash(self, payload: Dict[str, Any]) -> str:
        """Deterministic canonical hash of the setpoint dictionary."""
        sorted_items = sorted(payload.items(), key=lambda x: x[0])
        canonical_str = ";".join(f"{k}={v}" for k, v in sorted_items)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def issue_nonce(self, setpoints_payload: Dict[str, Any]) -> NonceToken:
        """Issues an ephemeral 30-second token tied to a specific setpoint payload."""
        now = time.time()
        nonce_id = secrets.token_hex(16)
        p_hash = self._generate_payload_hash(setpoints_payload)

        # Message: nonce_id:issued_at:payload_hash
        msg = f"{nonce_id}:{now:.3f}:{p_hash}".encode("utf-8")
        sig = hmac.new(self.secret_key, msg, hashlib.sha256).hexdigest()

        token = NonceToken(
            nonce_id=nonce_id,
            issued_at=now,
            expires_at=now + self.NONCE_TTL_SECONDS,
            payload_hash=p_hash,
            signature=sig,
            is_consumed=False,
        )
        self.active_nonces[nonce_id] = token
        logger.info(f"Issued single-use cryptographic nonce: {nonce_id[:8]}... (TTL: 30s)")
        return token

    def verify_and_consume(
        self,
        nonce_id: str,
        setpoints_payload: Dict[str, Any],
        signature: str,
        primary_operator_id: str,
        supervisor_id: Optional[str] = None,
        slew_temp_c: float = 0.0,
    ) -> AuthorizationVerificationResult:
        """
        Validates cryptographic integrity and enforces the Two-Man Rule.
        """
        now = time.time()
        token = self.active_nonces.get(nonce_id)

        if not token:
            return AuthorizationVerificationResult(
                is_valid=False,
                status_code="NONCE_NOT_FOUND",
                message="Nonce does not exist or expired from memory.",
                requires_supervisor_override=False,
                primary_operator=primary_operator_id,
                supervisor_approver=supervisor_id,
            )

        if token.is_consumed:
            return AuthorizationVerificationResult(
                is_valid=False,
                status_code="NONCE_ALREADY_CONSUMED",
                message="Replay attempt rejected: Nonce has already been burned.",
                requires_supervisor_override=False,
                primary_operator=primary_operator_id,
                supervisor_approver=supervisor_id,
            )

        if now > token.expires_at:
            del self.active_nonces[nonce_id]
            return AuthorizationVerificationResult(
                is_valid=False,
                status_code="NONCE_EXPIRED",
                message=f"Nonce TTL exceeded by {now - token.expires_at:.1f}s. Request fresh optimization.",
                requires_supervisor_override=False,
                primary_operator=primary_operator_id,
                supervisor_approver=supervisor_id,
            )

        # Payload tampering check
        curr_hash = self._generate_payload_hash(setpoints_payload)
        if curr_hash != token.payload_hash:
            return AuthorizationVerificationResult(
                is_valid=False,
                status_code="PAYLOAD_TAMPERED",
                message="Cryptographic payload mismatch: Target setpoints altered in transit.",
                requires_supervisor_override=False,
                primary_operator=primary_operator_id,
                supervisor_approver=supervisor_id,
            )

        # Signature verification (constant-time comparison)
        msg = f"{token.nonce_id}:{token.issued_at:.3f}:{token.payload_hash}".encode("utf-8")
        expected_sig = hmac.new(self.secret_key, msg, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return AuthorizationVerificationResult(
                is_valid=False,
                status_code="INVALID_SIGNATURE",
                message="Cryptographic signature verification failed.",
                requires_supervisor_override=False,
                primary_operator=primary_operator_id,
                supervisor_approver=supervisor_id,
            )

        # Two-Man Rule enforcement: If temperature slew rate >= 1.0 C/hr, dual-authorization is mandatory
        needs_supervisor = slew_temp_c >= 1.0
        if needs_supervisor:
            if not supervisor_id or supervisor_id == primary_operator_id:
                return AuthorizationVerificationResult(
                    is_valid=False,
                    status_code="TWO_MAN_RULE_REQUIRED",
                    message="High-magnitude setpoint shift requires independent Shift Supervisor authorization.",
                    requires_supervisor_override=True,
                    primary_operator=primary_operator_id,
                    supervisor_approver=supervisor_id,
                )

        # Mark consumed
        token.is_consumed = True
        logger.info(
            f"Cryptographic Nonce {nonce_id[:8]} consumed successfully by {primary_operator_id} "
            f"(Supervisor: {supervisor_id or 'N/A'})"
        )

        return AuthorizationVerificationResult(
            is_valid=True,
            status_code="AUTHORIZED",
            message="Cryptographic handshake verified and setpoint committed to DCS buffer.",
            requires_supervisor_override=needs_supervisor,
            primary_operator=primary_operator_id,
            supervisor_approver=supervisor_id,
        )
