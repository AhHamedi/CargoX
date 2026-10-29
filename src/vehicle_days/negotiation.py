"""Small, observation-local negotiation protocol and conserved credit ledger.

The ledger is authoritative runtime state.  Only ``make_observation`` exports an
agent's own funds; rendering never receives another agent's private task/contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from threading import RLock
from typing import Any, Mapping


# Normalise numerical clock drift before the discontinuous deadline predicate.
# This absolute one-nanosecond rule never changes protocol/delivery expiry.
DEADLINE_TOLERANCE_SECONDS = 1e-9


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, minimum: float = 0.) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def deadline_lateness(duration: float, target_seconds: float) -> float:
    """Snap at most 1ns of positive clock drift to the task deadline.

    Lateness outside that absolute tolerance is charged in full. In particular,
    this is not a relative tolerance, a grace period or an expiry rule.
    """
    duration=_number(duration,"duration");target_seconds=_number(target_seconds,"target_seconds")
    return 0. if duration<=target_seconds+DEADLINE_TOLERANCE_SECONDS else duration-target_seconds


@dataclass(frozen=True)
class Goal:
    target_seconds: float
    travel_cost: float = 1.
    missed_cost: float = 0.
    late_cost: float = 0.

    def __post_init__(self):
        for name, value in asdict(self).items():
            _number(value, name)

    def deadline_cost(self, duration: float) -> float:
        late=deadline_lateness(duration,self.target_seconds)
        return self.missed_cost*(late>0.)+self.late_cost*late

    def cost(self, duration: float, *, failure: bool = False, timeout: float | None = None) -> float:
        """Task costs use scheduled timeout on failure; failure adds 100."""
        duration = _number(duration, "duration")
        if failure:
            if timeout is None:
                raise ValueError("a failed task requires its scheduled timeout")
            duration = _number(timeout, "timeout")
        return self.travel_cost*duration+self.deadline_cost(duration)+100.*bool(failure)


@dataclass(frozen=True)
class Contract:
    """One planner authority epoch; versions increase whenever authority changes.

    spend_limit includes all reserved/settled spending in this version.  Funds
    received do not replenish that permission.  reserve is a hard wallet floor.
    """
    spend_limit: int
    reserve: int = 0
    max_yield_seconds: float = 8.
    allow_gift: bool = False
    version: int = 1

    def __post_init__(self):
        _integer(self.spend_limit, "spend_limit")
        _integer(self.reserve, "reserve")
        _integer(self.version, "version", 1)
        if _number(self.max_yield_seconds, "max_yield_seconds") > 60.:
            raise ValueError("max_yield_seconds exceeds the pilot's 60-second bound")
        if type(self.allow_gift) is not bool:
            raise ValueError("allow_gift must be boolean")
        if self.allow_gift and self.max_yield_seconds == 0:
            raise ValueError("allow_gift contradicts a zero yielding limit")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Contract:
        try:
            return cls(**value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported or contradictory planner instructions: {exc}") from exc


class Ledger:
    """Integer credits, atomic reservations, idempotent delivery/refund events."""

    def __init__(self, balances: Mapping[str, int]):
        if not balances or any(not isinstance(k, str) or not k for k in balances):
            raise ValueError("balances require nonempty owner IDs")
        self.balances = {k: _integer(v, "balance") for k, v in balances.items()}
        self.initial = dict(self.balances)
        self.reservations: dict[str, dict] = {}
        self.events: list[dict] = []
        self._lock = RLock()

    def locked(self, owner: str) -> int:
        return sum(r["price"] for r in self.reservations.values()
                   if r["buyer"] == owner and r["status"] == "running")

    def available(self, owner: str) -> int:
        return self.balances[owner] - self.locked(owner)

    def active(self, owner: str) -> dict | None:
        return next((dict(r) for r in self.reservations.values()
                     if owner in (r["seller"], r["buyer"]) and r["status"] == "running"), None)

    def committed(self, owner: str, version: int) -> int:
        return sum(r["price"] for r in self.reservations.values()
                   if r["buyer"] == owner and r["payer_contract_version"] == version
                   and r["status"] in ("running", "delivered"))

    def reserve(self, uid: str, payer: str, recipient: str, amount: int, expires: float,
                *, now: float = 0., merge_id: str = "", authority_version: int | None = None) -> dict:
        amount = _integer(amount, "amount")
        now, expires = _number(now, "now"), _number(expires, "expires")
        if not isinstance(uid, str) or not uid or payer == recipient or amount > sum(self.initial.values()) or expires <= now:
            raise ValueError("invalid reservation identity, price, parties or window")
        if payer not in self.balances or recipient not in self.balances:
            raise ValueError("unknown credit owner")
        if authority_version is not None:
            _integer(authority_version, "authority_version", 1)
        with self._lock:
            existing = self.reservations.get(uid)
            if existing is not None:
                expected = (payer, recipient, amount, expires, merge_id, authority_version)
                actual = tuple(existing[k] for k in ("buyer", "seller", "price", "expires_at", "merge_id", "payer_contract_version"))
                if actual != expected:
                    raise ValueError("reservation ID reused with different terms")
                return dict(existing)
            if self.active(payer) or self.active(recipient):
                raise ValueError("a party already has an executable commitment")
            if self.available(payer) < amount:
                raise ValueError("insufficient unreserved credits")
            reservation = {"id": uid, "merge_id": merge_id, "seller": recipient,
                           "buyer": payer, "price": amount, "accepted_at": now,
                           "expires_at": expires, "payer_contract_version": authority_version,
                           "status": "running"}
            self.reservations[uid] = reservation
            self.events.append({"kind": "reserve", **reservation})
            return dict(reservation)

    def settle(self, uid: str, now: float, *, delivered: bool = True, evidence: str = "rear_cleared_exit") -> dict:
        now = _number(now, "now")
        if type(delivered) is not bool:
            raise ValueError("delivered must be boolean")
        with self._lock:
            reservation = self.reservations[uid]
            if reservation["status"] != "running":
                return dict(reservation)
            if now < reservation["accepted_at"]:
                raise ValueError("delivery predates acceptance")
            if now >= reservation["expires_at"]:
                return self.refund(uid, "expired", now=now)
            if not delivered:
                return self.refund(uid, "blocked", now=now)
            if not isinstance(evidence, str) or not evidence:
                raise ValueError("delivery requires an external evidence reference")
            self.balances[reservation["buyer"]] -= reservation["price"]
            self.balances[reservation["seller"]] += reservation["price"]
            reservation.update(status="delivered", resolved_at=now)
            self.events.append({"kind": "settle", "id": uid, "now": now, "evidence": evidence})
            return dict(reservation)

    def refund(self, uid: str, reason: str = "cancelled", *, now: float | None = None) -> dict:
        if reason not in ("cancelled", "expired", "blocked", "failure"):
            raise ValueError("unknown refund reason")
        if now is not None:
            now = _number(now, "now")
        with self._lock:
            reservation = self.reservations[uid]
            if reservation["status"] != "running":
                return dict(reservation)
            if now is not None and now < reservation["accepted_at"]:
                raise ValueError("refund predates acceptance")
            reservation.update(status=reason, resolved_at=now)
            self.events.append({"kind": "refund", "id": uid, "reason": reason, "now": now})
            return dict(reservation)

    def expire(self, now: float) -> None:
        now = _number(now, "now")
        for uid, reservation in list(self.reservations.items()):
            if reservation["status"] == "running" and now >= reservation["expires_at"]:
                self.refund(uid, "expired", now=now)

    def audit(self) -> dict:
        """Reconstruct balances/reservations from events independently of methods."""
        balances, active, seen, replayed, errors = dict(self.initial), {}, set(), {}, []
        for index, event in enumerate(self.events):
            uid, kind = event["id"], event["kind"]
            if kind == "reserve":
                if uid in seen or type(event["price"]) is not int or not 0 <= event["price"] <= sum(self.initial.values()):
                    errors.append(f"event {index}: duplicate or invalid reservation")
                owners = {event["buyer"], event["seller"]}
                if any(owners & {r["buyer"], r["seller"]} for r in active.values()):
                    errors.append(f"event {index}: overlapping commitment")
                if event["buyer"] not in balances or event["seller"] not in balances:
                    errors.append(f"event {index}: unknown owner")
                    continue
                if balances[event["buyer"]] < event["price"]:
                    errors.append(f"event {index}: unaffordable reservation")
                active[uid] = event
                replayed[uid] = {k: v for k, v in event.items() if k != "kind"}
                seen.add(uid)
            elif kind in ("settle", "refund"):
                reservation = active.pop(uid, None)
                if reservation is None:
                    errors.append(f"event {index}: resolution without active reservation")
                elif kind == "settle":
                    if not reservation["accepted_at"] <= event["now"] < reservation["expires_at"] or not event.get("evidence"):
                        errors.append(f"event {index}: unsupported/out-of-window delivery")
                    balances[reservation["buyer"]] -= reservation["price"]
                    balances[reservation["seller"]] += reservation["price"]
                    replayed[uid].update(status="delivered", resolved_at=event["now"])
                else:
                    if (event.get("reason") not in ("cancelled", "expired", "blocked", "failure")
                            or event.get("now") is not None and event["now"] < reservation["accepted_at"]):
                        errors.append(f"event {index}: invalid refund")
                    replayed[uid].update(status=event.get("reason"), resolved_at=event.get("now"))
            else:
                errors.append(f"event {index}: unknown event")
        if balances != self.balances:
            errors.append("replayed balances differ")
        running = {uid for uid, r in self.reservations.items() if r["status"] == "running"}
        if set(active) != running:
            errors.append("replayed reservations differ")
        if replayed != self.reservations:
            errors.append("replayed commitment terms or status differ")
        if sum(balances.values()) != sum(self.initial.values()) or min(balances.values()) < 0:
            errors.append("nonconserved or negative credits")
        return {"ok": not errors, "errors": errors, "balances": balances,
                "total": sum(balances.values()), "events": len(self.events)}

