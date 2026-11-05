"""Provider keys, SimHash buckets and keyed bucket directions.

The watermark lives in a *direction field*: every SimHash bucket ``b`` of the
embedding space gets a provider-secret unit vector ``v_k(b)`` derived from the
provider key ``k``. The recipes below are bit-for-bit those of the paper code,
including the legacy ``numpy.random.RandomState`` generators, so directions
(and therefore watermarked corpora) produced by either implementation are
interchangeable.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
import secrets
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

__all__ = [
    "SimHashCoder",
    "DirectionSource",
    "DirectionManager",
    "OrthoDirectionManager",
    "generate_provider_keys",
    "save_provider_keys",
    "load_provider_keys",
    "ProviderKeys",
    "UnknownProviderError",
]


class SimHashCoder:
    """Public SimHash bucketizer with ``bits`` random hyperplanes.

    The hyperplanes come from ``RandomState(seed)`` and are public by design:
    secrecy lives entirely in the per-bucket directions. ``code`` returns the
    bucket id as ASCII-hex ``bytes`` (bits packed LSB-first per byte), which is
    also the HMAC payload used to derive directions.
    """

    def __init__(self, dim: int, bits: int = 96, seed: int = 2025):
        self.dim = int(dim)
        self.bits = int(bits)
        self.seed = int(seed)
        rng = np.random.RandomState(self.seed)
        self.R = rng.normal(0.0, 1.0, size=(self.bits, self.dim)).astype(np.float32)

    def code(self, u: np.ndarray) -> bytes:
        """Bucket code of one embedding."""
        bits = (self.R @ u.astype(np.float32)) >= 0.0
        pad = (-len(bits)) % 8
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=bool)])
        byte_vals = (bits.reshape(-1, 8) * (1 << np.arange(8))).sum(axis=1).astype(np.uint8)
        return bytearray(byte_vals).hex().encode("ascii")


class DirectionSource(Protocol):
    """Anything that maps a bucket code to a unit direction."""

    def direction(self, bucket_code: bytes) -> np.ndarray: ...


def _hmac_sha256(key: bytes, payload: bytes) -> bytes:
    return hmac.new(key, payload, hashlib.sha256).digest()


def _keyed_gaussian(key: bytes, payload: bytes, dim: int) -> np.ndarray:
    seed = int.from_bytes(_hmac_sha256(key, payload)[:8], "big", signed=False) & 0x7FFFFFFF
    v = np.random.RandomState(seed).normal(0.0, 1.0, size=(dim,)).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    return v


class DirectionManager:
    """Per-provider keyed directions (the ``naive`` scheme used in the paper).

    ``v_k(b) = normalize(N(0, I_d))`` drawn from ``RandomState(seed)`` with
    ``seed = HMAC-SHA256(k, b)[:8] & 0x7fffffff``. Directions are cached.
    """

    def __init__(self, key_hex: str, dim: int):
        self.key = bytes.fromhex(key_hex)
        self.dim = int(dim)
        self._cache: dict[bytes, np.ndarray] = {}

    def direction(self, bucket_code: bytes) -> np.ndarray:
        v = self._cache.get(bucket_code)
        if v is None:
            v = _keyed_gaussian(self.key, bucket_code, self.dim)
            self._cache[bucket_code] = v
        return v


class OrthoDirectionManager:
    """Jointly orthonormalised directions for a fixed set of providers (ablation).

    For each bucket the ``P`` providers' keyed base vectors (HMAC payload
    ``b"dirbucket:base:" + code``) are orthonormalised with a QR decomposition,
    with signs fixed to agree with the base vectors. Providers are ordered by
    ``sorted(keys)``. The paper found this reduces watermark strength without a
    meaningful calibration benefit, so ``naive`` is the default everywhere.
    """

    def __init__(self, keys_by_provider: Mapping[Hashable, str], dim: int):
        self.dim = int(dim)
        self.keys = {pid: bytes.fromhex(k) for pid, k in keys_by_provider.items()}
        self.pids = sorted(self.keys)
        self._index = {pid: i for i, pid in enumerate(self.pids)}
        if len(self.pids) > self.dim:
            raise ValueError(f"cannot orthogonalise {len(self.pids)} providers in dim={self.dim}")
        self._cache: dict[bytes, np.ndarray] = {}

    def _bucket(self, bucket_code: bytes) -> np.ndarray:
        V = self._cache.get(bucket_code)
        if V is None:
            M = np.stack(
                [_keyed_gaussian(self.keys[pid], b"dirbucket:base:" + bucket_code, self.dim) for pid in self.pids]
            ).astype(np.float32)
            Q, _ = np.linalg.qr(M.T.astype(np.float64, copy=False))
            V = Q[:, : len(self.pids)].T.astype(np.float32)
            signs = np.where((V * M).sum(axis=1) >= 0.0, 1.0, -1.0).astype(np.float32)
            V = V * signs[:, None]
            V = (V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
            self._cache[bucket_code] = V
        return V

    def direction_for(self, provider_id: Hashable, bucket_code: bytes) -> np.ndarray:
        return self._bucket(bucket_code)[self._index[provider_id]]

    def for_provider(self, provider_id: Hashable) -> DirectionSource:
        """A :class:`DirectionSource` view for one provider."""
        return _OrthoView(self, provider_id)


class _OrthoView:
    def __init__(self, mgr: OrthoDirectionManager, provider_id: Hashable):
        self.mgr = mgr
        self.provider_id = provider_id

    def direction(self, bucket_code: bytes) -> np.ndarray:
        return self.mgr.direction_for(self.provider_id, bucket_code)


def generate_provider_keys(n: int, seed: int | None = None) -> list[str]:
    """Generate ``n`` 256-bit provider keys as hex strings.

    With ``seed=None`` (the default) keys come from :mod:`secrets` and are
    suitable for real use. With an integer seed, keys are
    ``random.Random(seed).randbytes(32)`` -- the deterministic scheme used to
    build the paper corpora (``seed=13``). Seeded keys are public and must
    never be used to protect real data.
    """
    if seed is None:
        return [secrets.token_hex(32) for _ in range(n)]
    prng = random.Random(seed)
    return [prng.randbytes(32).hex() for _ in range(n)]


def save_provider_keys(path: str | Path, keys: Sequence[str], **meta: Any) -> None:
    """Write keys in the ``provider_keys.json`` layout used by the paper code."""
    payload = {"keys_hex": list(keys), "providers": len(keys), **meta}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_provider_keys(path: str | Path) -> list[str]:
    """Read ``keys_hex`` from a ``provider_keys.json`` file (index = provider id)."""
    return list(json.loads(Path(path).read_text(encoding="utf-8"))["keys_hex"])


class UnknownProviderError(KeyError, ValueError):
    """A document or reference names a provider that has no key."""

    def __str__(self) -> str:  # KeyError would repr-quote the message
        return str(self.args[0]) if self.args else ""


def _parse_providers(providers: int | str | Iterable[Hashable]) -> list[Hashable]:
    if isinstance(providers, bool):
        raise TypeError("providers must be a count or a list of names")
    if isinstance(providers, int):
        return list(range(providers))
    if isinstance(providers, str):
        s = providers.strip()
        if s.isdigit():
            return list(range(int(s)))
        return [p.strip() for p in s.split(",") if p.strip()]
    return list(providers)


class ProviderKeys(Mapping[Hashable, str]):
    """Secret keys for any number of named providers.

    A read-only mapping ``provider id -> key hex``. Provider ids are names
    (``"acme"``) or, for the paper's key files, integers ``0..n-1``. Lookups
    accept either spelling (``keys["0"]`` finds provider ``0``), so ids read
    from CSV or directory names resolve correctly. ``repr`` never shows keys.

    The JSON file written by :meth:`save` is a superset of the paper's
    ``provider_keys.json`` (``keys_hex`` in provider order, plus
    ``provider_names`` for named providers), so both formats load here and in
    :func:`load_provider_keys`.

    Example::

        keys = ProviderKeys.generate(["acme", "globex", "initech"])   # secret keys
        keys.save("provider_keys.json")
        keys = ProviderKeys.load("provider_keys.json")
        keys["acme"]   # -> '3f9c...'
    """

    def __init__(self, keys: Mapping[Hashable, str] | Sequence[str]):
        items = list(keys.items()) if isinstance(keys, Mapping) else list(enumerate(keys))
        self._keys: dict[Hashable, str] = {}
        for pid, k in items:
            k = str(k).strip().lower()
            try:
                if not k or len(bytes.fromhex(k)) < 16:
                    raise ValueError
            except ValueError:
                raise ValueError(f"provider {pid!r}: key must be a hex string of at least 16 bytes") from None
            if pid in self._keys:
                raise ValueError(f"duplicate provider id {pid!r}")
            self._keys[pid] = k

    @classmethod
    def generate(cls, providers: int | str | Iterable[Hashable], *, seed: int | None = None) -> ProviderKeys:
        """Fresh 256-bit keys for ``providers`` (a count, ``"a,b,c"`` or a list of names).

        ``seed=None`` draws keys from the OS CSPRNG. An integer seed gives the
        deterministic, public keys of :func:`generate_provider_keys` (the paper
        used ``seed=13``); never protect real data with them.
        """
        ids = _parse_providers(providers)
        if len(set(ids)) != len(ids):
            raise ValueError("provider names must be unique")
        return cls(dict(zip(ids, generate_provider_keys(len(ids), seed=seed))))

    @classmethod
    def load(cls, path: str | Path) -> ProviderKeys:
        """Read a key file written by :meth:`save` or by the paper code.

        Also accepted: a JSON list of hex keys (providers ``0..n-1``),
        ``{"keys": {name: hex}}`` and a flat ``{name: hex}`` mapping.
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: not a provider key file (invalid JSON: {e.msg}); create one with `rentarag keygen`") from e
        if isinstance(data, dict) and isinstance(data.get("keys"), dict):
            return cls(data["keys"])
        if isinstance(data, list):
            return cls(data)
        if isinstance(data, dict) and isinstance(data.get("keys_hex"), list):
            keys = list(data["keys_hex"])
            names = data.get("provider_names")
            if names is not None and len(names) != len(keys):
                raise ValueError(f"{path}: provider_names and keys_hex have different lengths")
            return cls(dict(zip(names, keys)) if names else keys)
        if isinstance(data, dict) and data and all(isinstance(v, str) and _is_hex(v) for v in data.values()):
            return cls(data)
        raise ValueError(
            f"{path}: not a provider key file (expected 'keys_hex', 'keys', a list of keys or {{name: hex key}}); "
            "create one with `rentarag keygen`"
        )

    def save(self, path: str | Path, **meta: Any) -> None:
        """Write the keys as JSON (keep this file secret)."""
        payload: dict[str, Any] = {"keys_hex": list(self._keys.values()), "providers": len(self._keys)}
        if not self.is_indexed:
            payload["provider_names"] = [str(p) for p in self._keys]
        payload.update(meta)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @property
    def is_indexed(self) -> bool:
        """True for paper-style ids ``0..n-1``."""
        return list(self._keys) == list(range(len(self._keys)))

    @property
    def ids(self) -> list[Hashable]:
        return list(self._keys)

    @property
    def keys_hex(self) -> list[str]:
        return list(self._keys.values())

    def resolve(self, provider: Hashable) -> Hashable:
        """Canonical id for ``provider`` (``"0"`` -> ``0`` for indexed keys, ``0`` -> ``"0"`` for names)."""
        if provider in self._keys:
            return provider
        if isinstance(provider, str):
            s = provider.strip()
            if s in self._keys:
                return s
            if s.lstrip("-").isdigit() and int(s) in self._keys:
                return int(s)
        elif isinstance(provider, int) and not isinstance(provider, bool) and str(provider) in self._keys:
            return str(provider)
        n = len(self._keys)
        if self.is_indexed:
            raise UnknownProviderError(f"provider {provider!r} out of range 0..{n - 1} ({n} keys)")
        shown = ", ".join(map(str, list(self._keys)[:12])) + (", ..." if n > 12 else "")
        raise UnknownProviderError(f"unknown provider {provider!r} (known: {shown})")

    def __getitem__(self, provider: Hashable) -> str:
        return self._keys[self.resolve(provider)]

    def __contains__(self, provider: object) -> bool:
        try:
            self.resolve(provider)  # type: ignore[arg-type]
            return True
        except (UnknownProviderError, TypeError):
            return False

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        return f"ProviderKeys({self.ids!r})"


def _is_hex(s: str) -> bool:
    try:
        bytes.fromhex(s.strip())
    except ValueError:
        return False
    return bool(s.strip())


def as_provider_keys(keys: ProviderKeys | Mapping[Hashable, str] | Sequence[str] | str | Path) -> ProviderKeys:
    """Coerce a key file path, mapping, list or :class:`ProviderKeys`."""
    if isinstance(keys, ProviderKeys):
        return keys
    if isinstance(keys, (str, Path)):
        return ProviderKeys.load(keys)
    return ProviderKeys(keys)
