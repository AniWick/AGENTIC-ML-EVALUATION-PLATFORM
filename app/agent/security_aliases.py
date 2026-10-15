"""Canonical security-vocabulary helpers shared by the router and the planner.

The platform's canonical identifiers use underscores (``feature_squeezing``)
while users type prose ("feature squeezing"). Every alias pattern therefore runs
against a normalized string where ``_``, ``-`` and repeated whitespace all
collapse to a single space, so both forms resolve identically.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Framework capability maps (authoritative)
# ----------------------------------------------------------------------

ATTACKS_BY_FRAMEWORK: Dict[str, List[str]] = {
    "sklearn": ["boundary_attack", "hopskipjump", "zoo"],
    "pytorch": ["boundary_attack", "hopskipjump", "zoo", "fgsm", "pgd", "cw"],
    "onnx": ["boundary_attack", "hopskipjump", "zoo"],
}

HARDENING_METHODS_BY_FRAMEWORK: Dict[str, List[str]] = {
    "pytorch": ["fgsm", "pgd"],
    "sklearn": ["boundary_attack", "hopskipjump"],
    "onnx": ["boundary_attack", "hopskipjump"],
}

ATTACK_LABELS: Dict[str, str] = {
    "boundary_attack": "Boundary Attack",
    "hopskipjump": "HopSkipJump Attack",
    "zoo": "ZOO Attack",
    "fgsm": "FGSM",
    "pgd": "PGD",
    "cw": "C&W",
}

DEFENSE_LABELS: Dict[str, str] = {
    "gaussian_augmentation": "Gaussian Augmentation",
    "spatial_smoothing": "Spatial Smoothing",
    "feature_squeezing": "Feature Squeezing",
    "clip_values": "Clip Values",
    "reverse_sigmoid": "Reverse Sigmoid",
    "class_labels": "Class Labels",
    "high_confidence_masking": "High Confidence Masking",
}

#: Preprocessor defenses are non-differentiable, so gradient-based attacks
#: cannot see a meaningful gradient through them.
NON_DIFFERENTIABLE_DEFENSES = {
    "gaussian_augmentation",
    "spatial_smoothing",
    "feature_squeezing",
    "clip_values",
}

GRADIENT_ATTACKS = {"fgsm", "pgd", "cw"}

# Aliases are matched against NORMALIZED text, so a single space separates
# words and underscores/hyphens are already folded away.
ATTACK_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("fgsm", (r"\bfast gradient sign method\b", r"\bfgsm\b")),
    ("pgd", (r"\bprojected gradient descent\b", r"\bpgd\b")),
    ("cw", (r"\bc&w\b", r"\bcarlini wagner\b", r"\bcw\b")),
    ("hopskipjump", (r"\bhop skip jump\b", r"\bhopskipjump\b")),
    ("boundary_attack", (r"\bboundary attack\b", r"\bboundaryattack\b")),
    ("zoo", (r"\bzoo attack\b", r"\bzoo\b")),
)

DEFENSE_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("feature_squeezing", (r"\bfeature squeezing\b",)),
    ("spatial_smoothing", (r"\bspatial smoothing\b",)),
    ("gaussian_augmentation", (r"\bgaussian augmentation\b",)),
    ("high_confidence_masking", (r"\bhigh confidence masking\b",)),
    ("reverse_sigmoid", (r"\breverse sigmoid\b",)),
    ("class_labels", (r"\bclass labels?\b",)),
    ("clip_values", (r"\bclip values\b",)),
)

_HARDENING_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("fgsm", (r"\bfast gradient sign method\b", r"\bfgsm\b")),
    ("boundary_attack", (r"\bboundary attack\b",)),
    ("hopskipjump", (r"\bhop skip jump\b", r"\bhopskipjump\b")),
    ("pgd", (r"\bprojected gradient descent\b", r"\bpgd\b")),
)


class IncompatibleSecurityRequest(ValueError):
    """Raised when a request cannot be executed against the active model."""


# ----------------------------------------------------------------------
# Normalization + matching
# ----------------------------------------------------------------------

def normalize(text: Optional[str]) -> str:
    """Fold case, underscores, hyphens and repeated whitespace into one space."""
    raw = (text or "").lower()
    raw = re.sub(r"[_]+", " ", raw)
    raw = re.sub(r"-+", " ", raw)
    raw = re.sub(r"[^a-z0-9&.\s]", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def _match(text: str, aliases: Tuple[Tuple[str, Tuple[str, ...]], ...]) -> Optional[str]:
    for canonical, patterns in aliases:
        if any(re.search(p, text) for p in patterns):
            return canonical
    return None


def extract_attack(text: str, default: str = "fgsm") -> str:
    return _match(normalize(text), ATTACK_ALIASES) or default


def extract_defense(text: str, default: str = "feature_squeezing") -> str:
    return _match(normalize(text), DEFENSE_ALIASES) or default


def extract_hardening_method(text: str, default: str = "pgd") -> str:
    """The retraining method.

    "harden against fgsm using pgd" means measure with FGSM but *retrain* with
    PGD, so an alias that directly follows a preposition wins over an earlier
    one. Without this the first alias in the prompt would always be chosen and
    the retraining method would be silently wrong.
    """
    n = normalize(text)
    positioned = [
        (m.start(), canonical)
        for canonical, patterns in _HARDENING_ALIASES
        for p in patterns
        for m in re.finditer(p, n)
    ]
    if not positioned:
        return default
    qualified = [
        (pos, name) for pos, name in positioned
        if re.search(
            r"\b(?:using|via|with)\s+(?:the\s+)?(?:adversarial\s+)?\w*$",
            n[:pos],
        )
    ]
    if qualified:
        # "using" and "via" name the method unambiguously; "with" can equally
        # introduce a defense ("with feature_squeezing"), so it ranks lower.
        strong = [
            (pos, name) for pos, name in qualified
            if re.search(r"\b(?:using|via)\s+(?:the\s+)?(?:adversarial\s+)?\w*$", n[:pos])
        ]
        return max(strong or qualified)[1]
    return min(positioned)[1]


def _match_all(
    text: str,
    aliases: Tuple[Tuple[str, Tuple[str, ...]], ...],
) -> List[str]:
    """Every distinct canonical match, in prompt order (first appearance)."""
    found: List[str] = []
    for canonical, patterns in aliases:
        positions = [m.start() for p in patterns for m in re.finditer(p, text)]
        if positions and canonical not in found:
            found.append(canonical)
            found.sort(key=lambda name: min(
                pos
                for c, pats in aliases if c == name
                for p in pats
                for pos in [m.start() for m in re.finditer(p, text)]
            ))
    return found


def extract_attacks(text: str) -> List[str]:
    """Every distinct attack named in ``text``, in the order first mentioned."""
    return _match_all(normalize(text), ATTACK_ALIASES)


def extract_defenses(text: str) -> List[str]:
    """Every distinct defense named in ``text``, in the order first mentioned."""
    return _match_all(normalize(text), DEFENSE_ALIASES)


#: N is only read from explicit forms so unrelated tuning numbers such as
#: ``eps=0.1`` or ``max_iter=10`` are never mistaken for a step count.
_N_PATTERNS: Tuple[str, ...] = (
    r"\bn\s*=\s*(\d+)",
    r"\b(\d+)\s*(?:attack\s*defen[sc]e\s*)?(?:combination|combinations|step|steps|round|rounds|iteration|iterations|cycle|cycles)\b",
    r"\b(?:run|apply|do|execute|perform|try)\s+(\d+)\s+(?:attack|attacks|defense|defenses|combination|combinations|step|steps|round|rounds)\b",
)


def extract_explicit_step_count(text: str, cap: int = 5) -> Optional[int]:
    """The N the user actually stated, or None when they stated none.

    Read only from explicit forms so tuning parameters such as ``eps=0.1`` or
    ``max_iter=10`` are never mistaken for a step count.
    """
    # ``normalize`` strips '=' as punctuation, which would break ``n=4``, so the
    # explicit ``key = value`` form is matched on a whitespace-folded copy that
    # keeps the separator.
    n = re.sub(r"\s+", " ", (text or "").lower()).strip()
    for pattern in _N_PATTERNS:
        match = re.search(pattern, n)
        if not match:
            continue
        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if value >= 1:
            return min(value, cap)
    return None


def extract_step_count(text: str, default: int = 2, cap: int = 5) -> int:
    """Number of attack/defense combinations to run.

    Unstated means ``default``; the result never exceeds ``cap``.
    """
    explicit = extract_explicit_step_count(text, cap)
    if explicit is not None:
        return explicit
    return min(default, cap)


def contains_attack(text: str) -> bool:
    n = normalize(text)
    if _match(n, ATTACK_ALIASES):
        return True
    return bool(re.search(r"\battack\w*\b|\bevasion\b|\badversarial\b", n))


def contains_defense(text: str) -> bool:
    n = normalize(text)
    if _match(n, DEFENSE_ALIASES):
        return True
    return bool(re.search(r"\bdefen[sc]\w*\b|\bprotect\w*\b", n))


def is_hardening_request(text: str) -> bool:
    n = normalize(text)
    return bool(
        re.search(
            r"\bharden\w*\b"
            r"|\badversarial (train|retrain)\w*\b"
            r"|\brobust(ness)? training\b",
            n,
        )
    )


def is_compound_defended_attack_request(text: str) -> bool:
    return contains_attack(text) and contains_defense(text)


def is_hardening_loop_request(text: str) -> bool:
    """A hardening request that iterates attack -> retrain over several rounds.

    Checked before ``is_hardening_request`` because a loop is a hardening
    request, but it needs the cumulative workflow rather than a single retrain.
    """
    n = normalize(text)
    if not is_hardening_request(text):
        return False
    return bool(
        re.search(
            r"\b(round|rounds|iteration|iterations|cycle|cycles|loop|loops)\b"
            r"|\brepeatedly\b"
            r"|\bprogressively\b"
            r"|\biterative\w*\b",
            n,
        )
    )


# ----------------------------------------------------------------------
# Capability checks
# ----------------------------------------------------------------------

def supported_attacks(framework: Optional[str]) -> List[str]:
    return ATTACKS_BY_FRAMEWORK.get((framework or "").lower(), [])


def supported_hardening_methods(framework: Optional[str]) -> List[str]:
    return HARDENING_METHODS_BY_FRAMEWORK.get((framework or "").lower(), [])


def assert_attack_supported(attack_name: str, framework: Optional[str]) -> None:
    """Refuse an attack the active model cannot execute, naming valid options."""
    allowed = supported_attacks(framework)
    label = ATTACK_LABELS.get(attack_name, attack_name)
    if not allowed:
        raise IncompatibleSecurityRequest(
            f"No adversarial attack support is available for a "
            f"{framework or 'unknown'} model in this environment. "
            f"Please use a supported model type."
        )
    if attack_name not in allowed:
        options = ", ".join(ATTACK_LABELS.get(a, a) for a in allowed)
        raise IncompatibleSecurityRequest(
            f"{label} cannot be executed against this project: the active model "
            f"is a {framework} model, which only supports {options}. "
            f"Please upload or select a {framework}-compatible model that supports "
            f"{label}, or choose one of: {options}."
        )


def assert_hardening_method_supported(method: str, framework: Optional[str]) -> None:
    allowed = supported_hardening_methods(framework)
    label = ATTACK_LABELS.get(method, method)
    if not allowed:
        raise IncompatibleSecurityRequest(
            f"Adversarial retraining is not supported for a "
            f"{framework or 'unknown'} model in this environment."
        )
    if method not in allowed:
        options = ", ".join(ATTACK_LABELS.get(a, a) for a in allowed)
        raise IncompatibleSecurityRequest(
            f"{label} adversarial retraining cannot be executed against this "
            f"project: the active model is a {framework} model, which only "
            f"supports {options}. Please use a {framework}-compatible model, "
            f"or choose one of: {options}."
        )


def warns_gradient_masking(attack_name: str, defense_name: str) -> bool:
    return attack_name in GRADIENT_ATTACKS and defense_name in NON_DIFFERENTIABLE_DEFENSES
