"""Upload safety: detect executable / malicious files before they enter chat.

A chat that relays files is a malware vector, so anything that *runs* is
blocked outright (category ``cybersecurity``) — by extension, by magic bytes
(so renaming ``virus.exe`` to ``virus.jpg`` doesn't help), and by the EICAR
test signature (the industry-standard harmless string for exercising AV
pipelines — this is also what our tests use).

This is a gate, not an antivirus: pair it with a real scanner (e.g. ClamAV)
for production. Archives are blocked too because we can't see inside them.
"""
from __future__ import annotations

import os

from .result import ModerationResult, ACTION_BLOCK
from . import taxonomy as T

# Files that execute (or script) — never relayed.
EXECUTABLE_EXTS = frozenset({
    ".exe", ".dll", ".so", ".dylib", ".msi", ".scr", ".com", ".pif",
    ".bat", ".cmd", ".ps1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".hta",
    ".sh", ".bash", ".zsh", ".py", ".rb", ".pl", ".php",
    ".jar", ".apk", ".ipa", ".appimage", ".deb", ".rpm",
})

# Opaque containers we can't inspect — blocked rather than waved through.
ARCHIVE_EXTS = frozenset({
    ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".iso", ".img",
    ".cab", ".dmg",
})

# Magic bytes of executable formats (extension renames don't help).
_EXEC_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"MZ", "Windows PE executable"),
    (b"\x7fELF", "ELF executable"),
    (b"\xfe\xed\xfa\xce", "Mach-O executable"),
    (b"\xfe\xed\xfa\xcf", "Mach-O executable"),
    (b"\xcf\xfa\xed\xfe", "Mach-O executable"),
    (b"\xca\xfe\xba\xbe", "Mach-O fat / Java class"),
    (b"#!", "script with shebang"),
)

# The EICAR anti-virus test signature (harmless by design, universally flagged).
EICAR_SIGNATURE = (b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-"
                   b"ANTIVIRUS-TEST-FILE!$H+H*")


def check_file_safety(filename: str, data: bytes) -> ModerationResult | None:
    """Return a blocking ``ModerationResult`` if the file is dangerous,
    else ``None`` (the file may proceed to modality moderation)."""
    ext = os.path.splitext(filename or "")[1].lower()
    head = data[:4096]

    reason = None
    if EICAR_SIGNATURE in data:
        reason = "EICAR test signature (malware scanner check)"
    elif ext in EXECUTABLE_EXTS:
        reason = f"executable file type ({ext})"
    elif ext in ARCHIVE_EXTS:
        reason = f"archive ({ext}) — contents can't be inspected"
    else:
        for magic, label in _EXEC_MAGIC:
            if head.startswith(magic):
                reason = f"{label} (magic bytes)"
                break

    if reason is None:
        return None
    return ModerationResult(
        modality="file",
        flagged=True,
        action=ACTION_BLOCK,
        categories=[T.CYBERSECURITY],
        scores={T.CYBERSECURITY: 1.0},
        reasons=[f"blocked: {reason}"],
        detectors=["filecheck"],
    )
