"""Comprueba que un clip no empiece ni termine a mitad de una frase o de una palabra."""
from streamclipper.models import Transcript

TOL = 0.05
_TERMINAL = (".", "?", "!", "…")


def check_boundaries(transcript: Transcript, start: float, end: float, tol: float = TOL) -> dict:
    """Devuelve {'errors': [...], 'warnings': [...]}.

    errors:   el inicio/fin cae dentro de una frase o palabra transcrita (corte seguro de habla).
    warnings: indicios de corte semántico (última frase sin puntuación, primera en minúscula).
    """
    errors: list[str] = []
    warnings: list[str] = []
    for label, t in (("inicio", start), ("fin", end)):
        for s in transcript.segments:
            if s.start + tol < t < s.end - tol:
                errors.append(f"El {label} del clip ({t:.2f}s) cae dentro de la frase "
                              f"«{s.text[:50]}» ({s.start:.2f}-{s.end:.2f}s)")
                break
        else:
            for s in transcript.segments:
                for w in s.words:
                    if w.start + tol < t < w.end - tol:
                        errors.append(f"El {label} del clip ({t:.2f}s) cae dentro de la palabra «{w.text}»")
    inside = [s for s in transcript.segments if s.start >= start - tol and s.end <= end + tol]
    if inside:
        if not inside[-1].text.rstrip().endswith(_TERMINAL):
            warnings.append(f"La última frase del clip no termina en puntuación: «{inside[-1].text[-40:]}»")
        first = inside[0].text.lstrip()
        if first and first[0].islower():
            warnings.append(f"La primera frase del clip empieza en minúscula (¿mitad de oración?): «{first[:40]}»")
    return {"errors": errors, "warnings": warnings}
