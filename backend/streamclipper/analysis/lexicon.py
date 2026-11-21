"""Léxico de reacciones verbales (es/en), configurable. Reglas documentadas en docs/lexicon.md.

- strong / mild: expresiones (regex sobre texto en minúsculas, sin acentos ni puntuación) que suman
  1.0 / 0.5 a la reacción de una frase.
- ambiguous: palabras que son cotidianas en una frase y una reacción en otra (p. ej. «vamos»). No suman por
  sí solas: se decide con evidencia (contexto, repetición confiable, alargamiento, volumen, exclamación).
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from streamclipper.core.errors import InvalidInputError

_STRONG = [
    r"oh my god", r"omg", r"no way", r"holy (shit|crap|cow|moly)?", r"what the (fuck|hell)", r"wtf",
    r"let'?s go+", r"are you kidding", r"i can'?t believe", r"that'?s insane", r"so close",
    r"no puede ser", r"dios mio", r"no lo puedo creer", r"que locura", r"la puta madre",
    r"que haces", r"increible", r"mierda", r"joder", r"hijo de", r"ganamos", r"gane",
]
_MILD = [
    r"insane", r"crazy", r"unreal", r"clutch", r"nice", r"wow", r"bro", r"dude", r"dead", r"ace",
    r"got (him|them)", r"killed", r"win", r"lose", r"lost", r"no no", r"yes yes", r"oh no",
    r"loco", r"brutal", r"tremendo", r"boludo", r"nooo+", r"siii+", r"dale", r"mato", r"morir",
]


@dataclass
class AmbiguousRule:
    name: str
    pattern: str                    # regex que debe coincidir con UNA palabra normalizada completa
    elongated: str | None = None    # regex de la variante alargada («vamooos»): señal fuerte por sí sola
    mundane_next: list[str] = field(default_factory=list)  # si la sigue una de estas palabras: uso cotidiano
    strong_points: int = 4          # puntos de evidencia para contar como reacción fuerte (1.0)
    mild_points: int = 2            # puntos para contar como reacción leve (0.5)


DEFAULT_AMBIGUOUS = [
    AmbiguousRule(
        name="vamos", pattern=r"vamo+s", elongated=r"vamo{3,}s",
        # «vamos a revisar», «vamos para la isla», «vamos con calma»…: perífrasis o movimiento, no reacción
        mundane_next=["a", "al", "para", "pa", "con", "por", "hacia", "en", "de", "hasta", "tras", "sobre"],
    ),
]


@dataclass
class Lexicon:
    strong: list[str] = field(default_factory=lambda: list(_STRONG))
    mild: list[str] = field(default_factory=lambda: list(_MILD))
    ambiguous: list[AmbiguousRule] = field(default_factory=lambda: [AmbiguousRule(**vars(r)) for r in DEFAULT_AMBIGUOUS])

    def __post_init__(self) -> None:
        try:
            self.strong_re = [re.compile(rf"\b{p}\b") for p in self.strong]
            self.mild_re = [re.compile(rf"\b{p}\b") for p in self.mild]
            self.ambiguous_re = [(r, re.compile(rf"^{r.pattern}$"),
                                  re.compile(rf"^{r.elongated}$") if r.elongated else None) for r in self.ambiguous]
        except re.error as e:
            raise InvalidInputError(f"Expresión regular inválida en el léxico: {e}") from None

    @classmethod
    def from_file(cls, path: str | Path) -> "Lexicon":
        """Parte del léxico por defecto y aplica el JSON:

        {"extend_strong": [...], "extend_mild": [...], "remove": ["regex exacta", ...],
         "ambiguous": {"nombre": {"pattern": ..., "elongated": ..., "mundane_next": [...],
                                  "strong_points": 4, "mild_points": 2}},
         "disable_ambiguous": ["nombre", ...]}
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise InvalidInputError(f"No se pudo leer el léxico {path}: {e}") from None
        if not isinstance(data, dict):
            raise InvalidInputError("El léxico debe ser un objeto JSON")
        known = {"extend_strong", "extend_mild", "remove", "ambiguous", "disable_ambiguous"}
        if extra := set(data) - known:
            raise InvalidInputError(f"Claves desconocidas en el léxico: {sorted(extra)}")
        lex = cls()
        remove = set(data.get("remove", []))
        lex.strong = [p for p in lex.strong if p not in remove] + list(data.get("extend_strong", []))
        lex.mild = [p for p in lex.mild if p not in remove] + list(data.get("extend_mild", []))
        disabled = set(data.get("disable_ambiguous", []))
        rules = {r.name: r for r in lex.ambiguous if r.name not in disabled}
        for name, spec in data.get("ambiguous", {}).items():
            try:
                rules[name] = AmbiguousRule(name=name, **spec)
            except TypeError as e:
                raise InvalidInputError(f"Regla ambigua '{name}' inválida: {e}") from None
        return cls(strong=lex.strong, mild=lex.mild, ambiguous=list(rules.values()))


def load_lexicon(path: str | None) -> Lexicon:
    return Lexicon.from_file(path) if path else Lexicon()
