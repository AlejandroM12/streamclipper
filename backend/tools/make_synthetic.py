"""Genera un MP4 sintético (voz SAPI de Windows + ruido + video de prueba) para validar el pipeline técnico.

NO sirve para evaluar la calidad del detector con gameplay real. Uso:
    python tools/make_synthetic.py salida.mp4
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from streamclipper.core.config import Settings

# (segundo, texto, ganancia). Ganancia alta = momento "emocionante".
TIMELINE = [
    (3, "Okay, so we are dropping into the city now.", 0.7),
    (10, "Let me check the map and loot this building before the circle closes.", 0.7),
    (19, "I think there is a squad on the north side, so I will stay quiet.", 0.7),
    (30, "Oh my god! No way! He is right behind me! That was insane!", 3.0),
    (42, "Alright, that fight went well. Now we need to heal and find a better position near the hill.", 0.7),
    (53, "The storm is coming, so let us rotate to the south and hold the high ground.", 0.7),
    (64, "I am checking my inventory, I have two shields and a few bandages left.", 0.7),
    (73, "Let's go! We won! Oh my god, that was unreal! I can't believe it!", 3.0),
    (88, "Okay, thanks for watching everyone. That was a good game, see you next time.", 0.7),
]
BURSTS = [(30, 1.2), (73, 1.5)]  # (segundo, duración) de ruido fuerte (explosión)
TOTAL = 105


def _tts(text: str, out: Path, voice: str = "Microsoft Zira Desktop") -> None:
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SelectVoice('{voice}');"
        f"$s.SetOutputToWaveFile('{out}');"
        "$s.Speak($env:SC_TTS_TEXT);$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True,
                   env={**os.environ, "SC_TTS_TEXT": text})


def main(dst: Path) -> None:
    ff = str(Settings().ffmpeg)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inputs, filters, labels = [], [], []
        for i, (t, text, gain) in enumerate(TIMELINE):
            wav = tmp / f"u{i}.wav"
            _tts(text, wav)
            inputs += ["-i", str(wav)]
            filters.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=mono,volume={gain},adelay={t * 1000}:all=1[s{i}]")
            labels.append(f"[s{i}]")
        n = len(TIMELINE)
        # fondo: ruido marrón suave + ráfagas de ruido en los momentos fuertes
        inputs += ["-f", "lavfi", "-i", f"anoisesrc=color=brown:amplitude=0.04:sample_rate=48000:duration={TOTAL}"]
        labels.append(f"[{n}:a]")
        for j, (t, d) in enumerate(BURSTS):
            inputs += ["-f", "lavfi", "-i", f"anoisesrc=color=white:amplitude=0.5:sample_rate=48000:duration={d}"]
            filters.append(f"[{n + 1 + j}:a]afade=t=out:st={d * 0.4}:d={d * 0.6},adelay={t * 1000}:all=1[b{j}]")
            labels.append(f"[b{j}]")
        filters.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0,alimiter=limit=0.95,atrim=0:{TOTAL}[a]")
        mixed = tmp / "mix.wav"
        subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex",
                        ";".join(filters), "-map", "[a]", "-ar", "48000", "-ac", "1", str(mixed)], check=True)
        subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", f"testsrc2=size=1920x1080:rate=30:duration={TOTAL}",
                        "-i", str(mixed), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(dst)], check=True)
    print(f"Generado: {dst}")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic.mp4").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    main(out)
