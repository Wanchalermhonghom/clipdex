# clipdex

Durchsuche dein eigenes Footage mit lokaler KI. Finde einen Clip über das, was gesagt wird, was zu sehen ist oder was passiert.
clipdex läuft auf deinem Rechner. Kein Video verlässt den Rechner.
Claude und andere KI-Agenten können den Index über MCP durchsuchen.

Jeder Treffer ist ein **Moment**: eine Datei plus Start und Ende in Sekunden.

[English version](README.md)

## Was clipdex findet

| Kanal | Modell | Findet |
|---|---|---|
| `visual` | [SigLIP 2](https://huggingface.co/google/siglip2-so400m-patch14-384) | Was auf einem Bild zu sehen ist: Objekte, Orte, Licht |
| `scenes` | [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | Was in ein paar Sekunden passiert: Handlungen |
| `speech` | [Whisper large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) + SQLite FTS5 | Genau gesprochene Wörter, mit Wort-Zeitstempel |
| `speech_sem` | Qwen3-VL-Embedding | Worüber gesprochen wird, auch mit anderen Worten |

Suchanfragen gehen auf Deutsch, Englisch und in den anderen Sprachen der Modelle.

## Schnellstart (Mac mit Apple Silicon, macOS 14 oder neuer)

```bash
brew install ffmpeg uv
uv tool install git+https://github.com/Wanchalermhonghom/clipdex
clipdex setup /Volumes/T7                # erkennt den Mac, schreibt die Config, lädt die Modelle
clipdex index /Volumes/T7 --limit 3      # Test mit 3 Dateien
clipdex index /Volumes/T7                # alles indexieren (Ctrl-C stoppt, derselbe Befehl macht weiter)
clipdex search "jemand gießt Kaffee ein" /Volumes/T7 --mode action
```

## Wie der Index entsteht

Für jede Datei läuft diese Kette:

1. `ffprobe` liest Länge, Auflösung und Tonspur.
2. `ffmpeg` speichert alle 2 Sekunden ein Bild (512 px). Zuerst dekodiert es nur Keyframes. Das ist schnell. Bei großen Lücken dekodiert es die ganze Datei.
3. SigLIP 2 berechnet einen Vektor pro Bild.
4. [Silero VAD](https://github.com/snakers4/silero-vad) sucht die Sprache. Whisper transkribiert nur diese Stellen. Dateien ohne Sprache überspringt es.
5. Whisper erkennt die Sprache pro Datei. Es hört dort hin, wo die Sprache beginnt.
6. clipdex teilt jede Datei in kurze Szenen (max. 10 s). Eine neue Szene beginnt, wenn sich das Bild stark ändert.
7. Qwen3-VL-Embedding berechnet einen Vektor pro Szene und pro Transkript-Abschnitt (20 s).

Alles liegt in einer SQLite-Datei und einem Ordner mit kleinen JPEG-Bildern: ungefähr 55 MB pro Stunde Footage.

## An deinen Rechner anpassen

`clipdex setup` erkennt Chip, Speicher und GPU. Dann wählt es ein Profil:

| Profil | Rechner | Modelle | Spitzen-RAM (Schätzung) |
|---|---|---|---|
| `apple-small` | Apple Silicon, < 16 GB | SigLIP 2 base, ohne Qwen | Indexer ~4 GB |
| `apple-medium` | Apple Silicon, 16–47 GB | SigLIP 2 so400m, Qwen 2B | Indexer ~10 GB, MCP ~8 GB |
| `apple-large` | Apple Silicon, ≥ 48 GB | SigLIP 2 so400m, Qwen 8B | Indexer ~24 GB, MCP ~21 GB |
| `nvidia-small/medium/large` | NVIDIA-GPU nach VRAM | wie oben | wie oben |
| `cpu` | keine GPU | SigLIP 2 base, Whisper small, ohne Qwen | Indexer ~3 GB |

Die RAM-Werte kommen aus den Modellgrößen. Sie sind nicht auf deinem Rechner gemessen.
Der Indexer zeigt am Ende jedes Laufs den echten Spitzenwert.

**Lass das deinen KI-Agenten machen.** Öffne das Repo in Claude Code, Codex oder Cursor. Sag:
„Richte clipdex für diesen Rechner ein.“ Der Agent folgt dem Ablauf in [AGENTS.md](AGENTS.md):
Tools prüfen, Profil wählen, 3 Dateien testen, Speed und RAM messen, Suche kalibrieren, MCP verbinden.
In Claude Code geht auch `/clipdex-setup`.

## Suchen im Terminal

```bash
clipdex search "Laptop auf einem Holztisch" /Volumes/T7
clipdex search '"claude code"' /Volumes/T7 --mode speech        # exakte Wörter in doppelten Anführungszeichen
clipdex search "Sonnenuntergang am Strand" /Volumes/T7 --in Thailand
clipdex transcript C0012.MP4 /Volumes/T7
clipdex scenes C0012.MP4 /Volumes/T7 --changes-only            # jeder Bildwechsel
clipdex search "rotes Auto" /Volumes/T7 --json                  # für Skripte
```

Modi: `auto` (alle Kanäle, Standard), `visual`, `action` und `speech`.

## Mit Claude verbinden (MCP)

`clipdex setup` zeigt den genauen Befehl für deinen Rechner. Er sieht so aus:

```bash
claude mcp add --transport stdio --scope user clipdex -- clipdex mcp /Volumes/T7
```

Für Claude Desktop trägst du den Server in `~/Library/Application Support/Claude/claude_desktop_config.json` ein.
Nimm den vollen Pfad aus `which clipdex`:

```json
{
  "mcpServers": {
    "clipdex": { "command": "/Users/DEINNAME/.local/bin/clipdex", "args": ["mcp", "/Volumes/T7"] }
  }
}
```

Der Agent bekommt diese Tools:

| Tool | Zweck |
|---|---|
| `search_footage` | Momente suchen (Datei, Start, Ende, bester Zeitpunkt) |
| `view_frames` | Vorschaubilder ansehen, um Treffer zu prüfen |
| `get_transcript` | Transkript einer Datei mit Zeitstempeln, auch pro Wort |
| `get_scenes` | Szenengrenzen einer Datei, mit dem Grund für jeden Schnitt |
| `similar_shots` | Ähnlich aussehende Shots finden |
| `list_files` | Dateien mit Länge, Auflösung und Sprache |
| `index_status` | Statistik und fehlerhafte Dateien |

Der MCP-Server liest den Index nur. Du kannst suchen, während der Indexer läuft.
Auf einem Mac mit 16–24 GB startest du den Server beim Indexieren mit `mcp ... --no-qwen`. Dann braucht er weniger RAM.

## Den Index aktuell halten

- Starte `clipdex index /Volumes/T7` erneut. Es bearbeitet nur neue und geänderte Dateien.
- Verschobene und umbenannte Dateien behalten ihren Index.
- Gelöschte Dateien markiert clipdex als `missing`. `--prune` entfernt sie. Mehr als die Hälfte des Index entfernt clipdex nur mit `--force`.
- Defekte Dateien überspringt clipdex. `clipdex status /Volumes/T7` zeigt die Fehler. `--retry-failed` versucht sie erneut.
- Nach einer geänderten Einstellung startet `--redo STAGE` diesen Schritt neu, z. B. `--redo scenes` nach einer Szenen-Einstellung.
- Der Index liegt in `/Volumes/T7/.clipdex`. Er wandert mit der SSD. Mit `--index-dir` liegt er woanders.

## Die Suche anpassen

Vektorsuche liefert immer einen nächsten Treffer, auch bei Unsinn. Darum filtert clipdex schwache Treffer pro Kanal aus.
Sieh dir die Werte mit `--explain` an:

```bash
clipdex search "rotes Auto" /Volumes/T7 --explain
```

Trage neue Werte in `/Volumes/T7/.clipdex/config.toml` ein, z. B. `scene_floor = 0.38`.
[clipdex.example.toml](clipdex.example.toml) zeigt alle Einstellungen mit ihren Standardwerten.

## Grenzen

- Die Standard-Schwellen stammen aus einem kleinen Test mit Beispielbildern und dem 2B-Modell. Prüfe sie mit deinem Footage.
- Whisper wählt eine Sprache pro Datei. Für Dateien mit gemischter Sprache setzt du `--language`.
- Log- und HDR-Footage sieht in den gespeicherten Bildern flach aus. Die Bildsuche kann dann schwächer sein.
- ffmpeg kann BRAW und R3D nicht lesen. Diese Dateien erscheinen als Fehler.
- Ein Modellwechsel braucht `clipdex index ... --reembed visual` oder `--reembed qwen`.

## Entwicklung

```bash
git clone https://github.com/Wanchalermhonghom/clipdex && cd clipdex
uv sync --extra dev
uv run pytest                  # schnelle Tests mit Fake-Modellen, ca. 30 s
uv run pytest -m models -s     # echte Modelle auf Beispielmedien (lädt ca. 6 GB)
```

Lies [AGENTS.md](AGENTS.md), bevor du Code änderst. Dort stehen die Projektkarte und die Regeln.

## Lizenz

Apache-2.0. Siehe [LICENSE](LICENSE) und [NOTICE](NOTICE). Die Modelle behalten ihre eigenen Lizenzen:
SigLIP 2 und Qwen3-VL-Embedding (Apache-2.0), Whisper und Silero VAD (MIT).
