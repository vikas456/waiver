"""Draw the image link previews show, web/og-image.png.

Run it again whenever the headline changes. It needs Pillow, which the site
itself does not:

    python -m pip install pillow
    python tools/make_og_image.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 630
PAPER, INK, INK_2, PINE = "#0e1211", "#e6ebe8", "#a2aca7", "#6cc3a0"
OUT = Path(__file__).resolve().parent.parent / "web" / "og-image.png"

# The site's face is Archivo, which is not installed locally; these are close
# enough for a preview image and ship with macOS.
BOLD = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"]
REGULAR = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"]


def font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def wrap(draw: ImageDraw.ImageDraw, text: str, face, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=face) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line]


def main() -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(img)

    # The same W as the favicon, scaled up from its 32-unit grid.
    scale, x0, y0 = 2.6, 80, 70
    mark = [(7, 10), (11.5, 22), (16, 13), (20.5, 22), (25, 10)]
    draw.line([(x0 + x * scale, y0 + y * scale) for x, y in mark], fill=PINE, width=9, joint="curve")
    draw.text((x0 + 88, y0 + 20), "Waiver", font=font(BOLD, 44), fill=PINE)

    headline = font(BOLD, 80)
    draw.text((80, 230), "Who should you pick up?", font=headline, fill=INK)
    sub = font(REGULAR, 36)
    y = 350
    for line in wrap(draw, "Fantasy football waiver picks from an AI model trained on every "
                     "NFL play and tested against expert rankings.", sub, 1000):
        draw.text((80, y), line, font=sub, fill=INK_2)
        y += 50
    draw.text((80, 530), "fantasywaiverpicks.com", font=font(REGULAR, 30), fill=INK_2)

    img.save(OUT, optimize=True)
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
