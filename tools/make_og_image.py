"""Draw the image link previews show, web/og-image.png.

Run it again whenever the headline changes. It needs Pillow, which the site
itself does not:

    python -m pip install pillow
    python tools/make_og_image.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 630
PAPER, INK, INK_2, INK_3, PINE = "#0e1211", "#e6ebe8", "#a2aca7", "#7b8581", "#6cc3a0"
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


def card(headline: str, blurb: str, figures: list[tuple[str, str]] | None = None) -> Image.Image:
    """One preview image: the wordmark, a headline, and optionally a row of
    figures. Everything the site publishes shares this frame, so a link from
    any page looks like it came from the same place."""
    img = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(img)
    _wordmark(draw)

    title = font(BOLD, 72 if len(headline) > 28 else 80)
    y = 220
    for line in wrap(draw, headline, title, 1040)[:2]:
        draw.text((80, y), line, font=title, fill=INK)
        y += 88

    sub = font(REGULAR, 34)
    y = max(y + 10, 400)
    for line in wrap(draw, blurb, sub, 1040)[:2]:
        draw.text((80, y), line, font=sub, fill=INK_2)
        y += 46

    # A few numbers carry more than another sentence does.
    if figures:
        x = 80
        for label, value in figures[:3]:
            draw.text((x, 505), value, font=font(BOLD, 46), fill=PINE)
            draw.text((x, 562), label.upper(), font=font(REGULAR, 24), fill=INK_3)
            x += 330
    else:
        draw.text((80, 530), "fantasywaiverpicks.com", font=font(REGULAR, 30), fill=INK_2)
    return img


def save(img: Image.Image, path: Path) -> int:
    """Write a card, small enough that hundreds of them cost nothing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.quantize(colors=48).save(path, optimize=True)
    return path.stat().st_size


def _wordmark(draw: ImageDraw.ImageDraw) -> None:
    # The same W as the favicon, scaled up from its 32-unit grid.
    scale, x0, y0 = 2.6, 80, 70
    mark = [(7, 10), (11.5, 22), (16, 13), (20.5, 22), (25, 10)]
    draw.line([(x0 + x * scale, y0 + y * scale) for x, y in mark], fill=PINE, width=9, joint="curve")
    # The wordmark as the site sets it: "W" and "ver" recede, "ai" in pine.
    x, name = x0 + 88, font(BOLD, 44)
    for part, colour in (("W", INK_3), ("ai", PINE), ("ver", INK_3)):
        draw.text((x, y0 + 20), part, font=name, fill=colour)
        x += draw.textlength(part, font=name)


def main() -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(img)

    # The same W as the favicon, scaled up from its 32-unit grid.
    scale, x0, y0 = 2.6, 80, 70
    mark = [(7, 10), (11.5, 22), (16, 13), (20.5, 22), (25, 10)]
    draw.line([(x0 + x * scale, y0 + y * scale) for x, y in mark], fill=PINE, width=9, joint="curve")
    # The wordmark as the site sets it: "W" and "ver" recede, "ai" in pine.
    x, name = x0 + 88, font(BOLD, 44)
    for part, colour in (("W", INK_3), ("ai", PINE), ("ver", INK_3)):
        draw.text((x, y0 + 20), part, font=name, fill=colour)
        x += draw.textlength(part, font=name)

    headline = font(BOLD, 80)
    draw.text((80, 230), "Who should you pick up?", font=headline, fill=INK)
    sub = font(REGULAR, 36)
    y = 350
    for line in wrap(draw, "Fantasy football waiver picks from an AI model trained on every "
                     "NFL play and tested against expert rankings.", sub, 1000):
        draw.text((80, y), line, font=sub, fill=INK_2)
        y += 50
    draw.text((80, 530), "fantasywaiverpicks.com", font=font(REGULAR, 30), fill=INK_2)

    # Five flat colours plus their anti-aliased edges fit easily in a small
    # palette, which shrinks the file without any visible change.
    img.quantize(colors=48).save(OUT, optimize=True)
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
