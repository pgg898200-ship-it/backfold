"""Draw the before-and-after readings from the README example on the sharing card."""

from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont

FONTS = Path("/usr/share/fonts/truetype/dejavu")
OUT = Path(__file__).resolve().parents[1] / "docs" / "og-card.png"


def main() -> None:
    paper, ink, rust, trace = "#f6f0dc", "#253831", "#bd582c", "#c2c5af"
    image = Image.new("RGB", (1200, 630), paper)
    draw = ImageDraw.Draw(image)
    serif = ImageFont.truetype(str(FONTS / "DejaVuSerif.ttf"), 80)
    value = ImageFont.truetype(str(FONTS / "DejaVuSansMono.ttf"), 33)
    label = ImageFont.truetype(str(FONTS / "DejaVuSansMono.ttf"), 23)
    original = (4, 40, 10)
    corrected = (4, 7, 10)

    draw.text((54, 45), "Backfold", font=serif, fill=ink)
    draw.text((57, 158), "A late correction at time 20", font=value, fill=ink)
    draw.line((56, 237, 1144, 237), fill=ink, width=2)
    columns = (370, 572, 775)
    for x, time in zip(columns, (10, 20, 30)):
        draw.text((x, 283), f"t = {time}", font=label, fill=ink, anchor="mm")
    draw.text((1053, 283), "MEAN", font=label, fill=ink, anchor="mm")
    draw.line((930, 264, 930, 573), fill=trace, width=2)

    for y, name, readings in ((368, "Observed", original), (517, "Corrected", corrected)):
        draw.text((55, y), name, font=value, fill=ink, anchor="lm")
        draw.line((columns[0], y, columns[-1], y), fill=trace, width=3)
        for x, reading in zip(columns, readings):
            edited = name == "Corrected" and x == columns[1]
            draw.ellipse((x - 44, y - 44, x + 44, y + 44),
                         fill=rust if edited else paper, outline=rust if edited else ink, width=3)
            draw.text((x, y), str(reading), font=value, fill=paper if edited else ink, anchor="mm")
        draw.text((1053, y), str(mean(readings)), font=serif, fill=rust if name == "Corrected" else ink, anchor="mm")
    draw.line((572, 416, 572, 463), fill=rust, width=3)
    draw.line([(563, 453), (572, 463), (581, 453)], fill=rust, width=3)
    draw.line((56, 586, 1144, 586), fill=ink, width=2)
    image.save(OUT, optimize=True)


if __name__ == "__main__":
    main()
