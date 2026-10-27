"""
Generate diverse benchmark test images locally with Pillow for Task 3 testing:
1. Bakery Logo (SupriLicious Cakes) - with text, badge, cupcake, balloons, bunting.
2. Landscape - mountains, trees, lake, sunset sky.
3. Animal - golden retriever / dog illustration in grassy field.
4. Person - student at desk with laptop and books.
"""

import math
import os
from PIL import Image, ImageDraw, ImageFont


def get_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("segoeui.ttf", size)
        except Exception:
            return ImageFont.load_default()


def create_bakery_logo(filepath: str):
    """Create a bakery logo with turquoise circular badge, pink cupcake, bunting, balloons, and text."""
    width, height = 700, 700
    img = Image.new("RGB", (width, height), color=(255, 228, 235))  # Light pink background
    draw = ImageDraw.Draw(img)

    # 1. Decorative Bunting at top
    bunting_colors = [(255, 105, 180), (64, 224, 208), (255, 215, 0), (147, 112, 219), (255, 160, 122)]
    for i, x in enumerate(range(50, 650, 60)):
        col = bunting_colors[i % len(bunting_colors)]
        draw.polygon([(x, 30), (x + 50, 30), (x + 25, 75)], fill=col)
    draw.line([(30, 30), (670, 30)], fill=(180, 180, 180), width=2)

    # 2. Colorful balloons on sides
    balloon_colors = [(255, 99, 132), (54, 162, 235), (255, 206, 86), (75, 192, 192), (153, 102, 255)]
    # Left balloons
    for i, (bx, by) in enumerate([(80, 150), (120, 200), (70, 260)]):
        col = balloon_colors[i % len(balloon_colors)]
        draw.ellipse([bx, by, bx + 50, by + 65], fill=col)
        draw.line([(bx + 25, by + 65), (100, 340)], fill=(200, 200, 200), width=1)
    # Right balloons
    for i, (bx, by) in enumerate([(570, 150), (530, 200), (580, 260)]):
        col = balloon_colors[(i + 2) % len(balloon_colors)]
        draw.ellipse([bx, by, bx + 50, by + 65], fill=col)
        draw.line([(bx + 25, by + 65), (560, 340)], fill=(200, 200, 200), width=1)

    # 3. Celebratory fireworks / sparkles
    for fx, fy in [(180, 120), (520, 120), (120, 480), (580, 480)]:
        draw.line([(fx - 15, fy), (fx + 15, fy)], fill=(255, 215, 0), width=2)
        draw.line([(fx, fy - 15), (fx, fy + 15)], fill=(255, 215, 0), width=2)
        draw.line([(fx - 10, fy - 10), (fx + 10, fy + 10)], fill=(255, 215, 0), width=2)
        draw.line([(fx - 10, fy + 10), (fx + 10, fy - 10)], fill=(255, 215, 0), width=2)

    # 4. Scalloped outer edge & Turquoise circular badge
    cx, cy, radius = 350, 360, 240
    # Scallops around circle
    num_scallops = 40
    for i in range(num_scallops):
        angle = 2 * math.pi * i / num_scallops
        sx = cx + int((radius + 10) * math.cos(angle))
        sy = cy + int((radius + 10) * math.sin(angle))
        draw.ellipse([sx - 18, sy - 18, sx + 18, sy + 18], fill=(64, 224, 208))

    # Outer turquoise circle
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(64, 224, 208))
    # White border ring
    draw.ellipse([cx - radius + 12, cy - radius + 12, cx + radius - 12, cy + radius - 12], outline=(255, 255, 255), width=4)
    # Inner turquoise circle
    draw.ellipse([cx - radius + 20, cy - radius + 20, cx + radius - 20, cy + radius - 20], fill=(72, 209, 204))

    # 5. Pink Cupcake with Heart
    # Cupcake wrapper (trapezoid)
    draw.polygon([(290, 340), (410, 340), (390, 420), (310, 420)], fill=(245, 222, 179))
    draw.line([(330, 340), (325, 420)], fill=(210, 180, 140), width=2)
    draw.line([(350, 340), (350, 420)], fill=(210, 180, 140), width=2)
    draw.line([(370, 340), (365, 420)], fill=(210, 180, 140), width=2)

    # Cupcake pink frosting
    draw.ellipse([280, 280, 420, 350], fill=(255, 105, 180))
    draw.ellipse([300, 250, 400, 310], fill=(255, 182, 193))
    draw.ellipse([325, 225, 375, 270], fill=(255, 105, 180))

    # Red heart on top of cupcake
    draw.ellipse([342, 205, 354, 217], fill=(220, 20, 60))
    draw.ellipse([350, 205, 362, 217], fill=(220, 20, 60))
    draw.polygon([(342, 213), (362, 213), (352, 226)], fill=(220, 20, 60))

    # 6. Prominent Business Name Text
    font_name = get_font(38)
    name_text = "SupriLicious Cakes"
    bbox = draw.textbbox((0, 0), name_text, font=font_name)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw // 2, 450), name_text, fill=(255, 255, 255), font=font_name)

    # 7. Slogan Text
    font_slogan = get_font(21)
    slogan_text = "Custom Cakes made with Love and Passion"
    sbbox = draw.textbbox((0, 0), slogan_text, font=font_slogan)
    stw = sbbox[2] - sbbox[0]
    draw.text((cx - stw // 2, 508), slogan_text, fill=(255, 255, 255), font=font_slogan)

    img.save(filepath, format="PNG", quality=95)
    return img


def create_landscape_sample(filepath: str):
    """Scenic mountain landscape under sunrise sky."""
    img = Image.new("RGB", (640, 480), color=(255, 200, 150))
    draw = ImageDraw.Draw(img)

    # Orange/yellow sun rising
    draw.ellipse([260, 120, 380, 240], fill=(255, 230, 100))

    # Far mountains
    draw.polygon([(0, 300), (180, 160), (360, 300)], fill=(120, 110, 150))
    draw.polygon([(250, 300), (450, 140), (640, 300)], fill=(100, 90, 130))

    # Near green hills
    draw.polygon([(0, 350), (220, 240), (500, 350)], fill=(46, 139, 87))
    draw.polygon([(200, 350), (420, 250), (640, 350)], fill=(34, 110, 65))

    # Calm blue lake at bottom
    draw.rectangle([0, 350, 640, 480], fill=(70, 130, 180))
    # Mountain reflection
    draw.polygon([(180, 350), (320, 420), (460, 350)], fill=(60, 110, 160))

    img.save(filepath, format="JPEG", quality=95)
    return img


def create_animal_sample(filepath: str):
    """Dog / golden retriever in an outdoor green grass field."""
    img = Image.new("RGB", (640, 480), color=(176, 224, 230))  # Light cyan sky
    draw = ImageDraw.Draw(img)

    # Sun
    draw.ellipse([500, 40, 580, 120], fill=(255, 220, 80))

    # Green grass field
    draw.rectangle([0, 240, 640, 480], fill=(60, 179, 113))

    # Dog body (golden brown)
    dog_gold = (218, 165, 32)
    dog_dark = (184, 134, 11)
    # Body
    draw.ellipse([220, 250, 420, 360], fill=dog_gold)
    # 4 legs
    draw.rectangle([250, 330, 275, 410], fill=dog_dark)
    draw.rectangle([290, 330, 315, 410], fill=dog_dark)
    draw.rectangle([340, 330, 365, 410], fill=dog_dark)
    draw.rectangle([380, 330, 405, 410], fill=dog_dark)
    # Head
    draw.ellipse([380, 200, 480, 290], fill=dog_gold)
    # Floppy ear
    draw.polygon([(390, 220), (430, 210), (410, 280)], fill=dog_dark)
    # Snout
    draw.ellipse([450, 240, 510, 285], fill=(238, 195, 112))
    # Black nose
    draw.ellipse([495, 250, 512, 265], fill=(20, 20, 20))
    # Eye
    draw.ellipse([440, 225, 455, 240], fill=(30, 30, 30))
    # Tail wagging up
    draw.line([(230, 270), (180, 210)], fill=dog_gold, width=16)

    img.save(filepath, format="JPEG", quality=95)
    return img


def create_person_sample(filepath: str):
    """Person working at a wooden desk with a laptop."""
    img = Image.new("RGB", (640, 480), color=(240, 240, 245))  # Office wall
    draw = ImageDraw.Draw(img)

    # Window in background
    draw.rectangle([80, 60, 240, 220], fill=(173, 216, 230), outline=(150, 150, 150), width=4)
    draw.line([(160, 60), (160, 220)], fill=(150, 150, 150), width=2)
    draw.line([(80, 140), (240, 140)], fill=(150, 150, 150), width=2)

    # Wooden desk
    draw.rectangle([0, 320, 640, 480], fill=(139, 90, 43))

    # Person head & torso
    # Torso (blue shirt)
    draw.rectangle([270, 240, 410, 360], fill=(65, 105, 225))
    # Neck
    draw.rectangle([325, 215, 355, 245], fill=(238, 200, 170))
    # Head
    draw.ellipse([305, 130, 375, 220], fill=(238, 200, 170))
    # Dark hair
    draw.ellipse([300, 120, 380, 170], fill=(40, 30, 25))

    # Laptop on desk
    # Screen
    draw.rectangle([300, 280, 390, 345], fill=(50, 50, 50), outline=(180, 180, 180), width=2)
    draw.rectangle([305, 285, 385, 335], fill=(220, 230, 245))
    # Keyboard base
    draw.polygon([(280, 345), (410, 345), (430, 370), (260, 370)], fill=(190, 190, 190))

    # Coffee mug
    draw.rectangle([480, 330, 515, 370], fill=(220, 60, 60))

    img.save(filepath, format="JPEG", quality=95)
    return img


def main():
    benchmarks_dir = os.path.join(os.path.dirname(__file__), "benchmark_samples")
    os.makedirs(benchmarks_dir, exist_ok=True)

    create_bakery_logo(os.path.join(benchmarks_dir, "bakery_logo.png"))
    create_landscape_sample(os.path.join(benchmarks_dir, "landscape_sample.jpg"))
    create_animal_sample(os.path.join(benchmarks_dir, "animal_sample.jpg"))
    create_person_sample(os.path.join(benchmarks_dir, "person_sample.jpg"))

    print("Generated 4 benchmark images successfully in:", benchmarks_dir)


if __name__ == "__main__":
    main()
