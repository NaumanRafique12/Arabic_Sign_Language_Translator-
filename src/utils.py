# src/utils.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import textwrap

def draw_info_text(image, text, gender, font_path='Arial.ttf'):
    """
    Draws Arabic text with auto-wrapping and a modern semi-transparent UI.
    """
    # 1. Setup Pillow Image
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil, 'RGBA')
    w, h = img_pil.size
    
    # 2. Load Font (Adjust size dynamically if needed, keeping 32 for readability)
    try:
        font = ImageFont.truetype(font_path, 32)
        small_font = ImageFont.truetype(font_path, 20)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # 3. Process Arabic Text
    reshaped_text = reshape(text)
    display_text = get_display(reshaped_text)

    # 4. Create Dynamic Overlay
    # We want a bar at the bottom for subtitles, cinema-style
    margin = 20
    
    # Wrap text logic (simple character count estimation for wrapping)
    # Note: proper RTL wrapping is complex; we use a simple visual heuristic here
    wrapper = textwrap.TextWrapper(width=50) 
    lines = wrapper.wrap(display_text)
    
    # Calculate box height based on number of lines
    line_height = 40
    box_height = (len(lines) * line_height) + 30
    
    # Draw Semi-Transparent Background (Bottom)
    overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    # Bottom Subtitle Bar
    draw_overlay.rectangle(
        [(0, h - box_height - 20), (w, h)], 
        fill=(0, 0, 0, 180)  # Black with ~70% opacity
    )
    
    # Top Status Bar (Gender & Info)
    draw_overlay.rectangle(
        [(0, 0), (w, 40)], 
        fill=(0, 0, 0, 150)
    )
    
    # Merge overlay
    img_pil = Image.alpha_composite(img_pil.convert('RGBA'), overlay)
    draw = ImageDraw.Draw(img_pil)

    # 5. Draw Text
    # Draw Subtitles (Bottom, Centered)
    y_text = h - box_height - 10
    for line in lines:
        # Calculate text width to center it
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x_text = (w - text_w) / 2
        
        draw.text((x_text, y_text), line, font=font, fill=(255, 255, 255))
        y_text += line_height

    # Draw Header Info (Top)
    # Gender Indicator
    gender_color = (255, 105, 180) if gender == "Female" else (100, 149, 237) # Pink vs Blue
    gender_text = get_display(reshape(f"Gender: {gender}"))
    draw.text((w - 150, 5), gender_text, font=small_font, fill=gender_color)
    
    # App Title
    title_text = ""
    draw.text((20, 5), title_text, font=small_font, fill=(200, 200, 200))

    return cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)