import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import math

# --- Configuration ---
IMG_WIDTH = 224
IMG_HEIGHT = 224
BG_COLOR = (255, 255, 255)  # White background
TEXT_COLOR = (0, 0, 0)      # Black text

# --- Dyslexic vs. Normal Effect Probabilities and Parameters ---
# These probabilities determine how often an effect is applied,
#  'dyslexic' is for the dyslexic set, 'normal' for the normal set.
EFFECT_PROBABILITIES = {
    'thicker_lines': {'dyslexic': 0.8, 'normal': 0.05},
    'crossed_out_words': {'dyslexic': 0., 'normal': 0.}, # Probability per word for just strikethrough
    'strike_then_reprint_word': {'dyslexic': 0.05, 'normal': 0.0}, # Probability per word to strike AND reprint
    'slanted_writing': {'dyslexic': 0.7, 'normal': 0.05},
    'baseline_deviation': {'dyslexic': 0.9, 'normal': 0.1}, # Probability per character
    'larger_writing': {'dyslexic': 0.9, 'normal': 0.0}, 
    'letter_correction': {'dyslexic': 0.05, 'normal': 0.0}, # Probability per character for double-write effect
}

# Parameters for the effects
EFFECT_PARAMS = {
    'thicker_lines_offset': 1, # Pixels for drawing multiple times for thicker lines
    'slanted_writing_angle_range': (-5, 5), # Degrees for rotation (for the entire text block)
    'baseline_deviation_range': (-8, 8), # Pixels for random vertical offset per character
    'base_font_size': 20, # Starting font size
    'dyslexic_font_size_multiplier': 1.3, # How much larger dyslexic text is
    'normal_font_size_multiplier': 1.0,
    'line_spacing_multiplier': 1.2, # Multiplier for space between lines
    'padding': 10, # Padding from image edges
    'strikethrough_width': 1.5, # Width of the strikethrough line
    'reprint_word_spacing': 3, # Pixels space between struck word and reprinted word
    'letter_correction_angle_range': (-15, 15), # Degrees for small letter rotation
    'letter_correction_offset_range': (-2, 2), # Pixels for small letter offset
}


# --- Function to load fonts ---
def load_fonts(font_dir):
    """Loads all .ttf font files from the specified directory."""
    fonts = []
    if not os.path.exists(font_dir):
        print(f"Error: Font directory '{font_dir}' not found. Please create it and add font files.")
        return fonts
    for filename in os.listdir(font_dir):
        if filename.lower().endswith(('.ttf', '.otf')):
            fonts.append(os.path.join(font_dir, filename))
    if not fonts:
        print(f"Warning: No font files found in '{font_dir}'. Please add some .ttf or .otf files.")
    return fonts

# --- Function to generate a single handwriting image ---
def generate_handwriting_image(text, font_path, style):
    """
    Generates a handwriting image with specified style (normal or dyslexic).
    Applies effects based on probabilities defined for each style.
    """
    font_size_multiplier = EFFECT_PARAMS['dyslexic_font_size_multiplier'] if style == 'dyslexic' else EFFECT_PARAMS['normal_font_size_multiplier']
    font_size = int(EFFECT_PARAMS['base_font_size'] * font_size_multiplier)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Could not load font {font_path}. Skipping image generation for this font.")
        return None
    
    # Decide once for the whole sentence if thicker lines will be applied
    apply_thicker_lines = random.random() < EFFECT_PROBABILITIES['thicker_lines'][style]

    # Create a temporary large image to draw text on without clipping issues
    temp_img_width = IMG_WIDTH * 3 # Allow for large text and rotation
    temp_img_height = IMG_HEIGHT * 3
    temp_img = Image.new('RGB', (temp_img_width, temp_img_height), BG_COLOR)
    temp_draw = ImageDraw.Draw(temp_img)

    # Initial text position for drawing
    start_x_drawing = EFFECT_PARAMS['padding'] + temp_img_width // 2 - IMG_WIDTH // 2
    start_y_drawing = EFFECT_PARAMS['padding'] + temp_img_height // 2 - IMG_HEIGHT // 2
    
    current_x = start_x_drawing
    current_y = start_y_drawing
    
    words = text.split(' ')
    word_metrics = [] # Store word info including calculated position and size

    # Pre-calculate word and character positions for all effects
    # Also handle line wrapping
    
    # Measure typical character height for line spacing
    char_height_ref = font.getbbox("Tg")[3] - font.getbbox("Tg")[1] # Use "Tg" for good height reference
    
    for word_idx, word in enumerate(words):
        # Measure word length with current font
        word_render_width = temp_draw.textlength(word, font=font)
        
        # Check if adding the word exceeds image width
        if current_x + word_render_width + EFFECT_PARAMS['padding'] > temp_img_width - (temp_img_width // 2 - IMG_WIDTH // 2):
            # Start a new line
            current_x = start_x_drawing
            current_y += char_height_ref * EFFECT_PARAMS['line_spacing_multiplier']

        # Determine if this word needs strike-then-reprint
        do_strike_then_reprint = random.random() < EFFECT_PROBABILITIES['strike_then_reprint_word'][style]
        # Determine if this word just needs strikethrough (if not striking and reprinting)
        do_crossed_out = not do_strike_then_reprint and random.random() < EFFECT_PROBABILITIES['crossed_out_words'][style]

        word_info = {
            'word_text': word,
            'x': current_x,
            'y': current_y,
            'width': word_render_width,
            'height': char_height_ref, # Using ref height for more consistent line handling
            'strike_then_reprint': do_strike_then_reprint,
            'crossed_out': do_crossed_out,
            'chars_to_correct': [] # Store indices of chars that get double-written
        }
        
        # Identify characters for the "letter correction" effect within this word
        for char_idx, char in enumerate(word):
            if random.random() < EFFECT_PROBABILITIES['letter_correction'][style]:
                word_info['chars_to_correct'].append(char_idx)
        
        word_metrics.append(word_info)
        
        # Predict width if struck and reprinted
        predicted_width = (word_render_width * 2) + EFFECT_PARAMS['reprint_word_spacing'] + font_size // 3 if do_strike_then_reprint else word_render_width + font_size // 3

        # Check if it fits before placing
        if current_x + predicted_width + EFFECT_PARAMS['padding'] > temp_img_width - (temp_img_width // 2 - IMG_WIDTH // 2):
            current_x = start_x_drawing
            current_y += char_height_ref * EFFECT_PARAMS['line_spacing_multiplier']

        # Then move the cursor forward
        current_x += predicted_width

    # Now, draw everything based on the pre-calculated metrics
    all_drawn_elements_bboxes = [] # To calculate overall content bounding box

    for word_info in word_metrics:
        word = word_info['word_text']
        base_x = word_info['x']
        base_y = word_info['y']
        word_width = word_info['width']
        word_height = word_info['height']
        
        current_word_drawing_x = base_x
        
        # --- Draw the initial (potentially struck) word ---
        # Apply baseline deviation for the whole word's characters
        word_y_offset = 0
        if random.random() < EFFECT_PROBABILITIES['baseline_deviation'][style]:
            word_y_offset = random.randint(EFFECT_PARAMS['baseline_deviation_range'][0], EFFECT_PARAMS['baseline_deviation_range'][1])
        
        # In effect this draws it twice at a slight offset to make it look thicker if apply_thicker_lines is true
        if apply_thicker_lines:
            temp_draw.text((current_word_drawing_x, base_y + word_y_offset), word, font=font, fill=TEXT_COLOR)
            temp_draw.text((current_word_drawing_x + EFFECT_PARAMS['thicker_lines_offset'], base_y + word_y_offset), word, font=font, fill=TEXT_COLOR)
        else:
            temp_draw.text((current_word_drawing_x, base_y + word_y_offset), word, font=font, fill=TEXT_COLOR)

        # Record bounding box for this drawing, ie we store (left, top, right, bottom) for this word 
        drawn_bbox = temp_draw.textbbox((current_word_drawing_x, base_y + word_y_offset), word, font=font)
        all_drawn_elements_bboxes.append(drawn_bbox)

        # Apply strikethrough if required
        if word_info['crossed_out'] or word_info['strike_then_reprint']:
            line_start_x = current_word_drawing_x
            line_end_x = current_word_drawing_x + word_width
            bbox = temp_draw.textbbox((current_word_drawing_x, base_y + word_y_offset), word, font=font)
            line_y = int((bbox[1] + bbox[3]) / 2)
            line_width = int(EFFECT_PARAMS['strikethrough_width'])
            temp_draw.line([(line_start_x, line_y), (line_end_x, line_y)], fill=TEXT_COLOR, width=line_width) 
            # Record bounding box for strikethrough
            all_drawn_elements_bboxes.append((line_start_x, line_y - EFFECT_PARAMS['strikethrough_width']/2, line_end_x, line_y + EFFECT_PARAMS['strikethrough_width']/2))

        # --- If 'strike_then_reprint', draw the word again without strikethrough ---
        if word_info['strike_then_reprint']:
            current_word_drawing_x += word_width + EFFECT_PARAMS['reprint_word_spacing']
            
            if apply_thicker_lines:
                temp_draw.text((current_word_drawing_x, base_y + word_y_offset), word, font=font, fill=TEXT_COLOR)
                temp_draw.text((current_word_drawing_x + EFFECT_PARAMS['thicker_lines_offset'], base_y + word_y_offset), word, font=font, fill=TEXT_COLOR)
            else:
                temp_draw.text((current_word_drawing_x, base_y + word_y_offset), word, font=font, fill=TEXT_COLOR)
            
            # Record bounding box for the reprinted word
            drawn_bbox = temp_draw.textbbox((current_word_drawing_x, base_y + word_y_offset), word, font=font)
            all_drawn_elements_bboxes.append(drawn_bbox)


        # --- Apply individual letter correction for the original word (not the reprinted one) ---
        # This will be drawn on top of the already drawn word
        current_char_x_offset_in_word = 0
        for char_idx, char_to_correct in enumerate(word):
            # Calculate char width using the font
            char_width = int(temp_draw.textlength(char_to_correct, font=font))
            
            if char_idx in word_info['chars_to_correct']:
                # Calculate the exact position of this character on the temp_img
                char_abs_x = base_x + current_char_x_offset_in_word
                char_abs_y = base_y + word_y_offset # Same baseline as the word

                # Create a small temporary image for the single character
                # Use a larger buffer for rotation, and ensure transparent background
                buffer_for_char_rotation = max(char_width, char_height_ref) // 2 + 5
                char_temp_img_size = (char_width + buffer_for_char_rotation * 2, char_height_ref + buffer_for_char_rotation * 2)
                char_temp_img = Image.new('RGBA', char_temp_img_size, (255, 255, 255, 0))
                char_temp_draw = ImageDraw.Draw(char_temp_img)
                
                # Draw the character at a relative position within its temp image (centered)
                draw_char_x_in_temp = buffer_for_char_rotation + (char_temp_img_size[0] - char_width - buffer_for_char_rotation * 2) / 2
                draw_char_y_in_temp = buffer_for_char_rotation + (char_temp_img_size[1] - char_height_ref - buffer_for_char_rotation * 2) / 2
                
                # PIL's text() origin is top-left, but getbbox() includes descenders/ascenders
                # Adjust to make sure the char is roughly centered in its temp image
                char_temp_draw.text((buffer_for_char_rotation, buffer_for_char_rotation), char_to_correct, font=font, fill=TEXT_COLOR)

                # Apply rotation
                angle = random.uniform(EFFECT_PARAMS['letter_correction_angle_range'][0], EFFECT_PARAMS['letter_correction_angle_range'][1])
                rotated_char_img = char_temp_img.rotate(angle, expand=True, fillcolor=(255, 255, 255, 0))  # transparent
                
                # Calculate small offset
                offset_x = random.randint(EFFECT_PARAMS['letter_correction_offset_range'][0], EFFECT_PARAMS['letter_correction_offset_range'][1])
                offset_y = random.randint(EFFECT_PARAMS['letter_correction_offset_range'][0], EFFECT_PARAMS['letter_correction_offset_range'][1])

                # Calculate paste position for the rotated image onto the main temp_img
                # This needs to align the center of the rotated char with the center of the original char position
                paste_x = int(char_abs_x + offset_x - (rotated_char_img.width - char_width) / 2)
                paste_y = int(char_abs_y + offset_y - (rotated_char_img.height - char_height_ref) / 2)
                
                # Paste this new char img onto temp_img
                temp_img.paste(rotated_char_img, (paste_x, paste_y), rotated_char_img)
                
                # Add the rotated character's bbox to the overall elements for cropping
                all_drawn_elements_bboxes.append((paste_x, paste_y, paste_x + rotated_char_img.width, paste_y + rotated_char_img.height))

            current_char_x_offset_in_word += char_width # Advance for the next character within the word


    # --- Calculate the overall bounding box of all drawn content ---
    if not all_drawn_elements_bboxes:
        print(f"Warning: No text drawn for '{text}'. Skipping image generation.")
        return None

    min_x_content = min(bbox[0] for bbox in all_drawn_elements_bboxes)
    min_y_content = min(bbox[1] for bbox in all_drawn_elements_bboxes)
    max_x_content = max(bbox[2] for bbox in all_drawn_elements_bboxes)
    max_y_content = max(bbox[3] for bbox in all_drawn_elements_bboxes)

    # Add padding to the content bounding box for cropping
    buffer = EFFECT_PARAMS['padding'] // 2
    crop_box = (max(0, min_x_content - buffer), max(0, min_y_content - buffer),
                min(temp_img_width, max_x_content + buffer), min(temp_img_height, max_y_content + buffer))
    
    # Check crop box has positive dimensions
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        print(f"Warning: Invalid final crop box for text '{text}'. Skipping image generation.")
        return None

    cropped_text_img = temp_img.crop(crop_box)

    # Slanted writing effect (rotate the cropped image)
    if random.random() < EFFECT_PROBABILITIES['slanted_writing'][style]:
        angle = random.uniform(EFFECT_PARAMS['slanted_writing_angle_range'][0], EFFECT_PARAMS['slanted_writing_angle_range'][1])
        cropped_text_img = cropped_text_img.rotate(angle, expand=True, fillcolor=BG_COLOR)

    # Create final image and paste the text
    final_img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), BG_COLOR)
    
    # Calculate paste position to center the text
    paste_x = (IMG_WIDTH - cropped_text_img.width) // 2
    paste_y = (IMG_HEIGHT - cropped_text_img.height) // 2
    
    # Ensure paste coordinates are non-negative
    paste_x = max(0, paste_x)
    paste_y = max(0, paste_y)

    final_img.paste(cropped_text_img, (paste_x, paste_y))

    return final_img