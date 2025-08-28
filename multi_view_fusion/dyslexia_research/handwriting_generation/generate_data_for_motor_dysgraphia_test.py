import pandas as pd
import os

from generate_text import *
from generate_images_new import *

NUMBER_OF_SENTENCES = 125
DATA_DIR = 'data_dysgraphia'
CSV_OUTPUT_FILE = 'dyslexia_and_dysgraphia_handwriting_dataset.csv'

# Generate non-dyslexic sentences
sentences = generate_sentences(NUMBER_OF_SENTENCES)

# Print the first 10 sentences to verify 
print(f"Generated {len(sentences)} sentences. Here are the first 10:")
for i, sentence in enumerate(sentences[:10]):
    print(f"{i+1}. {sentence}")

# Save these to a file
with open(f"../{DATA_DIR}/random_sentences.txt", "w") as f:
    for sentence in sentences:
        f.write(sentence + "\n")
print(f"\nAll {NUMBER_OF_SENTENCES} sentences saved to random_sentences.txt")

# Generate the modified sentences from these 
dyslexic_sentences = []
for sentence in sentences:
    dyslexic_sentences.append(generate_dyslexic_sentence(sentence))
print(dyslexic_sentences)

# Save sentences to file
with open(f"../{DATA_DIR}/random_dyslexic_sentences.txt", "w") as f:
    for sentence in dyslexic_sentences:
        f.write(sentence + "\n")
print(f"\nAll {NUMBER_OF_SENTENCES} sentences saved to /{DATA_DIR}/random_dyslexic_sentences.txt")

# Now create the images for these sentences 
FONT_DIR = 'fonts' # Directory where font files are located
OUTPUT_NORMAL_DIR = f'../{DATA_DIR}/normal_handwriting_images'
OUTPUT_DYSLEXIC_DIR = f'../{DATA_DIR}/dyslexic_handwriting_images'

# Create output directories if they don't exist
os.makedirs(OUTPUT_NORMAL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DYSLEXIC_DIR, exist_ok=True)

# Load available handwriting fonts
available_fonts = load_fonts(FONT_DIR)
if not available_fonts:
    print("No fonts found. Please add font files to the 'fonts' directory and try again.")
    exit()

file_path = f'../{DATA_DIR}/random_sentences.txt'
dyslexic_file_path = f'../{DATA_DIR}/random_dyslexic_sentences.txt'

with open(file_path, 'r') as file:
    text_samples = file.readlines()

with open(dyslexic_file_path, 'r') as file:
    dyslexic_text_samples = file.readlines()

# Remove newline characters from the end of each line
text_samples = [line.strip() for line in text_samples]
dyslexic_text_samples = [line.strip() for line in dyslexic_text_samples]

print(len(text_samples))
assert len(text_samples) == NUMBER_OF_SENTENCES

print(f"Found {len(available_fonts)} font(s) in '{FONT_DIR}'.")
print(f"Generating {len(text_samples)} normal and {len(text_samples)} dyslexic handwriting images...")

# Here for the non-dyslexic but considered motor-dysgraphic we are creating the non-dyslexic text
# but dyslexic handwriting
for i, sample_text in enumerate(text_samples):
    # Choose a random font for this sample
    selected_font = random.choice(available_fonts)

    # Generate Handwriting Image with dyslexic traits
    normal_img = generate_handwriting_image(sample_text, selected_font, 'dyslexic')
    if normal_img:
        normal_filename = os.path.join(OUTPUT_NORMAL_DIR, f'normal_sample_{i+1:03d}.jpg')
        normal_img.save(normal_filename)
        # print(f"Saved: {normal_filename}")
    
    if (i + 1) % 10 == 0 or (i + 1) == len(text_samples):
        print(f"Processed {i + 1}/{len(text_samples)} text samples.")

# Create the dyslexic samples
for i, sample_text in enumerate(dyslexic_text_samples):
    # Choose a random font for this sample
    selected_font = random.choice(available_fonts)

    # Generate Dyslexic Handwriting Image
    dyslexic_img = generate_handwriting_image(sample_text, selected_font, 'dyslexic')
    if dyslexic_img:
        dyslexic_filename = os.path.join(OUTPUT_DYSLEXIC_DIR, f'dyslexic_sample_{i+1:03d}.jpg')
        dyslexic_img.save(dyslexic_filename)
        # print(f"Saved: {dyslexic_filename}")
    
    if (i + 1) % 10 == 0 or (i + 1) == len(text_samples):
        print(f"Processed {i + 1}/{len(text_samples)} text samples.")

print(f"\nImage generation complete. Images saved in '{OUTPUT_NORMAL_DIR}' and '{OUTPUT_DYSLEXIC_DIR}'.")

# Now create csv containing file path of all images, the text transcript and the classification 

# --- Configuration ---
NUM_SAMPLES = NUMBER_OF_SENTENCES * 2 # As specified, NUMBER_OF_SENTENCES samples for each category

# Base directories for generated images
NORMAL_IMG_DIR = os.path.join('..', DATA_DIR, 'normal_handwriting_images')
DYSLEXIC_IMG_DIR = os.path.join('..', DATA_DIR, 'dyslexic_handwriting_images')

# Ensure the lengths match the expected number of samples
if len(sentences) != NUM_SAMPLES or len(dyslexic_sentences) != NUM_SAMPLES:
    print(f"Warning: Expected {NUM_SAMPLES} sentences but found {len(sentences)} normal and {len(dyslexic_sentences)} dyslexic.")
    print(f"Please ensure 'sentences' and 'dyslexic_sentences' lists contain {NUMBER_OF_SENTENCES} entries each.")
    
    # Proceed with the available number of sentences.
    NUM_SAMPLES = min(len(sentences), len(dyslexic_sentences))

# --- Prepare Data for CSV ---
data = []

# Add handwriting samples (classification = 0) but here these will actually be non-dyslexic sentences but 
# the handwriting image will have dyslexic traits to imitate motor dysgraphia
print(f"Collecting data for {NUM_SAMPLES} motor dysgraphia/normal handwriting samples...")
for i in range(NUM_SAMPLES):
    # Construct file path, ensuring formatting (e.g., 'normal_sample_001.jpg')
    # The image file names are 1-indexed, so we use i+1
    image_filename = f'normal_sample_{i+1:03d}.jpg'
    file_path = os.path.join(NORMAL_IMG_DIR, image_filename)
    
    # Get the corresponding text
    text_content = sentences[i]
    
    data.append({
        'presence_of_dyslexia': 0,
        'file_path': file_path,
        'text': text_content
    })

# Add dyslexic handwriting samples (classification = 1)
print(f"Collecting data for {NUM_SAMPLES} dyslexic handwriting samples...")
for i in range(NUM_SAMPLES):
    # Construct file path
    image_filename = f'dyslexic_sample_{i+1:03d}.jpg'
    file_path = os.path.join(DYSLEXIC_IMG_DIR, image_filename)
    
    # Get the corresponding text
    text_content = dyslexic_sentences[i]
    
    data.append({
        'presence_of_dyslexia': 1,
        'file_path': file_path,
        'text': text_content
    })

# --- Create DataFrame and Save to CSV ---
print("Creating DataFrame...")
df = pd.DataFrame(data)

print(f"Saving data to {CSV_OUTPUT_FILE}...")
df.to_csv(CSV_OUTPUT_FILE, index=False)

print(f"\nCSV file '{CSV_OUTPUT_FILE}' created successfully with {len(df)} entries.")
print("First 5 rows of the CSV:")
print(df.head())
print("\nLast 5 rows of the CSV:")
print(df.tail())