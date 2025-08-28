# Import all necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import os
import pandas as pd
import pydicom
import re
import numpy as np
import matplotlib.pyplot as plt 
import time
import copy
from typing import Any
import pathlib

class MammogramDataset:
    """
    Custom dataset class for loading mammogram DICOM (.dcm) files
    from the CBIS-DDSM dataset structure, grouping images by individual breast i.e. left or right and per region of interest 
    (which can be identified by the abnormality ID).
    Each item in this dataset represents a unique Region of Interest ROI set (Patient ID + Breast Side + Abnormality ID),
    and contains all associated DICOM images found.
    The dataset contains calcifications and masses, we concentrate on calcifications as there are more usable 
    There can be more than one region of interest for one patient's particular breast. 

    This filters entities based on a provided CSV file, ensuring
    only Region of Interest (ROI) with both 'MLO' and 'CC' views are included, i.e. we need both of these files. It also includes
    'breast density' and 'pathology' for each ROI entity, with consistency checks.
    """

    def __init__(self, root_dir, csv_path=None, is_mass=False):
        """
        Initialize the dataset: collecting and grouping all relevant DICOM file paths
        and their parsed metadata by unique ROI entity.

        Args:
            root_dir (str): The path to the top-level 'CBIS-DDSM' directory.
            csv_path (str, optional): The path to a single description CSV file
                                        (e.g., mass_case_description_train_set.csv).
                                        If provided, only ROI entities with both
                                        'MLO' and 'CC' views as indicated in this
                                        CSV will be included, and 'breast density'
                                        and 'pathology' will be added.
        """
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.roi_entities = [] # List of unique ROI entities (Patient ID + Breast Side + Abnormality)
        self.is_mass = is_mass
        # Ensure the provided root directory exists
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Error: Root directory not found: '{self.root_dir}'. "
                                    f"Please ensure the 'CBIS-DDSM' folder path is correct.")

        self._collect_files()

    def _parse_folder_name(self, folder_name) -> tuple[str | None, str | None, str | None, str | None]:
        """
        Parses a folder name (e.g., 'Calc-Test_P_00038_LEFT_CC_1') to extract
        the patient ID (including the 'P_' prefix), breast side, and image view.
        This now correctly captures views like 'CC_1' or 'MLO_1'.

        Args:
            folder_name (str): The name of the directory containing patient/series info.

        Returns:
            tuple: (patient_id, breast_side, image_view, abnormality_id) or (None, None, None, None) if no match.
        """
        # Regex updated:
        # P_(\d+) for patient ID with 'P_' prefix.
        # (LEFT|RIGHT) for breast side.
        # ((?:CC|MLO)(?:_\d+)?) for image view, allowing 'CC', 'MLO', 'CC_1', 'MLO_1', etc.
        # (?:_.*)?$ for optional trailing characters after the main view component.
        match = re.match(r'^(?:Calc|Mass|Calc-Test|Mass-Test|Calc-Training|Mass-Training)_(P_\d+)_(LEFT|RIGHT)_((?:CC|MLO)(?:_(\d+))?)(?:_.*)?$', folder_name, re.IGNORECASE)
        if match:
            patient_id = match.group(1)   # Captured patient ID (e.g., 'P_00038')
            breast_side = match.group(2)  # Captured breast side (e.g., 'LEFT')
            image_view = match.group(3)   # Captured image view (e.g., 'CC', 'MLO_1')
            abnormality_id = match.group(4) # The number after the CC/MLO
            return patient_id, breast_side, image_view, abnormality_id
        return None, None, None, None  # Return None if the folder name doesn't match the expected pattern

    def _load_and_filter_csv_metadata(self) -> dict[tuple[str, str, str | None], Any]:
        """
        Loads the single CSV file, extracts relevant image paths, and identifies
        patient-ROI entities that have both MLO and CC views according to the CSV data.
        Also extracts and checks consistency for 'breast density' and 'pathology'.
        (This csv could be for calcifications or masses.)

        Returns:
            dict: A dictionary where keys are (patient_id, breast_side, abnormality id) tuples
                  and values are dictionaries containing 'views', 'density', and 'pathology'.
                  Only ROIs with both 'CC' and 'MLO' are included.
        """
        if not self.csv_path:
            return {} # No CSV provided, no CSV-based filtering

        if not os.path.exists(self.csv_path):
            print(f"Warning: CSV file not found: '{self.csv_path}'. Skipping CSV filtering.")
            return {}
            
        all_csv_data = pd.read_csv(self.csv_path)
        if self.is_mass:
            all_csv_data = all_csv_data.rename(columns={'breast_density' : 'breast density'})
        roi_attributes_from_csv = {} # Key: (patient_id, breast_side), Value: {'views': set(), 'density': value, 'pathology': value}
        
        # Check for required columns
        required_cols = ['image file path', 'breast density', 'pathology', 'abnormality id']

        # Add optional path columns, might use these in the future
        optional_path_cols = ['cropped image file path', 'ROI mask file path']

        # Ensure primary required columns are present
        if not all(col in all_csv_data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in all_csv_data.columns]
            print(f"Error: Missing expected columns {missing_cols} in CSV. Cannot filter/extract by CSV data.")
            return {}

        for _, row in all_csv_data.iterrows():
            current_density = row['breast density']
            current_pathology = row['pathology']
            current_abnormality_id = row['abnormality id']
            
            # Collect potential folder names from all relevant path columns
            potential_series_folder_names = []
            
            # Process 'image file path'
            if 'image file path' in row and pd.notna(row['image file path']):
                potential_series_folder_names.append(str(row['image file path']).split(os.sep)[0])
            
            # Process 'cropped image file path'
            if 'cropped image file path' in row and pd.notna(row['cropped image file path']):
                potential_series_folder_names.append(str(row['cropped image file path']).split(os.sep)[0])

            # Process 'ROI mask file path'
            if 'ROI mask file path' in row and pd.notna(row['ROI mask file path']):
                potential_series_folder_names.append(str(row['ROI mask file path']).split(os.sep)[0])

            # Iterate through all potential folder names found in the current row
            for series_folder_name_from_csv in potential_series_folder_names:
                patient_id, breast_side, image_view_from_csv, abnormality_id = self._parse_folder_name(series_folder_name_from_csv)

                # we skip if any of the key values are None because we are simply collecting
                # keys for which there may be a cropped ROI MLO and CC view
                if patient_id and breast_side and image_view_from_csv and abnormality_id:
                    key = (patient_id, breast_side, abnormality_id)
                    
                    if key not in roi_attributes_from_csv:
                        roi_attributes_from_csv[key] = {
                            'views': set(),
                            'breast density': None,
                            'pathology': None,
                        }
                    
                    # Check for consistency of density and pathology within the same roi_key
                    # Density 
                    if roi_attributes_from_csv[key]['breast density'] is None:
                        roi_attributes_from_csv[key]['breast density'] = current_density
                    elif roi_attributes_from_csv[key]['breast density'] != current_density:
                        print(f"Warning: inconsistent 'breast density' ('{current_density}' vs previously "
                              f"'{roi_attributes_from_csv[key]['breast density']}') found for breast {key}. "
                              f"Using '{current_density}'.")
                        roi_attributes_from_csv[key]['breast density'] = current_density # Overwrite with last encountered

                    # Pathology
                    if roi_attributes_from_csv[key]['pathology'] is None:
                        roi_attributes_from_csv[key]['pathology'] = current_pathology
                    elif roi_attributes_from_csv[key]['pathology'] != current_pathology:
                        '''print(f"Warning: inconsistent 'pathology' ('{current_pathology}' vs previously "
                              f"'{roi_attributes_from_csv[key]['pathology']}') found for breast {key}. "
                              f"Using '{current_pathology}'.")'''
                        roi_attributes_from_csv[key]['pathology'] = current_pathology # Overwrite with last encountered
                    
                    base_view = image_view_from_csv.split('_')[0].upper()
                    if base_view in ['CC', 'MLO']:
                        roi_attributes_from_csv[key]['views'].add(base_view)
            
        filtered_roi_attributes_from_csv = {}
        for key, attributes in roi_attributes_from_csv.items():
            if 'CC' in attributes['views'] and 'MLO' in attributes['views']:
                # Ensure density and pathology were actually populated
                if attributes['breast density'] is not None and attributes['pathology'] is not None:
                    filtered_roi_attributes_from_csv[key] = attributes
                else:
                    print(f"Warning: ROI {key} has both CC/MLO views but missing 'breast density' or 'pathology' from CSV. Skipping.")
            else:
                # Breast {key} does not have both CC/MLO views, we skip
                pass

        return filtered_roi_attributes_from_csv

    def _collect_files(self):
        """
        Walks through the root dir, finds all '.dcm' files, and groups them
        by unique ROI entity (patient_id, breast_side, abnormality_id). 
        Applies CSV-based filtering (if CSV paths are provided).
        """        
        # Determine valid entities based on CSV (if provided)
        # i.e. iterate over the CSV rows and directories and return entities that
        # group files for the same patient/side/abnormality id
        print("In collect_files")
        valid_csv_lookup = self._load_and_filter_csv_metadata()
        if self.csv_path and not valid_csv_lookup:
            print("Warning: No entities with both CC and MLO views found in provided CSV and matching criteria. "
                  "The dataset will be empty as per filtering criteria.")
            return

        temp_roi_data = {}
        found_files_count_raw = 0
        found_files_count_filtered = 0

        # iterate over the files in the directories, and if the file is 
        # an image in a directory we have an entity for, load into memory
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.lower().endswith('.dcm'):
                    # For mass dataset, many CC and MLO have two subdirs instead of one with the name ROI mask images only, so we need to check if there is a sibling dir with name cropped images, this will be the ROI image we require
                    if self.is_mass:
                        if "ROI mask images" in dirpath:
                            parent_path = pathlib.Path(f"{dirpath}/../..")
                            cropped_images_dir = list(parent_path.rglob("*"))                           
                            if any("cropped images" in str(x) for x in cropped_images_dir):
                                print (f"INFO: Appear to have sibling folder to {dirpath}, skipping")
                                continue

                    dcm_file_path = os.path.join(dirpath, filename)
                    found_files_count_raw += 1
                    
                    series_info_dir_path = os.path.dirname(os.path.dirname(dirpath))
                    series_folder_name = os.path.basename(series_info_dir_path)
                    
                    # parse the directory to build a key to check for in lookup
                    patient_id, breast_side, image_view, abnormality_id = self._parse_folder_name(series_folder_name)

                    # as before, we collect the folders with no abnormality id
                    if patient_id and breast_side and image_view:
                        roi_key = (patient_id, breast_side, abnormality_id)
                        
                        # Apply CSV filtering if CSV was provided
                        if self.csv_path:
                            if roi_key not in valid_csv_lookup:
                                # Skip this file, the folder was not mapped
                                # to a row in the csv; either it isn't a CC/MLO
                                # folder, or there was not a sibling CC/MLO for it
                                continue 

                        base_view_type = image_view.split('_')[0].upper()
                        if base_view_type not in ['CC', 'MLO']:
                            continue

                        if roi_key not in temp_roi_data:
                            temp_roi_data[roi_key] = {
                                'patient_id': patient_id,
                                'breast_side': breast_side,
                                'abnormality id': abnormality_id,
                                'images': [] # This list will hold details for each image of this breast
                            }
                            # Add breast density and pathology from CSV if available
                            if self.csv_path and roi_key in valid_csv_lookup:
                                temp_roi_data[roi_key]['breast density'] = valid_csv_lookup[roi_key]['breast density']
                                temp_roi_data[roi_key]['pathology'] = valid_csv_lookup[roi_key]['pathology']
                            else:
                                temp_roi_data[roi_key]['breast density'] = 'N/A'
                                temp_roi_data[roi_key]['pathology'] = 'N/A'
                                    
                        # Add the images    
                        temp_roi_data[roi_key]['images'].append({
                            'file_path': dcm_file_path,
                            'image_view': image_view,
                            'series_folder_name': series_folder_name # Full folder name for context
                        })
                        found_files_count_filtered += 1
        
        '''print(f"Initially found {found_files_count_raw} DICOM files in directories.")
        print(f"After initial filtering (CSV/view type), collected {found_files_count_filtered} images.")'''

        final_roi_entities = []
        for roi_key, roi_info in temp_roi_data.items():
            actual_base_views_found = set()
            for img_data in roi_info['images']:
                base_view = img_data['image_view'].split('_')[0].upper()
                actual_base_views_found.add(base_view)
            
            # Both views needed
            if 'CC' in actual_base_views_found and 'MLO' in actual_base_views_found:
                final_roi_entities.append(roi_info)
            else:
                print(f"Warning: ROI (Region of Interest) {roi_key} was filtered out. Expected both CC and MLO views based on file system check, "
                      f"but only found {actual_base_views_found}. This might indicate missing files despite CSVs.")
        
        self.roi_entities = final_roi_entities
        print(f"Final dataset contains {len(self.roi_entities)} unique mammogram ROI collection entities "
              f"(each with both CC and MLO views, and matching CSV criteria if provided).")

    def __len__(self):
        """
        Returns the total number of unique entities in the dataset.
        """
        return len(self.roi_entities)

    def __getitem__(self, idx):
        """
        Loads and returns all DICOM images (as NumPy arrays) and their associated
        metadata for a given unique ROI entity (identified by index).
        DICOM files are read on demand.

        Args:
            idx (int): The index of the ROI entity to retrieve.

        Returns:
            dict: A dictionary containing the patient ID, breast side, abnormality id
                  breast density, pathology, and a list of dictionaries, where each
                  inner dictionary holds the image data (NumPy array) and various
                  metadata for a single image.
                  Returns None if the index is out of bounds or an error
                  occurs during file processing.
        """
        if not (0 <= idx < len(self.roi_entities)):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.roi_entities)}")

        roi_data = self.roi_entities[idx]
        processed_images = []

        for image_info in roi_data['images']:
            file_path = image_info['file_path']
            try:
                dicom_data = pydicom.dcmread(file_path)           
                image_array = dicom_data.pixel_array.astype(np.float32)
                
                # Construct data entry for this specific image
                processed_images.append({
                    'image': image_array, # The processed image pixel data (NumPy array)
                    'file_path': file_path,
                    'image_view': image_info['image_view'],
                    'series_folder_name': image_info['series_folder_name'],
                })
            except Exception as e:
                print(f"Error reading or processing DICOM file '{file_path}': {e}")
                # Currently, we're just skipping problematic individual images within a ROI entity, 
                # but might need to log more severely if we add logging file in future.

        # Return the  ROI entity with all its processed images
        return {
            'patient_id': roi_data['patient_id'],
            'breast_side': roi_data['breast_side'],
            'breast density': roi_data['breast density'], 
            'abnormality id': roi_data['abnormality id'], 
            'pathology': roi_data['pathology'],          
            'images': processed_images # List of all images (NumPy arrays + their specific metadata) for this ROI
        }

    def display_roi_images(self, roi_entity_or_idx):
        """
        Displays all roi images associated with a given entity.

        Args:
            roi_entity_or_idx (dict or int): Either a dictionary representing
                a ROI entity (as returned by __getitem__) or an integer index
                into the dataset.
        """
        if isinstance(roi_entity_or_idx, int):
            roi_data = self.__getitem__(roi_entity_or_idx)
        elif isinstance(roi_entity_or_idx, dict) and 'patient_id' in roi_entity_or_idx and 'images' in roi_entity_or_idx:
            roi_data = roi_entity_or_idx
        else:
            print("Error: Invalid input. Please provide a valid index.")
            return

        if not roi_data or not roi_data['images']:
            print(f"No images found for ROI entity: Patient ID {roi_data.get('patient_id', 'N/A')}, "
                  f"Side {roi_data.get('breast_side', 'N/A')}")
            return

        num_images = 0
        for image_info in roi_data['images']:
            if image_info['file_path'].endswith("1-1.dcm"):
                num_images +=1     
        # Determine optimal subplot grid
        if num_images == 1:
            rows, cols = 1, 1
        elif num_images == 2:
            rows, cols = 1, 2
        elif num_images == 3:
            rows, cols = 1, 3
        else: # 4 or more images, arrange in a 2x2 grid, or more if needed
            rows = (num_images + 1) // 2
            cols = 2
            
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3)) 
        axes = axes.flatten()
        fig.suptitle(f"Patient: {roi_data['patient_id']}, Breast: {roi_data['breast_side']}\n"
                     f"Density: {roi_data.get('breast density', 'N/A')}, Pathology: {roi_data.get('pathology', 'N/A')}", 
                     fontsize=10)

        found_count = 0
        plot_index = 0
        for image_info in roi_data['images']:
            if image_info['file_path'].endswith("1-1.dcm"):                
                found_count +=1
                if plot_index < len(axes):                   
                    ax = axes[plot_index]
                    image_array = image_info['image']
                    image_view = image_info['image_view']                    
                    ax.imshow(image_array, cmap='gray', aspect='auto')
                    ax.set_title(f"View: {image_view}")
                    ax.axis('off') 
                    plot_index +=1
                
        # Hide any unused subplots if num_images is not a fit for the grid
        for j in range(found_count + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to stop title overlap
        plt.show()