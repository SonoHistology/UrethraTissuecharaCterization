#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:07:37 2023

@author: haoweitai
"""

import os

def rename_files_in_directory(directory_path):
    # Listing subdirectories
    for subfolder in os.listdir(directory_path):
        subfolder_path = os.path.join(directory_path, subfolder)
        
        # Checking if it's a directory
        if os.path.isdir(subfolder_path):
            # Fetching the list of files, excluding .DS_Store
            files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f)) and f != ".DS_Store"]
            
            # Sorting the files
            sorted_files = sorted(files)
            
            # Renaming files
            for idx, filename in enumerate(sorted_files, 1):
                old_filepath = os.path.join(subfolder_path, filename)
                new_filepath = os.path.join(subfolder_path, f"img{idx}.jpg")  # Assuming files are .jpg. Change the extension if different.
                os.rename(old_filepath, new_filepath)

if __name__ == "__main__":
    main_folder_path = './backup/'
    rename_files_in_directory(main_folder_path)
