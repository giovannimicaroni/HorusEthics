#!/bin/bash

# List of directories
dirs=("ipad" "iphone" "s9" "s22" "tabs6")

# Loop through each directory
for dir in "${dirs[@]}"; do
  # Initialize counter
  count=0
  # Loop through each file in the directory
  for file in "$dir"/*; do
    # Generate new name with leading zeroes
    new_name=$(printf "%s/%s%02d.jpg" "$dir" "$dir" "$count")
    # Rename the file
    mv "$file" "$new_name"
    # Increment counter
    count=$((count + 1))
  done
done

