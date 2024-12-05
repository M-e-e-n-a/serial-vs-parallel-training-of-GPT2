def convert_rtf_to_csv(input_content):
    """
    Converts RTF-formatted CSV content into proper CSV format.
    """
    import re
    
    # Extract the actual CSV data from RTF content
    # First, find the actual data portion after the RTF header
    data_start = input_content.find("PatientID")
    if data_start == -1:
        raise ValueError("Could not find CSV header in RTF content")
        
    raw_data = input_content[data_start:]
    
    # Clean up RTF formatting
    cleaned_data = (raw_data
        .replace('\\', '')  # Remove backslashes
        .replace('{', '')   # Remove curly braces
        .replace('}', '')
        .replace(',,,,,,,,,,,,', ',')  # Clean up repeated commas
        .replace(',,,,,,,,,,', ',')
        .replace(',,,,,,,,,', ',')
        .replace(',,,,,,,,', ',')
        .replace(',,,,,,,', ',')
        .replace(',,,,,,', ',')
        .replace(',,,,,', ',')
        .replace(',,,,', ',')
        .replace(',,,', ',')
        .replace(',,', ',')
    )
    
    # Split into lines and clean each line
    lines = cleaned_data.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip():  # Only keep non-empty lines
            # Remove any remaining RTF commands
            line = re.sub(r'\\[a-z0-9]+\s?', '', line)
            cleaned_lines.append(line)
    
    # Write to new CSV file
    with open('cleaned_dataset.csv', 'w') as f:
        f.write('\n'.join(cleaned_lines))
    
    return 'cleaned_dataset.csv'

# Read the original file
try:
    print("Starting conversion process...")
    with open('big_dataset.csv', 'r') as f:
        content = f.read()
    
    # Convert and save as CSV
    output_file = convert_rtf_to_csv(content)
    print(f"\nFile has been converted and saved as {output_file}")
    
    # Verify the conversion by reading a few lines
    print("\nVerifying converted data:")
    with open(output_file, 'r') as f:
        print(f.readline().strip(), "- (Header)")
        print(f.readline().strip(), "- (First data row)")
        
except Exception as e:
    print(f"An error occurred: {e}")