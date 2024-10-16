import re
import requests
import hashlib
import os
from tqdm import tqdm  # Import tqdm for progress bar

# Path to your bash script file
bash_script_path = r"c:\Users\bobel\OneDrive - Université de Namur\Scripts\ClimateDataRetrieval\wget_script_2024-10-14_13-39-4.sh"

# Directory where the files will be downloaded
download_directory = r"C:\Users\bobel\OneDrive - Université de Namur\Data\CMIS"

# Create the download directory if it doesn't exist
if not os.path.exists(download_directory):
    os.makedirs(download_directory)

# Extract the first 5 URLs, filenames, and checksums from the bash script
def extract_download_info(bash_script_path, limit=5):
    download_files = []
    
    # Regular expression pattern to match each line with the format of filename, URL, checksum type, and checksum
    pattern = re.compile(r"'(?P<filename>.*?)' '(?P<url>.*?)' 'SHA256' '(?P<checksum>.*?)'")
    
    with open(bash_script_path, "r") as script:
        for line in script:
            match = pattern.search(line)
            if match:
                download_files.append(match.groupdict())
            if len(download_files) == limit:
                break
    
    return download_files

# Function to download a file with a progress bar and save it locally
def download_file(url, filename):
    filepath = os.path.join(download_directory, filename)  # Save to the specified directory
    print(f"Downloading {filename}...")

    # Start downloading the file with a stream
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))  # Get the total file size from headers
        block_size = 1024  # Set block size for chunks
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)  # Initialize tqdm progress bar

        with open(filepath, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))  # Update progress bar
                f.write(data)
        progress_bar.close()

        print(f"Downloaded {filename} to {download_directory}.")
    else:
        print(f"Failed to download {filename}, status code: {response.status_code}")

# Function to verify the SHA256 checksum of the downloaded file
def verify_checksum(filename, expected_checksum):
    filepath = os.path.join(download_directory, filename)  # Use the specified directory
    print(f"Verifying checksum for {filename}...")
    sha256 = hashlib.sha256()
    
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    
    calculated_checksum = sha256.hexdigest()
    if calculated_checksum == expected_checksum:
        print(f"Checksum matches for {filename}.")
        return True
    else:
        print(f"Checksum mismatch for {filename}!")
        print(f"Expected: {expected_checksum}")
        print(f"Got: {calculated_checksum}")
        return False

# Main function to handle downloading and verification
def main():
    # Extract download files information from the bash script (limit to 5 for testing)
    download_files = extract_download_info(bash_script_path, limit=5)
    
    for file in download_files:
        filename = file["filename"]
        url = file["url"]
        #checksum = file["checksum"]

        # Download the file with a progress bar
        download_file(url, filename)

        # Verify the checksum
        #if not verify_checksum(filename, checksum):
            #print(f"Deleting {filename} due to checksum mismatch.")
            #os.remove(os.path.join(download_directory, filename))

if __name__ == "__main__":
    main()
