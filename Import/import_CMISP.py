import re
import requests
import hashlib
import os
from tqdm import tqdm  # Import tqdm for progress bar
import time  # Import time for delay in retry

# Path to your bash script file
bash_script_path = r"c:\github\PHD-Paper1\Import-Data\sh\wget_script_2024-10-14_13-39-4.sh"

# Directory where the files will be downloaded
download_directory = r"c:\ThesisData\CMISP"

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

# Function to download a file with a progress bar and retry logic
def download_file(url, filename, max_retries=50, retry_delay=5):
    filepath = os.path.join(download_directory, filename)  # Save to the specified directory
    print(f"Downloading {filename}...")

    retries = 0
    while retries < max_retries:
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
            return True  # Download succeeded

        elif response.status_code == 502:
            retries += 1
            print(f"502 Bad Gateway Error. Retrying... ({retries}/{max_retries})")
            time.sleep(retry_delay)  # Wait before retrying

        else:
            print(f"Failed to download {filename}, status code: {response.status_code}")
            return False  # Other failure

    print(f"Max retries reached. Failed to download {filename} after {max_retries} attempts.")
    return False  # If retries are exhausted

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
    # Extract download files information from the bash script (limit to 100 for testing)
    download_files = extract_download_info(bash_script_path, limit=1000)
    
    for file in download_files:
        filename = file["filename"]
        url = file["url"]
        #checksum = file["checksum"]

        # Download the file with a progress bar and retry logic
        if download_file(url, filename):
            # Verify the checksum if download succeeds
            #if not verify_checksum(filename, checksum):
                #print(f"Deleting {filename} due to checksum mismatch.")
                #os.remove(os.path.join(download_directory, filename))
            pass

if __name__ == "__main__":
    main()
