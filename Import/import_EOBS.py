import requests
import os
from tqdm import tqdm  # Import tqdm for progress bar
import time  # Import time for delay in retry

# Directory where the files will be downloaded
download_directory = r"c:\ThesisData\RainfallEnsembles"

# Create the download directory if it doesn't exist
if not os.path.exists(download_directory):
    os.makedirs(download_directory)

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

# Main function to handle downloading for each year
def main():
    # Loop through the years from 1950 to 2024
    for year in range(1950, 2025):
        # Construct the URL by replacing the year in the URL pattern
        url = f"https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/Grid_0.1deg_reg_ensemble/rr_ensemble_0.1deg_reg_{year}_v30.0e.nc"
        filename = f"rr_ensemble_0.1deg_reg_{year}_v30.0e.nc"

        # Download the file with a progress bar and retry logic
        if download_file(url, filename):
            print(f"Successfully downloaded {filename}")
        else:
            print(f"Failed to download {filename}")

if __name__ == "__main__":
    main()
