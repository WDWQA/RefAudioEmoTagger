import os
import argparse

def rename_wav_with_txt(directory):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.lab'):
                basename = os.path.splitext(filename)[0]
                wav_path = os.path.join(root, f"{basename}.wav")
                txt_path = os.path.join(root, filename)
                
                if os.path.exists(wav_path):
                    with open(txt_path, 'r', encoding='utf-8') as txt_file:
                        new_name = txt_file.read().strip()
                        
                        if os.path.isfile(new_name) or os.path.isdir(new_name) or not os.path.basename(new_name):
                            print(f"Skipping invalid filename: {new_name}")
                        else:
                            try:
                                os.rename(wav_path, os.path.join(root, f"{new_name}.wav"))
                                print(f"Renamed {wav_path} to {new_name}.wav")
                            except OSError as e:
                                print(f"Error renaming {wav_path}: {e}")
                else:
                    print(f"No matching WAV file found for {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename WAV files based on corresponding TXT files')
    parser.add_argument('directory', help='Path to the directory containing WAV and TXT files')
    args = parser.parse_args()
    
    directory_path = args.directory
    if os.path.isdir(directory_path):
        rename_wav_with_txt(directory_path)
    else:
        print("Invalid directory path!")
