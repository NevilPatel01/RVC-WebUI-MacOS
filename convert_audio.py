import os
from pydub import AudioSegment
import sys


def convert_audio(input_path, output_path):
    try:
        # Load the audio file
        audio = AudioSegment.from_file(input_path)

        # Convert to WAV format with specific parameters
        audio = audio.set_channels(2)  # Stereo
        audio = audio.set_frame_rate(44100)  # 44.1kHz
        audio = audio.set_sample_width(2)  # 16-bit

        # Export as WAV
        audio.export(output_path, format="wav")
        print(f"Successfully converted {input_path} to {output_path}")
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_audio.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_audio(input_file, output_file)
