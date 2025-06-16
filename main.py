import streamlit as st
import pandas as pd
import tempfile
import time
import sys
import re
import os

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline, AutoTokenizer
from torchaudio.transforms import Resample
import soundfile as sf
import torchaudio
import yt_dlp
import torch

class Interface:
    @staticmethod
    def get_header(title: str, description: str) -> None:
        """
        Display the header of the application.
        """
        st.set_page_config(
            page_title="Audio Summarization",
            page_icon="🗣️",
        )

        hide_decoration_bar_style = """<style>header {visibility: hidden;}</style>"""
        st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
        hide_streamlit_footer = """
        <style>#MainMenu {visibility: hidden;}
        footer {visibility: hidden;}</style>
        """
        st.markdown(hide_streamlit_footer, unsafe_allow_html=True)
        
        st.title(title)
            
        st.info(description)
        st.write("\n")

    @staticmethod
    def get_audio_file() -> str:
        """
        Upload an audio file for transcription and summarization.
        """
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav"],
            help="Upload an audio file for transcription and summarization.",
        )
        if uploaded_file is None:
            return None
        
        if uploaded_file.name.endswith(".wav"):
            st.audio(uploaded_file, format="audio/wav")
        else:
            st.warning("Please upload a valid .wav audio file.")
            return None
        
        return uploaded_file
    
    @staticmethod
    def get_approach() -> None:
        """
        Select the approach for input audio summarization.
        """
        approach = st.selectbox(
            "Select the approach for input audio summarization",
            options=["Youtube Link", "Input Audio File"],
            index=1,
            help="Choose the approach you want to use for summarization.",
        )

        return approach
    
    @staticmethod
    def get_link_youtube() -> str:
        """
        Input a YouTube link for audio summarization.
        """
        youtube_link = st.text_input(
            "Enter the YouTube link",
            placeholder="https://www.youtube.com/watch?v=example",
            help="Paste the YouTube link of the video you want to summarize.",
        )
        if youtube_link.strip():
            st.video(youtube_link)

        return youtube_link
    
    @staticmethod
    def get_sidebar_input(state: dict) -> str:
        """
        Handles sidebar interaction and returns the audio path if available.
        """
        with st.sidebar:
            st.markdown("### Select Approach")
            approach = Interface.get_approach()
            state['session'] = 1

            audio_path = None

            if approach == "Input Audio File" and state['session'] == 1:
                audio = Interface.get_audio_file()
                if audio is not None:
                    audio_path = Utils.temporary_file(audio)
                    state['session'] = 2

            elif approach == "Youtube Link" and state['session'] == 1:
                youtube_link = Interface.get_link_youtube()
                if youtube_link:
                    audio_path = Utils.download_youtube_audio_to_tempfile(youtube_link)
                    if audio_path is not None:
                        with open(audio_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format="audio/wav")
                        state['session'] = 2
            
            generate = False
            if state['session'] == 2 and 'audio_path' in locals() and audio_path:
                generate = st.button("🚀 Generate Result !!")

            return audio_path, generate

class Utils:
    @staticmethod
    def temporary_file(uploaded_file: str) -> str:
        """
        Create a temporary file for the uploaded audio file.
        """
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            return temp_file_path
        
    @staticmethod   
    def clean_transcript(text: str) -> str:
        """
        Clean the transcript text by removing unwanted characters and formatting.
        """
        text = text.replace(",", " ")
        text = re.sub(r'(?<=[a-zA-Z])\.(?=[a-zA-Z])', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*\.\s*', '. ', text)
        return text.strip()
    
    @staticmethod
    def preprocess_audio(input_path: str) -> str:
        """
        Preprocess the audio file by converting it to mono and resampling to 16000 Hz.
        """
        waveform, sample_rate = torchaudio.load(input_path)
        print(f"📢 Original waveform shape: {waveform.shape}")
        print(f"📢 Original sample rate: {sample_rate}")

        # Convert to mono (average if stereo)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            print("✅ Converted to mono.")

        # Resample to 16000 Hz if needed
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            print(f"✅ Resampled to {target_sample_rate} Hz.")
            sample_rate = target_sample_rate

        # Create a temporary file for the output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            output_path = tmpfile.name

        torchaudio.save(output_path, waveform, sample_rate)
        print(f"✅ Saved preprocessed audio to temporary file: {output_path}")

        return output_path
    
    @staticmethod
    def _format_filename(input_string, chunk_number=0):
        """
        Format the input string to create a valid filename.
        Replaces non-alphanumeric characters with underscores, removes extra spaces,
        and appends a chunk number if provided.
        """
        input_string = input_string.strip()
        formatted_string = re.sub(r'[^a-zA-Z0-9\s]', '_', input_string)
        formatted_string = re.sub(r'[\s_]+', '_', formatted_string)
        formatted_string = formatted_string.lower()
        formatted_string += f'_chunk_{chunk_number}'
        return formatted_string

    @staticmethod
    def download_youtube_audio_to_tempfile(youtube_url):
        """
        Download audio from a YouTube video and save it as a WAV file in a temporary directory.
        Returns the path to the saved WAV file.
        """
        try:
            cookies_path = '/root/audio_summarization_favian_test/audio_summarization_v2/cookies.txt'
            temp_dir = tempfile.mkdtemp()

            # First, verify cookies file exists
            if not os.path.exists(cookies_path):
                raise FileNotFoundError(f"Cookies file not found: {cookies_path}")

            # Use the video title as filename
            with yt_dlp.YoutubeDL({
                'quiet': True,
                'cookies': cookies_path,
                'user_agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                'nocheckcertificate': True,
                'forceipv4': True,
            }) as ydl:
                info_dict = ydl.extract_info(youtube_url, download=False)
                title = info_dict.get('title', 'audio')
                # More aggressive sanitization
                sanitized_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                sanitized_title = sanitized_title.replace(" ", "_")[:100]  # Limit length
                output_path_no_ext = os.path.join(temp_dir, sanitized_title)

            # Enhanced download options - match your working command line version exactly
            ydl_opts = {
                'format': 'bestaudio/best',
                'cookies': cookies_path,
                'user_agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                'nocheckcertificate': True,
                'forceipv4': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'outtmpl': output_path_no_ext + '.%(ext)s',
                'quiet': False,  # Enable logging to see what's happening
                'noplaylist': True,
                # Additional options that might help
                'extract_flat': False,
                'cookiesfrombrowser': None,  # Explicitly disable browser cookies
                'no_warnings': False,
            }

            print(f"Attempting to download: {youtube_url}")
            print(f"Using cookies from: {cookies_path}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])

            expected_output = output_path_no_ext + ".wav"
            timeout = 30  # Increased timeout
            print(f"Waiting for file: {expected_output}")
            
            while not os.path.exists(expected_output) and timeout > 0:
                time.sleep(1)
                timeout -= 1
                if timeout % 5 == 0:  # Progress indicator
                    print(f"Still waiting... {timeout}s remaining")

            if not os.path.exists(expected_output):
                # Check if any files were created in temp_dir
                files_in_temp = os.listdir(temp_dir)
                print(f"Files in temp directory: {files_in_temp}")
                raise FileNotFoundError(f"Audio file was not saved as expected: {expected_output}")

            print(f"Audio downloaded to: {expected_output}")
            return expected_output

        except Exception as e:
            print(f"Failed to download {youtube_url}: {e}")
            return None
        
class Generation:
    def __init__(
            self, 
            summarization_model: str = "vian123/brio-finance-finetuned-v2",
            speech_to_text_model: str = "nyrahealth/CrisperWhisper", 
    ):
        self.summarization_model = summarization_model
        self.speech_to_text_model = speech_to_text_model
        self.device = "cpu"
        self.dtype = torch.float32
        self.processor_speech = AutoProcessor.from_pretrained(speech_to_text_model)
        self.model_speech = AutoModelForSpeechSeq2Seq.from_pretrained(
            speech_to_text_model,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="eager",
        ).to(self.device)
        self.summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model)

    def transcribe_audio_pytorch(self, file_path: str) -> str:
        """
        transcribe audio using the PyTorch-based speech-to-text model.
        """
        converted_path = Utils.preprocess_audio(file_path)
        waveform, sample_rate = torchaudio.load(converted_path)
        duration = waveform.shape[1] / sample_rate
        if duration < 1.0:
            print("❌ Audio too short to process.")
            return ""

        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_speech,
            tokenizer=self.processor_speech.tokenizer,
            feature_extractor=self.processor_speech.feature_extractor,
            chunk_length_s=5,
            batch_size=1,
            return_timestamps=None,
            torch_dtype=self.dtype,
            device=self.device,
            model_kwargs={"language": "en"},
        )

        try:
            hf_pipeline_output = pipe(converted_path)
            print("✅ HF pipeline output:", hf_pipeline_output)
            return hf_pipeline_output.get("text", "")
        except Exception as e:
            print("❌ Pipeline failed with error:", e)
            return ""

    def summarize_string(self, text: str) -> str:
        """
        Summarize the input text using the summarization model.
        """
        summarizer = pipeline("summarization", model=self.summarization_model, tokenizer=self.summarization_model)
        try:
            if len(text.strip()) < 10:
                return ""

            inputs = self.summarization_tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            truncated_text = self.summarization_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

            word_count = len(truncated_text.split())
            min_len = max(int(word_count * 0.5), 30)
            max_len = max(min_len + 20, int(word_count * 0.75))

            summary = summarizer(
                truncated_text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )
            return summary[0]['summary_text']
        except Exception as e:
            return f"Error: {e}"
        
def main():
  Interface.get_header(
    title="Financial YouTube Video Audio Summarization",
    description="🎧 Upload an financial audio file or financial YouTube video link to 📝 transcribe and 📄 summarize its content using CrisperWhisper and Financial Fine-tuned BRIO 🤖."
  )

  generate = False  
  state = dict(session=0)
  
  audio_path, generate = Interface.get_sidebar_input(state)

  if generate and state['session'] == 2:
      with st.spinner("Generating ..."):
          generation = Generation()
          transcribe = generation.transcribe_audio_pytorch(audio_path)

      with st.expander("Transcription Text", expanded=True):
          st.text_area("Transcription:", transcribe, height=300)

      summarization = generation.summarize_string(transcribe)
      with st.expander("Summarization Text", expanded=True):
          st.text_area("Summarization:", summarization, height=300)

if __name__ == "__main__":
    main()
