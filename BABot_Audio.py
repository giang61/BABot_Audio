import os
import streamlit as st
import assemblyai as aai
import tempfile
import json
from dotenv import load_dotenv, find_dotenv

# Streamlit page configuration
st.set_page_config(page_title="Audio Transcription", page_icon="🎙️", layout="wide")

# Load environment variables
load_dotenv(find_dotenv('.env'))
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
aai.settings.api_key = ASSEMBLYAI_API_KEY

# File to save speaker mappings for reuse
SPEAKER_MAPPING_FILE = "speaker_mapping.json"

# Cache the transcription function
@st.cache_resource
def transcribe_audio(file_path):
    config = aai.TranscriptionConfig(
        language_code="fr",
        speech_model=aai.SpeechModel.nano,
        speaker_labels=True,
    )
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(file_path)
    return transcript

# Save or load speaker mappings
def load_speaker_mappings():
    if os.path.exists(SPEAKER_MAPPING_FILE):
        with open(SPEAKER_MAPPING_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}

def save_speaker_mappings(mapping):
    with open(SPEAKER_MAPPING_FILE, "w", encoding="utf-8") as file:
        json.dump(mapping, file)

# Generate HTML for chat-like UI
def generate_message_html(speaker, text, speaker_mapping):
    real_name = speaker_mapping.get(speaker, speaker)
    color = hash(real_name) % 360  # Unique color per speaker
    alignment = "left" if hash(real_name) % 2 == 0 else "right"  # Alternate alignment
    return f'''
        <div style="display: flex; justify-content: {alignment}; margin: 10px 0;">
            <div style="
                background-color: hsl({color}, 70%, 85%);
                border: 1px solid hsl({color}, 70%, 60%);
                border-radius: 10px;
                padding: 10px 15px;
                max-width: 60%;
                text-align: left;
                word-wrap: break-word;">
                <b style="color: hsl({color}, 70%, 40%);">{real_name}</b><br>
                {text}
            </div>
        </div>
    '''

# Generate plain text transcription
def generate_plain_text_transcription(transcript, speaker_mapping):
    plain_text = ""
    for utterance in transcript.utterances:
        speaker_name = speaker_mapping.get(utterance.speaker, utterance.speaker)
        plain_text += f"{speaker_name}: {utterance.text}\n"
    return plain_text

# Streamlit app

# Add CSS for centering the titles
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .centered-subtitle {
        text-align: center;
        font-size: 20px;
        color: gray;
        margin-bottom: 20px;
    }
    .centered-element {
        text-align: center;
        font-size: 30px;
        color: gray;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Centered titles
st.markdown('<div class="centered-title">BABot Audio</div>', unsafe_allow_html=True)
st.markdown('<div class="centered-subtitle">Transcrivez vos réunions et visualisez vos conversations.</div>', unsafe_allow_html=True)

# Initialize session state
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "speaker_mapping" not in st.session_state:
    st.session_state.speaker_mapping = {}
if "transcription" not in st.session_state:
    st.session_state.transcription = None

# File uploader for audio
uploaded_audio = st.file_uploader("Sélectionnez un fichier audio", type=None)

if uploaded_audio is not None:
    # Check if a new file has been uploaded
    if uploaded_audio.name != st.session_state.last_uploaded_file:
        st.session_state.last_uploaded_file = uploaded_audio.name
        st.session_state.speaker_mapping = {}  # Reset speaker mapping
        st.session_state.transcription = None  # Reset transcription

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_audio.read())
            temp_file_path = temp_file.name

        # Transcribe the audio file
        st.write("Transcription en cours...")
        st.session_state.transcription = transcribe_audio(temp_file_path)

    # Handle transcription result
    transcript = st.session_state.transcription

    if transcript.status == aai.TranscriptStatus.error:
        st.error(f"Erreur dans la transcription: {transcript.error}")
    else:
        st.session_state.unique_speakers = set()

        # Extract speakers and text
        for utterance in transcript.utterances:
            st.session_state.unique_speakers.add(utterance.speaker)

        # Load or initialize speaker mappings
        speaker_mapping = st.session_state.speaker_mapping

        # User input for updating speaker names
        st.write("### Modifiez les noms des interlocuteurs")
        for speaker in sorted(st.session_state.unique_speakers):
            speaker_mapping[speaker] = st.text_input(f"Nom pour {speaker} :", value=speaker_mapping.get(speaker, speaker))

        # Save speaker mappings
        if st.button("Sauvegarder les noms"):
            save_speaker_mappings(speaker_mapping)
            st.session_state.speaker_mapping = speaker_mapping  # Update session state
            st.success("Les noms des interlocuteurs ont été mis à jour.")

        # Display transcription as a chat
        st.markdown('<div class="centered-element">Transcription</div>', unsafe_allow_html=True)
        chat_html = ""
        for utterance in transcript.utterances:
            chat_html += generate_message_html(utterance.speaker, utterance.text, speaker_mapping)

        # Render the chat UI
        st.markdown(chat_html, unsafe_allow_html=True)

        # Generate plain text transcription
        plain_text_transcription = generate_plain_text_transcription(transcript, speaker_mapping)

        # Download button for the plain text transcription
        st.download_button(
            label="Télécharger la transcription",
            data=plain_text_transcription,
            file_name="transcription.txt",
            mime="text/plain",
        )
