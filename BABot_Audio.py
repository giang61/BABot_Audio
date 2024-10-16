import os
import streamlit as st
import assemblyai as aai
import tempfile
import openai
from openai import Client
from dotenv import load_dotenv, find_dotenv

# Streamlit page configuration
st.set_page_config(page_title="Audio Transcription", page_icon="🎙️", layout="centered")

load_dotenv(find_dotenv('.env'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# AssemblyAI API Key setup
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')  # Or directly set your API key here
aai.settings.api_key = ASSEMBLYAI_API_KEY

# Cache the OpenAI client initialization
@st.cache_resource
def init_openai_client(api_key):
    openai.api_key = api_key
    return openai.Client()

client = init_openai_client(OPENAI_API_KEY)

# Function to transcribe audio file using AssemblyAI
# Cache the AssemblyAI transcription process
@st.cache_resource
def transcribe_audio(file_path):
    config = aai.TranscriptionConfig(
        language_code="fr",  # Specify the language code
        speech_model=aai.SpeechModel.nano,  # Select the model (nano is fast but less accurate)
        speaker_labels=True,  # Enable speaker labels
    )

    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(file_path)

    return transcript

# Fonction pour résumer la transcription avec ChatOpenAI
def summarize_transcription(text):
    # Make a request to the OpenAI API using the chat model
    response = client.chat.completions.create(
        model="gpt-4o",  # Specify the model
        messages=[
            {"role": "system", "content": "Tu es un assistant qui résumes des textes"},
            {"role": "user", "content": f"Résumer ce texte: {text}"}
        ],
        temperature=0.5,  # Limit for the summary length
        max_tokens=300,  # Adjust for creativity
    )

    # Extract and return the summary from the response
    summary = response.choices[0].message.content
    return summary

# Streamlit File Uploader for audio file
st.title("BABot_Audio")

st.write("### Transcrire et Résumer vos réunions")
uploaded_audio = st.file_uploader("Sélectionner un fichier audio", type=None)  # Allow all audio file types

if uploaded_audio is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_audio.read())
        temp_file_path = temp_file.name

    # Transcribe the audio file using AssemblyAI
    st.write("Transcription du fichier audio ...")

    # Run transcription and display results
    transcript = transcribe_audio(temp_file_path)

    if transcript.status == aai.TranscriptStatus.error:
        st.error(f"Erreur dans la transcription: {transcript.error}")
    else:
        # Store the transcription in session state to avoid re-running the transcription
        st.session_state.transcription = transcript
        st.session_state.transcription_text = ""
        st.session_state.unique_speakers = set()

        # Iterate over each utterance and build the transcription text
        for utterance in transcript.utterances:
            st.session_state.transcription_text += f'{utterance.speaker}: {utterance.text}\n\n'
            st.session_state.unique_speakers.add(utterance.speaker)  # Collect all unique speakers

        # Step 2: allow users to map speaker labels to real names
    st.write("### Résultats de la Transcription")

    # Display the transcription
    for utterance in st.session_state.transcription.utterances:
        st.write(f"**{utterance.speaker}:** {utterance.text}")

    st.write("### Qui sont les interlocuteurs?")

    # Display detected speakers
    st.write("**Interlocuteurs détectés**")
    st.write(", ".join(st.session_state.unique_speakers))  # Show all detected speakers

    # Collect real names in a single text area, one per line or comma-separated
    speaker_names_input = st.text_area(
        "Entrez les noms des interlocuteurs dans l'ordre (un par ligne ou séparés par une virgule)", "")

    # Step 3: Process the input names and update speaker names
    if st.button("Mettre à jour les noms"):
        input_names = [name.strip() for name in speaker_names_input.replace(",", "\n").splitlines()]

        # Ensure that the number of input names matches the number of unique speakers
        if len(input_names) != len(st.session_state.unique_speakers):
            st.error(
                f"Vous avez spécifié {len(input_names)} noms, mais {len(st.session_state.unique_speakers)} interlocuteurs ont été détectés. Veuillez réessayer.")
        else:
            # Step 3 Optimization: Initialize session_state variables if they don't exist
            if 'updated_transcription_text' not in st.session_state:
                st.session_state.updated_transcription_text = ""
            if 'speaker_mapping' not in st.session_state:
                st.session_state.speaker_mapping = {}

            # Only update speaker names if they haven't already been mapped or if the mapping has changed
            if st.session_state.speaker_mapping != dict(zip(st.session_state.unique_speakers, input_names)):

                # Map each detected speaker to the corresponding real name
                st.session_state.speaker_mapping = dict(zip(st.session_state.unique_speakers, input_names))

                # Replace speaker labels with real names in the transcription
                st.session_state.updated_transcription_text = ""

                for utterance in st.session_state.transcription.utterances:
                    real_name = st.session_state.speaker_mapping.get(utterance.speaker, utterance.speaker)
                    st.write(f"**{real_name}:** {utterance.text}")
                    st.session_state.updated_transcription_text += f'{real_name}: {utterance.text}\n\n'

            # Save the updated transcription to a file
            temp_dir = './temp'
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, 'transcription.txt')
            with open(temp_file_path, "w", encoding="utf-8") as file:
                file.write(st.session_state.updated_transcription_text)

            st.success("Transcription sauvegardée dans 'transcription.txt'. Vous pouvez le télécharger.")

            # Add a download button
            with open(temp_file_path, "rb") as file:
                st.download_button(
                    label="Télécharger",
                    data=file,
                    file_name="transcription.txt",  # The name of the file when downloaded
                    mime="text/plain; charset=utf-8"  # MIME type for plain text files
                )

else:
    st.info("Téléchargez votre fichier audio pour démarrer la transcription.")