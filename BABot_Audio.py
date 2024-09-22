import os
import streamlit as st
import assemblyai as aai
import tempfile
import openai
from openai import Client

from dotenv import load_dotenv, find_dotenv

# Streamlit page configuration
st.set_page_config(page_title="Audio Transcription", page_icon="🎙️", layout="wide")

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# AssemblyAI API Key setup
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')  # Or directly set your API key here
aai.settings.api_key = ASSEMBLYAI_API_KEY

openai.api_key=OPENAI_API_KEY
client=Client()

# Function to transcribe audio file using AssemblyAI
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
    # summary = response['choices'][0]['message']['content'].strip()
    summary = response.choices[0].message.content
    return summary

# Streamlit File Uploader for audio file
st.title("BABot_Audio")

st.write("### Transcrire et Résumer vos réunions")
uploaded_audio = st.file_uploader("Sélectionner un fichier audio (mp3, wav)", type=["mp3", "wav"])

if uploaded_audio is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(uploaded_audio.read())
        temp_file_path = temp_file.name

    # Transcribe the audio file using AssemblyAI
    st.write("Transcription du fichier audio ...")

    # Run transcription and display results
    transcript = transcribe_audio(temp_file_path)

    if transcript.status == aai.TranscriptStatus.error:
        st.error(f"Erreur dans la transcription: {transcript.error}")
    else:
        st.write("### Résultats de la Transcription")
        transcription_text = ""
        for utterance in transcript.utterances:
            st.write(f"**Interlocuteur {utterance.speaker}:** {utterance.text}")
            transcription_text += f"**Interlocuteur {utterance.speaker}: **{utterance.text}\n "

        # Option pour sauvegarder la transcription dans un fichier
        with open("./data/Audio/transcription.txt", "w", encoding="utf-8") as file:
            file.write(transcription_text)

        st.success("Transcription sauvegardée dans 'transcription.txt'.")

        # Résumer la transcription avec ChatGPT (OpenAI)
        st.write("Résumé de la transcription en cours...")
        summary = summarize_transcription(transcription_text)

        st.write("### Résumé du texte transcrit:")
        st.write(summary)
else:
    st.info("Téléchargez votre fichier audio pour démarrer la transcription.")
