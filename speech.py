from transformers import pipeline
import speech_recognition as sr
from googletrans import Translator

class UserQueryProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.translator = Translator()
        self.text_processor = None  

    def load_text_processor(self):
        if self.text_processor is None:
            self.text_processor = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    def process_text(self, text):
        print(f"\nReceived text: {text}")

        try:
            detected_lang = self.translator.detect(text).lang
            translated_text = text  

            if detected_lang != 'en':
                print(f"Translating from {detected_lang} to English...")
                translated_text = self.translator.translate(text, dest='en').text
                print(f"Translated Text: {translated_text}")

            self.load_text_processor() 
            analysis = self.text_processor(translated_text)
            print("Analysis:",analysis)

            return {
                'original_text': text,
                'detected_language': detected_lang,
                'translated_text': translated_text if detected_lang != 'en' else "No translation needed",
                'analysis': analysis
            }

        except Exception as e:
            print(f"Translation Error: {e}")
            return {'error': 'Translation failed'}


    def process_voice(self):
        with sr.Microphone() as source:
            print("\nListening for voice input...")
            self.recognizer.adjust_for_ambient_noise(source)

            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                print(f"\nRecognized: {text}")
                return self.process_text(text)

            except sr.UnknownValueError:
                print("Could not understand the audio.")
            except sr.RequestError:
                print("Could not request results, check your internet connection.")
            except sr.WaitTimeoutError:
                print("Listening timed out. Try again.")
            
            return None

if __name__ == "__main__":
    processor = UserQueryProcessor()

    while True:
        choice = input("\nEnter 'T' for text input, 'V' for voice input, or 'exit' to quit: ").strip().lower()

        if choice == 'exit':
            print("Exiting...")
            break
        elif choice == 't':
            user_text = input("Enter your text: ")
            result = processor.process_text(user_text)
            if result:
                print("\nProcessed Text Output:", result)
        elif choice == 'v':
            result = processor.process_voice()
            if result:
                print("\nProcessed Voice Output:", result)
        else:
            print("Invalid choice! Please enter 'T' or 'V'.")