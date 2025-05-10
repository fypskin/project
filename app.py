from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from skimage.restoration import denoise_wavelet
from skimage.util import img_as_ubyte
import base64
import os
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from speech import UserQueryProcessor
import speech_recognition as sr
from chatbot import Chatbot
from tensorflow.keras.preprocessing.image import load_img, img_to_array


app = Flask(__name__)
tf.keras.backend.clear_session()

MODEL_PATH = "/Users/geetika/Desktop/FYP/client_1.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print(f" Model loaded successfully: {type(model)}")

conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name]
if not conv_layers:
    raise ValueError("No convolutional layers found in the model!")
last_conv_layer_name = conv_layers[-1]
print(f" Last Conv Layer: {last_conv_layer_name}")

CLASS_NAMES = ["Acne and Rosacea Photos", 
"Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions", 
"Atopic Dermatitis Photos", 
"Bullous Disease Photos", 
"Cellulitis Impetigo and other Bacterial Infections", 
"Eczema Photos", 
"Exanthems and Drug Eruptions", 
"Hair Loss Photos Alopecia and other Hair Diseases", 
"Herpes HPV and other STDs Photos", 
"Light Diseases and Disorders of Pigmentation", 
"Lupus and other Connective Tissue diseases", 
"Melanoma Skin Cancer Nevi and Moles", 
"Nail Fungus and other Nail Disease", 
"Poison Ivy Photos and other Contact Dermatitis", 
"Psoriasis pictures Lichen Planus and related diseases", 
"Scabies Lyme Disease and other Infestations and Bites", 
"Seborrheic Keratoses and other Benign Tumors", 
"Systemic Disease", 
"Tinea Ringworm Candidiasis and other Fungal Infections", 
"Urticaria Hives", 
"Vascular Tumors", 
"Vasculitis Photos", 
"Warts Molluscum and other Viral Infections"]


def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode()

def resize(img, sz=(224, 224)):
    return cv2.resize(img, sz, interpolation=cv2.INTER_AREA)

def remove_blackhat(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img, threshold, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

def denoise(img):
    ch = [denoise_wavelet(img[:, :, i], method='BayesShrink', mode='soft', wavelet_levels=1, rescale_sigma=True)
          for i in range(img.shape[2])]
    return img_as_ubyte(np.stack(ch, axis=-1))



def preprocess_image(image_path, img_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = np.expand_dims(img, axis=0) / 255.0
    return img


def compute_gradcam(model, img_array, layer_name):
    grad_model = Model(inputs=model.input, 
                       outputs=[model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])  
        loss = predictions[:, class_idx] 
    
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  
    
    conv_output = conv_output[0].numpy()
    pooled_grads = pooled_grads.numpy()
    
    heatmap = np.mean(conv_output * pooled_grads, axis=-1)


    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap, class_idx

def create_pixelated_heatmap(heatmap, output_size=(224, 224), pixel_size=16):
    small_size = (pixel_size, pixel_size)  
    heatmap_small = cv2.resize(heatmap, small_size, interpolation=cv2.INTER_NEAREST)
  
    heatmap_pixelated = cv2.resize(heatmap_small, output_size, interpolation=cv2.INTER_NEAREST)
    

    heatmap_colored = np.uint8(255 * heatmap_pixelated)
    return cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)


def overlay_heatmap(image_path, heatmap, alpha=0.6):
    img = cv2.imread(image_path)
    
    
    heatmap_smooth = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    heatmap_smooth = np.uint8(255 * heatmap_smooth)
    heatmap_smooth = cv2.applyColorMap(heatmap_smooth, cv2.COLORMAP_JET)
    
    return cv2.addWeighted(img, 1 - alpha, heatmap_smooth, alpha, 0)

def grad_cam(image_path):
    img_array = preprocess_image(image_path)
    heatmap, _ = compute_gradcam(model, img_array, last_conv_layer_name)

    heatmap_pixelated = create_pixelated_heatmap(heatmap, pixel_size=16)
    cv2.imwrite("static/gradcam_heatmap.jpg", heatmap_pixelated)

    result = overlay_heatmap(image_path, heatmap)
    cv2.imwrite("static/gradcam_output.jpg", result)





def generate_lime_explanation(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  

    explainer = lime_image.LimeImageExplainer()
    
    def model_predict_fn(images):
        return model.predict(np.array(images))  
    
    explanation = explainer.explain_instance(img_array, 
                                             model_predict_fn, 
                                             top_labels=1, 
                                             hide_color=0, 
                                             num_samples=1000) 

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                positive_only=True, 
                                                num_features=5, 
                                                hide_rest=False)
    
    lime_output = mark_boundaries(temp, mask)
    lime_output = (lime_output * 255).astype(np.uint8) 
    lime_output = cv2.cvtColor(lime_output, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite("static/lime_output.jpg", lime_output)




@app.route('/')
def index():
    return render_template("index.html")

@app.route('/query')
def query():
    return render_template("query.html")

@app.route('/mobiles')
def mobiles():
    return render_template("mobile.html")

@app.route('/feder')
def feder():
    return render_template("fed.html")

@app.route('/upload',methods=['GET','POST'])
def upload():
    return render_template("upload.html")

@app.route('/model_page')
def model_page():
    return render_template("model.html")



@app.route('/preprocess', methods=['POST'])
def preprocess():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    try:
        file = request.files['image']
        image_pil = Image.open(file).convert("RGB")
        img_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        blackhat_removed = remove_blackhat(img_cv2)
        denoised = denoise(blackhat_removed)
        resized = resize(denoised)

        def encode_image(img):
            _, buffer = cv2.imencode('.jpg', img)
            return base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "blackhat_removed": encode_image(blackhat_removed),
            "denoised": encode_image(denoised),
            "resized": encode_image(resized)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/xai')
def xai():
    image_path = "static/processed_image.jpg"
    grad_cam(image_path)
    generate_lime_explanation(image_path)
    return render_template('xai.html')


@app.route('/predict', methods=['POST'])
def predict():
    global model

    if not hasattr(model, "predict"):
        return jsonify({"error": "Model is not loaded properly!"}), 500

    try:
        processed_image_path = "static/processed_image.jpg"
        if not os.path.exists(processed_image_path):
            return jsonify({"error": "Processed image not found"}), 400

        img = image.load_img(processed_image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)  # Ensure model is valid
        predicted_class = CLASS_NAMES[np.argmax(predictions)]

        return jsonify({"predicted_class": predicted_class})

    except Exception as e:
        print("Exception:", str(e))  
        return jsonify({"error": str(e)}), 500



processor = UserQueryProcessor()
recognizer = sr.Recognizer()

@app.route('/chats')
def chats():
    return render_template("chatbot.html")


@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = processor.process_text(text)

    if isinstance(result, list):  
        result = result[0]  
    return jsonify({'result': result}) 


@app.route('/process_voice', methods=['POST'])
def process_voice():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)  
        recognizer.energy_threshold = 400 
        recognizer.pause_threshold = 1.0  
        
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)  
            text = recognizer.recognize_google(audio)
            
            result = processor.process_text(text)
            
            return jsonify({
                'recognized_text': text,
                'result': {
                    'translated_text': result.get('translated_text', ''),
                    'analysis': result.get('analysis', [])
                }
            })
        except sr.UnknownValueError:
            return jsonify({'error': 'Could not understand the audio'}), 400
        except sr.RequestError:
            return jsonify({'error': 'Speech recognition service unavailable'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500







PDF_FILES = ["/Users/geetika/Desktop/FYP/book.pdf"]
chatbot = Chatbot(PDF_FILES)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    print("Received Data:", data)  

    query = data.get("query", "")
    chat_history = data.get("chat_history", [])

    if not query:
        return jsonify({"error": "Query is required."}), 400

    response = chatbot.get_chatbot_response(query, chat_history)

    if isinstance(response, list) and len(response) == 2:
        book_answer, web_answer = response
    else:
        book_answer = "No book answer available"
        web_answer = "No web answer available"

    print("Generated Response:")
    print(f"Book Answer: {book_answer}")
    print(f"Web Answer: {web_answer}")

    return jsonify({"book_answer": book_answer, "web_answer": web_answer})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
