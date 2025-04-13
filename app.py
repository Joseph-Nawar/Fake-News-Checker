import gradio as gr
import joblib

# Load models
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

def predict_news(text):
    if not text.strip():
        return "⚠️ Please enter some text to analyze!"
    
    # Transform and predict
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    
    return "✅ The News is Real!" if prediction[0] == 1 else "❌ The News is Fake!"

# Gradio interface
interface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(label="Paste News Article", placeholder="Enter text here..."),
    outputs="text",
    title="Fake News Detector",
    description="Enter a news article to check if it's fake or real."
)

interface.launch()