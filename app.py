import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Cache the model loading function to optimize performance
@st.cache_resource
def load_model_and_tokenizer(model_name, tokenizer_name):
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = M2M100Tokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer

# Function to generate translation
def translate(text, model, tokenizer, source_lang, target_lang):
    tokenizer.src_lang = source_lang
    encoded_text = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

# Streamlit App
st.title("Machine Translation System")

st.sidebar.title("Project Details")
st.sidebar.write("Developed by: Md. Azizul Hakim")
st.sidebar.write("Institution: Bangladesh Sweden Polytechnic Institute")
st.sidebar.write("Department: CST")
st.sidebar.write("Semester: 5th")
st.sidebar.write("Project repository: [GitHub](https://github.com/HAKIM-ML)")
st.sidebar.write("---")
st.sidebar.write("### About the Project")
st.sidebar.write("This project is a machine translation system that translates text between Bengali and English. The system uses a deep learning model trained on parallel corpora of Bengali and English sentences, allowing it to understand and translate between the two languages.")
st.sidebar.write("The user interface is built using the Streamlit framework, which provides an interactive platform for users to input text in one language and receive the translated text in the other language. The application supports both Bengali to English and English to Bengali translations.")
st.sidebar.write("The project showcases the application of deep learning in natural language processing tasks, particularly in machine translation. It can be extended or customized to include more languages or to improve the translation accuracy based on additional data and training.")


translation_direction = st.selectbox("Select Translation Direction:", options=["Bangla to English", "English to Bangla"])

if translation_direction == "Bangla to English":
    model_name = "finetuned_model_bn"
    tokenizer_name = 'finetuned_tokenizer_bn'
    source_lang = "bn"
    target_lang = "en"
elif translation_direction == "English to Bangla":
    model_name = "finetuned_model_en"
    tokenizer_name = 'finetuned_tokenizer_en'
    source_lang = "en"
    target_lang = "bn"

# Load the appropriate model and tokenizer
model, tokenizer = load_model_and_tokenizer(model_name, tokenizer_name)

input_text = st.text_input("Enter Text in Source Language:")

if st.button("Translate"):
    with st.spinner(f"Translating to {'English' if target_lang == 'en' else 'Bangla'}..."):
        translated_text = translate(input_text, model, tokenizer, source_lang, target_lang)
        st.write(f"Translated Text: {translated_text}")
