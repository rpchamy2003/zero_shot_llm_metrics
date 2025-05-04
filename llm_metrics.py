'''
Streamlit dashboard now includes:
Token & system metrics
Performance stats (latency, throughput, token usage)
NLI entailment scoring (logical consistency)
Toxicity detection with visual indicators
'''
import time
import psutil
import torch
import streamlit as st
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from detoxify import Detoxify
from transformers import AutoTokenizer as HFTokenizer
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# --- Config ---
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
LLM_ENDPOINT = "http://localhost:8000/v1"
FAKE_KEY = "not-needed"

st.set_page_config(layout="wide")
st.title("🦙 LLaMA 3.3 LangChain Dashboard (vLLM)")

# --- Sidebar Inputs ---
prompt = st.sidebar.text_area("Prompt", "What is the capital of Germany?")
use_json = st.sidebar.checkbox("Use guided JSON output", True)
run_button = st.sidebar.button("Run LLaMA")

# --- Tokenizer ---
tokenizer = HFTokenizer.from_pretrained(MODEL_NAME)
input_ids = tokenizer(prompt)["input_ids"]
token_count = len(input_ids)
max_len = tokenizer.model_max_length

with st.expander("Tokenizer Metrics"):
    st.metric("Input Token Count", token_count)
    st.metric("Max Context Length", max_len)

# --- System Usage ---
with st.expander("System Usage"):
    st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
    st.metric("RAM Usage", f"{psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        st.metric("GPU VRAM Allocated", f"{torch.cuda.memory_allocated() / 1e6:.2f} MB")
        st.metric("GPU VRAM Reserved", f"{torch.cuda.memory_reserved() / 1e6:.2f} MB")

# --- LangChain Model ---
llm = ChatOpenAI(
    openai_api_base=LLM_ENDPOINT,
    openai_api_key=FAKE_KEY,
    model_name=MODEL_NAME,
    temperature=0.0,
)

# --- Optional JSON Output ---
if use_json:
    schema = [
        ResponseSchema(name="answer", description="Answer to the question"),
        ResponseSchema(name="confidence", description="Confidence level: High, Medium, or Low")
    ]
    parser = StructuredOutputParser.from_response_schemas(schema)
    prompt_for_llm = f"{prompt}\n{parser.get_format_instructions()}"
else:
    prompt_for_llm = prompt

# --- Inference ---
if run_button:
    st.subheader("Sending prompt to LLaMA 3.3")
    with st.spinner("Running model..."):
        start = time.time()
        response = llm([HumanMessage(content=prompt_for_llm)])
        duration = time.time() - start
        response_text = response.content

    st.success(f"Response in {duration:.2f}s")

    # Token Metrics
    output_ids = tokenizer(response_text)["input_ids"]
    completion_tokens = len(output_ids)
    total_tokens = token_count + completion_tokens
    tps = completion_tokens / duration if duration > 0 else 0

    with st.expander("Model Performance Metrics"):
        st.metric("Latency", f"{duration:.2f} sec")
        st.metric("Completion Tokens", completion_tokens)
        st.metric("Total Tokens", total_tokens)
        st.metric("Throughput", f"{tps:.2f} tokens/sec")

    st.subheader("Raw Output")
    st.code(response_text, language="markdown")

    if use_json:
        try:
            parsed = parser.parse(response_text)
            st.subheader("Parsed JSON")
            st.json(parsed)
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")

    # --- Generation Quality Manual Inputs ---
    st.subheader("Generation Quality (Manual)")
    col1, col2, col3 = st.columns(3)
    with col1:
        relevance = st.select_slider("Relevance", options=["Poor", "Okay", "Good", "Excellent"], value="Good")
    with col2:
        coherence = st.select_slider("Coherence", options=["Poor", "Okay", "Good", "Excellent"], value="Good")
    with col3:
        confidence = st.select_slider("Confidence", options=["Low", "Medium", "High"], value="High")
    factual = st.checkbox("Factual?")
    toxic = st.checkbox("Toxic content?")

    st.markdown("###Manual Summary")
    st.markdown(f"- **Relevance**: {relevance}")
    st.markdown(f"- **Coherence**: {coherence}")
    st.markdown(f"- **Confidence**: {confidence}")
    st.markdown(f"- **Factual**: {'Yes' if factual else 'No'}")
    st.markdown(f"- **Toxic**: {'Yes' if toxic else 'No'}")

    # --- NLI Consistency ---
    @st.cache_resource
    def load_nli_model():
        model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
        tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        return model.eval(), tokenizer

    def get_nli_score(premise, hypothesis):
        model, tokenizer = load_nli_model()
        inputs = tokenizer.encode(premise, hypothesis, return_tensors="pt", truncation=True)
        logits = model(inputs)[0]
        probs = F.softmax(logits, dim=-1)
        labels = ['entailment', 'neutral', 'contradiction']
        return dict(zip(labels, probs[0].detach().numpy().tolist()))

    st.subheader("NLI Consistency")
    nli = get_nli_score(prompt, response_text)
    nli_label = max(nli, key=nli.get)
    nli_status = "Entailment" if nli_label == "entailment" else "Neutral" if nli_label == "neutral" else "Contradiction"
    st.markdown(f"**Prediction:** {nli_status}")
    st.bar_chart(nli)

    # --- Toxicity Detection ---
    @st.cache_resource
    def load_toxicity_model():
        return Detoxify('original')

    def get_toxicity_scores(text):
        model = load_toxicity_model()
        return model.predict(text)

    st.subheader("Toxicity Detection")
    toxicity = get_toxicity_scores(response_text)
    toxic_flag = any(val > 0.5 for key, val in toxicity.items() if key in ["toxicity", "insult", "identity_attack"])
    toxicity_status = "Toxic content detected!" if toxic_flag else "Clean output"
    st.markdown(f"**Status:** {toxicity_status}")
    top_tox = {k: v for k, v in sorted(toxicity.items(), key=lambda item: -item[1])[:5]}
    st.bar_chart(top_tox)
	
Model Performance Metrics:

Accuracy: Shows how many times the model’s response matches the expected response.

F1 Score: Measures the harmonic mean of precision and recall.

BLEU Score: Measures the overlap of n-grams between the generated and reference response.

ROUGE Score: Measures recall of n-grams in the generated response compared to the reference.

Perplexity: Measures how confident the model is in predicting the next word.

Generation Quality Metrics:

Truthfulness: Evaluates if the response is factually correct using a fact-checking API.

Relevance: Measures how relevant the response is to the complaint using Sentence-BERT for semantic similarity.

Bias Detection: Identifies biased or harmful content in the response using a bias detection model.

Hallucination Rate: Detects fabricated information by comparing the generated response with factual data from a trusted source.

Factual Consistency: Assesses if the response is consistent with factual knowledge, using a consistency-checking API.

Sentiment Analysis:

Evaluates if the model’s tone is appropriate for customer complaint scenarios, such as measuring toxicity.
import time
import streamlit as st
import torch
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, GPT2LMHeadModel
from detoxify import Detoxify
from sentence_transformers import SentenceTransformer
import requests

# --- Define Financial Complaint Dataset ---
complaints = [
    {
        "complaint": "I was charged an overdraft fee despite having sufficient balance. Why was I charged?",
        "expected_response": "We apologize for the inconvenience. We will review your account and reverse the overdraft fee if it was charged in error."
    },
    {
        "complaint": "My credit card was charged for a service I never subscribed to. How do I get a refund?",
        "expected_response": "We are sorry to hear about this issue. Please contact our customer support team, and we will initiate a refund for the unauthorized charge."
    }
]

# --- Load Pre-trained Model ---
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
model = GPT2LMHeadModel.from_pretrained('gpt2')  # Example for perplexity

# --- Sentiment Analysis for Tone ---
def sentiment_analysis(text):
    model = Detoxify('original')
    result = model.predict(text)
    return result

# --- Functions for Evaluation Metrics ---
def calculate_accuracy(predicted, actual):
    return accuracy_score([actual], [predicted])

def calculate_f1(predicted, actual):
    return f1_score([actual], [predicted], average='weighted')

def calculate_bleu(reference, predicted):
    return sentence_bleu([reference], predicted)

def calculate_rouge(reference, predicted):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, predicted)
    return scores

def calculate_perplexity(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()

# --- Truthfulness, Relevance, and Bias Detection ---
def check_truthfulness(response):
    # Use a fact-checking API (e.g., ClaimBuster, FactCheck, or external tools)
    fact_check_api = f"https://factcheckapi.com/verify?text={response}"
    response = requests.get(fact_check_api)
    return response.json()

def check_relevance(response, complaint):
    # Use pre-trained semantic models like Sentence-BERT to check relevance
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    relevance_score = model.similarity(complaint, response)
    return relevance_score

def check_bias(response):
    # Use a bias classifier model or API
    bias_classifier = AutoModelForSequenceClassification.from_pretrained("HuggingFace/bias-detection")
    return bias_classifier(response)

def check_hallucination_rate(response):
    # Compare the response with facts from a trusted knowledge base or database.
    hallucination_api = f"https://api.factcheck.org/hallucinate?text={response}"
    response = requests.get(hallucination_api)
    return response.json()

def check_factual_consistency(response, previous_data):
    # Compare response with the stored data (like KB) or use a model like FactCC.
    consistency_api = f"https://api.factcheck.org/consistency?text={response}"
    response = requests.get(consistency_api)
    return response.json()

# --- Streamlit UI ---
st.title("Customer Complaint Response Evaluation")

complaint_idx = st.selectbox("Select a complaint to evaluate", range(len(complaints)))
selected_complaint = complaints[complaint_idx]

# Show selected complaint
st.subheader("Complaint")
st.write(selected_complaint["complaint"])

# --- Generate Model Response ---
response_text = "We are sorry to hear about the inconvenience. Please contact customer service."  # Replace with actual model output from LLaMA 3.3
st.subheader("Model Response")
st.write(response_text)

# --- Metrics Calculation ---
accuracy = calculate_accuracy(response_text, selected_complaint["expected_response"])
f1 = calculate_f1(response_text, selected_complaint["expected_response"])
bleu = calculate_bleu([selected_complaint["expected_response"]], response_text.split())
rouge = calculate_rouge(selected_complaint["expected_response"], response_text)
perplexity = calculate_perplexity(response_text, model, tokenizer)
sentiment = sentiment_analysis(response_text)

# --- New Metrics for Financial Complaints ---
truthfulness = check_truthfulness(response_text)
relevance = check_relevance(response_text, selected_complaint["complaint"])
bias = check_bias(response_text)
hallucination = check_hallucination_rate(response_text)
factual_consistency = check_factual_consistency(response_text, selected_complaint["complaint"])

# --- Display Model Performance Metrics ---
st.metric("Accuracy", f"{accuracy * 100:.2f}%")
st.metric("F1 Score", f"{f1:.2f}")
st.metric("BLEU Score", f"{bleu:.2f}")
st.json(rouge)
st.metric("Perplexity", f"{perplexity:.2f}")

# --- Display Generation Quality Metrics ---
st.subheader("Truthfulness")
st.write(truthfulness)

st.subheader("Relevance")
st.write(f"Relevance Score: {relevance:.2f}")

st.subheader("Bias Detection")
st.write(bias)

st.subheader("Hallucination Rate")
st.write(hallucination)

st.subheader("Factual Consistency")
st.write(factual_consistency)

# --- Sentiment Analysis Result ---
st.subheader("Sentiment Analysis (Tone)")
st.write(f"Sentiment: {sentiment['toxicity']:.2f} - Toxicity score")
st.write(f"Sentiment: {sentiment['severe_toxicity']:.2f} - Severe Toxicity")
