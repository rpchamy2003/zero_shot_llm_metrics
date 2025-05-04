'''
Streamlit dashboard now includes:
Token & system metrics
Performance stats (latency, throughput, token usage)
NLI entailment scoring (logical consistency)
Toxicity detection with visual indicators
'''
'''Model Performance Metrics:
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
'''
