import argparse
import math
import os
import csv
import warnings
import re
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, stop_after_attempt, wait_exponential

# --- RAG Pipeline Settings ---
CHROMA_PATH = "chroma_db"
LLM_MODEL = ""
TOP_K = 3
SIMILARITY_THRESHOLD = 0.5

# --- Embedding Model Settings ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# --- LLM Generation Settings ---
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_MAX_TOKENS = 512

# --- BERTScore Settings ---
BERTSCORE_MODEL = "roberta-large"  
BERTSCORE_USE_IDF = False
BERTSCORE_RESCALE_BASELINE = False  

# --- Query Validation ---
MAX_QUERY_LENGTH = 1000
MIN_QUERY_LENGTH = 3

# --- Prompt Template ---
PROMPT_TEMPLATE = """
You are an agricultural expert. Answer the question using ONLY the provided context below.

CRITICAL REQUIREMENT: You MUST cite every fact by including [Source ID: X] immediately after each statement, where X is the source number (1, 2, or 3).

Context:
{context}

---

Question: {question}

Instructions:
- Provide a complete, accurate answer
- After EVERY factual claim, add [Source ID: X] where X is 1, 2, or 3
- Use multiple sources if they support different parts of your answer
- Do not make claims without citing a source

Answer with citations:
"""


GROUND_TRUTH = {
    # Maize
    "what are the main planting seasons for maize?": "As a rainfed crop, maize is grown in June-July or August-September. The irrigated crop is raised in January-February.",
    "what are some important hybrid and composite varieties of maize?": "Hybrids: Ganga Hybrid-1, Ganga Hybrid- 101, Deccan hybrid, Renjit, Hi-Starch. Composite varieties: Kissan Composite, Amber, Vijay, Vikram, Sona, Jawahar.",
    "what are the optimal environmental and soil conditions required for maize cultivation?": "Maize can be grown throughout the year at altitude ranging from sea level to about 300 m. Maize grows best in areas with rainfall of 600- 900 mm. It requires fertile, well-drained soil with a pH ranging from 5.5-8.0, but pH 6.0- 7.0 is optimum.",
    "how frequently should maize be irrigated during the initial growth period?": "Irrigate the crop on the day of sowing and on third day. Subsequent irrigations may be given at 10-15 days intervals.",
    "what is the recommended timing for weeding operations in maize crops?": "Hand hoeing and weeding on the 21st and 45th day after sowing.",
    "what is the recommended fertilizer application schedule and dosage for maize cultivation?": "Apply FYM/compost @ 25 t ha-1 at the time of preparation of land. The recommended fertilizer dose is 135 kg nitrogen, 65 kg phosphorus and 15 kg potash per ha. Apply full dose of phosphorus and potash and 1/3 dose of nitrogen as basal. Apply 1/3 nitrogen, 30-40 days and the rest 60-70 days after sowing.",
    # Sweet Potato
    "what are the main planting seasons for sweet potato?": "Sweet potatoes can be grown as rainfed crops during June-July and September-October, while irrigated crops are typically planted in October-November for uplands and January-February for lowlands.",
    "what are the varieties of sweet potato?": "Improved sweet potato varieties include H-41 and H-42 (both with excellent cooking quality and 120-day duration), early maturing varieties like Sree Nandini and Sree Vardhini (100-105 days), carotene-rich varieties such as Sree Rethna (90-105 days) and Sree Kanaka (75-85 days with very high carotene content), short-duration varieties like Sree Bhadra and Sree Arun (90 days), highly palatable Sree Varun, the KAU selection Kanjanghad (105-120 days), and local varieties including Badrakali Chuvala, Kottayam Chuvala, Chinavella, Chakaravalli, and Anakomban.",
    "how frequently should sweet potato be irrigated during the initial growth period?": "For irrigated sweet potato crops, provide irrigation every 2 days for the first 10 days after planting, then reduce to once every 7-10 days, stop irrigation 3 weeks before harvest (with an optional final irrigation 2 days before harvest), maintain an IW/CPE ratio of 1.2 (approximately 11-day intervals) for higher tuber yield during non-rainy periods, and apply 50 kg/ha each of nitrogen and potassium oxide (K2O) as fertilizer.",
    "what is the recommended timing for weeding operations in sweet potato crops?": "Conduct two weeding and earthing up operations about 2 weeks and 5 weeks after planting.",
    "what is the recommended manuring and fertilizer application schedule and dosage for sweet potato cultivation?": "For sweet potato manuring, apply 10 tonnes per hectare of cattle manure or compost during ridge preparation, use the recommended NPK dosage of 75:50:75 kg/ha (or 50:25:50 kg/ha for reclaimed alluvial soils of Kuttanad), apply nitrogen in two equal split doses at planting and 4-5 weeks after planting, and apply the full doses of phosphorus (P2O5) and potassium (K2O) at planting time.",
    # Ragi
    "what are the different growing seasons for ragi cultivation?": "Ragi is not a season-bound crop and can be cultivated throughout the year if moisture is available, with typical growing seasons including the main season from June to September, late season from July to October, and summer season from December-January to March-April.",
    "what are the main varieties of ragi?": "PR-202, K-2, Co-2, Co-7,Co-8, Co-9, Co-10.",
    "what are the optimal environmental and soil conditions required for ragi cultivation?": "Ragi is suited for cultivation in areas with annual rainfall of 700-1200 mm. It does not tolerate heavy rainfall and requires a dry spell at the time of grain ripening. It grows well in altitudes of 1000-2000 m with average temperature of 27°C. Ragi is cultivated mostly in red lateritic soils. Relatively fertile and well drained soils are the most suitable.",
    "what is the recommended irrigation schedule for ragi after transplantation?": "Irrigate the field on the day of transplantation. Irrigation at weekly intervals increases growth rate and yield.",
    "what is the recommended timing for weeding operations in ragi crops?": "Weeding should be done three weeks after sowing and completed before top dressing.",
    "what is the recommended fertilizer application schedule and dosage for ragi cultivation?": "Plough the field 3-4 times and incorporate FYM orcompost 5 tha-1. Apply nitrogen, phosphorus and potash @ 22.5 kg ha-1 each before sowing or planting. Topdress nitrogen @ 22.5 kg ha-1 21 days after sowing or planting.",
    # Cotton
    "what are the different growing seasons for cotton cultivation?": "Cotton can be grown as a winter crop planted in August-September or as a summer crop planted in February-March.",
    "what are the main varieties of cotton?": "Cotton varieties include MCU 5/MCU 5 VT, hybrid varieties TCHB 213 and Savita, and LRA 5166.",
    "what are the optimal environmental and soil conditions required for cotton cultivation?": "Cotton is grown from sea level to moderate elevations not exceeding 1000 meters in tropical climates with 500-750 mm rainfall, though excessive rain at any stage is harmful to the crop, and while it can be cultivated in a wide variety of soils, deep, homogeneous, and fertile soil is most desirable for optimal growth.",
    "how frequently should cotton be irrigated during the initial growth period?": "For irrigated cotton crops, irrigate the plants once every two weeks, with copious irrigation during flowering being essential to ensure good pod setting and high fiber quality.",
    "what is the recommended manuring and fertilizer application schedule and dosage for cotton cultivation?": "For cotton manuring, apply farmyard manure (FYM) or compost at 12.5 tonnes per hectare for rainfed crops and 25 tonnes per hectare for irrigated crops, apply nitrogen, phosphorus (P2O5), and potassium (K2O) each at 35 kg per hectare as basal dressing, and top dress with an additional 35 kg nitrogen per hectare about 45 days after sowing.",
    # Groundnut
    "what are the different growing seasons for groundnut cultivation?": "Groundnut can be grown as a rainfed crop from May-June to September-October or as an irrigated crop from January to May.",
    "what are some important varieties of groundnut?": "Groundnut varieties include bunch types TMV-2, TMV-7, TG-3, TG-14, Sneha, and Snigdha, as well as Spanish Improved.",
    "what are the sowing specifications for groundnut?": "For groundnut sowing, plough the field three to four times to achieve a fine tilth, sow the seeds by dibbling in ploughed furrows at 15 cm x 15 cm spacing, and treat the seeds with rhizobial culture before sowing.",
    "how frequently should groundnut be irrigated during the initial growth period?": "For groundnut cultivation, irrigate the crop once every 7 days, weed the crop 10-15 days after seed germination through light hoeing, perform another light hoeing or raking when applying lime, and avoid disturbing the soil after 45 days of sowing.",
    "what is the recommended manuring and fertilizer application schedule and dosage for groundnut cultivation?": "For groundnut manuring, apply 2 tonnes per hectare of cattle manure or compost and fertilizers at 10:75:75 kg/ha (N:P2O5:K2O) as basal dressing and incorporate well into the soil, then apply 1-1.5 tonnes per hectare of lime at flowering time and mix with the soil through light hoeing or raking."
}

def get_embedding_model() -> SentenceTransformer:
    """Lazy load the embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def print_hyperparameters():
    """Prints the key hyperparameters to the console."""
    print("=" * 50)
    print("RAG System Hyperparameters")
    print("-" * 50)
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Chroma DB Path: {CHROMA_PATH}")
    print(f"Retrieval Top-K: {TOP_K}")
    print(f"GT Match Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print("=" * 50 + "\n")


def validate_query(query: str) -> str:
    """Validates and sanitizes user query."""
    if not query or len(query.strip()) < MIN_QUERY_LENGTH:
        raise ValueError(f"Query must be at least {MIN_QUERY_LENGTH} characters long")
    
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query too long (max {MAX_QUERY_LENGTH} characters)")
    
    return query.strip()


@contextmanager
def get_chroma_db():
    """Context manager for Chroma DB connection."""
    db = None
    try:
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        yield db
    except Exception as e:
        print(f"Error connecting to Chroma DB: {e}")
        raise
    finally:
        
        pass


def semantic_match_query(query: str, gt_lookup: Dict[str, str]) -> Optional[str]:
    """Finds the best ground truth query match using semantic similarity."""
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    
    best_match = None
    best_score = 0.0
    
    for gt_query in gt_lookup:
        gt_embedding = embedding_model.encode(gt_query, convert_to_tensor=True)
        score = util.cos_sim(query_embedding, gt_embedding).item()
        
        if score > best_score:
            best_score = score
            best_match = gt_query
    
    return best_match if best_score > SIMILARITY_THRESHOLD else None


def detect_citations(response: str, source_ids: List[str]) -> List[str]:
    """
    Robustly detect cited sources in response using regex.
    
    """
    cited = []
    response_lower = response.lower()
    
    for sid in source_ids:
        sid_str = str(sid)
        
        # Pattern 1: "Source ID: X" or "Source: X"
        pattern1 = rf'\bsource\s*(?:id\s*)?:?\s*{re.escape(sid_str)}\b'
        
        # Pattern 2: "[X]" or "(X)"
        pattern2 = rf'[\[\(]\s*{re.escape(sid_str)}\s*[\]\)]'
        
        
        if (re.search(pattern1, response_lower, re.IGNORECASE) or 
            re.search(pattern2, response_lower, re.IGNORECASE)):
            cited.append(sid)
    
    return cited


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts using embeddings."""
    embedding_model = get_embedding_model()
    emb1 = embedding_model.encode(text1, convert_to_tensor=True)
    emb2 = embedding_model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


def export_metrics_to_csv(metrics: Dict, filename: str = "metrics_log.csv"):
    """Appends a dictionary of metrics to a CSV file with error handling."""
    try:
        keys = list(metrics.keys())
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
        
        print(f"Metrics exported to {filename}")
    except Exception as e:
        print(f"Failed to export metrics to CSV: {e}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def invoke_llm_with_retry(model: OllamaLLM, prompt: str) -> str:
    """Invoke LLM with retry logic for robustness."""
    return model.invoke(prompt)


def calculate_retrieval_metrics(
    retrieved_texts: List[str], 
    references: List[str],
    top_k: int
) -> Dict[str, float]:
    """
    Calculate comprehensive retrieval metrics.
    Uses NDCG formula: DCG = Σ(rel_i / log2(i + 2)) for i in [0, k-1]
    """
    embedding_model = get_embedding_model()
    
    # Get embeddings
    ref_embedding = embedding_model.encode(references[0], convert_to_tensor=True)
    doc_embeddings = embedding_model.encode(retrieved_texts, convert_to_tensor=True)
    similarities = util.cos_sim(ref_embedding, doc_embeddings)[0].cpu().numpy()
    
    # Binary relevance labels: 1 if similarity >= threshold, else 0
    hits = [1 if sim >= SIMILARITY_THRESHOLD else 0 for sim in similarities]
    
    # Precision@K: fraction of retrieved docs that are relevant
    precision_at_k = np.mean(hits) if hits else 0.0
    
    # Coverage@K: relevant docs that were retrieved
    coverage_at_k = sum(hits)
    
    # Mean Reciprocal Rank (MRR): 1 / rank of first relevant doc
    if 1 in hits:
        first_relevant_rank = hits.index(1) + 1  # 1-indexed
        mrr = 1.0 / first_relevant_rank
    else:
        mrr = 0.0
    
    # Normalized Discounted Cumulative Gain (NDCG)
    # DCG: rewards relevant docs, with higher weight for top positions
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(hits))
    
    # IDCG: ideal ordering (all relevant docs at top)
    ideal_hits = sorted(hits, reverse=True)
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_hits))
    
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    # Clamp to [0, 1] to prevent floating-point errors
    ndcg = min(max(ndcg, 0.0), 1.0)
    mrr = min(max(mrr, 0.0), 1.0)
    
    return {
        "precision_at_k": precision_at_k,
        "coverage_at_k": coverage_at_k,
        "mrr": mrr,
        "ndcg": ndcg,
        "similarities": similarities.tolist(),
        "hits": hits
    }


def calculate_generation_metrics(
    response: str, 
    references: List[str]
) -> Dict[str, float]:
    """Calculate generation quality metrics: BLEU, ROUGE, BERTScore."""
    
    # BLEU Score with smoothing
    smoothing = SmoothingFunction().method4
    bleu = sentence_bleu(
        [references[0].split()], 
        response.split(), 
        smoothing_function=smoothing
    )
    
    # ROUGE Scores
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(references[0], response)
    rouge1 = rouge_scores['rouge1'].fmeasure
    rouge2 = rouge_scores['rouge2'].fmeasure
    rougel = rouge_scores['rougeL'].fmeasure
    
    # BERTScore
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P, R, F1 = bert_score(
            [response], 
            references, 
            lang="en", 
            rescale_with_baseline=False, 
            verbose=False
        )
    
    bert_p = P.mean().item()
    bert_r = R.mean().item()
    bert_f1 = F1.mean().item()
    
    # Semantic Similarity (additional metric)
    semantic_sim = calculate_semantic_similarity(response, references[0])
    
    return {
        "bleu": bleu,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougel": rougel,
        "bert_precision": bert_p,
        "bert_recall": bert_r,
        "bert_f1": bert_f1,
        "semantic_similarity": semantic_sim
    }


def calculate_faithfulness_metrics(
    response: str,
    source_ids: List[str]
) -> Dict[str, float]:
    """Calculate attribution metrics."""
    
    # Attribution Score: fraction of sources that were cited
    cited_sources = detect_citations(response, source_ids)
    attribution_score = len(set(cited_sources)) / len(source_ids) if source_ids else 0.0
    
    return {
        "cited_sources": cited_sources,
        "attribution_score": attribution_score
    }


def query_rag(query_text: str):
    """
    Executes the RAG pipeline: retrieves documents, generates an answer,
    evaluates it against ground truth, and logs the metrics.
    """
    
    # Validate query
    try:
        query_text = validate_query(query_text)
    except ValueError as e:
        print(f"Invalid query: {e}")
        return
    
    print(f"Query: {query_text}\n")
    
    # Track timing
    start_time = time.time()
    
    # --- 1. Retrieval Phase ---
    retrieval_start = time.time()
    
    try:
        with get_chroma_db() as db:
            results = db.similarity_search_with_score(query_text, k=TOP_K)
    except Exception as e:
        print(f"Retrieval failed: {e}")
        return
    
    retrieval_time = time.time() - retrieval_start
    
    retrieved_docs = [doc for doc, _ in results]
    retrieved_texts = [doc.page_content for doc in retrieved_docs]
    
    # Keep original IDs for logging/debugging
    original_ids = [doc.metadata.get("id", "Unknown") for doc in retrieved_docs]
    simple_ids = [str(i+1) for i in range(len(retrieved_docs))]  # ["1", "2", "3"]
    
    # Create a mapping for later verification
    id_mapping = dict(zip(simple_ids, original_ids))
    
    print(f"Retrieved {len(simple_ids)} documents in {retrieval_time:.2f}s")
    print(f"Source Mapping:")
    for simple_id, orig_id in id_mapping.items():
        print(f"   [{simple_id}] -> {orig_id}")
    print()
    
    # --- 2. Generation Phase ---
    generation_start = time.time()
    
    # Use simple IDs (1, 2, 3) in context for LLM
    context_text = "\n\n---\n\n".join([
        f"Source ID: {simple_id}\nContent: {doc.page_content}" 
        for simple_id, doc in zip(simple_ids, retrieved_docs)
    ])
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, 
        question=query_text
    )
    
    try:
        model = OllamaLLM(model=LLM_MODEL)
        response_text = invoke_llm_with_retry(model, prompt)
    except Exception as e:
        print(f"Generation failed: {e}")
        return
    
    generation_time = time.time() - generation_start
    
    print(f"Generated Answer ({generation_time:.2f}s):")
    print(f"{response_text}\n")
    
    # Answer length metrics
    response_word_count = len(response_text.split())
    response_char_count = len(response_text)
    
    # --- 3. Evaluation Phase ---
    gt_lookup = {k.lower(): v for k, v in GROUND_TRUTH.items()}
    processed_query = query_text.lower()
    matched_query = None

    # First, check for a direct, case-insensitive match
    if processed_query in gt_lookup:
       print("Found an exact match in ground truth.")
       matched_query = processed_query
    else:
       # If no exact match, fall back to semantic search
       print("No exact match found. Performing semantic search for ground truth...")
       matched_query = semantic_match_query(processed_query, gt_lookup)
    
    if not matched_query:
        print("No matched ground truth found for evaluation.")
        print(f"Total time: {time.time() - start_time:.2f}s")
        return
    
    references = [gt_lookup[matched_query]]
    print(f"Matched Ground Truth: {matched_query}")
    print(f"Reference: {references[0][:100]}...\n")
    
    # --- 3a. Retrieval Metrics ---
    print("--- Retrieval Evaluation ---")
    retrieval_metrics = calculate_retrieval_metrics(retrieved_texts, references, TOP_K)
    
    print(f"Precision@{TOP_K}: {retrieval_metrics['precision_at_k']:.4f}")
    print(f"Coverage@{TOP_K}: {retrieval_metrics['coverage_at_k']:.4f}")
    print(f"MRR: {retrieval_metrics['mrr']:.4f}")
    print(f"NDCG: {retrieval_metrics['ndcg']:.4f}")
    print(f"Similarity Scores: {[f'{s:.3f}' for s in retrieval_metrics['similarities']]}\n")
    
    # --- 3b. Generation Metrics ---
    print("--- Generation Quality ---")
    generation_metrics = calculate_generation_metrics(response_text, references)
    
    print(f"BLEU: {generation_metrics['bleu']:.4f}")
    print(f"ROUGE-1: {generation_metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {generation_metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {generation_metrics['rougel']:.4f}")
    print(f"BERTScore: P={generation_metrics['bert_precision']:.4f}, "
          f"R={generation_metrics['bert_recall']:.4f}, "
          f"F1={generation_metrics['bert_f1']:.4f}")
    print(f"Semantic Similarity: {generation_metrics['semantic_similarity']:.4f}")
    print(f"Response Length: {response_word_count} words, {response_char_count} chars\n")
    
    # --- 3c. Attribution ---
    print("--- Attribution ---")
    # Use simple_ids for citation detection (what LLM was given)
    faithfulness_metrics = calculate_faithfulness_metrics(response_text, simple_ids)
    
    cited = faithfulness_metrics['cited_sources']
    
    # Map back to original IDs for display
    cited_original = [id_mapping.get(cid, cid) for cid in cited]
    
    print(f"Cited Sources (Simple IDs): {cited if cited else 'None'}")
    print(f"Cited Sources (Original): {cited_original if cited_original else 'None'}")
    print(f"Attribution Score: {faithfulness_metrics['attribution_score']:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTotal Pipeline Time: {total_time:.2f}s")
    print(f"Retrieval: {retrieval_time:.2f}s | Generation: {generation_time:.2f}s")
    
    # --- 4. Export Metrics ---
    metrics_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "llm_model": LLM_MODEL,
        "query": query_text,
        "matched_gt": matched_query,
        "cited_sources": ", ".join(cited) if cited else "None",
        "cited_sources_original": ", ".join(cited_original) if cited_original else "None",
        "response_words": response_word_count,
        "response_chars": response_char_count,
        "retrieval_time": f"{retrieval_time:.2f}",
        "generation_time": f"{generation_time:.2f}",
        "total_time": f"{total_time:.2f}",
        "bleu": f"{generation_metrics['bleu']:.4f}",
        "rouge1": f"{generation_metrics['rouge1']:.4f}",
        "rouge2": f"{generation_metrics['rouge2']:.4f}",
        "rougel": f"{generation_metrics['rougel']:.4f}",
        "bert_p": f"{generation_metrics['bert_precision']:.4f}",
        "bert_r": f"{generation_metrics['bert_recall']:.4f}",
        "bert_f1": f"{generation_metrics['bert_f1']:.4f}",
        "semantic_sim": f"{generation_metrics['semantic_similarity']:.4f}",
        "precision_at_k": f"{retrieval_metrics['precision_at_k']:.4f}",
        "coverage_at_k": f"{retrieval_metrics['coverage_at_k']:.4f}",
        "mrr": f"{retrieval_metrics['mrr']:.4f}",
        "ndcg": f"{retrieval_metrics['ndcg']:.4f}",
        "attribution": f"{faithfulness_metrics['attribution_score']:.4f}"
    }
    
    export_metrics_to_csv(metrics_dict)


def main():
    """Parses command-line arguments and runs the RAG query."""
    parser = argparse.ArgumentParser(
        description="Query a RAG model and evaluate the response.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_data.py "what are the main varieties of ragi?"
  python query_data.py "how to irrigate maize crops?"
        """
    )
    parser.add_argument("query_text", type=str, help="The user query to process.")
    args = parser.parse_args()
    
    print_hyperparameters()
    
    try:
        query_rag(args.query_text)
    except KeyboardInterrupt:
        print("\n\nQuery interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()