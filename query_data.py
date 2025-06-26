import argparse
import math
import os
import csv
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util

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
    "what are some important varieties of groundnut?": "Groundnut varieties include bunch types TMV-2, TMV-7, TG-3, TG-14, Sneha, and Snigtha, as well as Spanish Improved.",
    "what are the sowing specifications for groundnut?": "For groundnut sowing, plough the field three to four times to achieve a fine tilth, sow the seeds by dibbling in ploughed furrows at 15 cm x 15 cm spacing, and treat the seeds with rhizobial culture before sowing.",
    "how frequently should groundnut be irrigated during the initial growth period?": "For groundnut cultivation, irrigate the crop once every 7 days, weed the crop 10-15 days after seed germination through light hoeing, perform another light hoeing or raking when applying lime, and avoid disturbing the soil after 45 days of sowing.",
    "what is the recommended manuring and fertilizer application schedule and dosage for groundnut cultivation?": "For groundnut manuring, apply 2 tonnes per hectare of cattle manure or compost and fertilizers at 10:75:75 kg/ha (N:P2O5:K2O) as basal dressing and incorporate well into the soil, then apply 1-1.5 tonnes per hectare of lime at flowering time and mix with the soil through light hoeing or raking."
}

CHROMA_PATH = "chroma_db"
PROMPT_TEMPLATE = """
You are an expert in agriculture. Based only on the information in the following context, provide a complete and accurate answer to the user's question.

Context:
{context}

---

Question: {question}
Answer:
"""

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_match_query(query, gt_lookup):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    best_match = None
    best_score = 0.0
    for gt_query in gt_lookup:
        gt_embedding = sbert_model.encode(gt_query, convert_to_tensor=True)
        score = util.cos_sim(query_embedding, gt_embedding).item()
        if score > best_score:
            best_score = score
            best_match = gt_query
    return best_match if best_score > 0.5 else None

def export_metrics_to_csv(metrics, filename="metrics_log.csv"):
    keys = list(metrics.keys())
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = OllamaLLM(model="")

    results = db.similarity_search_with_score(query_text, k=5)
    retrieved_docs = [doc for doc, _ in results]
    retrieved_texts = [doc.page_content for doc in retrieved_docs]

    context_text = "\n\n---\n\n".join(retrieved_texts)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context_text, question=query_text)
    response = model.invoke(prompt)

    print(f"\nGenerated Answer:\n{response}")
    print(f"\nRetrieved Sources: {[doc.metadata.get('id') for doc in retrieved_docs]}")

    gt_lookup = {k.lower(): v for k, v in GROUND_TRUTH.items()}
    matched_query = semantic_match_query(query_text.lower(), gt_lookup)

    if matched_query:
        references = [gt_lookup[matched_query]]

        print(f"\nMatched Ground Truth Query: {matched_query}")
        print(f"Ground Truth Answer: {references[0]}")

        # Compute Precision@K, Recall@K
        hits = [
            1 if any(
                util.cos_sim(
                    sbert_model.encode(ref, convert_to_tensor=True),
                    sbert_model.encode(doc.page_content, convert_to_tensor=True)
                ).item() >= 0.5 for ref in references
            ) else 0
            for doc in retrieved_docs
        ]
        precision_at_k = sum(hits) / len(hits)
        recall_at_k = sum(hits)

        print("\nRetrieval Evaluation Metrics:")
        print(f"Precision@K: {precision_at_k:.2f}")
        print(f"Recall@K: {recall_at_k:.2f}")

        print("\nSemantic Similarity Debug Info:")
        for i, doc in enumerate(retrieved_docs):
            sims = [
                util.cos_sim(
                    sbert_model.encode(ref, convert_to_tensor=True),
                    sbert_model.encode(doc.page_content, convert_to_tensor=True)
                ).item()
                for ref in references
            ]
            max_sim = max(sims)
            print(f"Doc {i + 1} similarity: {max_sim:.2f} | Match: {'Found' if max_sim >= 0.5 else 'Not found'}")

        # MRR & NDCG
        mrr = 1 / (hits.index(1) + 1) if 1 in hits else 0
        dcg = sum([rel / math.log2(idx + 2) for idx, rel in enumerate(hits)])
        idcg = sum([1.0 / math.log2(i + 2) for i in range(min(1, len(hits)))])
        ndcg = dcg / idcg if idcg > 0 else 0
        print(f"MRR: {mrr:.2f}")
        print(f"NDCG: {ndcg:.2f}")

        # Generation Quality Metrics
        print("\nGeneration Quality Metrics:")
        smoothing = SmoothingFunction().method4
        bleu = max(sentence_bleu([ref.split()], response.split(), smoothing_function=smoothing) for ref in references)
        print(f"BLEU Score: {bleu:.2f}")

        rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = [rouge.score(ref, response) for ref in references]
        rouge1 = max(score['rouge1'].fmeasure for score in rouge_scores)
        rougel = max(score['rougeL'].fmeasure for score in rouge_scores)
        print(f"ROUGE-1: {rouge1:.2f}, ROUGE-L: {rougel:.2f}")

        # BERTScore
        try:
            P, R, F1 = bert_score(
                [response] * len(references), 
                references, 
                lang="en", 
                rescale_with_baseline=True,
                verbose=False
            )
            print(f"BERTScore: P={P.mean():.2f}, R={R.mean():.2f}, F1={F1.mean():.2f}")
            bert_f1 = F1.mean().item()
        except Exception as e:
            print(f"BERTScore error: {e}")
            bert_f1 = 0.0

        export_metrics_to_csv({
            "Query": query_text,
            "Matched_GT": matched_query,
            "BLEU": bleu,
            "ROUGE-1": rouge1,
            "ROUGE-L": rougel,
            "BERTScore_F1": bert_f1,
            "Precision@K": precision_at_k,
            "Recall@K": recall_at_k,
            "MRR": mrr,
            "NDCG": ndcg,
        })

    else:
        print("\nNo matched ground truth found for evaluation.")

    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)

if __name__ == "__main__":
    main()