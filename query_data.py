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
        
    # Sorghum
    "what are the main planting seasons for sorghum?": "Sorghum is grown as a rainfed crop during May-August and as an irrigated crop during January-April.",
    "what are the main varieties and hybrids of sorghum?": "Sorghum cultivation involves varieties such as Co.1, Co-10, Co-12, Co-17, K-1, and K-2, along with hybrid varieties ranging from CSH-1 to CSH-4, including Co-11 and Co-1.",
    "what are the optimal environmental and soil conditions required for sorghum cultivation?": "Sorghum is a plant of hot and warm localities. The optimum temperature for growth is 30°C and it needs about 250-400 mm rainfall. Excess moisture and prolonged drought are harmful. It is fairly tolerant to alkalinity and salinity.",
    "how frequently should sorghum be irrigated during the initial growth period?": "Irrigate the crop on the day of sowing and thereafter at 10 days interval.",
    "what is the recommended timing for weeding operations in sorghum crops?": "Thinning, weeding and hoeing may be done on the 20th day after sowing.",
    "what is the recommended fertilizer application schedule and dosage for sorghum cultivation?": "For sorghum cultivation, FYM or compost should be applied at 5 tonnes per hectare for both irrigated and rainfed crops, with fertilizer application rates of 90 kg nitrogen, 45 kg phosphorus, and 45 kg potash per hectare for irrigated crops, and 45 kg nitrogen, 25 kg phosphorus, and 25 kg potash per hectare for rainfed crops, where FYM and the entire quantity of phosphorus and potash are applied as basal dose while nitrogen is split into two equal applications - half as basal and the remainder 30 days after sowing.",

    # Ragi
    "what are the different growing seasons for ragi cultivation?": "Ragi is not a season-bound crop and can be cultivated throughout the year if moisture is available, with typical growing seasons including the main season from June to September, late season from July to October, and summer season from December-January to March-April.",
    "what are the main varieties of ragi?": "PR-202, K-2, Co-2, Co-7,Co-8, Co-9, Co-10.",
    "what are the optimal environmental and soil conditions required for ragi cultivation?": "Ragi is suited for cultivation in areas with annual rainfall of 700-1200 mm. It does not tolerate heavy rainfall and requires a dry spell at the time of grain ripening. It grows well in altitudes of 1000-2000 m with average temperature of 27°C. Ragi is cultivated mostly in red lateritic soils. Relatively fertile and well drained soils are the most suitable.",
    "what is the recommended irrigation schedule for ragi after transplantation?": "Irrigate the field on the day of transplantation. Irrigation at weekly intervals increases growth rate and yield.",
    "what is the recommended timing for weeding operations in ragi crops?": "Weeding should be done three weeks after sowing and completed before top dressing.",
    "what is the recommended fertilizer application schedule and dosage for ragi cultivation?": "Plough the field 3-4 times and incorporate FYM orcompost 5 tha-1. Apply nitrogen, phosphorus and potash @ 22.5 kg ha-1 each before sowing or planting. Topdress nitrogen @ 22.5 kg ha-1 21 days after sowing or planting.",

    # French bean
    "what are the different growing seasons for french bean cultivation?": "In the high ranges of elevation more than 1000 m, this crop can be grown throughout the year. The crop being susceptible to ground frost in higher altitudes (above 1400 m), adequate protection should be given during January-February.",
    "what are the main varieties of french bean?": "There are two types of French beans - pole beans and bush beans - where Kentucky Wonder is a popular pole bean variety, while bush bean varieties include Contender, Premier, YCD-1, Arka Komal, and Tender Green.",
    "what are the optimal soil conditions required for french bean cultivation?": "Light sandy-loam to clayey-loam soils with good drainage are best suited for the crop.",
    "what are the sowing specifications for different types of french beans?": "Prepare land thoroughly by ploughing. Raised beds are not essential for bush beans. For pole beans, raised beds are advantageous. Spacing of 30 cm x 20 cm is recommended.",
    "what is the recommended timing for weeding operations in french bean crops?": "First weeding can be given about 4 weeks after sowing and second weeding will be essential 50 days later.",
    "what is the recommended fertilizer application schedule and dosage for french bean cultivation?": "Apply basal dose of 20 t ha-1 of FYM and N:P,0,K,0 @ 30:40:60 kg ha-1. Top dressing with 30 kg N ha-1 may be given 20 days after sowing.",

    # Green gram
    "what are some important varieties of green gram?": "Philippines, Madiera, Pusa Baisakhi, NP-24, Co-2, Pusa-8973 (Pusa-8973 is suited to the summer rice fallows of Onattukara; tolerant to pod borer; duration 66 days).",
    "what are the different cropping systems and cultivation methods used for growing green gram?": "Green gram is grown as a pure crop in rice fallows after the harvest of the first or second crop of paddy. It can also be grown as amixed crop with tapioca, colocasia, yam and banana or as an intercrop in coconut gardens.",
    "what are the sowing specifications for green gram?": "Plough the land 2-3 times thoroughly and remove weeds and stubbles. Channels, 30 cm broad and 15 cm deep, are drawn at 2 mapart to drain off excess rain water during Kharif season and provide irrigation during summer season. The seeds may be sown broadcast.",
    "what is the recommended fertilizer application schedule and dosage for green gram cultivation?": "For green gram cultivation, the recommended nutrient management includes applying FYM at 20 tonnes per hectare as basal dose, lime at 250 kg per hectare or dolomite at 400 kg per hectare during first ploughing, nitrogen at 20 kg per hectare, phosphorus at 30 kg per hectare, and potash at 30 kg per hectare, where half the nitrogen quantity along with full phosphorus and potash should be applied during last ploughing, and the remaining 10 kg nitrogen should be applied as foliar spray using 2 percent urea solution in two equal doses on the 15th and 30th day after sowing."

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