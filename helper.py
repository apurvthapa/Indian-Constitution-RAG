import re
from difflib import get_close_matches
import ast
from pathlib import Path
from prompts import prompt
from model_selection import get_llm, get_retriever
from langchain_core.documents import Document
import json

def normalize_roman(text):
    ROMAN_MAP = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
    "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"
    }
    for r, n in ROMAN_MAP.items():
        text = re.sub(rf'\b{r}\b', n, text)
    return text


def fuzzy_replace(text, keywords, correct_word):
    words = text.split()
    corrected = []

    for w in words:
        match = get_close_matches(w, keywords, n=1, cutoff=0.75)
        if match:
            corrected.append(correct_word)
        else:
            corrected.append(w)

    return " ".join(corrected)


def expand_range(start, end):
    try:
        start, end = int(start), int(end)
        if start <= end and end - start <= 50:
            return [str(i) for i in range(start, end + 1)]
    except:
        pass
    return []


def extract_entities(query: str):
    ARTICLE_KEYWORDS = ["article", "art"]
    SCHEDULE_KEYWORDS = ["schedule", "sch"]
    q = query.lower()

    # --- typo handling ---
    q = fuzzy_replace(q, ARTICLE_KEYWORDS, "article")
    q = fuzzy_replace(q, SCHEDULE_KEYWORDS, "schedule")

    # --- roman normalization ---
    q = normalize_roman(q)

    articles = set()
    schedules = set()

    # -----------------------------
    # step 1: schedules first
    # -----------------------------
    schedule_spans = list(re.finditer(r'schedule\s*([\d,\sand]+)', q))

    for match in schedule_spans:
        block = match.group(1)
        nums = re.findall(r'\d+', block)
        schedules.update(nums)

    # 7th schedule
    schedules.update(re.findall(r'(\d+)\s*th\s*schedule', q))

    # remove schedule spans
    clean_q = q
    for span in schedule_spans:
        clean_q = clean_q.replace(span.group(), '')

    # -----------------------------
    # step 2: articles
    # -----------------------------
    article_blocks = re.findall(
        r'article\s*(?:no\.?|number)?\s*([\d+a-z,\sandto]+)', clean_q
    )

    for block in article_blocks:

        # range
        ranges = re.findall(r'(\d+)\s*to\s*(\d+)', block)
        for start, end in ranges:
            articles.update(expand_range(start, end))

        # normal numbers
        nums = re.findall(r'\d+[a-z]?', block)
        articles.update(nums)

    # -----------------------------
    # step 3: fallback
    # -----------------------------
    if not articles and not schedules:
        nums = re.findall(r'\b\d+[a-z]?\b', q)
        articles.update(nums)

    # -----------------------------
    # step 4: no type leak
    # -----------------------------
    articles = articles - schedules

    # -----------------------------
    # step 5: normalize caps
    # -----------------------------
    articles = {a.upper() for a in articles}
    schedules = {s.upper() for s in schedules}

    return {
        "articles": sorted(articles),
        "schedules": sorted(schedules)
    }

def embedded_or_not(query):
    result = extract_entities(query)

    return {
        "use_rag": not any(result.values()),
        "entities": result
    }

def parse_llm_output(content):
    try:
        parsed = ast.literal_eval(content)

        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed

    except Exception:
        pass

    return [] 

def query_enhancer(query):
    chain_1 = prompt | get_llm()
    result = chain_1.invoke({"user_query": query})
    return parse_llm_output(result.content)

def load_docs(path="docs.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"]
        )
        for item in data
    ]

def context_from_query_list(query_list):
    context = set()
    retriever = get_retriever()

    for query in query_list:
        results = retriever.invoke(query)
    
        for doc in results:
            metadata = doc.metadata
        
            start_page = metadata.get("start_page", "")
            doc_type = metadata.get("document_type", "")
            schedule = metadata.get("schedule_number").replace("None","")
            article = metadata.get("article_number").replace("None", "")
        
            text = (
                f"{doc_type} "
                f"{schedule} "
                f"{article} "
                f"[Page Number {start_page}] "
                f"{doc.page_content}"
            )
        
            context.add(text)

    context = list(context)
    return context

def direct_fetch(entities, final_docs):
    articles = set(entities.get("articles", []))
    schedules = set(entities.get("schedules", []))

    results = []

    for doc in final_docs:
        meta = doc.metadata

        doc_article = str(meta.get("article_number", "")).upper()
        doc_schedule = str(meta.get("schedule_number", "")).upper()
        doc_type = meta.get("document_type", "")

        # --- match articles ---
        if articles and doc_type == "article":
            if doc_article in articles:
                results.append(doc)
                continue  # avoid duplicate match

        # --- match schedules ---
        if schedules and doc_type == "schedule":
            if doc_schedule in schedules:
                results.append(doc)

    return results


BASE_DIR = Path(__file__).resolve().parent
RERANKER_DIR = BASE_DIR / "models" / "bge-reranker"
DOCS_PATH = BASE_DIR / "docs.json"

_reranker_model = None
_final_docs = None


def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(str(RERANKER_DIR))
    return _reranker_model

def context_reranker(query, context):
    model = get_reranker_model()
    pairs = [[query, doc] for doc in context]
    scores = model.predict(pairs)
    ranked = sorted(zip(context, scores), key=lambda x: x[1], reverse=True)
    top_k = [doc for doc, _ in ranked[:3]]
    return(top_k)


def get_final_docs():
    global _final_docs
    if _final_docs is None:
        _final_docs = load_docs(str(DOCS_PATH))
    return _final_docs

def context_outputer(query):
    route = embedded_or_not(query)

    context = str()
    if route["use_rag"]:
        list_of_queries = query_enhancer(query)
        result = context_from_query_list(list_of_queries)
        return " ".join(context_reranker(query = query, context= result))
    else:
        entities = route['entities']
        docs = direct_fetch(entities, get_final_docs())

        context_parts = []

        for doc in docs:
            doc_type = doc.metadata["document_type"]
            number = doc.metadata['schedule_number'] if doc.metadata['article_number'] == "None" else doc.metadata['article_number']
            page_number = doc.metadata["start_page"]
            data = doc.page_content

            context_parts.append(
                f"{doc_type} {number} [Page Number {page_number}] {data}"
            )

        context = " ".join(context_parts)
        return context
