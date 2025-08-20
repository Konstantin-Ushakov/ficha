import os
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
_MODEL = None
_NLI_MODEL = None
_NLI_TOKENIZER = None
_NLI2_MODEL = None  # optional second NLI model (disabled unless configured)
_NLI2_TOKENIZER = None
_NLI3_MODEL = None  # deprecated
_NLI3_TOKENIZER = None  # deprecated

NLI_MODEL_DIR = os.environ.get(
    "NLI_MODEL_DIR",
    "/app/model/models--MoritzLaurer--mDeBERTa-v3-base-xnli-multilingual-nli-2mil7/snapshots/b5113eb38ab63efdd7f280f8c144ea8b13f978ce",
)

NLI2_MODEL_DIR = os.environ.get("NLI2_MODEL_DIR")  # set to local path of joeddav/xlm-roberta-large-xnli snapshot to enable
NLI3_MODEL_DIR = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        try:
            offline_hf = os.environ.get("HF_HUB_OFFLINE", "").strip()
            offline_tr = os.environ.get("TRANSFORMERS_OFFLINE", "").strip()
            hf_home = os.environ.get("HF_HOME", "") or os.environ.get("HF_DATASETS_CACHE", "")
            st_home = os.environ.get("SENTENCE_TRANSFORMERS_HOME", "")
            print(f"[EMB] Loading embedding model: {MODEL_NAME} (HF_HUB_OFFLINE={offline_hf}, TRANSFORMERS_OFFLINE={offline_tr})")
            if hf_home or st_home:
                print(f"[EMB] Cache dirs: HF_HOME={hf_home or '-'}; SENTENCE_TRANSFORMERS_HOME={st_home or '-'}")
        except Exception:
            pass
        _MODEL = SentenceTransformer(MODEL_NAME)
        try:
            print(f"[EMB] Embedding model ready: {MODEL_NAME}")
        except Exception:
            pass
    return _MODEL


def embedding_score(text_a: str, text_b: str) -> float:
    """
    Вычисляет семантическую близость между двумя текстами с помощью эмбеддингов.

    Функция embedding_score использует модель SentenceTransformer для преобразования двух входных текстов
    в векторные представления (эмбеддинги), после чего вычисляет косинусное сходство между этими векторами.
    Результат — число от 0 до 1, где 1 означает максимальную семантическую схожесть, а 0 — отсутствие схожести.

    Для чего это нужно:
    - Оценка степени смысловой близости между двумя фразами, предложениями или абзацами.
    - Поиск дубликатов, перефразированных или схожих по смыслу текстов.
    - Используется в задачах кластеризации, поиска похожих документов, автоматической проверки перефразирования и др.

    Args:
        text_a (str): Первый текст для сравнения.
        text_b (str): Второй текст для сравнения.

    Returns:
        float: Значение косинусного сходства между эмбеддингами двух текстов (от 0 до 1).
    """
    model = get_model()
    embedding_a = model.encode([text_a], convert_to_tensor=True, normalize_embeddings=True)
    embedding_b = model.encode([text_b], convert_to_tensor=True, normalize_embeddings=True)
    cosine_similarity = float(util.cos_sim(embedding_a, embedding_b)[0][0].cpu().item())
    return max(0.0, min(1.0, cosine_similarity))

def get_nli():
    """
    Загружает и возвращает модель NLI (Natural Language Inference) и соответствующий токенизатор.

    Функция get_nli обеспечивает ленивую инициализацию модели и токенизатора для задачи распознавания
    логических отношений между двумя текстами (NLI). Модель и токенизатор загружаются только один раз
    из локальной директории (строгий оффлайн-режим), после чего повторно используются при последующих вызовах.

    Для чего это нужно:
    - Определение отношения между парой предложений: противоречие, следование, нейтральность.
    - Используется в задачах фактчекинга, анализа аргументации, автоматического вывода и др.

    Returns:
        Tuple[torch.nn.Module, transformers.PreTrainedTokenizer]:
            Загруженные модель и токенизатор для NLI.
    """
    global _NLI_MODEL, _NLI_TOKENIZER
    if _NLI_MODEL is None or _NLI_TOKENIZER is None:
        # Строгий оффлайн-режим: загрузка только из локальных путей
        _NLI_TOKENIZER = AutoTokenizer.from_pretrained(NLI_MODEL_DIR, local_files_only=True)
        _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_DIR, local_files_only=True)
        _NLI_MODEL.eval()
        try:
            print(f"[NLI] Primary model loaded: {NLI_MODEL_DIR}")
        except Exception:
            pass
    return _NLI_MODEL, _NLI_TOKENIZER


def get_nli2():
    """Return second NLI model if configured, otherwise raise.

    To enable, set env `NLI2_MODEL_DIR` to a local path (offline cache) of
    joeddav/xlm-roberta-large-xnli (or another compatible NLI model).
    """
    global _NLI2_MODEL, _NLI2_TOKENIZER
    if not NLI2_MODEL_DIR:
        raise RuntimeError("NLI2 not configured")
    if _NLI2_MODEL is None or _NLI2_TOKENIZER is None:
        _NLI2_TOKENIZER = AutoTokenizer.from_pretrained(NLI2_MODEL_DIR, local_files_only=True)
        _NLI2_MODEL = AutoModelForSequenceClassification.from_pretrained(NLI2_MODEL_DIR, local_files_only=True)
        _NLI2_MODEL.eval()
        try:
            print(f"[NLI] Secondary model loaded: {NLI2_MODEL_DIR}")
        except Exception:
            pass
    return _NLI2_MODEL, _NLI2_TOKENIZER


def get_nli3():
    raise RuntimeError("NLI3 disabled")


@torch.no_grad()
def nli_contradiction_prob(premise: str, hypothesis: str) -> float:
    """
    Вычисляет вероятность противоречия (contradiction) между двумя текстами с помощью NLI-модели.

    Для чего это нужно:
    - Автоматическое определение, противоречит ли гипотеза (hypothesis) исходному утверждению (premise).
    - Используется в задачах фактчекинга, анализа аргументации, выявления логических несостыковок и др.

    Args:
        premise (str): Исходное утверждение (premise).
        hypothesis (str): Гипотеза (hypothesis), которую нужно проверить на противоречие.

    Returns:
        float: Вероятность класса "contradiction" (от 0 до 1).
    """
    # Primary model
    model1, tok1 = get_nli()
    inputs1 = tok1(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
    logits1 = model1(**inputs1).logits
    probs1 = F.softmax(logits1, dim=-1)[0]
    id2label1 = model1.config.id2label
    contra_idx1 = None
    for idx, label in id2label1.items():
        if str(label).lower().startswith("contradiction"):
            contra_idx1 = int(idx)
            break
    if contra_idx1 is None:
        contra_idx1 = 2
    p1 = float(probs1[contra_idx1].item())

    # Optional second model
    p2 = None
    try:
        model2, tok2 = get_nli2()
        inputs2 = tok2(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
        logits2 = model2(**inputs2).logits
        probs2 = F.softmax(logits2, dim=-1)[0]
        id2label2 = model2.config.id2label
        contra_idx2 = None
        for idx, label in id2label2.items():
            if str(label).lower().startswith("contradiction"):
                contra_idx2 = int(idx)
                break
        if contra_idx2 is None:
            contra_idx2 = 2
        p2 = float(probs2[contra_idx2].item())
    except Exception:
        p2 = None

    # Conservative aggregation: require both to be high for strong flags
    if p2 is not None:
        if os.environ.get("NLI_DEBUG", "0") == "1":
            try:
                print(f"[NLI] contradiction probs: primary={p1:.4f} secondary={p2:.4f} -> used={min(p1, p2):.4f}")
            except Exception:
                pass
        return min(p1, p2)
    return p1


@torch.no_grad()
def nli_entailment_prob(premise: str, hypothesis: str) -> float:
    """
    Вычисляет вероятность следования (entailment) между двумя текстами с помощью NLI-модели.

    Для чего это нужно:
    - Автоматическое определение, следует ли гипотеза (hypothesis) из исходного утверждения (premise).
    - Используется в задачах фактчекинга, анализа аргументации, автоматического вывода и др.

    Args:
        premise (str): Исходное утверждение (premise).
        hypothesis (str): Гипотеза (hypothesis), которую нужно проверить на следование.

    Returns:
        float: Вероятность класса "entailment" (от 0 до 1).
    """
    try:
        # Primary
        model1, tok1 = get_nli()
        inputs1 = tok1(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
        logits1 = model1(**inputs1).logits
        probs1 = F.softmax(logits1, dim=-1)[0]
        id2label1 = model1.config.id2label
        entail_idx1 = None
        for idx, label in id2label1.items():
            if str(label).lower().startswith("entail"):
                entail_idx1 = int(idx)
                break
        if entail_idx1 is None:
            entail_idx1 = 0
        p1 = float(probs1[entail_idx1].item())

        # Optional second
        p2 = None
        try:
            model2, tok2 = get_nli2()
            inputs2 = tok2(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
            logits2 = model2(**inputs2).logits
            probs2 = F.softmax(logits2, dim=-1)[0]
            id2label2 = model2.config.id2label
            entail_idx2 = None
            for idx, label in id2label2.items():
                if str(label).lower().startswith("entail"):
                    entail_idx2 = int(idx)
                    break
            if entail_idx2 is None:
                entail_idx2 = 0
            p2 = float(probs2[entail_idx2].item())
        except Exception:
            p2 = None

        # Conservative aggregation: make entailment harder (max) to reduce false contradictions
        if p2 is not None:
            if os.environ.get("NLI_DEBUG", "0") == "1":
                try:
                    print(f"[NLI] entailment probs: primary={p1:.4f} secondary={p2:.4f} -> used={max(p1, p2):.4f}")
                except Exception:
                    pass
            return max(p1, p2)
        return p1
    except Exception:
        return 0.0

