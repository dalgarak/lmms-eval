import os
import re
from pathlib import Path
import yaml
from PIL import Image

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)

def _load_cache_name(yaml_filename):
    yaml_path = Path(__file__).parent / yaml_filename
    if not yaml_path.exists():
        return ""
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            lines = [line for line in f if "!function" not in line]
            config = yaml.safe_load("".join(lines))
            return config.get("dataset_kwargs", {}).get("cache_dir", "")
    except Exception:
        return ""

cache_name = _load_cache_name("rxnbench_vqa_en.yaml") or ""

def rxnbench_doc_to_visual(doc):
    image = doc["image"]
    
    if cache_name:
        full_path = os.path.join(base_cache_dir, cache_name, image)
        if os.path.exists(full_path):
            return [full_path]
    return [image]

def rxnbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    choices = doc["choices"]
    while len(choices) < 4:
        choices.append("")
    choices = choices[:4]
    options = ["A", "B", "C", "D"]
    choices_str = "\n".join(f"{opt}. {choice}" for opt, choice in zip(options, choices))
    
    default_post = "\nBased on the image and the question, choose the most appropriate answer.\n**Only output a single letter (A, B, C, or D)**. Do NOT output any other text or explanation."
    
    pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post = lmms_eval_specific_kwargs.get("post_prompt", default_post)
    
    return f"{pre}Question: {doc['question']}\nChoices:\n{choices_str}{post}"

def _get_target(doc):
    return ["A", "B", "C", "D"][int(doc["answer"])]


# \boxed{} 출력과 서술형 출력에서 선택지 글자를 뽑는다. (utils_zh.py 와 같은 내용을 유지한다)
# \boxed{...} 안쪽(중첩 중괄호 1단계까지)
_BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
# 'Answer: X' / 'answer is X' / 'option X'. 대문자는 어디서든, 소문자는 응답 맨 끝일 때만 인정한다("the answer is a mixture" 방지).
_MARKER_RE = re.compile(
    r"(?i:answer|option|choice)(?:\s+is)?\s*[:：]?\s*\(?\**\s*([A-D]|[a-d](?=\s*[.)*]*\s*$))(?![A-Za-z])"
)
# 단독 대문자 글자. 'C=O' 같은 화학식은 제외한다.
_LETTER_RE = re.compile(r"(?<![A-Za-z0-9=])([A-D])(?![A-Za-z0-9=])")
_STRIP_RE = re.compile(r"\\(?:text|textbf|mathrm|mathbf)\s*\{|[{}()\s.\*]")


def _extract_choice(pred):
    # 1) 마지막 \boxed{} 안이 글자 하나일 때만 인정한다(\boxed{Acetone} 같은 단어는 제외). 대소문자 무시.
    for inner in reversed(_BOXED_RE.findall(pred)):
        cleaned = _STRIP_RE.sub("", inner)
        if len(cleaned) == 1 and cleaned.lower() in "abcd":
            return cleaned.lower()
    # 2) 마지막 명시적 표지
    marked = _MARKER_RE.findall(pred)
    if marked:
        return marked[-1].lower()
    # 3) 단독 대문자 글자의 마지막 매칭(결론은 뒤에 온다)
    letters = _LETTER_RE.findall(pred)
    return letters[-1].lower() if letters else ""


def rxnbench_process_results(doc, results):
    raw_pred = results[0] if results else ""

    pred = raw_pred.strip()
    target = _get_target(doc).strip().lower()
    
    score = 0.0

    if pred.lower() == target:
        score = 1.0
    else:
        if _extract_choice(pred) == target:
            score = 1.0

    prompt_text = rxnbench_doc_to_text(doc)
    
    return {
        "exact_match": score,
        "raw_output": raw_pred,
        "question": prompt_text
    }

