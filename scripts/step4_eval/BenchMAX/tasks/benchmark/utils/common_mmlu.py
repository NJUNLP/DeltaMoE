# This file is copied from the OpenAI repository:
# https://github.com/openai/simple-evals/blob/main/common.py
# Licensed under the MIT License.

import io
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.pool import ThreadPool
from typing import Any, Callable

import jinja2
import numpy as np
import requests
from tqdm import tqdm

MMLU_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

MMLU_QUERY_TEMPLATE_HU = """
Válaszolj a következő többszörös választásos kérdésre. A válaszod utolsó sora a következő formátumban legyen: 'Válasz: $BETŰ' (idézőjelek nélkül), ahol BETŰ az A, B, C vagy D lehet. Gondolkodj lépésről lépésre, mielőtt válaszolsz.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()
MMLU_QUERY_TEMPLATE_AR = """
أجب عن سؤال الاختيار المتعدد التالي. يجب أن يكون السطر الأخير من إجابتك بالشكل التالي: 'الإجابة: $الحرف' (بدون علامات الاقتباس) حيث يكون الحرف أحد الأحرف ABCD. فكر خطوة بخطوة قبل الإجابة.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

MMLU_QUERY_TEMPLATE_BN = """
নিম্নলিখিত বহু নির্বাচনী প্রশ্নের উত্তর দিন। আপনার উত্তরের শেষ লাইনটি নিম্নলিখিত ফর্ম্যাটে হতে হবে: 'উত্তর: $LETTER' (উদ্ধৃতি ছাড়া) যেখানে LETTER ABCD এর মধ্যে একটি। উত্তর দেওয়ার আগে ধাপে ধাপে চিন্তা করুন।

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

MMLU_QUERY_TEMPLATE_SR_CYRILLIC = """
Одговорите на следеће питање вишеструког избора. Последњи ред вашег одговора мора бити у следећем формату: 'Одговор: $LETTER' (без наводника) где је LETTER једно од ABCD. Размислите корак по корак пре него што одговорите.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# 西班牙语 (es) MMLU Prompt
MMLU_QUERY_TEMPLATE_ES = """
Responde a la siguiente pregunta de opción múltiple. La última línea de tu respuesta debe tener el siguiente formato: 'Respuesta: $LETRA' (sin comillas), donde LETRA es una de ABCD. Piensa paso a paso antes de responder.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# 中文 (zh) MMLU Prompt
MMLU_QUERY_TEMPLATE_ZH = """
请回答以下选择题。你的回答最后一行应遵循以下格式：'答案：$字母'（不含引号），其中 字母 是ABCD中的一个。在回答前请逐步思考。

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# 法语 (fr) MMLU Prompt
MMLU_QUERY_TEMPLATE_FR = """
Répondez à la question à choix multiples suivante. La dernière ligne de votre réponse doit être au format suivant : 'Réponse: $LETTRE' (sans guillemets), où LETTRE est l'une des options ABCD. Réfléchissez étape par étape avant de répondre.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

MMLU_QUERY_TEMPLATE_JA = """
次の選択問題に答えてください。回答の最終行は「答え：$LETTER」（引用符なし）の形式にしてください。ここで、$LETTERはABCDのいずれかです。回答する前に、順を追って考えてください。

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

MMLU_QUERY_TEMPLATE_CS = """
Odpovězte na následující otázku s výběrem odpovědí. Poslední řádek vaší odpovědi musí být v následujícím formátu: 'Odpověď: $PÍSMENO' (bez uvozovek), kde PÍSMENO je jedno z ABCD. Před odpovědí přemýšlejte krok za krokem.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# 越南语 (vi) MMLU Prompt
MMLU_QUERY_TEMPLATE_VI = """
Trả lời câu hỏi trắc nghiệm sau đây. Dòng cuối cùng trong câu trả lời của bạn phải có định dạng sau: 'Đáp án: $CHỮ_CÁI' (không có dấu ngoặc kép), trong đó CHỮ_CÁI là một trong các chữ ABCD. Hãy suy nghĩ từng bước trước khi trả lời.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# 泰卢固语 (te) MMLU Prompt
MMLU_QUERY_TEMPLATE_TE = """
కింది బహుళైచ్ఛిక ప్రశ్నకు సమాధానం ఇవ్వండి. మీ ప్రతిస్పందన యొక్క చివరి పంక్తి కింది ఆకృతిలో ఉండాలి: 'జవాబు: $LETTER' (కోట్స్ లేకుండా), ఇక్కడ LETTER ABCDలో ఒకటి. సమాధానం ఇచ్చే ముందు దశలవారీగా ఆలోచించండి.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()



# 更新后的 MMLU 字典
MMLU_TEMPLATE_DICT = {
    "en": MMLU_QUERY_TEMPLATE,
    "hu": MMLU_QUERY_TEMPLATE_HU,
    "ar": MMLU_QUERY_TEMPLATE_AR,
    "bn": MMLU_QUERY_TEMPLATE_BN,
    "sr": MMLU_QUERY_TEMPLATE_SR_CYRILLIC,
    "es": MMLU_QUERY_TEMPLATE_ES,
    "zh": MMLU_QUERY_TEMPLATE_ZH,
    "fr": MMLU_QUERY_TEMPLATE_FR,
    "ja": MMLU_QUERY_TEMPLATE_JA,
    "cs": MMLU_QUERY_TEMPLATE_CS,
    "vi": MMLU_QUERY_TEMPLATE_VI,
    "te": MMLU_QUERY_TEMPLATE_TE,
    
}


ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"
)
# All the different ways "Answer" is written in different languages
MULTILINGUAL_ANSWER_REGEXES = [
    "Answer\s*:",
    "Answer\s*:​​​​​​",  # Korean invisible character
    "উত্তর\s*:",
    "उत्तर\s*:",
    "উত্তরঃ",
    "উত্তর\s*:",
    "Antwort\s*:",
    "Válasz\s*:",  # Hungarian
    "Oдговор\s*:",  # Serbian (Cyrillic)
    "Odgovor\s*:",  # Serbian (Latin)
    "Одговор\s*:",
    "답변\s*:",
    "정답\s*:",
    "답\s*:",
    "答案\s*：",
    "答案\s*:",
    "答\s*：",
    "答\s*:",
    "答复\s*：",
    "答曰\s*：",
    "الإجابة:",
    "الجواب:",
    "إجابة:",
    "الإجابة النهائية:",
    "الإجابة الصحيحة:",
    "الإجابة هي:",
    "الجواب النهائي:",
    "Respuesta\s*:",
    "Risposta\s*:",
    "答え\s*:",
    "答え\s*：",
    "回答\s*:",
    "回答\s*：",
    "解答\s*:",
    "Jawaban\s*:",
    "Réponse\s*:",
    "Resposta\s*:",
    "Jibu\s*:",
    "Idahun\s*:",
    "Ìdáhùn\s*:",
    "Idáhùn\s*:",
    "Àmọ̀nà\s*:",
    "Àdáhùn\s*:",
    "Ànúgọ\s*:",
    "Àṣàyàn\s*:",
    "Đáp án\s*:",          # Vietnamese
    "Odpověď\s*:",         # Czech
    "జవాబు\s*:",           # Telugu

]

EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()


HTML_JINJA = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{{ message_to_html(next_message) | safe }}
<h3>Results</h3>
<p>Correct Answer: {{ correct_answer }}</p>
<p>Extracted Answer: {{ extracted_answer }}</p>
<p>Score: {{ score }}</p>
"""

def format_multichoice_question(row):
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def _compute_stat(values: list, stat: str):
    if stat == "mean":
        return np.mean(values)
    elif stat == "std":
        return np.std(values)
    elif stat == "min":
        return np.min(values)
    elif stat == "max":
        return np.max(values)
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        return np.std(
            [np.mean(np.random.choice(values, len(values))) for _ in range(1000)]
        )
    else:
        raise ValueError(f"Unknown {stat =}")



def normalize_response(response: str) -> str:
    """
    Normalize the response by removing markdown and LaTeX formatting that may prevent a match.
    """

    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # In arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )

