"""Standalone comparison server for Project VOICE models.

Compares three models side-by-side:
  1. Local gemma3-270m-pred-dpo (on-device)
  2. Tuned v11 model (Vertex AI endpoint)
  3. Gemini Flash (Vertex AI)
"""

import difflib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import flask
from google import genai
from google.genai import types

app = flask.Flask(__name__, static_folder='static')

# ---------------------------------------------------------------------------
# Vertex AI config (override via environment variables)
# ---------------------------------------------------------------------------
VERTEX_PROJECT = os.environ.get('VERTEX_PROJECT', 'your-gcp-project-id')
VERTEX_LOCATION = os.environ.get('VERTEX_LOCATION', 'us-central1')
TUNED_ENDPOINT = os.environ.get(
    'TUNED_ENDPOINT',
    'projects/YOUR_PROJECT_NUMBER/locations/us-central1/'
    'endpoints/YOUR_ENDPOINT_ID',
)

_vertex_client = None

def _get_vertex_client():
    global _vertex_client
    if _vertex_client is None:
        _vertex_client = genai.Client(
            vertexai=True,
            project=VERTEX_PROJECT,
            location=VERTEX_LOCATION,
        )
    return _vertex_client

# ---------------------------------------------------------------------------
# v11 prompt building
# ---------------------------------------------------------------------------
V11_PROMPT_HEADER = (
    "キーボードの予測変換として[---]に続く言葉を予測変換してください。"
    "[---]より前はこれまでのユーザー入力です。\n"
    "ユーザー入力と予測変換の間には境界 [---]を入れてください。"
)
V11_MARKER_LINE = "ーーーー以下が予測変換対象ーーーー"

V11_TONE_PROMPTS = {
    'dev': (
        "【トーン: dev】開発の文脈で予測変換してください。"
        "開発寄り語彙（例: PR/レビュー/デプロイ/issue/バグ/再現/修正/ログ/確認）を優先。"
        "語尾はフラットで自然（です/ますでも可）。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
    'meeting': (
        "【トーン: meeting】ミーティングの文脈で予測変換してください。"
        "会議語彙（例: 議題/アジェンダ/共有/確認事項/宿題/決定/進捗/次回）を優先。"
        "結びは『〜します』『〜しましょう』『〜いかがでしょうか』など会議っぽく。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
    'casual': (
        "【トーン: casual】カジュアルな文脈で予測変換してください。"
        "砕けた口語（例: だよ/だね/しよ/しよう/かな/だと思う）を優先し、敬語はなるべく避ける。"
        "ただし乱暴な表現は避けて自然に。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
    'business': (
        "【トーン: business】ビジネスの文脈で予測変換してください。"
        "実務的で丁寧（例: 恐れ入りますが/ご確認のほど/差し支えなければ/よろしくお願いいたします）を優先。"
        "長くなりすぎないように [---]より後は原則 20 トークン（/区切りで20個）以内を目安。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
    'polite': (
        "【トーン: polite】丁寧・敬語の文脈で予測変換してください。"
        "です/ます調＋クッション言葉（例: お手数ですが/恐れ入りますが/ありがとうございます）を優先。"
        "ビジネスほど堅くしすぎず、丁寧さを保った自然文に。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
    'friendly': (
        "【トーン: friendly】親しみやすい文脈で予測変換してください。"
        "柔らかい語尾（例: 〜ですね/〜だと嬉しいです/〜しよう）や感謝を入れてもよい。"
        "ただし過剰に長くしない。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
    'concise': (
        "【トーン: concise】短く要点だけの文脈で予測変換してください。"
        "冗長な前置きは避け、[---]より後は原則 8〜12 トークン（/区切りで8〜12個）程度を目安に短く。"
        "敬語は必要最低限に。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
    'enthusiastic': (
        "【トーン: enthusiastic】明るく前向きな文脈で予測変換してください。"
        "前向き語彙（例: いいですね/助かります/楽しみ/最高/嬉しい）を適度に使い、"
        "必要なら『！』を1個だけ入れてもよい（多用しない）。"
        "同じ語句や同じ文の繰り返しは禁止。"
        "[---]より前は一文字も変更せず、[---]より後のみを / 区切りで出力してください。"
    ),
}

V11_TONE_TEMPERATURES = {
    'dev': 0.28, 'meeting': 0.25, 'casual': 0.55, 'business': 0.20,
    'polite': 0.22, 'friendly': 0.35, 'concise': 0.20, 'enthusiastic': 0.65,
}

# Gemini Flash prompt template (same as SentenceJapaneseLong20250424.jinja2)
FLASH_PROMPT_TEMPLATE = """あなたはALSやSMAや脳機能障害などでコミュニケーションに困難を抱えるユーザーの会話を支援するボットです。ユーザーが入力中の「{text}」で始まる文（読点"。"や感嘆符"！"、"？"で終わるもの）を{num}つ推測して番号付きのリストにしてください。あなたの出力はそのままユーザーに選択肢として表示されるので、出力には余分な補足や説明、スペース（空白）は一切含めないでください。

以下ルールです。
- 各文章はなるべく異なる内容にしてください。
- 「{text}」は入力途中の場合もあります。単語で終わっていない場合は文字の補足もしたうえで、続きうる文章を作ってください。名前など、固有名詞であるケースも想定してください。
- 「{text}」の文章は通常漢字やカタカナで書かれるものが、ひらがなのままなケースもあります。「漢字、あるいはカタカナで書いてあれば」という想定もしてください。漢字であることを想定して作成した回答では、回答内の表示も想定した漢字で表記してください。その際どう想定したか、という補足や読みの説明は不要です。
- 「{text}」に続く最初の単語、または助詞は回答ごとに極力異なるものにしてください。ただし、あまりにマイナーな語彙は特に指示のない限り避けてください。
- 「{text}」には不要な句読点やスペース、漢字の読み方（）の注釈などは含めないでください。
{persona_section}{conversation_section}{emotion_section}回答:
"""


def _build_flash_prompt(text, num='3', persona='', conversation_history='',
                        emotion=''):
    persona_section = ''
    if persona:
        persona_section = (
            f'\n参考までに、このユーザのプロフィールは以下のとおりです:\n'
            f'{persona}\n'
        )
    conversation_section = ''
    if conversation_history:
        conversation_section = (
            f'\n以下はユーザとその相手との会話の履歴です:\n'
            f'{conversation_history}\n'
        )
    emotion_section = ''
    if emotion:
        emotion_section = (
            f'なお、ユーザーは{emotion}文の入力を意図しています。'
            f'「{text}」に入力されている文章を元に、{emotion}文になるよう'
            f'書き換えてください。必要であれば文章の冒頭から書き換えてください。\n\n'
        )
    return FLASH_PROMPT_TEMPLATE.format(
        text=text, num=num,
        persona_section=persona_section,
        conversation_section=conversation_section,
        emotion_section=emotion_section,
    )


# ---------------------------------------------------------------------------
# v11 prompt / parse helpers
# ---------------------------------------------------------------------------
def _build_v11_prompt(history, last_sentence, prefix, tone_prompt,
                      persona='', conversation_history='', emotion=''):
    sections = []
    if persona:
        sections.append(f"<ペルソナ>\n{persona}")
    if conversation_history:
        sections.append(f"<会話履歴>\n{conversation_history}")
    if emotion and emotion != 'statement':
        labels = {'question': '質問文', 'request': '依頼・お願い', 'negative': '否定文'}
        sections.append(f"<文のタイプ>\n{labels.get(emotion, emotion)}")
    if history:
        sections.append(f"<文脈>\n{history}")
    ctx = ''
    if sections:
        ctx = "\n" + "\n\n".join(sections) + "\n"
    return (f"{V11_PROMPT_HEADER}\n{tone_prompt}{ctx}\n"
            f"{V11_MARKER_LINE}\n{last_sentence}[---]{prefix}")


def _parse_v11_output(output_text):
    if not output_text or '[---]' not in output_text:
        return None
    _, after = output_text.split('[---]', 1)
    tokens = [t.strip() for t in after.split('/') if t.strip()]
    return '/'.join(tokens) if tokens else None


def _similarity(a, b):
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _select_diverse(suggestions, n=3):
    if len(suggestions) <= n:
        return [s[1] for s in suggestions]
    selected = [suggestions[0]]
    remaining = suggestions[1:]
    while len(selected) < n and remaining:
        best_idx, best_sim = -1, 1.0
        for i, (_, c) in enumerate(remaining):
            sim = min(_similarity(c, s[1]) for s in selected)
            if sim < best_sim:
                best_sim, best_idx = sim, i
        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
        else:
            break
    return [s[1] for s in selected]


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------
def _generate_one_tone(client, history, last_sentence, prefix, tone_id,
                       persona='', conversation_history='', emotion=''):
    tone_prompt = V11_TONE_PROMPTS.get(tone_id, '')
    temperature = V11_TONE_TEMPERATURES.get(tone_id, 0.3)
    prompt = _build_v11_prompt(history, last_sentence, prefix, tone_prompt,
                               persona, conversation_history, emotion)
    try:
        resp = client.models.generate_content(
            model=TUNED_ENDPOINT, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature, max_output_tokens=96,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        if resp.text:
            pred = _parse_v11_output(resp.text.strip())
            if pred:
                return (tone_id, pred)
    except Exception as e:
        print(f'[WARN] Tone {tone_id} failed: {e}')
    return None


def run_tuned(last_sentence, prefix, history='', persona='',
              conversation_history='', emotion=''):
    """Run tuned v11 model with 8 tones in parallel, select 3 diverse."""
    client = _get_vertex_client()
    suggestions = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(_generate_one_tone, client, history, last_sentence,
                      prefix, tid, persona, conversation_history, emotion): tid
            for tid in V11_TONE_PROMPTS
        }
        for f in as_completed(futures):
            r = f.result()
            if r:
                suggestions.append(r)
    if not suggestions:
        return []
    return _select_diverse(suggestions, 3)


def run_flash(text, persona='', conversation_history='', emotion=''):
    """Run Gemini Flash via Vertex AI."""
    client = _get_vertex_client()
    prompt = _build_flash_prompt(text, '3', persona, conversation_history,
                                 emotion)
    resp = client.models.generate_content(
        model='gemini-2.5-flash', contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3, top_p=0.5,
            safety_settings=[
                types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH',
                                    threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                                    threshold='BLOCK_NONE'),
            ],
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    if not resp.text:
        return []
    cleaned = resp.text.replace('*', '')
    cleaned = re.sub(r'([^\w;:,.?]) +(\W)', r'\1\2', cleaned, flags=re.ASCII)
    results = []
    for line in cleaned.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        content = re.sub(r'^\d+\.\s*', '', line)
        if content:
            results.append(content)
    return results


# --- Local model ---
_local_model = None
_local_tokenizer = None


def _get_local_model():
    global _local_model, _local_tokenizer
    if _local_model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_id = 'katsukiono/gemma3-270m-pred-dpo'
        _local_tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                         use_fast=True)
        if _local_tokenizer.pad_token is None:
            _local_tokenizer.pad_token = _local_tokenizer.eos_token
        _local_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype='auto', device_map='auto').eval()
    return _local_model, _local_tokenizer


def run_local(last_sentence, prefix, history='', persona='',
              conversation_history='', emotion='', num_sequences=3):
    """Run local gemma3-270m model, generate multiple diverse predictions."""
    import torch
    prompt = _build_v11_prompt(history, last_sentence, prefix, '',
                               persona, conversation_history, emotion)
    model, tokenizer = _get_local_model()
    chat = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, do_sample=True, temperature=0.7, top_p=0.9,
            num_return_sequences=num_sequences, max_new_tokens=96,
        )
    predictions = []
    seen = set()
    input_len = inputs['input_ids'].shape[1]
    for i in range(outputs.shape[0]):
        gen = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
        first_line = gen.strip().split('\n')[0].strip()
        pred = _parse_v11_output(first_line)
        if pred and pred not in seen:
            seen.add(pred)
            predictions.append(pred)
    return predictions


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def _normalize_tuned_output(raw_list, last_sentence):
    full = [last_sentence + r.replace('/', '') for r in raw_list]
    return {'full': full, 'raw': raw_list}


def _normalize_normal_output(items, input_text):
    results = []
    for content in items:
        if content.startswith(input_text):
            continuation = content[len(input_text):]
        else:
            continuation = content
        results.append({'full': content, 'continuation': continuation})
    return results


# ---------------------------------------------------------------------------
# Common input parser
# ---------------------------------------------------------------------------
def _parse_input(data):
    last_sentence = data.get('last_sentence', '')
    prefix = data.get('prefix', '')
    history = data.get('history', '')
    persona = data.get('persona', '')
    conversation_history = data.get('conversationHistory', '')
    emotion = data.get('sentenceEmotion', '')
    text = last_sentence + prefix
    return last_sentence, prefix, history, persona, conversation_history, emotion, text


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return flask.send_file('index.html')


@app.route('/compare/local', methods=['POST'])
def compare_local():
    data = flask.request.get_json(force=True)
    last_sentence, prefix, history, persona, ch, emotion, text = _parse_input(data)
    t0 = time.monotonic()
    preds = run_local(last_sentence, prefix, history, persona, ch, emotion)
    ms = round((time.monotonic() - t0) * 1000)
    norm = _normalize_tuned_output(preds, last_sentence)
    return json.dumps({
        'full': norm['full'], 'raw': norm['raw'], 'ms': ms,
        'input': {'last_sentence': last_sentence, 'prefix': prefix, 'text': text},
    }, ensure_ascii=False)


@app.route('/compare/tuned', methods=['POST'])
def compare_tuned():
    data = flask.request.get_json(force=True)
    last_sentence, prefix, history, persona, ch, emotion, text = _parse_input(data)
    t0 = time.monotonic()
    preds = run_tuned(last_sentence, prefix, history, persona, ch, emotion)
    ms = round((time.monotonic() - t0) * 1000)
    norm = _normalize_tuned_output(preds, last_sentence)
    return json.dumps({
        'full': norm['full'], 'raw': norm['raw'], 'ms': ms,
        'input': {'last_sentence': last_sentence, 'prefix': prefix, 'text': text},
    }, ensure_ascii=False)


@app.route('/compare/normal', methods=['POST'])
def compare_normal():
    data = flask.request.get_json(force=True)
    last_sentence, prefix, history, persona, ch, emotion, text = _parse_input(data)
    t0 = time.monotonic()
    items = run_flash(text, persona, ch, emotion)
    ms = round((time.monotonic() - t0) * 1000)
    norm = _normalize_normal_output(items, text)
    return json.dumps({
        'items': norm, 'ms': ms,
        'input': {'last_sentence': last_sentence, 'prefix': prefix, 'text': text},
    }, ensure_ascii=False)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, threaded=True, port=5001,
            host=os.environ.get('FLASK_HOST', '127.0.0.1'))
