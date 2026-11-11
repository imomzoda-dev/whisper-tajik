---
language: [tg]
license: apache-2.0
base_model: abduaziz/whisper-small-tajik
pipeline_tag: automatic-speech-recognition
tags: [whisper, tajik, asr, speech-recognition]
datasets: [Peacockery/tajik-asr-youtube, google/fleurs]
metrics: [wer, cer]
---

# Whisper Small Tajik v2

Распознавание таджикской речи. Модель пишет **с заглавными буквами и пунктуацией**.
Это дообученная версия [abduaziz/whisper-small-tajik](https://huggingface.co/abduaziz/whisper-small-tajik)
(Whisper Small, 242M параметров). Веса в fp16, 484 МБ.

Tajik speech recognition, fine-tuned from abduaziz/whisper-small-tajik on 162 h of
clean Tajik speech. Outputs cased, punctuated text.

## Качество

WER — доля слов с ошибкой, CER — доля букв с ошибкой (ниже — лучше). Сравнение после
приведения текста к строчным буквам без пунктуации.

| модель | FLEURS test WER | FLEURS CER | YouTube (живая речь) WER | YouTube CER |
|---|---|---|---|---|
| abduaziz/whisper-small-tajik | 22.25% | 6.14% | 61.93% | 27.93% |
| **whisper-small-tajik-v2** (greedy) | **15.48%** | **5.06%** | **38.55%** | **19.93%** |
| **whisper-small-tajik-v2** (beam 5) | **14.99%** | **4.88%** | **37.88%** | **19.86%** |

- FLEURS test: 599 фраз чтения, эталонная ручная разметка.
- YouTube: 1625 клипов из 157 видео, отложенных до обучения (из
  [Peacockery/tajik-asr-corpus-v3](https://huggingface.co/datasets/Peacockery/tajik-asr-corpus-v3)).
  Разметка там машинная, поэтому эти цифры относительные; абсолютный ориентир — FLEURS.

Пример (живая речь):

| | |
|---|---|
| эталон | ва оҳиста оҳиста ин мавзӯҳо боз хоҳем гашт агар як нукта ҳам бошад… |
| старая модель | ва оиста иста ин мафзуҳо бос бархоем гашт ве як нуқтан бошад… |
| **v2** | Ва оҳиста-оҳиста ин мавзӯҳо боз бархоҳем гашт. Агар як нуқта ҳам бошад, … |

## Использование

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="./whisper-small-tajik-v2")
print(asr("audio.wav", generate_kwargs={"language": "tajik", "task": "transcribe"})["text"])
```

Для записей длиннее 30 секунд добавьте `chunk_length_s=30` в `pipeline(...)`.

## Обучение

- **Данные (161.9 ч):**
  - 153 ч из [Peacockery/tajik-asr-youtube](https://huggingface.co/datasets/Peacockery/tajik-asr-youtube)
    (CC-BY-4.0, 41 таджикский YouTube-канал, машинная разметка ElevenLabs Scribe).
    Взяты только клипы, где два независимых прохода разметки расходятся не больше чем на 6% по буквам;
    1–30 с; не больше 40 ч с одного канала; видео из тестового набора исключены.
  - FLEURS tg_tj train с оригинальной пунктуацией.
- **Чистка текста:** Unicode NFC; буквы из других алфавитов заменены таджикскими (һ→ҳ, ў→ӯ, ќ→қ, ӌ→ҷ,
  латинские двойники → кириллица).
- **Параметры:** 3 эпохи, 2718 шагов по 32 клипа, AdamW lr 1e-5, косинусное расписание, 500 шагов разогрева,
  SpecAugment, без dropout, bf16, gradient checkpointing. Apple M5 Pro, ~8.5 ч.
- dev loss (FLEURS dev): 0.703 → 0.176.

## Ограничения

- Живая разговорная речь, диалекты, смесь с русским языком и имена собственные ещё дают много ошибок (~38% WER).
- Пунктуация выучена из машинной разметки и не всегда совпадает с нормами орфографии.
- Модели Whisper могут «галлюцинировать» текст на тишине и музыке.

## Лицензии и благодарности

- Модель: Apache-2.0 (как у базовой модели).
- Базовая модель: [abduaziz/whisper-small-tajik](https://huggingface.co/abduaziz/whisper-small-tajik) ← [openai/whisper-small](https://huggingface.co/openai/whisper-small).
- Данные: [Peacockery/tajik-asr-youtube](https://huggingface.co/datasets/Peacockery/tajik-asr-youtube) (CC-BY-4.0),
  [Google FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0).
