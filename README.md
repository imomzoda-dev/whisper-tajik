# Whisper Tajik: дообучение

## Скачать модель
**[whisper-small-tajik-v2.zip](https://github.com/imomzoda-dev/whisper-tajik/releases/download/v2.0/whisper-small-tajik-v2.zip)**
(446 МБ, fp16; SHA-256 `338b6332bde866ed6770cb2570a71aa3ffb50e4948d45e20a03fe045b97cfdb2`).
Описание модели: [release/whisper-small-tajik-v2/README.md](release/whisper-small-tajik-v2/README.md).

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="./whisper-small-tajik-v2", chunk_length_s=30)
print(asr("audio.wav", generate_kwargs={"language": "tajik", "task": "transcribe"})["text"])
```

### VoiceFlow AI (whisper.cpp)
Для каждого опубликованного релиза GitHub Actions ([voiceflow-ggml.yml](.github/workflows/voiceflow-ggml.yml))
сам конвертирует модель в ggml, квантует (q5_1), проверяет её в whisper.cpp и добавляет в релиз
`ggml-tajik-small-q5_1.bin` и `voiceflow-model.json` (версия, размер, SHA-256).
VoiceFlow AI читает `releases/latest/download/voiceflow-model.json` и обновляет таджикскую модель сам.

Новый релиз: загрузите `whisper-small-tajik-*.zip` в релиз и опубликуйте его — остальное произойдёт автоматически
(~15 мин). Если модель меняет поведение (например, перестаёт ставить пунктуацию), поправьте
[.github/voiceflow.json](.github/voiceflow.json) до публикации.

Дообучаем [abduaziz/whisper-small-tajik](https://huggingface.co/abduaziz/whisper-small-tajik)
(Whisper Small, обучен только на FLEURS, ~8 ч) на большом наборе чистой таджикской речи
**с заглавными буквами и пунктуацией**.

## Данные
- **Peacockery/tajik-asr-youtube** (CC-BY-4.0): 1582 ч YouTube-речи (41 канал), машинная разметка ElevenLabs Scribe
  с оценкой согласия двух проходов. Берём только клипы с расхождением ≤ 6% (`scribe_cer`), 1–30 с,
  не больше 40 ч с одного канала → ~153 ч из 33 файлов (50 ГБ).
- **FLEURS tg_tj** train/dev: аудио из корпуса v3 + оригинальный текст с пунктуацией из google/fleurs.
- **Тест** (в обучение не попадает): FLEURS test (599 фраз, золотая разметка) и 1625 клипов из 157 отложенных
  YouTube-видео (Peacockery/tajik-asr-corpus-v3). Видео, которые есть в тесте, исключены из обучения по совпадению текста.

## Скрипты
| файл | что делает |
|---|---|
| `scripts/tajik_text.py` | чистка текста (һ→ҳ, ў→ӯ, ќ→қ, латиница-двойники и т.п.) и нормализация для WER |
| `scripts/scan_meta.py` | читает метаданные всех клипов без аудио |
| `scripts/prepare_data.py` | `select` / `fetch` / `fleurs`: отбор и загрузка клипов |
| `scripts/train.py` | обучение на Mac (MPS, bf16, gradient checkpointing) или CUDA |
| `scripts/evaluate.py` | WER/CER на тестах, предсказания в `results/` |
| `run_all.sh` | весь конвейер: загрузка → обучение → оценка (можно перезапускать, продолжит с места) |

## Управление
```bash
./tajik.sh start    # включить (продолжает с последнего сохранения)
./tajik.sh stop     # выключить: сохраняет прогресс, освобождает Mac
./tajik.sh pause    # быстрая пауза (память остаётся занятой)
./tajik.sh resume   # продолжить после паузы
./tajik.sh status   # шаг, loss, температура
./tajik.sh log      # журнал в реальном времени (Ctrl+C — выход)
```
Во время обучения работает `scripts/thermal_guard.py`: пауза при GPU/CPU ≥ 95°C или «serious»
от macOS, продолжение при ≤ 82°C. Журнал: `logs_thermal.txt`.

Важно: dropout должен быть 0. Модель обучена без него, и dropout 0.1 поднимает train loss с ~0.3 до ~7.

## Результаты (WER, ниже = лучше; строчные буквы, без пунктуации)
| модель | FLEURS test | YouTube (живая речь) |
|---|---|---|
| abduaziz/whisper-small-tajik (исходная) | 22.25% | 61.93% |
| **tajik-small-v2** (greedy) | **15.48%** | **38.55%** |
| **tajik-small-v2** (beam 5) | **14.99%** | **37.88%** |

CER (ошибка по буквам): FLEURS 6.14% → 4.88%, YouTube 27.93% → 19.86%.
Новая модель лучше старой в 1568 из 1625 YouTube-фраз (хуже в 25) и в 398 из 599 FLEURS-фраз (хуже в 77).
Кроме того, она пишет с заглавными буквами и пунктуацией (старая — только строчными без знаков).

Обучение: 161.9 ч (153 ч YouTube с расхождением разметчиков ≤ 6% + FLEURS train), 3 эпохи, 2718 шагов
по 32 клипа, lr 1e-5 cosine, SpecAugment, без dropout, bf16 на M5 Pro (~11 с/шаг, ~8.5 ч).
dev loss 0.703 → 0.176. Максимальная температура за обучение 94°C, пауз по перегреву не было.
Итоговая модель: `models/tajik-small-v2-final`.
