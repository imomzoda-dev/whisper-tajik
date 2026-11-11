#!/bin/zsh
# Управление обучением таджикского Whisper.
#   ./tajik.sh start    включить (продолжает с последнего сохранения)
#   ./tajik.sh stop     выключить: сохраняет прогресс и освобождает Mac
#   ./tajik.sh pause    быстрая пауза (память остаётся занятой)
#   ./tajik.sh resume   продолжить после паузы
#   ./tajik.sh status   что сейчас происходит
#   ./tajik.sh log      смотреть журнал в реальном времени (выход: Ctrl+C)
cd "$(dirname "$0")"

TRAIN_RE='^[^ ]*python[^ ]* -I scripts/(train|evaluate)\.py'
train_pid() { pgrep -f "$TRAIN_RE" | head -1; }
all_pids()  { local p; for p in $(pgrep -f "$TRAIN_RE"); do echo $p; pgrep -P $p; done; }
is_paused() { local p=$(train_pid); [[ -n $p ]] && ps -o stat= -p $p | grep -q T; }

status() {
  echo "────────── Обучение Tajik Whisper ──────────"
  if [[ -n $(train_pid) ]]; then
    if is_paused; then echo "Состояние:   ⏸  НА ПАУЗЕ  (./tajik.sh resume — продолжить)"
    else               echo "Состояние:   ▶  ИДЁТ"; fi
  elif pgrep -f "prepare_data.py fetch" >/dev/null; then echo "Состояние:   ⬇  СКАЧИВАНИЕ ДАННЫХ"
  elif pgrep -f run_all.sh >/dev/null; then echo "Состояние:   …  ЗАПУСКАЕТСЯ"
  elif grep -E "^== |STOPPED" logs_pipeline.txt 2>/dev/null | tail -1 | grep -q "ALL DONE"; then
    echo "Состояние:   ✅ ЗАВЕРШЕНО — обучение и оценка закончены, повторный start ничего не изменит"
    local w=$(grep -E "^(fleurs|youtube): " logs_pipeline.txt | tail -2 | awk '{print $1, $3}' | tr '\n' ' ')
    echo "Результат:   $w  (было: fleurs WER=22.25% youtube WER=61.93%)"
    echo "Модель:      models/tajik-small-v2-final"
    [[ -f logs_thermal.txt ]] && echo "Температура: $(tail -1 logs_thermal.txt | cut -c21-)"
    return
  else echo "Состояние:   ■  ВЫКЛЮЧЕНО  (./tajik.sh start — включить)"; fi
  local s=$(grep -E "^step [0-9]+/" logs_pipeline.txt 2>/dev/null | tail -1)
  [[ -n $s ]] && echo "Прогресс:    $(echo $s | awk '{print "шаг " $2 ", loss " $4 ", " $7 ", осталось ~" substr($9,1,length($9)-1) " ч"}')"
  local b=$(grep -E "dev_loss" logs_pipeline.txt 2>/dev/null | tail -1)
  [[ -n $b ]] && echo "Проверка:    $b"
  grep -E "STOPPED|ALL DONE|Traceback|resumed at" logs_pipeline.txt 2>/dev/null | tail -1 | sed 's/^/Последнее:   /'
  [[ -f logs_thermal.txt ]] && echo "Температура: $(tail -1 logs_thermal.txt | cut -c21-)"
  [[ -d models/tajik-small-v2/best ]] && echo "Лучшая модель: models/tajik-small-v2/best"
}

case "$1" in
  start)
    if [[ -n $(train_pid) ]] || pgrep -f run_all.sh >/dev/null; then
      is_paused && echo "Обучение на паузе. Продолжить: ./tajik.sh resume" || echo "Уже работает."
      exit 0
    fi
    echo "\n== $(date) START (вручную)" >> logs_pipeline.txt
    nohup caffeinate -i ./run_all.sh >> logs_pipeline.txt 2>&1 &!
    echo "▶ Обучение включено. Продолжит с последнего сохранения."
    echo "  Через ~2 минуты проверь: ./tajik.sh status"
    ;;
  stop)
    p=$(train_pid)
    pkill -f run_all.sh 2>/dev/null          # чтобы после остановки не началась оценка
    pkill -f "prepare_data.py fetch" 2>/dev/null
    if [[ -n $p ]]; then
      echo "⏳ Сохраняю прогресс и выключаю (до ~1 минуты)..."
      kill -TERM $p; for q in $(all_pids); do kill -CONT $q 2>/dev/null; done
      for i in {1..120}; do [[ -z $(train_pid) ]] && break; sleep 1; done
      if [[ -n $(train_pid) ]]; then
        echo "Не ответил за 2 минуты — останавливаю принудительно (откат к последнему сохранению)."
        for q in $(all_pids); do kill -KILL $q 2>/dev/null; done
      fi
    fi
    pkill -f thermal_guard.py 2>/dev/null
    pkill -f "caffeinate -i ./run_all.sh" 2>/dev/null
    echo "■ Обучение выключено. $(grep STOPPED logs_pipeline.txt | tail -1)"
    echo "  Включить снова: ./tajik.sh start"
    ;;
  pause)
    [[ -z $(train_pid) ]] && { echo "Обучение не запущено."; exit 0; }
    for q in $(all_pids); do kill -STOP $q; done
    echo "⏸ Пауза. GPU свободен, но память (~9 ГБ) остаётся занятой."
    echo "  Продолжить: ./tajik.sh resume     Полностью выключить: ./tajik.sh stop"
    ;;
  resume)
    [[ -z $(train_pid) ]] && { echo "Обучение не запущено. Включить: ./tajik.sh start"; exit 0; }
    for q in $(all_pids); do kill -CONT $q; done
    echo "▶ Продолжаю."
    ;;
  status) status ;;
  log)    tail -n 20 -f logs_pipeline.txt logs_thermal.txt ;;
  *)
    sed -n '2,8p' "$0" | sed 's/^# //'
    ;;
esac
