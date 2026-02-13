# Task 3: Пресеты ("скины") параметров

## Цель

Добавить набор пресетов параметров (разные "ощущения" дракона) и способ быстро переключать их клавишами во время работы.

## Файлы

- `main.py`

## Термины

- **Soft preset**: меняет только "безопасные" параметры, можно применять на лету.
- **Hard preset**: меняет структурные параметры (`N`, `L`, `leg_pairs`, и т.п.), после применения нужен `dragon.reset()`.

## План реализации

1. Ввести `PRESETS: dict[str, dict]` с именованными наборами параметров.
2. Ввести `HARD_KEYS` и функцию `apply_preset(dragon, preset)`.
3. Повесить `F1..F7` (или другие) в обработке `KEYDOWN` в `main()` на `apply_preset`.
4. Если реализованы ползунки из `task2.md`, решить проблему "двух источников правды":
   - Вариант A (предпочтительный): пресет обновляет `sliders[*].value`, после чего `apply_sliders()` применит значения к `dragon`.
   - Вариант B: пресет меняет `dragon`, а UI синхронизируется из `dragon` (иначе в следующем кадре `apply_sliders()` перезатрёт значения).

## Данные пресетов (готовые значения)

### Soft: “Silky Noodle” (тонкий змей, плавный)

```python
{
  "head_spring": 18.0, "head_damping": 9.5, "head_max_speed": 850.0,
  "wave_amp": 7.0, "wave_freq": 0.42, "wave_speed": 3.0,
  "leg_upper": 32.0, "leg_lower": 26.0, "leg_spread": 40.0, "leg_back": 16.0, "leg_hip_offset": 12.0,
  "step_threshold": 60.0, "step_duration": 0.30, "step_lift": 18.0,
  "leg_thickness": 4.0,
}
```

### Soft: “Imperial” (толстый/весомый, лапы мощные, шаги редкие)

```python
{
  "head_spring": 14.0, "head_damping": 11.5, "head_max_speed": 650.0,
  "wave_amp": 3.8, "wave_freq": 0.26, "wave_speed": 1.8,
  "leg_upper": 46.0, "leg_lower": 38.0, "leg_spread": 56.0, "leg_back": 24.0, "leg_hip_offset": 14.0,
  "step_threshold": 85.0, "step_duration": 0.42, "step_lift": 20.0,
  "leg_thickness": 7.0,
}
```

### Soft: “Agile” (быстрый/нервный, но контролируемый)

```python
{
  "head_spring": 32.0, "head_damping": 6.5, "head_max_speed": 1400.0,
  "wave_amp": 6.0, "wave_freq": 0.50, "wave_speed": 4.6,
  "leg_upper": 36.0, "leg_lower": 30.0, "leg_spread": 44.0, "leg_back": 18.0, "leg_hip_offset": 12.0,
  "step_threshold": 48.0, "step_duration": 0.22, "step_lift": 22.0,
  "leg_thickness": 5.0,
}
```

### Soft: “Heavy Beast” (очень тяжёлый, вязкий)

```python
{
  "head_spring": 10.0, "head_damping": 13.0, "head_max_speed": 520.0,
  "wave_amp": 2.8, "wave_freq": 0.22, "wave_speed": 1.2,
  "leg_upper": 52.0, "leg_lower": 44.0, "leg_spread": 60.0, "leg_back": 26.0, "leg_hip_offset": 16.0,
  "step_threshold": 100.0, "step_duration": 0.55, "step_lift": 16.0,
  "leg_thickness": 8.0,
}
```

### Soft: “Cartoon” (сверхвыразительный, высокие шаги)

```python
{
  "head_spring": 26.0, "head_damping": 7.5, "head_max_speed": 1200.0,
  "wave_amp": 10.5, "wave_freq": 0.62, "wave_speed": 5.4,
  "leg_upper": 40.0, "leg_lower": 34.0, "leg_spread": 52.0, "leg_back": 20.0, "leg_hip_offset": 13.0,
  "step_threshold": 55.0, "step_duration": 0.26, "step_lift": 28.0,
  "leg_thickness": 6.0,
}
```

### Hard: “Long Cloud Dragon” (очень длинный, плавный)

```python
{
  "N": 160, "L": 6.2,
  "radius_head": 20.0, "radius_tail": 3.0,
  "head_spring": 16.0, "head_damping": 10.0, "head_max_speed": 900.0,
  "wave_amp": 6.5, "wave_freq": 0.33, "wave_speed": 2.4,
}
```

### Hard: “Centipede” (многоножка)

```python
{
  "leg_pairs": 12,
  "leg_upper": 28.0, "leg_lower": 24.0, "leg_spread": 36.0, "leg_back": 14.0, "leg_hip_offset": 10.0,
  "step_threshold": 40.0, "step_duration": 0.20, "step_lift": 14.0,
  "leg_thickness": 3.0,
}
```

## Готовый каркас кода (применение пресета)

Добавить в `main.py`:

```python
PRESETS = {
    "noodle": {...},
    "imperial": {...},
    "agile": {...},
    "heavy": {...},
    "cartoon": {...},
    "cloud_long": {...},   # hard
    "centipede": {...},    # hard
}

HARD_KEYS = {
    "N", "L", "leg_pairs",
    "radius_head", "radius_tail",
    "rib_every", "rib_samples", "rib_arc", "rib_t_scale", "rib_width",
}

def apply_preset(dragon, preset: dict):
    hard = any(k in preset for k in HARD_KEYS)
    for k, v in preset.items():
        if hasattr(dragon, k):
            setattr(dragon, k, v)
    if hard:
        dragon.reset()
```

Рекомендуемые бинды:

- `F1` `noodle`
- `F2` `imperial`
- `F3` `agile`
- `F4` `heavy`
- `F5` `cartoon`
- `F6` `centipede`
- `F7` `cloud_long`

## Критерии готовности

- Переключение пресетов заметно меняет поведение/внешний вид.
- Hard пресеты не ломают приложение (после `dragon.reset()` всё стабильно).
- При наличии UI-ползунков пресеты не "откатываются" в следующий кадр из-за `apply_sliders()`.
