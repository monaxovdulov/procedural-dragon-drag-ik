# Task 2: Runtime-ползунки (sliders) для настройки параметров

## Цель

Добавить простой UI-панель с ползунками, чтобы в рантайме менять параметры `Dragon` мышью, без перезапуска.

## Файлы

- `main.py`

## Ограничения (важно)

- Параметры типа `N` (число позвонков), `L` (длина сегмента), `leg_pairs` безопаснее менять только через `dragon.reset()`, потому что требуется пересоздание `points`/лап.
- Ползунки должны "съедать" события мыши, чтобы не ломать остальное управление.

## План реализации

1. Добавить класс `Slider` (рядом с helper-функциями).
2. Сделать толщину лап параметром `Dragon.leg_thickness` и использовать её в `Dragon._draw_legs`.
3. В `Dragon.update()` обновлять длины костей у объектов `Leg` каждый кадр (чтобы ползунки влияли на IK).
4. В `main()` создать список слайдеров, `apply_sliders()` и флаг `ui_visible`.
5. В event loop:
   - `TAB` включает/выключает UI.
   - UI получает шанс обработать событие и при успехе делаем `continue`.
6. Каждый кадр: `apply_sliders(dragon)` перед `dragon.update(...)`, и отрисовать панель после `dragon.draw(...)`.

## Готовый код

### 1) Класс `Slider`

Добавить в `main.py` (после helper-функций, до классов `Leg`/`Dragon`):

```python
class Slider:
    def __init__(
        self,
        label: str,
        attr: str,
        x: int, y: int, w: int,
        vmin: float, vmax: float,
        value: float,
        fmt: str = "{:.2f}",
        is_int: bool = False,
    ):
        self.label = label
        self.attr = attr
        self.x, self.y, self.w = x, y, w
        self.vmin, self.vmax = float(vmin), float(vmax)
        self.value = float(value)
        self.fmt = fmt
        self.is_int = is_int

        self.h = 18
        self.dragging = False

    def _t_from_value(self) -> float:
        if self.vmax <= self.vmin:
            return 0.0
        return (self.value - self.vmin) / (self.vmax - self.vmin)

    def _value_from_t(self, t: float) -> float:
        t = clamp(t, 0.0, 1.0)
        v = self.vmin + (self.vmax - self.vmin) * t
        if self.is_int:
            return float(int(round(v)))
        return v

    def _knob_x(self) -> int:
        t = self._t_from_value()
        return int(self.x + t * self.w)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Возвращает True, если событие обработано UI."""
        mx, my = pygame.mouse.get_pos()

        track_rect = pygame.Rect(self.x, self.y + 10, self.w, 4)
        knob_x = self._knob_x()
        knob_rect = pygame.Rect(knob_x - 7, self.y + 4, 14, 16)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if knob_rect.collidepoint(mx, my) or track_rect.collidepoint(mx, my):
                self.dragging = True
                t = (mx - self.x) / float(self.w)
                self.value = self._value_from_t(t)
                return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True

        if event.type == pygame.MOUSEMOTION and self.dragging:
            t = (mx - self.x) / float(self.w)
            self.value = self._value_from_t(t)
            return True

        return False

    def draw(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        color_fg=(190, 190, 210),
        color_bg=(70, 70, 90),
    ):
        val_txt = self.fmt.format(self.value)
        text = font.render(f"{self.label}: {val_txt}", True, color_fg)
        screen.blit(text, (self.x, self.y - 2))

        pygame.draw.rect(screen, color_bg, (self.x, self.y + 10, self.w, 4), border_radius=2)

        kx = self._knob_x()
        pygame.draw.circle(screen, color_fg, (kx, self.y + 12), 7, 1)
```

### 2) Толщина лап как параметр `Dragon`

В `Dragon.__init__` добавить:

```python
self.leg_thickness = 5.0
```

В `Dragon._draw_legs` заменить расчёт `w1/w2` на:

```python
base = max(1, int(self.leg_thickness))
u = leg.attach_idx / (self.N - 1)
w1 = max(2, int(base * (1.0 - 0.45 * u)))
w2 = max(2, int((base - 1) * (1.0 - 0.55 * u)))
```

### 3) Обновление длин костей у `Leg` в рантайме

В `Dragon.update()` перед `leg.update(...)`:

```python
leg.upper_len = self.leg_upper
leg.lower_len = self.leg_lower
```

### 4) Слайдеры в `main()`

После `dragon = Dragon(...)`:

```python
ui_font = pygame.font.SysFont("consolas", 14)
ui_visible = True

panel_x = 12
panel_y = 36
panel_w = 260
row = 28

slider_specs = [
    ("head_spring",   "head_spring",   0.0,   60.0,  "{:.1f}", False),
    ("head_damping",  "head_damping",  0.0,   20.0,  "{:.1f}", False),
    ("head_max_spd",  "head_max_speed",100.0, 1600.0,"{:.0f}", False),

    ("wave_amp",      "wave_amp",      0.0,   16.0,  "{:.1f}", False),
    ("wave_freq",     "wave_freq",     0.0,   1.20,  "{:.2f}", False),
    ("wave_speed",    "wave_speed",    0.0,   8.0,   "{:.2f}", False),

    ("leg_upper",     "leg_upper",     10.0,  90.0,  "{:.0f}", False),
    ("leg_lower",     "leg_lower",     10.0,  90.0,  "{:.0f}", False),
    ("leg_spread",    "leg_spread",    0.0,   90.0,  "{:.0f}", False),
    ("leg_back",      "leg_back",      0.0,   60.0,  "{:.0f}", False),
    ("hip_offset",    "leg_hip_offset",0.0,   30.0,  "{:.0f}", False),

    ("step_thresh",   "step_threshold",5.0,   120.0, "{:.0f}", False),
    ("step_dur",      "step_duration", 0.05,  0.70,  "{:.2f}", False),
    ("step_lift",     "step_lift",     0.0,   40.0,  "{:.0f}", False),

    ("leg_thick",     "leg_thickness", 1.0,   10.0,  "{:.0f}", False),
]

sliders: list[Slider] = []
for i, (label, attr, vmin, vmax, fmt, is_int) in enumerate(slider_specs):
    y = panel_y + i * row
    sliders.append(
        Slider(
            label=label,
            attr=attr,
            x=panel_x, y=y, w=panel_w,
            vmin=vmin, vmax=vmax,
            value=float(getattr(dragon, attr)),
            fmt=fmt,
            is_int=is_int,
        )
    )

def apply_sliders(dr: Dragon):
    for s in sliders:
        setattr(dr, s.attr, s.value)
```

### 5) Event loop: `TAB` + "UI съедает события"

В `for event in pygame.event.get():` добавить:

```python
ui_consumed = False
if ui_visible:
    for s in sliders:
        if s.handle_event(event):
            ui_consumed = True
            break
if ui_consumed:
    continue
```

И в обработке `KEYDOWN`:

```python
if event.key == pygame.K_TAB:
    ui_visible = not ui_visible
```

### 6) Каждый кадр: применить и нарисовать UI

Перед `dragon.update(dt, mouse_pos)`:

```python
apply_sliders(dragon)
dragon.update(dt, mouse_pos)
```

После `dragon.draw(screen)`:

```python
if ui_visible:
    h = len(sliders) * row + 8
    pygame.draw.rect(screen, (20, 20, 30), (panel_x - 8, panel_y - 10, panel_w + 16, h), border_radius=8)
    pygame.draw.rect(screen, (60, 60, 80), (panel_x - 8, panel_y - 10, panel_w + 16, h), 1, border_radius=8)

    for s in sliders:
        s.draw(screen, ui_font)

    hint = ui_font.render("TAB: UI on/off | drag sliders", True, (150, 150, 170))
    screen.blit(hint, (panel_x - 4, panel_y - 28))
```

## Критерии готовности

- `TAB` показывает/скрывает панель.
- Перетаскивание ползунков меняет поведение дракона в рантайме без перезапуска.
- Клики/drag по UI не ломают остальные горячие клавиши.

## Примечание про `task3.md` (пресеты)

Если одновременно реализованы ползунки (эта задача) и пресеты (Task 3), нужно явно решить,
кто источник правды: либо пресеты обновляют `sliders[*].value`, либо UI не применяет слайдеры в кадр переключения
и синхронизируется с `dragon` после применения пресета.
