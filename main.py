# main.py
# Запуск: uv run main.py (после uv sync)
#
# Управление:
#   ESC  — выход
#   R    — reset (выстроить дракона горизонтально в центре)
#   D    — debug toggle (точки позвоночника, нормали, опоры лап)
#   1    — mode="bones"   (скелет + рёбра)
#   2    — mode="skin"    (лента/обводка тела)
#   3    — mode="hybrid"  (оба)

import math
import random
import pygame
from pygame.math import Vector2


def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x


def smoothstep01(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def safe_normalize(v: Vector2, fallback: Vector2 = Vector2(1, 0)) -> Vector2:
    if v.length_squared() > 1e-12:
        return v.normalize()
    return Vector2(fallback)


def rotate_rad(v: Vector2, angle_rad: float) -> Vector2:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return Vector2(v.x * c - v.y * s, v.x * s + v.y * c)


def v2i(v: Vector2) -> tuple[int, int]:
    return int(v.x), int(v.y)


def solve_two_bone_ik_2d(hip: Vector2, target: Vector2, upper_len: float, lower_len: float, bend_dir: float):
    """
    Аналитический 2-bone IK в 2D:
    Возвращает (knee_pos, foot_pos_clamped).

    bend_dir: +1 или -1 — какую из двух возможных конфигураций треугольника выбрать (в какую сторону "гнуть" колено).
    """
    to_target = target - hip
    d = to_target.length()
    if d < 1e-9:
        # Нечего решать — пусть направление будет фиктивным.
        to_target = Vector2(1.0, 0.0)
        d = 1e-9

    # Треугольник существует только если d в диапазоне [|a-b|, a+b].
    min_d = abs(upper_len - lower_len) + 1e-4
    max_d = (upper_len + lower_len) - 1e-4
    d_clamped = clamp(d, min_d, max_d)

    u = to_target / d  # единичное направление на цель (по старому d)

    # Закон косинусов для угла при бедре:
    # cos(phi) = (a^2 + d^2 - b^2) / (2 a d)
    a = upper_len
    b = lower_len
    cos_phi = (a * a + d_clamped * d_clamped - b * b) / (2.0 * a * d_clamped)
    cos_phi = clamp(cos_phi, -1.0, 1.0)
    phi = math.acos(cos_phi)

    # Верхняя кость: повернём направление на цель на ±phi, чтобы выбрать "локоть" (колено) нужной стороны.
    upper_dir = rotate_rad(u, bend_dir * phi)
    knee = hip + upper_dir * a

    # Нога может быть "недостижима": foot визуально зажмём на окружность достижимости.
    foot = hip + u * d_clamped
    return knee, foot


class Slider:
    def __init__(
        self,
        label: str,
        attr: str,
        x: int,
        y: int,
        w: int,
        vmin: float,
        vmax: float,
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


class Leg:
    def __init__(self, attach_idx: int, side: int, upper_len: float, lower_len: float):
        self.attach_idx = attach_idx
        self.side = side  # -1 / +1

        self.upper_len = upper_len
        self.lower_len = lower_len

        # Липкая точка опоры в мире:
        self.foot_target = Vector2(0, 0)

        # Параметры шага:
        self.stepping = False
        self.step_time = 0.0
        self.step_from = Vector2(0, 0)
        self.step_to = Vector2(0, 0)
        self.step_lift_dir = Vector2(0, -1)

        # Для отрисовки:
        self.hip = Vector2(0, 0)
        self.knee = Vector2(0, 0)
        self.foot = Vector2(0, 0)

    def reset(self, initial_foot: Vector2):
        self.foot_target = Vector2(initial_foot)
        self.stepping = False
        self.step_time = 0.0

    def begin_step(self, hip: Vector2, new_target: Vector2, lift_dir: Vector2):
        self.stepping = True
        self.step_time = 0.0
        self.step_from = Vector2(self.foot_target)
        self.step_to = Vector2(new_target)
        self.step_lift_dir = safe_normalize(lift_dir, Vector2(0, -1))

        # Чуть гарантируем достижимость конечной опоры:
        reach = self.upper_len + self.lower_len - 1e-3
        v = self.step_to - hip
        if v.length_squared() > reach * reach:
            self.step_to = hip + safe_normalize(v) * reach

        min_reach = abs(self.upper_len - self.lower_len) + 1e-3
        if v.length() < min_reach:
            self.step_to = hip + safe_normalize(v) * min_reach

    def update(self, dt: float, hip: Vector2, desired_foothold: Vector2,
               step_threshold: float, step_duration: float, step_lift: float, bend_dir: float):
        self.hip = Vector2(hip)

        if self.stepping:
            self.step_time += dt
            u = 1.0 if step_duration <= 1e-6 else (self.step_time / step_duration)

            if u >= 1.0:
                self.stepping = False
                self.foot_target = Vector2(self.step_to)
            else:
                s = smoothstep01(smoothstep01(u))
                pos = self.step_from * (1.0 - s) + self.step_to * s

                # Классическая "дуга шага": sin(pi*s) даёт 0 в концах и пик в середине.
                lift = math.sin(math.pi * s) * step_lift
                pos += self.step_lift_dir * lift

                self.foot_target = pos
        else:
            # Липкость: пока предполагаемая опора рядом, нога "держится" за старый foot_target.
            if (desired_foothold - self.foot_target).length_squared() > (step_threshold * step_threshold):
                # Направление подъёма: наружу от тела (слева/справа) + немного "вверх" по экрану.
                lift_dir = safe_normalize(desired_foothold - hip, Vector2(0, -1))
                self.begin_step(hip, desired_foothold, lift_dir)

        # 2-bone IK только для позы (визуала)
        self.knee, self.foot = solve_two_bone_ik_2d(self.hip, self.foot_target, self.upper_len, self.lower_len, bend_dir)


class Dragon:
    def __init__(self, screen_size: tuple[int, int]):
        self.w, self.h = screen_size

        # ----------------------------
        # Параметры (в одном месте)
        # ----------------------------
        self.N = 110              # число точек позвоночника
        self.L = 7.5              # длина сегмента

        # Голова: пружина на курсор + демпфирование скорости + лимит скорости
        self.head_spring = 22.0   # "жёсткость" пружины (чем больше, тем сильнее тянется)
        self.head_damping = 8.0   # демпфирование скорости (чем больше, тем суше)
        self.head_max_speed = 900.0  # px/s ограничение скорости

        # Визуал тела
        self.radius_head = 18.0
        self.radius_tail = 4.0

        # Микро-волна (не ломаем длину, т.к. применяем только при отрисовке)
        self.wave_amp = 6.0
        self.wave_freq = 0.33
        self.wave_speed = 2.6
        self.wave_phase_odd = 0.8

        # Рёбра/чешуя
        self.rib_every = 1           # рисовать каждую N-ю точку
        self.rib_samples = 6         # сколько отрезков в дуге
        self.rib_arc = 1.25          # "полуширина" дуги в радианах
        self.rib_width = 1
        self.rib_t_scale = 0.55      # насколько дуга тянется вдоль касательной

        # Лапки
        self.leg_pairs = 7           # пар лап (итого 2*leg_pairs)
        self.leg_upper = 34.0
        self.leg_lower = 28.0
        self.leg_hip_offset = 12.0   # отступ точки "бедра" от позвоночника
        self.leg_spread = 42.0       # куда в среднем тянется опора в сторону
        self.leg_back = 18.0         # куда в среднем тянется опора назад (к хвосту)
        self.step_threshold = 55.0
        self.step_duration = 0.28
        self.step_lift = 20.0
        self.leg_thickness = 5.0

        # Рендер/UX
        self.mode = "hybrid"         # "bones" | "skin" | "hybrid"
        self.debug = False
        self.time = 0.0

        # Состояние
        self.points: list[Vector2] = []
        self.head_vel = Vector2(0, 0)
        self.legs: list[Leg] = []

        self.reset()

    def reset(self):
        center = Vector2(self.w * 0.5, self.h * 0.5)
        self.points = []
        self.head_vel = Vector2(0, 0)
        self.time = 0.0

        # Выстраиваем позвоночник горизонтально: голова справа, хвост влево.
        for i in range(self.N):
            self.points.append(Vector2(center.x - i * self.L, center.y))

        # Создаём лапки (пары, лев/прав на одном индексе позвоночника)
        self.legs = []
        start_i = int(self.N * 0.18)
        end_i = int(self.N * 0.66)
        if self.leg_pairs < 1:
            self.leg_pairs = 1

        for j in range(self.leg_pairs):
            t = 0.0 if self.leg_pairs == 1 else (j / (self.leg_pairs - 1))
            attach_idx = int(start_i + (end_i - start_i) * t)

            # Лево/право (side = +1 / -1)
            for side in (+1, -1):
                leg = Leg(attach_idx=attach_idx, side=side, upper_len=self.leg_upper, lower_len=self.leg_lower)

                # Инициализируем опору в разумной позиции относительно тела.
                # В начальной позе касательная ~ (1, 0), нормаль ~ (0, 1).
                hip = self.points[attach_idx] + Vector2(0, 1) * (side * self.leg_hip_offset)
                foot = hip + Vector2(0, 1) * (side * self.leg_spread) + Vector2(-1, 0) * self.leg_back
                foot += Vector2(random.uniform(-6, 6), random.uniform(-6, 6))
                leg.reset(foot)

                self.legs.append(leg)

    def _radius_at(self, i: int) -> float:
        u = i / (self.N - 1)
        # лёгкая нелинейность, чтобы хвост быстрее сужался
        u2 = u * u
        return self.radius_head * (1.0 - u2) + self.radius_tail * u2

    def _frame_at(self, i: int):
        """
        Возвращает (tangent, normal) для точки i.
        Требование для рёбер: t = normalize(p[i-1] - p[i]); n = (-t_y, t_x).
        """
        if i <= 0:
            v = self.points[0] - self.points[1]
        else:
            v = self.points[i - 1] - self.points[i]

        t = safe_normalize(v, Vector2(1, 0))
        n = Vector2(-t.y, t.x)
        return t, n

    def update(self, dt: float, mouse_pos: tuple[int, int]):
        dt = clamp(dt, 0.0, 0.05)  # защита от огромного dt при alt-tab

        self.time += dt

        # ----------------------------
        # 1) Голова: пружина на курсор
        # ----------------------------
        head = self.points[0]
        target = Vector2(mouse_pos)

        to_target = target - head

        # "Пружина": ускорение пропорционально ошибке.
        accel = to_target * self.head_spring

        self.head_vel += accel * dt

        # Демпфирование скорости (экспоненциальное — стабильно по dt).
        self.head_vel *= math.exp(-self.head_damping * dt)

        # Ограничение скорости головы (без телепортов и чрезмерных рывков).
        spd = self.head_vel.length()
        if spd > self.head_max_speed:
            self.head_vel.scale_to_length(self.head_max_speed)

        head = head + self.head_vel * dt
        self.points[0] = head

        # ------------------------------------------------------------
        # 2) Drag IK (Follow-the-Leader) для позвоночника
        # ------------------------------------------------------------
        #
        # Для каждого сегмента i (от головы к хвосту) мы хотим:
        #   - оставить длину сегмента постоянной (L)
        #   - добиться эффекта "хвост догоняет голову"
        #
        # Почему dx = x_i - x_prev, dy = y_i - y_prev:
        #   Это компоненты вектора ОТ предыдущей точки (prev) К текущей точке (i)
        #   (причём текущая точка берётся в её СТАРОМ положении, т.е. до перетягивания).
        #   Такой вектор задаёт "куда сегмент смотрел" относительно обновлённого prev.
        #
        # Зачем atan2(dy, dx):
        #   atan2 даёт угол направления этого вектора с корректными квадрантами,
        #   в отличие от atan(dy/dx), который ломается на знаках и dx=0.
        #
        # Как cos/sin восстанавливают смещение по углу:
        #   (cos(angle), sin(angle)) — единичный вектор в направлении angle.
        #   Умножая на L, мы получаем смещение фиксированной длины.
        #
        # Почему порядок (i=1..N-1) тянет хвост за головой:
        #   Мы обновляем prev (p[i-1]) раньше, затем строим p[i] от него.
        #   Направление при этом берём на СТАРОЕ p[i], поэтому p[i] "не перепрыгивает"
        #   в произвольную конфигурацию, а естественно запаздывает, давая инерцию хвоста.
        #
        for i in range(1, self.N):
            prev = self.points[i - 1]
            cur = self.points[i]

            dx = cur.x - prev.x
            dy = cur.y - prev.y
            ang = math.atan2(dy, dx)

            cur.x = prev.x + math.cos(ang) * self.L
            cur.y = prev.y + math.sin(ang) * self.L
            self.points[i] = cur

        # ----------------------------
        # 3) Лапки: липкость + шаги
        # ----------------------------
        for leg in self.legs:
            idx = leg.attach_idx
            t, n = self._frame_at(idx)

            # Точка бедра (hip) сидит сбоку от позвоночника.
            hip = self.points[idx] + n * (leg.side * self.leg_hip_offset)

            # "Желаемая" опора: в сторону (spread) и немного в хвост (back).
            # Чуть добавим колебание (но это только желаемая точка; липкость удержит реальную).
            u = idx / (self.N - 1)
            side_phase = 1.35 if leg.side < 0 else 0.0
            wave = math.sin(self.time * (self.wave_speed * 0.8) - idx * 0.18 + side_phase) * (1.0 - u) * 4.0
            desired = hip + n * (leg.side * (self.leg_spread + wave)) + (-t) * (self.leg_back + 6.0 * (1.0 - u))

            # Bend dir: выбираем сторону сгиба колена (наружу).
            bend_dir = float(leg.side)

            leg.upper_len = self.leg_upper
            leg.lower_len = self.leg_lower
            leg.update(
                dt=dt,
                hip=hip,
                desired_foothold=desired,
                step_threshold=self.step_threshold,
                step_duration=self.step_duration,
                step_lift=self.step_lift,
                bend_dir=bend_dir
            )

    def draw(self, screen: pygame.Surface):
        # Цвета (светлые линии на чёрном фоне)
        col_bone = (210, 210, 235)
        col_rib = (170, 170, 205)
        col_skin = (200, 200, 230)
        col_leg = (200, 200, 235)
        col_debug = (120, 220, 140)
        col_debug2 = (220, 160, 120)

        if self.mode in ("skin", "hybrid"):
            self._draw_skin(screen, col_skin)

        if self.mode in ("bones", "hybrid"):
            self._draw_bones(screen, col_bone)
            self._draw_ribs(screen, col_rib)

        self._draw_legs(screen, col_leg, col_debug2 if self.debug else None)

        self._draw_head(screen, col_bone)

        if self.debug:
            self._draw_debug(screen, col_debug, col_debug2)

    def _draw_bones(self, screen: pygame.Surface, color):
        pts = [v2i(p) for p in self.points]
        pygame.draw.lines(screen, color, False, pts, 1)

    def _draw_skin(self, screen: pygame.Surface, color):
        left_pts = []
        right_pts = []

        for i in range(self.N):
            t, n = self._frame_at(i)
            r = self._radius_at(i)

            # Микроволна только в визуал (не меняем self.points)
            u = i / (self.N - 1)
            amp = self.wave_amp * (1.0 - u) ** 1.2
            phase = self.wave_phase_odd if (i & 1) else 0.0
            w = math.sin(self.time * self.wave_speed - i * self.wave_freq + phase) * amp

            # Чуть "дышим" толщиной и смещаем край по нормали
            rr = r * (1.0 + 0.04 * math.sin(self.time * 1.7 + i * 0.11))
            p = self.points[i] + n * (0.15 * w)

            left_pts.append(v2i(p + n * rr))
            right_pts.append(v2i(p - n * rr))

        poly = left_pts + right_pts[::-1]
        if len(poly) >= 3:
            pygame.draw.polygon(screen, color, poly, 1)

    def _draw_ribs(self, screen: pygame.Surface, color):
        # Рисуем "рёбра/чешую" как дуги в локальном базисе (t, n).
        # Важно: дуги зависят от ориентации сегмента (через t и n), иначе эффекта "скелета" не будет.
        for i in range(2, self.N - 1, self.rib_every):
            t, n = self._frame_at(i)
            u = i / (self.N - 1)

            base_r = self._radius_at(i)
            amp = self.wave_amp * (1.0 - u) ** 1.1
            phase = self.wave_phase_odd if (i & 1) else 0.0
            w = math.sin(self.time * self.wave_speed - i * self.wave_freq + phase) * amp

            # Центр ребра чуть “гуляет” по нормали (только визуал)
            center = self.points[i] + n * (0.35 * w)

            # Радиус дуги под тейперовку
            rib_r = base_r * (0.85 + 0.25 * (1.0 - u))

            # Дуга: параметризуем в локальной системе:
            # p(theta) = center + n*(side * (rib_r*(1 + 0.15*cos(theta))))
            #                 + t*(sin(theta) * rib_r * rib_t_scale)
            # theta идёт от -rib_arc до +rib_arc.
            for side in (+1, -1):
                pts = []
                for k in range(self.rib_samples + 1):
                    a = (-self.rib_arc) + (2.0 * self.rib_arc) * (k / self.rib_samples)
                    out = rib_r * (1.0 + 0.15 * math.cos(a))
                    along = math.sin(a) * rib_r * self.rib_t_scale
                    p = center + n * (side * out) + t * along
                    pts.append(v2i(p))

                pygame.draw.lines(screen, color, False, pts, self.rib_width)

    def _draw_legs(self, screen: pygame.Surface, color, debug_color=None):
        for leg in self.legs:
            hip = leg.hip
            knee = leg.knee
            foot = leg.foot

            # Небольшая тейперовка толщины лап в зависимости от положения по телу
            base = max(1, int(self.leg_thickness))
            u = leg.attach_idx / (self.N - 1)
            w1 = max(2, int(base * (1.0 - 0.45 * u)))
            w2 = max(2, int((base - 1) * (1.0 - 0.55 * u)))

            pygame.draw.line(screen, color, v2i(hip), v2i(knee), w1)
            pygame.draw.line(screen, color, v2i(knee), v2i(foot), w2)

            # "Коготки" — три коротких штриха вокруг стопы
            to_foot = safe_normalize(foot - knee, Vector2(1, 0))
            n = Vector2(-to_foot.y, to_foot.x)
            claw_len = 6.0 * (1.0 - 0.4 * u)
            for s in (-1, 0, +1):
                tip = foot + to_foot * claw_len + n * (s * 2.2)
                pygame.draw.line(screen, color, v2i(foot), v2i(tip), 1)

            if debug_color is not None:
                pygame.draw.circle(screen, debug_color, v2i(leg.foot_target), 3, 1)

    def _draw_head(self, screen: pygame.Surface, color):
        head = self.points[0]
        # Направление первой кости
        v = self.points[0] - self.points[1]
        t = safe_normalize(v, Vector2(1, 0))
        n = Vector2(-t.y, t.x)

        r = self.radius_head * 0.55
        pygame.draw.circle(screen, color, v2i(head), int(r), 1)

        # "Рога" — две V-ветки
        horn_base = head + t * (r * 0.6)
        horn1 = horn_base + t * (r * 0.9) + n * (r * 0.7)
        horn2 = horn_base + t * (r * 0.9) - n * (r * 0.7)
        pygame.draw.line(screen, color, v2i(horn_base), v2i(horn1), 1)
        pygame.draw.line(screen, color, v2i(horn_base), v2i(horn2), 1)

    def _draw_debug(self, screen: pygame.Surface, color, color2):
        # Точки позвоночника и локальные нормали
        for i in range(0, self.N, 6):
            p = self.points[i]
            pygame.draw.circle(screen, color, v2i(p), 2, 1)
            t, n = self._frame_at(i)
            pygame.draw.line(screen, color, v2i(p), v2i(p + n * 18), 1)

        # Индексы привязки лап
        for leg in self.legs:
            p = self.points[leg.attach_idx]
            pygame.draw.circle(screen, color2, v2i(p), 4, 1)


PRESETS: dict[str, dict[str, float | int]] = {
    "noodle": {
        "head_spring": 18.0,
        "head_damping": 9.5,
        "head_max_speed": 850.0,
        "wave_amp": 7.0,
        "wave_freq": 0.42,
        "wave_speed": 3.0,
        "leg_upper": 32.0,
        "leg_lower": 26.0,
        "leg_spread": 40.0,
        "leg_back": 16.0,
        "leg_hip_offset": 12.0,
        "step_threshold": 60.0,
        "step_duration": 0.30,
        "step_lift": 18.0,
        "leg_thickness": 4.0,
    },
    "imperial": {
        "head_spring": 14.0,
        "head_damping": 11.5,
        "head_max_speed": 650.0,
        "wave_amp": 3.8,
        "wave_freq": 0.26,
        "wave_speed": 1.8,
        "leg_upper": 46.0,
        "leg_lower": 38.0,
        "leg_spread": 56.0,
        "leg_back": 24.0,
        "leg_hip_offset": 14.0,
        "step_threshold": 85.0,
        "step_duration": 0.42,
        "step_lift": 20.0,
        "leg_thickness": 7.0,
    },
    "agile": {
        "head_spring": 32.0,
        "head_damping": 6.5,
        "head_max_speed": 1400.0,
        "wave_amp": 6.0,
        "wave_freq": 0.50,
        "wave_speed": 4.6,
        "leg_upper": 36.0,
        "leg_lower": 30.0,
        "leg_spread": 44.0,
        "leg_back": 18.0,
        "leg_hip_offset": 12.0,
        "step_threshold": 48.0,
        "step_duration": 0.22,
        "step_lift": 22.0,
        "leg_thickness": 5.0,
    },
    "heavy": {
        "head_spring": 10.0,
        "head_damping": 13.0,
        "head_max_speed": 520.0,
        "wave_amp": 2.8,
        "wave_freq": 0.22,
        "wave_speed": 1.2,
        "leg_upper": 52.0,
        "leg_lower": 44.0,
        "leg_spread": 60.0,
        "leg_back": 26.0,
        "leg_hip_offset": 16.0,
        "step_threshold": 100.0,
        "step_duration": 0.55,
        "step_lift": 16.0,
        "leg_thickness": 8.0,
    },
    "cartoon": {
        "head_spring": 26.0,
        "head_damping": 7.5,
        "head_max_speed": 1200.0,
        "wave_amp": 10.5,
        "wave_freq": 0.62,
        "wave_speed": 5.4,
        "leg_upper": 40.0,
        "leg_lower": 34.0,
        "leg_spread": 52.0,
        "leg_back": 20.0,
        "leg_hip_offset": 13.0,
        "step_threshold": 55.0,
        "step_duration": 0.26,
        "step_lift": 28.0,
        "leg_thickness": 6.0,
    },
    "cloud_long": {
        "N": 160,
        "L": 6.2,
        "radius_head": 20.0,
        "radius_tail": 3.0,
        "head_spring": 16.0,
        "head_damping": 10.0,
        "head_max_speed": 900.0,
        "wave_amp": 6.5,
        "wave_freq": 0.33,
        "wave_speed": 2.4,
    },
    "centipede": {
        "leg_pairs": 12,
        "leg_upper": 28.0,
        "leg_lower": 24.0,
        "leg_spread": 36.0,
        "leg_back": 14.0,
        "leg_hip_offset": 10.0,
        "step_threshold": 40.0,
        "step_duration": 0.20,
        "step_lift": 14.0,
        "leg_thickness": 3.0,
    },
}

HARD_KEYS = {
    "N",
    "L",
    "leg_pairs",
    "radius_head",
    "radius_tail",
    "rib_every",
    "rib_samples",
    "rib_arc",
    "rib_t_scale",
    "rib_width",
}


def apply_preset(dragon: Dragon, preset: dict[str, float | int], sliders: list[Slider] | None = None):
    hard = any(k in preset for k in HARD_KEYS)
    slider_by_attr = {s.attr: s for s in sliders} if sliders is not None else {}

    for k, v in preset.items():
        if hasattr(dragon, k):
            setattr(dragon, k, v)

        s = slider_by_attr.get(k)
        if s is not None:
            vv = float(v)
            vv = clamp(vv, s.vmin, s.vmax)
            if s.is_int:
                vv = float(int(round(vv)))
            s.value = vv

    if hard:
        dragon.reset()


def main():
    pygame.init()
    pygame.display.set_caption("Procedural Dragon (Drag IK + Ribs + Sticky Legs)")

    W, H = 1100, 650
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    # Текст (крошечный хелп)
    font = pygame.font.SysFont("consolas", 16)

    dragon = Dragon(screen_size=(W, H))
    ui_font = pygame.font.SysFont("consolas", 14)
    ui_visible = True

    panel_x = 12
    panel_y = 36
    panel_w = 260
    row = 28

    slider_specs = [
        ("head_spring", "head_spring", 0.0, 60.0, "{:.1f}", False),
        ("head_damping", "head_damping", 0.0, 20.0, "{:.1f}", False),
        ("head_max_spd", "head_max_speed", 100.0, 1600.0, "{:.0f}", False),
        ("wave_amp", "wave_amp", 0.0, 16.0, "{:.1f}", False),
        ("wave_freq", "wave_freq", 0.0, 1.20, "{:.2f}", False),
        ("wave_speed", "wave_speed", 0.0, 8.0, "{:.2f}", False),
        ("leg_upper", "leg_upper", 10.0, 90.0, "{:.0f}", False),
        ("leg_lower", "leg_lower", 10.0, 90.0, "{:.0f}", False),
        ("leg_spread", "leg_spread", 0.0, 90.0, "{:.0f}", False),
        ("leg_back", "leg_back", 0.0, 60.0, "{:.0f}", False),
        ("hip_offset", "leg_hip_offset", 0.0, 30.0, "{:.0f}", False),
        ("step_thresh", "step_threshold", 5.0, 120.0, "{:.0f}", False),
        ("step_dur", "step_duration", 0.05, 0.70, "{:.2f}", False),
        ("step_lift", "step_lift", 0.0, 40.0, "{:.0f}", False),
        ("leg_thick", "leg_thickness", 1.0, 10.0, "{:.0f}", False),
    ]

    sliders: list[Slider] = []
    for i, (label, attr, vmin, vmax, fmt, is_int) in enumerate(slider_specs):
        y = panel_y + i * row
        sliders.append(
            Slider(
                label=label,
                attr=attr,
                x=panel_x,
                y=y,
                w=panel_w,
                vmin=vmin,
                vmax=vmax,
                value=float(getattr(dragon, attr)),
                fmt=fmt,
                is_int=is_int,
            )
        )

    def apply_sliders(dr: Dragon):
        for s in sliders:
            setattr(dr, s.attr, s.value)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # dt в секундах

        for event in pygame.event.get():
            ui_consumed = False
            if ui_visible:
                for s in sliders:
                    if s.handle_event(event):
                        ui_consumed = True
                        break
            if ui_consumed:
                continue

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    dragon.reset()
                elif event.key == pygame.K_d:
                    dragon.debug = not dragon.debug
                elif event.key == pygame.K_1:
                    dragon.mode = "bones"
                elif event.key == pygame.K_2:
                    dragon.mode = "skin"
                elif event.key == pygame.K_3:
                    dragon.mode = "hybrid"
                elif event.key == pygame.K_TAB:
                    ui_visible = not ui_visible
                elif event.key == pygame.K_F1:
                    apply_preset(dragon, PRESETS["noodle"], sliders)
                elif event.key == pygame.K_F2:
                    apply_preset(dragon, PRESETS["imperial"], sliders)
                elif event.key == pygame.K_F3:
                    apply_preset(dragon, PRESETS["agile"], sliders)
                elif event.key == pygame.K_F4:
                    apply_preset(dragon, PRESETS["heavy"], sliders)
                elif event.key == pygame.K_F5:
                    apply_preset(dragon, PRESETS["cartoon"], sliders)
                elif event.key == pygame.K_F6:
                    apply_preset(dragon, PRESETS["centipede"], sliders)
                elif event.key == pygame.K_F7:
                    apply_preset(dragon, PRESETS["cloud_long"], sliders)

        mouse_pos = pygame.mouse.get_pos()
        apply_sliders(dragon)
        dragon.update(dt, mouse_pos)

        screen.fill((0, 0, 0))
        dragon.draw(screen)

        if ui_visible:
            h = len(sliders) * row + 8
            pygame.draw.rect(screen, (20, 20, 30), (panel_x - 8, panel_y - 10, panel_w + 16, h), border_radius=8)
            pygame.draw.rect(screen, (60, 60, 80), (panel_x - 8, panel_y - 10, panel_w + 16, h), 1, border_radius=8)

            for s in sliders:
                s.draw(screen, ui_font)

            hint = ui_font.render("TAB: UI on/off | drag sliders", True, (150, 150, 170))
            screen.blit(hint, (panel_x - 4, panel_y - 28))

        # Лёгкий overlay подсказок
        info = (
            f"mode={dragon.mode} | D=debug({dragon.debug}) | R=reset | "
            "1/2/3 modes | F1..F7 presets | ESC=quit"
        )
        surf = font.render(info, True, (150, 150, 170))
        screen.blit(surf, (12, 10))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
