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
    def __init__(
        self,
        attach_idx: int,
        side: int,
        upper_len: float,
        lower_len: float,
        pair_id: int = 0,
        is_front: bool = False,
        name: str = "",
    ):
        self.attach_idx = attach_idx
        self.side = side  # -1 / +1
        self.pair_id = pair_id
        self.is_front = is_front
        self.name = name
        # Диагональные пары для походки:
        #   diag 0: FL + BR
        #   diag 1: FR + BL
        self.diag_id = 0 if (bool(self.side > 0) == bool(self.is_front)) else 1

        self.upper_len = upper_len
        self.lower_len = lower_len

        # Точка крепления к телу (обновляется каждый кадр из Dragon.update).
        self.anchor = Vector2(0, 0)

        # Липкая точка опоры в мире:
        self.foot_pos = Vector2(0, 0)
        self.foot_target = Vector2(0, 0)

        # Параметры шага:
        self.stepping = False
        self.step_progress = 0.0
        self.time_since_step = 0.0
        self.step_start = Vector2(0, 0)
        self.step_target = Vector2(0, 0)
        self.step_perp = Vector2(0, -1)
        self.lift01 = 0.0  # 0..1..0 только для визуала во время swing

        # Для отрисовки:
        self.hip = Vector2(0, 0)
        self.knee = Vector2(0, 0)
        self.foot = Vector2(0, 0)

        # Движение точки крепления на позвоночнике (для direction-aware foothold).
        self.prev_attach_pos = Vector2(0, 0)
        self.attach_vel = Vector2(0, 0)

    def reset(self, initial_foot: Vector2):
        self.foot_pos = Vector2(initial_foot)
        self.foot_target = Vector2(initial_foot)
        self.stepping = False
        self.step_progress = 0.0
        self.time_since_step = 0.0
        self.attach_vel = Vector2(0, 0)
        self.lift01 = 0.0

    def begin_step(self, hip: Vector2, new_target: Vector2, out_dir: Vector2):
        self.stepping = True
        self.step_progress = 0.0
        self.time_since_step = 0.0
        self.step_start = Vector2(self.foot_pos)
        self.step_target = Vector2(new_target)

        # Чуть гарантируем достижимость конечной опоры:
        reach = self.upper_len + self.lower_len - 1e-3
        v = self.step_target - hip
        if v.length_squared() > reach * reach:
            self.step_target = hip + safe_normalize(v) * reach
            v = self.step_target - hip

        min_reach = abs(self.upper_len - self.lower_len) + 1e-3
        if v.length() < min_reach:
            self.step_target = hip + safe_normalize(v) * min_reach

        step_dir = safe_normalize(self.step_target - self.step_start, Vector2(1, 0))
        step_perp = Vector2(-step_dir.y, step_dir.x)
        out = safe_normalize(out_dir, Vector2(0, 1))
        if step_perp.dot(out) < 0.0:
            step_perp = -step_perp
        self.step_perp = safe_normalize(step_perp, Vector2(0, -1))

    def update(
        self,
        dt: float,
        anchor: Vector2,
        hip: Vector2,
        desired_foothold: Vector2,
        step_threshold: float,
        step_duration: float,
        step_lift: float,
        out_dir: Vector2,
        preferred_bend_dir: float,
        can_start_step: bool = True,
    ):
        self.anchor = Vector2(anchor)
        self.hip = Vector2(hip)
        out = safe_normalize(out_dir, Vector2(float(self.side), 0.0))

        if self.stepping:
            self.step_progress += 1.0 if step_duration <= 1e-6 else (dt / step_duration)
            p = clamp(self.step_progress, 0.0, 1.0)

            if p >= 1.0:
                self.stepping = False
                self.foot_pos = Vector2(self.step_target)
                self.lift01 = 0.0
            else:
                # Более "вязкая" кривая переноса: сглаживаем прогресс дважды.
                s = smoothstep01(smoothstep01(p))
                pos = self.step_start.lerp(self.step_target, s)

                # Дуга переноса: по перпендикуляру к направлению шага.
                self.lift01 = math.sin(math.pi * p)
                pos += self.step_perp * (self.lift01 * step_lift)

                self.foot_pos = pos
        else:
            self.time_since_step += dt
            self.lift01 = 0.0
            # Липкость: пока предполагаемая опора рядом, нога "держится" за старый foot_pos.
            if (
                can_start_step
                and (desired_foothold - self.foot_pos).length_squared() > (step_threshold * step_threshold)
            ):
                self.begin_step(hip, desired_foothold, out)

        self.foot_target = Vector2(self.foot_pos)

        # 2-bone IK для позы: выбираем ветку, где колено смотрит наружу от корпуса.
        knee_plus, foot_plus = solve_two_bone_ik_2d(self.hip, self.foot_pos, self.upper_len, self.lower_len, +1.0)
        knee_minus, foot_minus = solve_two_bone_ik_2d(self.hip, self.foot_pos, self.upper_len, self.lower_len, -1.0)

        score_plus = (knee_plus - self.hip).dot(out)
        score_minus = (knee_minus - self.hip).dot(out)

        if abs(score_plus - score_minus) < 1e-6:
            if preferred_bend_dir >= 0.0:
                self.knee, self.foot = knee_plus, foot_plus
            else:
                self.knee, self.foot = knee_minus, foot_minus
        elif score_plus > score_minus:
            self.knee, self.foot = knee_plus, foot_plus
        else:
            self.knee, self.foot = knee_minus, foot_minus


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
        self.leg_pairs = 2           # квадрупед: 2 пары (перед/зад), итого 4 лапы
        self.leg_upper = 34.0
        self.leg_lower = 28.0
        self.leg_hip_offset = 12.0   # отступ точки "бедра" от позвоночника
        self.leg_spread = 42.0       # куда в среднем тянется опора в сторону
        self.leg_back = 18.0         # куда в среднем тянется опора назад (к хвосту)
        self.step_threshold = 55.0
        self.step_duration = 0.28
        self.step_lift = 20.0
        self.leg_thickness = 5.0
        self.stride_front = 22.0
        self.stride_back = 16.0
        self.move_vel_eps = 40.0
        self.min_stance_time = 0.20
        self.max_step_per_pair = 1
        self.gait_diagonal = 0  # 0: FL+BR, 1: FR+BL
        self.gait_couple = 0.75  # если одна в диагонали шагает, вторую легче "подцепить"

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

        # Создаём квадрупед: две фиксированные пары крепления (front/back).
        self.legs = []
        self.leg_pairs = 2
        attach_front = clamp(int(self.N * 0.28), 0, self.N - 1)
        attach_back = clamp(int(self.N * 0.55), 0, self.N - 1)

        pair_specs = (
            (0, True, int(attach_front), "F"),
            (1, False, int(attach_back), "B"),
        )

        for pair_id, is_front, attach_idx, tag in pair_specs:
            # Лево/право (side = +1 / -1)
            for side in (+1, -1):
                side_tag = "L" if side > 0 else "R"
                leg = Leg(
                    attach_idx=attach_idx,
                    side=side,
                    upper_len=self.leg_upper,
                    lower_len=self.leg_lower,
                    pair_id=pair_id,
                    is_front=is_front,
                    name=f"{tag}{side_tag}",
                )

                # Инициализируем опору в разумной позиции относительно тела.
                # В начальной позе касательная ~ (1, 0), нормаль ~ (0, 1).
                r_body = self._radius_at(attach_idx)
                anchor = self.points[attach_idx] + Vector2(0, 1) * (side * r_body)
                hip = anchor + Vector2(0, 1) * (side * self.leg_hip_offset)
                foot = hip + Vector2(0, 1) * (side * self.leg_spread) + Vector2(-1, 0) * self.leg_back
                foot += Vector2(random.uniform(-6, 6), random.uniform(-6, 6))
                leg.reset(foot)
                leg.anchor = Vector2(anchor)
                leg.prev_attach_pos = Vector2(self.points[attach_idx])

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

    def _pair_partner(self, pair_id: int, side: int) -> Leg | None:
        for leg in self.legs:
            if leg.pair_id == pair_id and leg.side == -side:
                return leg
        return None

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

        for leg in self.legs:
            idx = leg.attach_idx
            attach_pos = self.points[idx]
            if dt > 1e-9:
                leg.attach_vel = (attach_pos - leg.prev_attach_pos) / dt
            else:
                leg.attach_vel = Vector2(0, 0)
            leg.prev_attach_pos = Vector2(attach_pos)

        # ----------------------------
        # 3) Лапки: липкость + шаги + простой gait controller (диагонали)
        # ----------------------------
        leg_ctx: list[tuple[Leg, Vector2, Vector2, Vector2, Vector2, float, float]] = []
        thr2 = self.step_threshold * self.step_threshold
        any_stepping = any(l.stepping for l in self.legs)

        for leg in self.legs:
            idx = leg.attach_idx
            t, n = self._frame_at(idx)

            # Точка крепления жёстко привязана к позвоночнику через радиус тела.
            r_body = self._radius_at(idx)
            anchor = self.points[idx] + n * (leg.side * r_body)
            hip = anchor + n * (leg.side * self.leg_hip_offset)

            # "Желаемая" опора: в сторону + назад + stride по реальному движению точки крепления.
            u = idx / (self.N - 1)
            side_phase = 1.35 if leg.side < 0 else 0.0
            pair_phase = 0.0 if leg.is_front else 1.1
            wave = (
                math.sin(self.time * (self.wave_speed * 0.8) - idx * 0.18 + side_phase + pair_phase)
                * (1.0 - u)
                * 4.0
            )

            speed = leg.attach_vel.length()
            move_dir = safe_normalize(leg.attach_vel, t)

            base_stride = self.stride_front if leg.is_front else self.stride_back
            move_t = smoothstep01((speed - self.move_vel_eps) / max(1e-6, self.move_vel_eps))
            stride = base_stride * move_t
            stride *= 0.85 + 0.15 * math.sin(self.time * (self.wave_speed * 0.7) + pair_phase)

            stance_back = self.leg_back + (4.0 if leg.is_front else 10.0) + 6.0 * (1.0 - u)
            desired = (
                hip
                + n * (leg.side * (self.leg_spread + wave))
                + move_dir * stride
                + (-t) * stance_back
            )

            # Bend dir: front чуть "вперёд по движению", back чуть "назад".
            forward_sign = 1.0 if move_dir.dot(t) >= 0.0 else -1.0
            bend_side = float(leg.side if leg.is_front else -leg.side)
            preferred_bend_dir = bend_side * forward_sign

            out_dir = n * float(leg.side)
            dist2 = (desired - leg.foot_pos).length_squared()
            leg_ctx.append((leg, anchor, hip, desired, out_dir, preferred_bend_dir, dist2))

        # Переключаем диагональ только когда все лапы в stance (иначе будут "две диагонали сразу").
        if not any_stepping:
            active_need = any(
                (leg.diag_id == self.gait_diagonal)
                and (dist2 > thr2)
                and (leg.time_since_step >= self.min_stance_time)
                for (leg, _anchor, _hip, _desired, _out, _pref_bend, dist2) in leg_ctx
            )
            other_need = any(
                (leg.diag_id != self.gait_diagonal)
                and (dist2 > thr2)
                and (leg.time_since_step >= self.min_stance_time)
                for (leg, _anchor, _hip, _desired, _out, _pref_bend, dist2) in leg_ctx
            )
            if (not active_need) and other_need:
                self.gait_diagonal = 1 - self.gait_diagonal

        # Если одна лапа в активной диагонали стартует, слегка облегчим старт второй.
        diag_trigger = (not any_stepping) and any(
            (leg.diag_id == self.gait_diagonal)
            and (dist2 > thr2)
            and (leg.time_since_step >= self.min_stance_time)
            for (leg, _anchor, _hip, _desired, _out, _pref_bend, dist2) in leg_ctx
        )

        for leg, anchor, hip, desired, out_dir, preferred_bend_dir, dist2 in leg_ctx:
            pair_step_count = 0
            for pair_leg in self.legs:
                if pair_leg.pair_id == leg.pair_id and pair_leg.stepping:
                    pair_step_count += 1

            partner = self._pair_partner(leg.pair_id, leg.side)
            partner_idle = (partner is None) or (not partner.stepping)

            diag_allowed = leg.diag_id == self.gait_diagonal
            can_start_step = (
                diag_allowed
                and (leg.time_since_step >= self.min_stance_time)
                and (pair_step_count < self.max_step_per_pair)
                and partner_idle
            )

            step_thr = self.step_threshold
            if diag_allowed and diag_trigger:
                step_thr *= self.gait_couple

            leg.upper_len = self.leg_upper
            leg.lower_len = self.leg_lower
            leg.update(
                dt=dt,
                anchor=anchor,
                hip=hip,
                desired_foothold=desired,
                step_threshold=step_thr,
                step_duration=self.step_duration,
                step_lift=self.step_lift,
                out_dir=out_dir,
                preferred_bend_dir=preferred_bend_dir,
                can_start_step=can_start_step,
            )

    def draw(self, screen: pygame.Surface):
        # Цвета (читаемая "плоть" + контуры поверх)
        col_bone = (210, 210, 235)
        col_rib = (170, 170, 205)
        col_body_outline = (205, 205, 235)
        col_body_fill = (118, 118, 145)
        col_leg_fill = (92, 92, 118)      # чуть темнее тела
        col_leg_outline = (185, 185, 215)
        col_debug = (120, 220, 140)
        col_debug2 = (220, 160, 120)

        # Лапы рисуем первыми, чтобы тело могло перекрыть "плечо/таз" (вид сверху).
        self._draw_legs(screen, col_leg_fill, col_leg_outline, col_debug2 if self.debug else None)

        if self.mode in ("skin", "hybrid"):
            self._draw_skin(screen, color_outline=col_body_outline, color_fill=col_body_fill)

        if self.mode in ("bones", "hybrid"):
            self._draw_bones(screen, col_bone)
            self._draw_ribs(screen, col_rib)

        self._draw_head(screen, col_bone)

        if self.debug:
            self._draw_debug(screen, col_debug, col_debug2)

    def _draw_bones(self, screen: pygame.Surface, color):
        pts = [v2i(p) for p in self.points]
        pygame.draw.lines(screen, color, False, pts, 1)

    def _draw_skin(self, screen: pygame.Surface, color_outline, color_fill=None):
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
            if color_fill is not None:
                pygame.draw.polygon(screen, color_fill, poly, 0)
            pygame.draw.polygon(screen, color_outline, poly, 1)

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

    def _draw_capsule(
        self,
        screen: pygame.Surface,
        a: Vector2,
        b: Vector2,
        r: float,
        color_fill,
        color_outline=None,
    ):
        r = float(r)
        if r <= 0.5:
            return

        d = b - a
        if d.length_squared() < 1e-9:
            rr = max(1, int(r))
            pygame.draw.circle(screen, color_fill, v2i(a), rr, 0)
            if color_outline is not None:
                pygame.draw.circle(screen, color_outline, v2i(a), rr, 1)
            return

        u = d.normalize()
        n = Vector2(-u.y, u.x)
        pts = [a + n * r, b + n * r, b - n * r, a - n * r]
        pts_i = [v2i(p) for p in pts]

        rr = max(1, int(r))
        pygame.draw.polygon(screen, color_fill, pts_i, 0)
        pygame.draw.circle(screen, color_fill, v2i(a), rr, 0)
        pygame.draw.circle(screen, color_fill, v2i(b), rr, 0)

        if color_outline is not None:
            pygame.draw.polygon(screen, color_outline, pts_i, 1)
            pygame.draw.circle(screen, color_outline, v2i(a), rr, 1)
            pygame.draw.circle(screen, color_outline, v2i(b), rr, 1)

    def _draw_legs(self, screen: pygame.Surface, color_fill, color_outline, debug_color=None):
        def mix(c0, c1, t: float):
            tt = clamp(t, 0.0, 1.0)
            return (
                int(c0[0] + (c1[0] - c0[0]) * tt),
                int(c0[1] + (c1[1] - c0[1]) * tt),
                int(c0[2] + (c1[2] - c0[2]) * tt),
            )

        for leg in self.legs:
            hip = leg.hip
            knee = leg.knee
            foot = leg.foot

            u = leg.attach_idx / (self.N - 1)
            lift = float(leg.lift01)  # 0..1..0 (swing)

            # Толщина: thigh толще shin, в swing чуть "ближе к камере".
            base_w = max(5.0, float(self.leg_thickness))
            taper = 1.10 - 0.22 * u
            shin_w = max(5.0, base_w * 1.15 * taper)
            thigh_w = max(shin_w + 2.0, base_w * 1.55 * taper)

            scale = 1.0 + 0.22 * lift
            shin_r = (shin_w * scale) * 0.5
            thigh_r = (thigh_w * scale) * 0.5

            seg_fill = mix(color_fill, color_outline, 0.35 * lift)
            seg_outline = mix(color_outline, (255, 255, 255), 0.10 * lift)

            # 1) Thigh (бедро): hip -> knee
            self._draw_capsule(screen, hip, knee, thigh_r, seg_fill, seg_outline)
            # 2) Shin (голень): knee -> foot
            self._draw_capsule(screen, knee, foot, shin_r, seg_fill, seg_outline)

            # Суставы (чтобы нога не была "палкой"):
            knee_r = max(2, int(shin_r * 0.85))
            hip_r = max(2, int(thigh_r * 0.75))
            joint_fill = mix(seg_fill, seg_outline, 0.55)
            pygame.draw.circle(screen, joint_fill, v2i(hip), hip_r, 0)
            pygame.draw.circle(screen, seg_outline, v2i(hip), hip_r, 1)
            pygame.draw.circle(screen, joint_fill, v2i(knee), knee_r, 0)
            pygame.draw.circle(screen, seg_outline, v2i(knee), knee_r, 1)

            # 3) Foot (стопа): клин + когти (без "стрелочек").
            fwd = safe_normalize(foot - knee, Vector2(1, 0))
            side = Vector2(-fwd.y, fwd.x)
            foot_len = max(12.0, shin_w * 2.2) * scale
            foot_w = max(7.0, shin_w * 1.35) * scale

            toe = foot + fwd * (foot_len * 0.70)
            heel = foot - fwd * (foot_len * 0.55)
            left = foot + side * (foot_w * 0.70) - fwd * (foot_len * 0.12)
            right = foot - side * (foot_w * 0.70) - fwd * (foot_len * 0.12)

            foot_fill = mix(seg_fill, (0, 0, 0), 0.18)
            foot_poly = [v2i(toe), v2i(left), v2i(heel), v2i(right)]
            pygame.draw.polygon(screen, foot_fill, foot_poly, 0)
            pygame.draw.polygon(screen, seg_outline, foot_poly, 1)

            claw_len = max(5.0, shin_w * 0.95) * (0.85 + 0.25 * lift)
            for s in (-1, 0, +1):
                base = toe + side * (s * foot_w * 0.18) - fwd * (foot_len * 0.06)
                tip = base + fwd * claw_len + side * (s * foot_w * 0.05)
                b1 = base + side * (foot_w * 0.10)
                b2 = base - side * (foot_w * 0.10)
                pygame.draw.polygon(screen, seg_outline, [v2i(tip), v2i(b1), v2i(b2)], 0)

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
            pygame.draw.circle(screen, color2, v2i(leg.anchor), 3, 1)
            pygame.draw.line(screen, color2, v2i(p), v2i(leg.anchor), 1)


PRESETS: dict[str, dict[str, float | int]] = {
    "quadruped_real": {
        "leg_pairs": 2,
        "head_spring": 18.0,
        "head_damping": 9.5,
        "head_max_speed": 820.0,
        "wave_amp": 4.2,
        "wave_freq": 0.30,
        "wave_speed": 2.1,
        "leg_upper": 44.0,
        "leg_lower": 36.0,
        "leg_spread": 52.0,
        "leg_back": 22.0,
        "leg_hip_offset": 14.0,
        "step_threshold": 62.0,
        "step_duration": 0.31,
        "step_lift": 17.0,
        "leg_thickness": 6.0,
        "stride_front": 24.0,
        "stride_back": 14.0,
        "min_stance_time": 0.24,
        "move_vel_eps": 55.0,
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
        ("stride_front", "stride_front", 0.0, 60.0, "{:.0f}", False),
        ("stride_back", "stride_back", 0.0, 60.0, "{:.0f}", False),
        ("move_vel_eps", "move_vel_eps", 0.0, 220.0, "{:.0f}", False),
        ("hip_offset", "leg_hip_offset", 0.0, 30.0, "{:.0f}", False),
        ("step_thresh", "step_threshold", 5.0, 120.0, "{:.0f}", False),
        ("step_dur", "step_duration", 0.05, 0.70, "{:.2f}", False),
        ("min_stance", "min_stance_time", 0.05, 0.70, "{:.2f}", False),
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
                    apply_preset(dragon, PRESETS["quadruped_real"], sliders)

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

            hint = ui_font.render("TAB: UI on/off | drag sliders | F1: quadruped_real", True, (150, 150, 170))
            screen.blit(hint, (panel_x - 4, panel_y - 28))

        # Лёгкий overlay подсказок
        info = (
            f"mode={dragon.mode} | D=debug({dragon.debug}) | R=reset | "
            "1/2/3 modes | F1 preset | ESC=quit"
        )
        surf = font.render(info, True, (150, 150, 170))
        screen.blit(surf, (12, 10))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
