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
                s = smoothstep01(u)
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
        self.leg_upper = 22.0
        self.leg_lower = 18.0
        self.leg_hip_offset = 9.0    # отступ точки "бедра" от позвоночника
        self.leg_spread = 24.0       # куда в среднем тянется опора в сторону
        self.leg_back = 12.0         # куда в среднем тянется опора назад (к хвосту)
        self.step_threshold = 30.0
        self.step_duration = 0.16
        self.step_lift = 14.0

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
            wave = math.sin(self.time * (self.wave_speed * 0.8) - idx * 0.18) * (1.0 - u) * 4.0
            desired = hip + n * (leg.side * (self.leg_spread + wave)) + (-t) * (self.leg_back + 6.0 * (1.0 - u))

            # Bend dir: выбираем сторону сгиба колена (наружу).
            bend_dir = float(leg.side)

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
            u = leg.attach_idx / (self.N - 1)
            w1 = max(1, int(2 - 1 * u))
            w2 = 1

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


def main():
    pygame.init()
    pygame.display.set_caption("Procedural Dragon (Drag IK + Ribs + Sticky Legs)")

    W, H = 1100, 650
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    # Текст (крошечный хелп)
    font = pygame.font.SysFont("consolas", 16)

    dragon = Dragon(screen_size=(W, H))

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # dt в секундах

        for event in pygame.event.get():
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

        mouse_pos = pygame.mouse.get_pos()
        dragon.update(dt, mouse_pos)

        screen.fill((0, 0, 0))
        dragon.draw(screen)

        # Лёгкий overlay подсказок
        info = f"mode={dragon.mode} | D=debug({dragon.debug}) | R=reset | 1/2/3 modes | ESC=quit"
        surf = font.render(info, True, (150, 150, 170))
        screen.blit(surf, (12, 10))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
