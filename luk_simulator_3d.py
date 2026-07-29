#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Симулятор слива акций в лук: 3D
================================
Первое лицо, рейкастинг (Wolfenstein-style), склад, который можно обойти
ногами, прилавок, за которым торгуешь луком, и Грей Тигра где-то рядом.

Как запустить:
    pip install pygame
    python3 luk_simulator_3d.py

Держи файл DejaVuSans.ttf в той же папке, что и скрипт — иначе кириллица
может отрисоваться кракозябрами (движок попробует запасной системный
шрифт, но это не гарантия).

Управление:
    W A S D   — ходьба / страфы
    Мышь      — обзор (ESC — отпустить/поймать курсор)
    Стрелки ← → — обзор без мыши
    E         — взаимодействие (прилавок / потыкать лук)
    TAB       — достижения
    R         — сброс сохранения
"""

import os
import sys
import json
import math
import random
import traceback

import pygame
import pygame.freetype


# ======================================================================
#  НАСТРОЙКИ
# ======================================================================
SCREEN_W, SCREEN_H = 1000, 650
FPS = 60

FOV = math.radians(66)
NUM_RAYS = 400
STRIP_W = SCREEN_W / NUM_RAYS
MAX_DEPTH = 24.0

PLAYER_RADIUS = 0.22
MOVE_SPEED = 3.0
STRAFE_SPEED = 2.6
TURN_SPEED = 2.2          # рад/сек, для стрелок
MOUSE_SENS = 0.0028       # рад/пиксель

INTERACT_RANGE = 2.6
POKE_RANGE = 1.3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(BASE_DIR, 'luk_save.json')
FONT_PATH = os.path.join(BASE_DIR, 'DejaVuSans.ttf')

# ---- палитра (взята из оригинальной веб-версии) ----
COL_BG = (15, 18, 22)
COL_PANEL = (24, 28, 34)
COL_PANEL_LIGHT = (36, 41, 50)
COL_PANEL_EDGE = (54, 60, 70)
COL_ACCENT = (232, 163, 61)
COL_ACCENT_DIM = (150, 108, 46)
COL_ACCENT2 = (111, 191, 115)
COL_DANGER = (224, 90, 90)
COL_TEXT = (238, 238, 238)
COL_MUTED = (153, 153, 153)

COL_WALL = (56, 66, 80)
COL_CRATE = (150, 100, 58)
COL_COUNTER = (232, 163, 61)
COL_FLOOR_NEAR = (54, 44, 34)
COL_FLOOR_FAR = (24, 20, 16)
COL_CEIL = (10, 12, 15)


# ======================================================================
#  КАРТА СКЛАДА
#  '#' стена, 'C' ящики с луком (препятствие), 'K' прилавок, '.' пол
# ======================================================================
GAME_MAP = [
    "#######KKK#######",
    "#...............#",
    "#...............#",
    "#...CC.....CC...#",
    "#...............#",
    "#...............#",
    "#...............#",
    "#...............#",
    "#...CC.....CC...#",
    "#...............#",
    "#...............#",
    "#################",
]
MAP_W = len(GAME_MAP[0])
MAP_H = len(GAME_MAP)

PLAYER_START = (8.5, 9.5)
PLAYER_START_ANGLE = -math.pi / 2  # смотрит на север, на прилавок
COMPANION_POS = (8.5, 2.2)
COUNTER_POINT = (8.5, 1.0)

ONION_PILE_SLOTS = [
    (5.0, 2.35), (5.0, 4.65), (3.7, 3.5), (6.3, 3.5),
    (4.0, 2.6), (6.0, 2.6), (4.0, 4.4), (6.0, 4.4),
    (12.0, 2.35), (12.0, 4.65), (10.7, 3.5), (13.3, 3.5),
    (11.0, 2.6), (13.0, 2.6), (11.0, 4.4), (13.0, 4.4),
    (5.0, 7.35), (5.0, 9.65), (3.7, 8.5), (6.3, 8.5),
    (4.0, 7.6), (6.0, 7.6), (4.0, 9.4), (6.0, 9.4),
    (12.0, 7.35), (12.0, 9.65), (10.7, 8.5), (13.3, 8.5),
    (11.0, 7.6), (13.0, 7.6), (11.0, 9.4), (13.0, 9.4),
]

ACHIEVEMENTS = {
    'first_deal': ('Первый слив', 'Продай первую партию лука.'),
    'poke_lord': ('Тыкатель', 'Потыркай лук 10 раз.'),
    'rich': ('Луковый магнат', 'Заработай 1000₽.'),
    'panic': ('Паника на бирже', 'Поймай цену ниже 8₽.'),
    'tiger': ('Грей Тигра одобряет', 'Поговори с Грей Тигрой.'),
}


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def map_cell(x, y):
    ix, iy = int(x), int(y)
    if ix < 0 or iy < 0 or ix >= MAP_W or iy >= MAP_H:
        return '#'
    return GAME_MAP[iy][ix]


def is_blocked(x, y):
    return map_cell(x, y) in '#CK'


def shade(color, factor):
    return tuple(clamp(int(c * factor), 0, 255) for c in color)


class FontAdapter:
    """Small wrapper that lets pygame.font and pygame.freetype behave alike."""

    def __init__(self, font, use_freetype=False):
        self.font = font
        self.use_freetype = use_freetype

    def render(self, text, antialias, color):
        if self.use_freetype:
            surface, _ = self.font.render(text, color)
            return surface
        return self.font.render(text, antialias, color)


class Toasts:
    def __init__(self):
        self.items = []

    def push(self, text, color=COL_ACCENT):
        self.items.append({'text': text, 'time': 3.0, 'color': color})

    def update(self, dt):
        for item in self.items:
            item['time'] -= dt
        self.items = [item for item in self.items if item['time'] > 0]

    def draw(self, surf, font):
        y = 92
        for item in self.items[-4:]:
            alpha = clamp(item['time'], 0, 1)
            text = font.render(item['text'], True, item['color'])
            rect = text.get_rect(topright=(SCREEN_W - 18, y))
            pad = 8
            bg = pygame.Surface((rect.w + pad * 2, rect.h + pad), pygame.SRCALPHA)
            bg.fill((10, 12, 15, int(210 * alpha)))
            surf.blit(bg, (rect.x - pad, rect.y - pad // 2))
            surf.blit(text, rect)
            y += rect.h + 10


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Симулятор слива акций в лук: 3D')
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = self.load_font(20)
        self.small = self.load_font(16)
        self.big = self.load_font(34)
        self.player_x, self.player_y = PLAYER_START
        self.angle = PLAYER_START_ANGLE
        self.mouse_locked = True
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        self.toasts = Toasts()
        self.show_achievements = False
        self.message = 'Добро пожаловать на склад луковых активов.'
        self.message_time = 4.0
        self.reset_runtime()
        self.load_save()

    def load_font(self, size):
        font_module = sys.modules.get('pygame.font')
        if font_module and font_module.get_init():
            if os.path.exists(FONT_PATH):
                return FontAdapter(font_module.Font(FONT_PATH, size))
            return FontAdapter(font_module.SysFont('dejavusans,arial', size))
        if not pygame.freetype.get_init():
            pygame.freetype.init()
        if os.path.exists(FONT_PATH):
            return FontAdapter(pygame.freetype.Font(FONT_PATH, size), use_freetype=True)
        return FontAdapter(pygame.freetype.SysFont('dejavusans,arial', size), use_freetype=True)

    def reset_runtime(self):
        self.cash = 120
        self.onions = 35
        self.price = 18.0
        self.reputation = 0
        self.pokes = 0
        self.deals = 0
        self.achieved = set()
        self.price_timer = 0

    def load_save(self):
        try:
            if not os.path.exists(SAVE_PATH):
                return
            with open(SAVE_PATH, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            self.cash = data.get('cash', self.cash)
            self.onions = data.get('onions', self.onions)
            self.reputation = data.get('reputation', self.reputation)
            self.pokes = data.get('pokes', self.pokes)
            self.deals = data.get('deals', self.deals)
            self.achieved = set(data.get('achieved', []))
            self.toasts.push('Сохранение загружено', COL_ACCENT2)
        except Exception:
            traceback.print_exc()
            self.toasts.push('Не смог прочитать сохранение', COL_DANGER)

    def save(self):
        data = {
            'cash': self.cash,
            'onions': self.onions,
            'reputation': self.reputation,
            'pokes': self.pokes,
            'deals': self.deals,
            'achieved': sorted(self.achieved),
        }
        with open(SAVE_PATH, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    def unlock(self, key):
        if key in self.achieved:
            return
        self.achieved.add(key)
        name, _ = ACHIEVEMENTS[key]
        self.toasts.push(f'🏆 {name}', COL_ACCENT)
        self.save()

    def set_message(self, text, seconds=3.2):
        self.message = text
        self.message_time = seconds

    def try_move(self, dx, dy):
        nx = self.player_x + dx
        ny = self.player_y + dy
        if not is_blocked(nx, self.player_y):
            self.player_x = nx
        if not is_blocked(self.player_x, ny):
            self.player_y = ny

    def update(self, dt):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.angle -= TURN_SPEED * dt
        if keys[pygame.K_RIGHT]:
            self.angle += TURN_SPEED * dt

        forward = (math.cos(self.angle), math.sin(self.angle))
        right = (-forward[1], forward[0])
        dx = dy = 0
        if keys[pygame.K_w]:
            dx += forward[0] * MOVE_SPEED * dt; dy += forward[1] * MOVE_SPEED * dt
        if keys[pygame.K_s]:
            dx -= forward[0] * MOVE_SPEED * dt; dy -= forward[1] * MOVE_SPEED * dt
        if keys[pygame.K_d]:
            dx += right[0] * STRAFE_SPEED * dt; dy += right[1] * STRAFE_SPEED * dt
        if keys[pygame.K_a]:
            dx -= right[0] * STRAFE_SPEED * dt; dy -= right[1] * STRAFE_SPEED * dt
        self.try_move(dx, dy)

        self.price_timer -= dt
        if self.price_timer <= 0:
            self.price_timer = random.uniform(1.5, 3.0)
            self.price = clamp(self.price + random.uniform(-3.2, 3.8), 4.0, 42.0)
            if self.price < 8:
                self.unlock('panic')
        self.message_time = max(0, self.message_time - dt)
        self.toasts.update(dt)

    def interact(self):
        if dist((self.player_x, self.player_y), COUNTER_POINT) <= INTERACT_RANGE:
            qty = min(self.onions, random.randint(4, 12))
            if qty <= 0:
                self.set_message('Лук кончился. Остались только слёзы инвестора.', 3)
                return
            revenue = int(qty * self.price)
            self.onions -= qty
            self.cash += revenue
            self.reputation += 1
            self.deals += 1
            self.set_message(f'Слил {qty} мешков по {self.price:.1f}₽. +{revenue}₽', 3)
            self.unlock('first_deal')
            if self.cash >= 1000:
                self.unlock('rich')
            self.save()
            return
        if dist((self.player_x, self.player_y), COMPANION_POS) <= INTERACT_RANGE:
            self.set_message('Грей Тигра: «Держи спред шире, а лук — ближе»', 4)
            self.unlock('tiger')
            return
        for slot in ONION_PILE_SLOTS:
            if dist((self.player_x, self.player_y), slot) <= POKE_RANGE:
                self.pokes += 1
                if random.random() < 0.35:
                    self.onions += 1
                    self.set_message('Ты потыкал лук и нашёл ещё один мешок.', 2.5)
                else:
                    self.set_message('Лук хрустит. Рынок нервничает.', 2.5)
                if self.pokes >= 10:
                    self.unlock('poke_lord')
                self.save()
                return
        self.set_message('Тут нечего делать. Ищи прилавок, лук или Грей Тигру.', 2.5)

    def cast_ray(self, angle):
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)
        depth = 0.02
        step = 0.025
        while depth < MAX_DEPTH:
            x = self.player_x + cos_a * depth
            y = self.player_y + sin_a * depth
            cell = map_cell(x, y)
            if cell != '.':
                return depth, cell
            depth += step
        return MAX_DEPTH, '.'

    def project_sprite(self, pos, color, radius, label=None):
        dx, dy = pos[0] - self.player_x, pos[1] - self.player_y
        distance = math.hypot(dx, dy)
        theta = math.atan2(dy, dx) - self.angle
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        if abs(theta) > FOV / 2 + 0.3 or distance < 0.1:
            return
        size = int(SCREEN_H / (distance + 0.2) * radius)
        sx = int((theta + FOV / 2) / FOV * SCREEN_W)
        sy = SCREEN_H // 2 + int(size * 0.4)
        shade_factor = clamp(1.25 - distance / 12, 0.35, 1.0)
        pygame.draw.circle(self.screen, shade(color, shade_factor), (sx, sy), max(3, size))
        pygame.draw.circle(self.screen, shade((255, 235, 180), shade_factor), (sx - size // 3, sy - size // 4), max(2, size // 4))
        if label and distance < 4:
            text = self.small.render(label, True, COL_TEXT)
            self.screen.blit(text, text.get_rect(center=(sx, sy - size - 14)))

    def draw_world(self):
        self.screen.fill(COL_BG)
        pygame.draw.rect(self.screen, COL_CEIL, (0, 0, SCREEN_W, SCREEN_H // 2))
        for y in range(SCREEN_H // 2, SCREEN_H):
            t = (y - SCREEN_H / 2) / (SCREEN_H / 2)
            col = tuple(int(COL_FLOOR_NEAR[i] * (1 - t) + COL_FLOOR_FAR[i] * t) for i in range(3))
            pygame.draw.line(self.screen, col, (0, y), (SCREEN_W, y))

        start = self.angle - FOV / 2
        for ray in range(NUM_RAYS):
            ray_angle = start + FOV * ray / NUM_RAYS
            depth, cell = self.cast_ray(ray_angle)
            depth *= math.cos(ray_angle - self.angle)  # fish-eye fix
            height = int(SCREEN_H / max(depth, 0.001))
            y = SCREEN_H // 2 - height // 2
            base = {'#': COL_WALL, 'C': COL_CRATE, 'K': COL_COUNTER}.get(cell, COL_WALL)
            factor = clamp(1.15 - depth / 11.5, 0.22, 1.0)
            rect = (int(ray * STRIP_W), y, int(STRIP_W) + 1, height)
            pygame.draw.rect(self.screen, shade(base, factor), rect)

        for slot in sorted(ONION_PILE_SLOTS, key=lambda p: dist((self.player_x, self.player_y), p), reverse=True):
            self.project_sprite(slot, (196, 124, 38), 0.18)
        self.project_sprite(COMPANION_POS, (120, 145, 210), 0.38, 'Грей Тигра')

    def draw_minimap(self):
        scale = 9
        ox, oy = 18, SCREEN_H - MAP_H * scale - 18
        for y, row in enumerate(GAME_MAP):
            for x, cell in enumerate(row):
                col = (32, 36, 42) if cell == '.' else {'#': COL_WALL, 'C': COL_CRATE, 'K': COL_COUNTER}[cell]
                pygame.draw.rect(self.screen, col, (ox + x * scale, oy + y * scale, scale - 1, scale - 1))
        px, py = ox + self.player_x * scale, oy + self.player_y * scale
        pygame.draw.circle(self.screen, COL_ACCENT2, (int(px), int(py)), 3)
        pygame.draw.line(self.screen, COL_ACCENT2, (px, py), (px + math.cos(self.angle) * 10, py + math.sin(self.angle) * 10), 2)

    def draw_hud(self):
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, SCREEN_W, 70))
        pygame.draw.rect(self.screen, COL_PANEL, (12, 10, 615, 50), border_radius=8)
        stats = f'₽ {self.cash}   Лук: {self.onions}   Цена: {self.price:.1f}₽   Репутация: {self.reputation}'
        self.screen.blit(self.font.render(stats, True, COL_TEXT), (26, 24))
        hint = 'E — действие • TAB — достижения • ESC — мышь • R — сброс'
        self.screen.blit(self.small.render(hint, True, COL_MUTED), (650, 25))
        pygame.draw.line(self.screen, COL_TEXT, (SCREEN_W // 2 - 7, SCREEN_H // 2), (SCREEN_W // 2 + 7, SCREEN_H // 2), 1)
        pygame.draw.line(self.screen, COL_TEXT, (SCREEN_W // 2, SCREEN_H // 2 - 7), (SCREEN_W // 2, SCREEN_H // 2 + 7), 1)
        if self.message_time > 0:
            text = self.font.render(self.message, True, COL_TEXT)
            rect = text.get_rect(center=(SCREEN_W // 2, SCREEN_H - 70))
            pygame.draw.rect(self.screen, (0, 0, 0), rect.inflate(28, 18), border_radius=8)
            self.screen.blit(text, rect)
        self.draw_minimap()
        self.toasts.draw(self.screen, self.small)

    def draw_achievements(self):
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.screen.blit(overlay, (0, 0))
        panel = pygame.Rect(190, 95, 620, 450)
        pygame.draw.rect(self.screen, COL_PANEL, panel, border_radius=12)
        pygame.draw.rect(self.screen, COL_PANEL_EDGE, panel, 2, border_radius=12)
        self.screen.blit(self.big.render('Достижения', True, COL_ACCENT), (panel.x + 28, panel.y + 22))
        y = panel.y + 86
        for key, (name, desc) in ACHIEVEMENTS.items():
            done = key in self.achieved
            mark = '✓' if done else '○'
            color = COL_ACCENT2 if done else COL_MUTED
            self.screen.blit(self.font.render(f'{mark} {name}', True, color), (panel.x + 34, y))
            self.screen.blit(self.small.render(desc, True, COL_TEXT if done else COL_MUTED), (panel.x + 62, y + 28))
            y += 70

    def draw(self):
        self.draw_world()
        self.draw_hud()
        if self.show_achievements:
            self.draw_achievements()
        pygame.display.flip()

    def toggle_mouse(self):
        self.mouse_locked = not self.mouse_locked
        pygame.event.set_grab(self.mouse_locked)
        pygame.mouse.set_visible(not self.mouse_locked)

    def reset_save(self):
        self.reset_runtime()
        if os.path.exists(SAVE_PATH):
            os.remove(SAVE_PATH)
        self.set_message('Сохранение сброшено. Лук снова свежий.', 3)
        self.toasts.push('Новая биржевая жизнь', COL_ACCENT2)

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.toggle_mouse()
                    elif event.key == pygame.K_e:
                        self.interact()
                    elif event.key == pygame.K_TAB:
                        self.show_achievements = not self.show_achievements
                    elif event.key == pygame.K_r:
                        self.reset_save()
                elif event.type == pygame.MOUSEMOTION and self.mouse_locked:
                    self.angle += event.rel[0] * MOUSE_SENS
            self.update(dt)
            self.draw()
        self.save()
        pygame.quit()


def main():
    try:
        Game().run()
    except KeyboardInterrupt:
        pygame.quit()
    except Exception:
        pygame.quit()
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
