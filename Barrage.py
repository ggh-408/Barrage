import argparse
import math
import tkinter.messagebox

import numpy as np
import pygame

from barrage_rl.dynamics import ACTION_VECTORS, normalized_direction, spawn_bullets


def settings():
    def center_text(event):
        global focus, text_dic
        text = text_dic[(focus, 1)].get("1.0", "end")[:-1]
        if not text.isdigit():
            text_dic[(focus, 1)].delete("end-2c")
        text_dic[(focus, 1)].tag_add("center", "1.0", "end")
        text_dic[(focus, 1)].tag_configure("center", justify="center")

    def handle_focus(event):
        global focus, text_dic
        for y in range(4):
            if event.widget == text_dic[(y, 1)]:
                focus = y
                break

    def button_c():
        Barrage.COLLISION = not Barrage.COLLISION
        if Barrage.COLLISION:
            button_collision.config(text="开启")
        else:
            button_collision.config(text="关闭")

    def button_i():
        Barrage.INVINCIBLE = not Barrage.INVINCIBLE
        if Barrage.INVINCIBLE:
            button_invincible.config(text="开启")
        else:
            button_invincible.config(text="关闭")

    def button_m():
        Barrage.MUSIC = not Barrage.MUSIC
        if Barrage.MUSIC:
            button_music.config(image=photo_on)
        else:
            button_music.config(image=photo_off)

    def close():
        init_list = list()
        for y_close in range(4):
            text = int(text_dic[(y_close, 1)].get("1.0", "end")[:-1])
            if y_close in [0, 1] and text < 100:
                tkinter.messagebox.showinfo(message="窗口边长不小于100!")
                return False
            elif y_close == 2 and text > 10:
                tkinter.messagebox.showinfo(message="子弹大小为1～10!")
                return False
            init_list.append(text)

        [Barrage.SCREEN_WIDTH, Barrage.SCREEN_HEIGHT, Barrage.BULLET_SIZE, Barrage.QUANTITY] = init_list
        Barrage.TimeSize = int(math.sqrt(Barrage.SCREEN_WIDTH * Barrage.SCREEN_HEIGHT) / 25)
        root.destroy()
        Barrage.start_game()

    global text_dic
    root = tkinter.Tk()
    focus = -1
    width, height = root.maxsize()
    root.geometry(f"{230}x{283}+{int(width / 2.4)}+{int(height / 3)}")
    root.title("Setting")
    root.resizable(False, False)
    root.bind("<FocusIn>", handle_focus)
    root.bind("<KeyRelease>", center_text)
    tkinter.Label(root, text="窗口宽度:", font="宋体 20").grid(row=0, column=0)
    tkinter.Label(root, text="窗口高度:", font="宋体 20").grid(row=1, column=0)
    tkinter.Label(root, text="子弹大小:", font="宋体 20").grid(row=2, column=0)
    tkinter.Label(root, text="子弹数量:", font="宋体 20").grid(row=3, column=0)
    tkinter.Label(root, text="墙体碰撞:", font="宋体 20").grid(row=4, column=0)
    tkinter.Label(root, text="子弹伤害:", font="宋体 20").grid(row=5, column=0)
    for i in range(4):
        text_dic[(i, 1)] = tkinter.Text(root, width=8, height=1, font="timesnewroman 22")
        text_dic[(i, 1)].grid(row=i, column=1, columnspan=2)
    text_dic[(0, 1)].insert("1.0", "820")
    text_dic[(1, 1)].insert("1.0", "820")
    text_dic[(2, 1)].insert("1.0", "5")
    text_dic[(3, 1)].insert("1.0", "50")
    for i in range(4):
        text_dic[(i, 1)].tag_add("center", "1.0", "end")
        text_dic[(i, 1)].tag_configure("center", justify="center")
    button_collision = tkinter.Button(root, text="开启", command=button_c, width=6, height=0, font="宋体 22")
    button_collision.grid(row=4, column=1)
    button_invincible = tkinter.Button(root, text="开启", command=button_i, width=6, height=0, font="宋体 22")
    button_invincible.grid(row=5, column=1)

    photo_on = tkinter.PhotoImage(file="image/music_on.png")
    photo_off = tkinter.PhotoImage(file="image/music_off.png")
    button_music = tkinter.Button(root, command=button_m, image=photo_on)
    button_music.grid(row=6, column=0)

    button_start = tkinter.Button(root, text="PLAY", command=close, width=6, height=0, font="timesnewroman  22")
    button_start.grid(row=6, column=1)
    root.mainloop()


class Barrage:
    window = None
    PLANE = None
    BULLET = None
    KEY = True
    COLLISION = False
    INVINCIBLE = True
    MUSIC = True
    SCREEN_WIDTH = int()
    SCREEN_HEIGHT = int()
    BULLET_SIZE = int()
    QUANTITY = int()
    TimeStart = int()
    TimeNow = int()
    TimeSize = int()
    TimeColor = str()
    TimeFont = None
    TimeText = None
    TimeTextKey = None
    RestartText = None
    FPS = 120
    PHYSICS_FPS = 120
    MAX_FRAME_TIME = 0.25
    PLANE_SPEED = 240.0
    BULLET_SPEED = 240.0
    TARGETED_BULLET_PROBABILITY = 0.0
    TARGETED_PREDICTION_SCALE_MIN = 0.65
    TARGETED_PREDICTION_SCALE_MAX = 1.0
    TARGETED_ANGULAR_NOISE = 0.08
    RNG = np.random.default_rng()
    AI_CONTROLLER = None
    AI_ACTION = 0

    @staticmethod
    def start_game():
        pygame.init()
        pygame.display.init()
        Barrage.window = pygame.display.set_mode((Barrage.SCREEN_WIDTH, Barrage.SCREEN_HEIGHT))
        pygame.display.set_caption("Barrage")

        # 字体只创建一次，避免在游戏循环中反复搜索和创建系统字体
        Barrage.TimeFont = pygame.font.SysFont("timesnewroman", Barrage.TimeSize)
        restart_font = pygame.font.SysFont("timesnewroman", int(0.7 * Barrage.TimeSize))
        Barrage.RestartText = restart_font.render(
            "PRESS SPACE OR ENTER TO RESTART", True, "#CD7F32"
        )
        Barrage.reset_game()

        if Barrage.MUSIC:
            Music("music/坂元信也,寺島里恵,前沢秀憲 - Starfield (ステージ2 BGM) - 沙羅曼蛇 (FC版).mp3")

        # 渲染最多120 FPS，物理始终使用固定120 Hz子步。
        clock = pygame.time.Clock()
        physics_step = 1.0 / Barrage.PHYSICS_FPS
        accumulator = 0.0

        while True:
            frame_time = min(
                clock.tick(Barrage.FPS) / 1000.0, Barrage.MAX_FRAME_TIME
            )
            switch_skin, direction, restarted = Barrage.get_event()
            if restarted:
                accumulator = 0.0
            if switch_skin:
                Barrage.PLANE.change_skin()

            accumulator += frame_time
            physics_steps = 0
            while accumulator >= physics_step:
                Barrage.advance_physics(direction, physics_step)
                accumulator -= physics_step
                physics_steps += 1

            Barrage.window.fill("#000000")
            plane = Barrage.PLANE.display()
            Barrage.BULLET.display()
            # The AI observes only the already-rendered surface.  Capture before
            # score text is drawn so UI glyphs cannot be mistaken for bullets.
            if Barrage.AI_CONTROLLER is not None and physics_steps > 0:
                Barrage.AI_ACTION = Barrage.AI_CONTROLLER.observe_surface(
                    Barrage.window, physics_steps=physics_steps
                )
            Barrage.get_time()
            pygame.display.flip()

    @staticmethod
    def reset_game():
        """重置一局游戏，避免重新进入 start_game 造成循环层层嵌套。"""
        Barrage.PLANE = Plane()
        Barrage.BULLET = Bullet()
        Bullet.LIST = [[] for _ in range(Barrage.QUANTITY)]
        for bullet_init in range(Barrage.QUANTITY):
            Bullet.bullet_update(bullet_init)

        Barrage.TimeStart = 0.0
        Barrage.TimeNow = 0.0
        Barrage.TimeColor = "#ffffff"
        Barrage.TimeText = None
        Barrage.TimeTextKey = None
        Barrage.AI_ACTION = 0
        if Barrage.AI_CONTROLLER is not None:
            Barrage.AI_CONTROLLER.reset()
            if Barrage.window is not None:
                Barrage.window.fill("#000000")
                Barrage.PLANE.display()
                Barrage.BULLET.display()
                Barrage.AI_ACTION = Barrage.AI_CONTROLLER.prime_surface(
                    Barrage.window
                )

    @staticmethod
    def get_time():
        if not Barrage.KEY:
            Barrage.TimeColor = "#CD7F32"
            Barrage.window.blit(Barrage.RestartText, (5, Barrage.TimeSize))

        # 每存活0.1秒计1分
        text = str(int(10 * (Barrage.TimeNow - Barrage.TimeStart)))
        text_key = (text, Barrage.TimeColor)
        # 数字或颜色不变时复用已经渲染的画面
        if text_key != Barrage.TimeTextKey:
            Barrage.TimeText = Barrage.TimeFont.render(text, True, Barrage.TimeColor)
            Barrage.TimeTextKey = text_key
        Barrage.window.blit(Barrage.TimeText, (5, 0))

    @classmethod
    def get_event(cls):
        key_list = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        restarted = False
        switch_skin = False
        if not Barrage.KEY and (
            key_list[pygame.K_SPACE] or key_list[pygame.K_RETURN]
        ):
            Barrage.KEY = True
            Barrage.reset_game()
            restarted = True

        if Barrage.AI_CONTROLLER is not None:
            direction = ACTION_VECTORS[Barrage.AI_ACTION].copy()
        else:
            direction = normalized_direction(
                float(key_list[pygame.K_RIGHT] or key_list[pygame.K_d])
                - float(key_list[pygame.K_LEFT] or key_list[pygame.K_a]),
                float(key_list[pygame.K_DOWN] or key_list[pygame.K_s])
                - float(key_list[pygame.K_UP] or key_list[pygame.K_w]),
            )
            if Barrage.KEY:
                for key in range(10):
                    if key_list[key + pygame.K_0] and key != Plane.SKIN:
                        Plane.SKIN = key
                        switch_skin = True
                        break

        return switch_skin, direction, restarted

    @staticmethod
    def advance_physics(direction, delta_time):
        """Advance movement and collision by exactly one fixed physics step."""
        if not Barrage.KEY:
            return
        Barrage.PLANE.move(direction, delta_time)
        Barrage.BULLET.update(Barrage.PLANE, delta_time)
        Barrage.TimeNow += delta_time


class Plane:
    # 默认使用独立的 plane(0).gif 外观
    SKIN = 0
    IMAGE = None

    def __init__(self):
        if Plane.SKIN:
            Plane.IMAGE = pygame.image.load(
                "image/plane(" + str(Plane.SKIN) + ").gif"
            ).convert_alpha()
        else:
            Plane.IMAGE = pygame.image.load(
                "image/plane(0).gif"
            ).convert_alpha()
        self.image = Plane.IMAGE
        self.rect = Plane.IMAGE.get_rect()
        self.rect.center = 0.5 * Barrage.SCREEN_WIDTH, 0.5 * Barrage.SCREEN_HEIGHT
        self.position = pygame.Vector2(self.rect.center)
        self.velocity = pygame.Vector2(0.0, 0.0)
        # 像素遮罩只在图片变化时生成一次
        self.mask = pygame.mask.from_surface(self.image)

    def change_skin(self):
        center = self.rect.center
        Plane.IMAGE = pygame.image.load(
            "image/plane(" + str(Plane.SKIN) + ").gif"
        ).convert_alpha()
        self.image = Plane.IMAGE
        self.rect = self.image.get_rect(center=center)
        self.rect.clamp_ip(pygame.Rect(0, 0, Barrage.SCREEN_WIDTH, Barrage.SCREEN_HEIGHT))
        self.position.update(self.rect.center)
        self.mask = pygame.mask.from_surface(self.image)

    def display(self):
        Barrage.window.blit(Plane.IMAGE, self.rect)

        return self

    def move(self, direction, delta_time):
        # 使用浮点位置和固定物理子步，使移动不再依赖实际渲染帧率。
        # Accept the old four-boolean test/human representation as well as the
        # shared normalized two-vector used by the AI.
        if len(direction) == 4:
            vector = normalized_direction(
                float(direction[1]) - float(direction[0]),
                float(direction[3]) - float(direction[2]),
            )
        else:
            vector = normalized_direction(float(direction[0]), float(direction[1]))
        previous = self.position.copy()
        self.position += pygame.Vector2(
            float(vector[0]), float(vector[1])
        ) * Barrage.PLANE_SPEED * delta_time

        half_width = self.rect.width / 2
        half_height = self.rect.height / 2
        self.position.x = min(max(self.position.x, half_width), Barrage.SCREEN_WIDTH - half_width)
        self.position.y = min(max(self.position.y, half_height), Barrage.SCREEN_HEIGHT - half_height)
        self.rect.center = self.position
        self.velocity = (self.position - previous) / max(float(delta_time), 1e-8)


class Bullet:
    LIST = list()

    def __init__(self):
        self.image = pygame.image.load(
            "image/bullet(" + str(Barrage.BULLET_SIZE) + ").gif"
        ).convert_alpha()
        self.rect = self.image.get_rect()
        # 所有子弹共用同一张图片，因此遮罩也只需创建一次
        self.mask = pygame.mask.from_surface(self.image)

    @staticmethod
    def bullet_update(bullet):
        plane_position = np.asarray(Barrage.PLANE.position, dtype=np.float32)
        plane_velocity = np.asarray(Barrage.PLANE.velocity, dtype=np.float32)
        positions, velocities, targeted = spawn_bullets(
            1,
            Barrage.SCREEN_WIDTH,
            Barrage.SCREEN_HEIGHT,
            Barrage.BULLET_SPEED,
            plane_position,
            plane_velocity,
            Barrage.TARGETED_BULLET_PROBABILITY,
            Barrage.TARGETED_PREDICTION_SCALE_MIN,
            Barrage.TARGETED_PREDICTION_SCALE_MAX,
            Barrage.TARGETED_ANGULAR_NOISE,
            Barrage.RNG,
        )
        Bullet.LIST[bullet] = [
            float(positions[0, 0]), float(positions[0, 1]),
            float(velocities[0, 0]), float(velocities[0, 1]),
            bool(targeted[0]),
        ]

    def update(self, plane, delta_time):
        if Barrage.KEY:
            for bullet in range(Barrage.QUANTITY):
                # 缓存当前子弹，减少热点循环中的重复列表索引
                bullet_data = Bullet.LIST[bullet]
                if Barrage.COLLISION:
                    if bullet_data[0] < 0:
                        bullet_data[0] *= -1
                        bullet_data[2] *= -1
                    elif bullet_data[0] > Barrage.SCREEN_WIDTH:
                        bullet_data[0] = 2 * Barrage.SCREEN_WIDTH - bullet_data[0]
                        bullet_data[2] *= -1

                    if bullet_data[1] < 0:
                        bullet_data[1] *= -1
                        bullet_data[3] *= -1
                    elif bullet_data[1] > Barrage.SCREEN_HEIGHT:
                        bullet_data[1] = 2 * Barrage.SCREEN_HEIGHT - bullet_data[1]
                        bullet_data[3] *= -1

                elif bullet_data[0] < 0 or bullet_data[0] > Barrage.SCREEN_WIDTH \
                        or bullet_data[1] < 0 or bullet_data[1] > Barrage.SCREEN_HEIGHT:
                    Bullet.bullet_update(bullet)
                    bullet_data = Bullet.LIST[bullet]

                bullet_data[0] += bullet_data[2] * delta_time
                bullet_data[1] += bullet_data[3] * delta_time
                self.rect.center = bullet_data[:2]
                # 先做便宜的矩形检测，接近飞机时才进行像素级检测
                if Barrage.INVINCIBLE and self.rect.colliderect(plane.rect) \
                        and pygame.sprite.collide_mask(self, plane):
                    Barrage.KEY = False
                    break

    def display(self):
        for bullet in Bullet.LIST:
            self.rect.center = bullet[:2]
            Barrage.window.blit(self.image, self.rect)


class Music:
    def __init__(self, bg):
        pygame.mixer.init()
        pygame.mixer.music.load(bg)
        pygame.mixer.music.play(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Barrage game")
    parser.add_argument("--ai-checkpoint", default="")
    parser.add_argument("--width", type=int, default=820)
    parser.add_argument("--height", type=int, default=820)
    parser.add_argument("--bullet-size", type=int, default=5)
    parser.add_argument("--bullets", type=int, default=50)
    parser.add_argument("--plane-speed", type=float, default=240.0)
    parser.add_argument("--bullet-speed", type=float, default=240.0)
    parser.add_argument("--targeted-probability", type=float, default=0.35)
    parser.add_argument("--targeted-prediction-min", type=float, default=0.65)
    parser.add_argument("--targeted-prediction-max", type=float, default=1.0)
    parser.add_argument("--targeted-angular-noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skin", type=int, default=0)
    parser.add_argument("--no-music", action="store_true")
    args = parser.parse_args()
    if args.ai_checkpoint:
        from barrage_rl.live_screen import LiveVisualController

        Barrage.SCREEN_WIDTH = args.width
        Barrage.SCREEN_HEIGHT = args.height
        Barrage.BULLET_SIZE = args.bullet_size
        Barrage.QUANTITY = args.bullets
        Barrage.PLANE_SPEED = args.plane_speed
        Barrage.BULLET_SPEED = args.bullet_speed
        Barrage.TARGETED_BULLET_PROBABILITY = args.targeted_probability
        Barrage.TARGETED_PREDICTION_SCALE_MIN = args.targeted_prediction_min
        Barrage.TARGETED_PREDICTION_SCALE_MAX = args.targeted_prediction_max
        Barrage.TARGETED_ANGULAR_NOISE = args.targeted_angular_noise
        Barrage.RNG = np.random.default_rng(args.seed)
        Barrage.MUSIC = not args.no_music
        Barrage.TimeSize = int(math.sqrt(args.width * args.height) / 25)
        Plane.SKIN = args.skin
        Barrage.AI_CONTROLLER = LiveVisualController(args.ai_checkpoint)
        Barrage.start_game()
    else:
        focus = 0
        text_dic = dict()
        settings()
