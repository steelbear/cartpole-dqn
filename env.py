import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from celluloid import Camera


class CartPoleEnv:
    def __init__(self):
        # world
        self.world_width = 4.8 # 이후 모든 길이는 world_width 기준으로 비율을 맞춤
        self.world_height = 6.4
        # mass
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = self.cart_mass + self.pole_mass
        # cart
        self.cart_width = 0.8
        self.cart_height = 0.48
        self.cart_x = 0.
        self.cart_y = self.cart_height / 2.
        self.cart_velocity = 0.
        # pole
        self.pole_length = 2.0
        self.pole_mass_length = self.pole_mass * 0.5
        self.pole_theta = 0.
        self.pole_angular_velocity = 0.
        # frame delta
        self.delta_time = 0.02
        # reward check
        self.steps_beyond_done = None
        # done condition
        self.x_threadhold = 2.4
        self.theta_threadhold = 12 * 2 * math.pi / 360.0
        # state / action size
        self.state_size = 4
        self.actions_num = 2
        # matplotlib
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlim=(-self.world_width, self.world_width), ylim=(0, self.world_height))
        # self.camera = Camera(self.fig)

    def reset(self):
        self.ax.cla()
        self.camera = Camera(self.fig)
        state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.cart_x, self.cart_velocity, self.pole_theta, self.pole_angular_velocity = state
        return state

    def step(self, action):
        theta_cos = np.cos(self.pole_theta)
        theta_sin = np.sin(self.pole_theta)
        force = 10.0 if action == 1 else -10.0

        # 가속도와 각가속도 계산
        temp = (
            force + self.pole_mass_length * self.pole_angular_velocity ** 2 * theta_sin
        ) / self.total_mass
        angular_acceleration = (self.gravity * theta_sin - theta_cos * temp) / (
            0.5 * (4.0 / 3.0 - self.pole_mass * theta_cos ** 2 / self.total_mass)
        )
        acceleration = temp - self.pole_mass_length * angular_acceleration * theta_cos / self.total_mass

        # 위치와 속도 계산
        self.cart_x = self.cart_x + self.delta_time * self.cart_velocity
        self.cart_velocity = self.cart_velocity + self.delta_time * acceleration
        # 각도와 각가속도 계산
        self.pole_theta = self.pole_theta + self.delta_time * self.pole_angular_velocity
        self.pole_angular_velocity = self.pole_angular_velocity + self.delta_time * angular_acceleration

        state = np.array([self.cart_x, self.cart_velocity, self.pole_theta, self.pole_angular_velocity])

        # 종료 조건을 만족하는지 확인
        done = bool(
            self.cart_x < -self.x_threadhold
            or self.cart_x > self.x_threadhold
            or self.pole_theta < -self.theta_threadhold
            or self.pole_theta > self.theta_threadhold
        )

        # 보상 계산
        if not done:
            reward = 0.1
        else:
            reward = -1.0

        return state, reward, done

    def render(self, episode=None, action=None, recording=False):
        # 화면 초기화
        if not recording:
            self.ax.cla()
        self.ax.set(xlim=(-self.world_width, self.world_width), ylim=(0, self.world_height))

        pole_angle = np.pi / 2. - self.pole_theta
        angle_cos = np.cos(pole_angle)
        angle_sin = np.sin(pole_angle)

        # 카트 그리기
        cart_coord = (self.cart_x - self.cart_width / 2., 0)
        cart = mpatches.Rectangle(cart_coord, self.cart_width, self.cart_height, color='black')
        self.ax.add_patch(cart)

        # 막대 그리기
        pole_x = [self.cart_x, self.cart_x + angle_cos * self.pole_length]
        pole_y = [self.cart_y, self.cart_y + angle_sin * self.pole_length]
        pole = Line2D(pole_x, pole_y, lw=5., color='brown')
        self.ax.add_line(pole)

        # 현재 에피소드 표시
        if episode is not None:
            self.ax.set_title("Episode {}".format(episode))

        # 현재 카트를 미는 방향 표시
        if action is not None:
            direction = 1 if action == 1 else -1
            arrow_x = self.cart_x + direction * self.cart_width
            arrow = mpatches.FancyArrowPatch((self.cart_x, self.cart_y), (arrow_x, self.cart_y), color='red',
                                             mutation_scale=10)
            self.ax.add_patch(arrow)

        if recording:
            self.camera.snap()
        else:
            # 화면에 표시 후 일정 시간동안 정지
            plt.pause(0.02)

    def show(self):
        anim = self.camera.animate(interval=10, blit=True)
        plt.show()

    def save(self, filename):
        anim = self.camera.animate(interval=50, blit=True)
        anim.save(filename)