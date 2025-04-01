import numpy as np
import matplotlib.pyplot as plt

class FallingSphere:
    def __init__(self, name="Sphere", resolution=1000,
                 start_time=0.0, end_time=10.0, distance_from_ground=100,
                 speed_towards_ground=0.0, g=9.825, mass=1.0, C=0.05,
                 positive_direction="upwards"):
        """
        Initialize the FallingSphere simulation.
        g = 9.825 is used as this is the actual value for Oslo
        """
        if positive_direction not in ["upwards", "downwards"]:
            raise ValueError(
                "direction must be either 'upwards' or 'downwards'")

        if distance_from_ground <= 0:
            raise ValueError(
                "Height above ground must be greater than zero.")
            
        if g < 0 or (g == 0 and speed_towards_ground <= 0):
            raise ValueError(
                "Gravity must be positive"
                "or zero with initial speed towards ground")
            
        if mass <= 0:
            raise ValueError(
                "Mass needs to be positive")
        
        if C < 0:
            raise ValueError(
                "C needs to be zero (for no resistance) or positive")
        
        if start_time >= end_time:
            raise ValueError(
                "End time needs to be larger than start time")

        # Controls positive direction and how calculations change
        self.positive_direction = positive_direction
        self.sign = 1 if positive_direction == "upwards" else -1

        # General properties of object
        self.name = name
        self.resolution = resolution

        # Time based parameters
        self.t = np.linspace(start_time, end_time, resolution) # All time values
        self.dt = self.t[1] - self.t[0]  # Time step size

        # Constants
        self.g = g
        self.mass = mass
        self.C = C

        # Initial conditions with positive direction in mind
        self.h0 = self.sign * distance_from_ground
        self.v0 = self.sign * speed_towards_ground
        self.a0 = self.sign * -g

        # Arrays for simulation (with and without air resistance)
        self.h = np.zeros(resolution)
        self.v = np.zeros(resolution)
        self.a = np.zeros(resolution)
        self.h_noair = np.zeros(resolution)
        self.v_noair = np.zeros(resolution)
        self.a_noair = np.zeros(resolution)

        # Initial conditions stored in array
        self.h[0] = self.h0
        self.v[0] = self.v0
        self.a[0] = self.a0

        # Impact times
        self.t_impact = None
        self.t_noair_impact = None

    def calculate(self):
        """
        Compute the motion of the sphere over time.
        """
        # For loop starts at index 1 as initial value at index 0 is set
        for i in range(1, self.resolution):
            self.h[i] = self.h[i - 1] + self.v[i - 1] * self.dt
            self.v[i] = self.v[i - 1] + self.a[i - 1] * self.dt
            self.a[i] = self.sign * (
                -self.g + (self.C * (self.v[i] ** 2)) / self.mass)

        # Without air resistance (vectorized calculations, does not need looping)
        self.h_noair = self.h[0] + (
            self.v0 * self.t + 0.5 * self.sign * -self.g * (self.t ** 2))
        self.v_noair = self.sign * -self.g * self.t
        self.a_noair = self.sign * -self.g * np.ones_like(self.t)

    def find_impact_times(self):
        """
        Find the time indices when the sphere impacts the ground.
        """
        self.t_impact = np.argmin(np.abs(self.h))
        self.t_noair_impact = np.argmin(np.abs(self.h_noair))
        
        # Checks if simulation is long enough for impact
        if abs(self.h_noair[self.t_noair_impact]) > 0.5:
            raise ValueError(
                "Object does not hit the ground, within current timeframe")
        elif abs(self.h[self.t_impact]) > 0.5:
            raise ValueError(
                "Object does not hit the ground with air resistance, "
                "within current timeframe")


    def display_results(self):
        """
        Print the time and velocity at impact for both cases.
        """
        print(
            f'The sphere hits the ground at '
            f'{round(self.t[self.t_impact], 3)}s, '
            f'reaching a speed of {round(self.v[self.t_impact], 3)}m/s'
            f'with {self.positive_direction} positive direction')
        print(
            f'and without air resistance at '
            f'{round(self.t[self.t_noair_impact], 3)}s, '
            f'reaching a speed of {round(self.v_noair[self.t_noair_impact], 3)}m/s '
            f'with {self.positive_direction} positive direction')

    def plot_height(self):
        """
        Plot height vs. time for both cases.
        """
        plt.figure(0)
        
        max_val = np.ceil(abs(self.h0) / 5.0) * 5
        y_step = max_val * 0.05
        min_val = 2 * y_step
        
        if self.sign == 1:
            plt.ylim(-min_val, max_val)       
            plt.text(
                self.t[self.t_impact],
                self.h[self.t_impact] + 1 * y_step,
                f'{round(self.t[self.t_impact], 2)}$s$',
                color="blue", fontsize=12, ha="center")
            plt.text(
                self.t[self.t_noair_impact],
                self.h_noair[self.t_noair_impact] + 1 * y_step,
                f'{round(self.t[self.t_noair_impact], 2)}$s$',
                color="red", fontsize=12, ha="center")
        else:
            plt.ylim(-max_val, min_val)
            plt.text(
                self.t[self.t_impact],
                self.h[self.t_impact] - 2 * y_step,
                f'{round(self.t[self.t_impact], 2)}$s$',
                color="blue", fontsize=12, ha="center")
            plt.text(
                self.t[self.t_noair_impact],
                self.h_noair[self.t_noair_impact] - 2 * y_step,
                f'{round(self.t[self.t_noair_impact], 2)}$s$',
                color="red", fontsize=12, ha="center")

        plt.plot(
            self.t, self.h,
            color="purple", label="With Air Resistance", zorder=3)
        plt.plot(
            self.t, self.h_noair,
            color="green", label="No Air Resistance", zorder=2)

        plt.scatter(
            self.t[self.t_impact], self.h[self.t_impact],
            color="blue", zorder=4, label="Impact Point (With Air)")
        plt.scatter(
            self.t[self.t_noair_impact], self.h_noair[self.t_noair_impact],
            color="red", zorder=3, label="Impact Point (No Air)")

        plt.grid(which="both", linestyle="-", color="gray", linewidth=1)
        plt.minorticks_on()
        plt.grid(which="minor", linestyle=":", color="black", linewidth=0.5)

        plt.title(
            f'Height over Time for {self.name} '
            f'({self.positive_direction.capitalize()} Positive)')
        plt.legend()
        plt.xlabel(r'Time ($s$)')
        plt.ylabel(r'Height ($m$)')
        plt.show()

    def plot_velocity(self):
        """
        Plot velocity vs. time for both cases.
        """
        plt.figure(1)
        max_val = np.ceil(abs(self.v_noair[self.t_noair_impact]) / 5.0) * 5
        y_step = max_val * 0.05
        min_val = 2 * y_step
        if self.sign == 1:
            plt.ylim(-max_val, min_val)
            plt.text(
                self.t[self.t_impact],
                self.v[self.t_impact] - 2 * y_step,
                f'{round(self.t[self.t_impact], 2)}$s$',
                color="blue", fontsize=12, ha="center")
            plt.text(
                self.t[self.t_impact],
                self.v[self.t_impact] - 3.5 * y_step, 
                f'{round(self.v[self.t_impact], 2)}$m/s$', 
                color="blue", fontsize=12, ha="center")
            plt.text(
                self.t[self.t_noair_impact],
                self.v_noair[self.t_noair_impact] + 2.5 * y_step, 
                f'{round(self.t[self.t_noair_impact], 2)}$s$', 
                color="red", fontsize=12, ha="center")
            plt.text(
                self.t[self.t_noair_impact],
                self.v_noair[self.t_noair_impact] + 1 * y_step, 
                f'{round(self.v_noair[self.t_noair_impact], 2)}$m/s$', 
                color="red", fontsize=12, ha="center")
        else:
            plt.ylim(-min_val, max_val)
            plt.text(
                self.t[self.t_impact], 
                self.v[self.t_impact] + 2.5 * y_step, 
                f'{round(self.t[self.t_impact], 2)}$s$', 
                color="blue", fontsize=12, ha="center")
            plt.text(
                self.t[self.t_impact], 
                self.v[self.t_impact] + 1 * y_step, 
                f'{round(self.v[self.t_impact], 2)}$m/s$', 
                color="blue", fontsize=12, ha="center")
            plt.text(
                self.t[self.t_noair_impact], 
                self.v_noair[self.t_noair_impact] - 2 * y_step, 
                f'{round(self.t[self.t_noair_impact], 2)}$s$', 
                color="red", fontsize=12, ha="center")
            plt.text(
                self.t[self.t_noair_impact], 
                self.v_noair[self.t_noair_impact] - 3.5 * y_step, 
                f'{round(self.v_noair[self.t_noair_impact], 2)}$m/s$', 
                color="red", fontsize=12, ha="center")


        plt.plot(
            self.t, self.v, 
            color="purple", label="With Air Resistance", zorder=3)
        plt.plot(
            self.t, self.v_noair, 
            color="green", label="No Air Resistance", zorder=2)

        plt.scatter(
            self.t[self.t_impact], self.v[self.t_impact], 
            color="blue", zorder=4, label="Impact Point (With Air)")
        plt.scatter(
            self.t[self.t_noair_impact], self.v_noair[self.t_noair_impact], 
            color="red", zorder=3, label="Impact Point (No Air)")

        plt.grid(which="both", linestyle="-", color="gray", linewidth=1)
        plt.minorticks_on()
        plt.grid(which="minor", linestyle=":", color="black", linewidth=0.5)

        plt.title(
            f'Velocity over Time for {self.name} '
            f'({self.positive_direction.capitalize()} Positive)')
        plt.legend()
        plt.xlabel(r'Time ($s$)')
        plt.ylabel(r'Velocity ($m/s$)')
        plt.show()
    
    def plot_acceleration(self):
        """
        Plot acceleration vs. time for both cases.
        """
        plt.figure(2)
        max_val = np.ceil(self.g / 2.0) * 2 + 1
        y_step = max_val * 0.05
        min_val = y_step
        if self.sign == 1:
            plt.ylim(-max_val, min_val)
            plt.text(
                self.t[self.t_impact], 
                self.a[self.t_impact] - 2 * y_step, 
                f'{round(self.t[self.t_impact], 2)}$s$', 
                color="blue", fontsize=12, ha="center")
            plt.text(
                self.t[self.t_impact], 
                self.a[self.t_impact] - 3.5 * y_step, 
                f'{round(self.a[self.t_impact], 2)}$m/s^2$', 
                color="blue", fontsize=12, ha="center")
            plt.text(
                self.t[self.t_noair_impact], 
                self.a_noair[self.t_noair_impact] + 2.5 * y_step, 
                f'{round(self.t[self.t_noair_impact], 2)}$s$', 
                color="red", fontsize=12, ha="center")
            plt.text(
                self.t[self.t_noair_impact], 
                self.a_noair[self.t_noair_impact] + 1 * y_step, 
                f'{round(self.a_noair[self.t_noair_impact], 2)}$m/s^2$', 
                color="red", fontsize=12, ha="center")
        else:
            plt.ylim(-min_val, max_val)
            plt.text(
                self.t[self.t_impact], 
                self.a[self.t_impact] + 2.5 * y_step, 
                f'{round(self.t[self.t_impact], 2)}$s$', 
                color="blue", fontsize=12, ha="center")
            plt.text(
                self.t[self.t_impact], 
                self.a[self.t_impact] + 1 * y_step, 
                f'{round(self.a[self.t_impact], 2)}$m/s^2$', 
                color="blue", fontsize=12, ha="center")
            plt.text(
                self.t[self.t_noair_impact], 
                self.a_noair[self.t_noair_impact] - 2 * y_step, 
                f'{round(self.t[self.t_noair_impact], 2)}$s$', 
                color="red", fontsize=12, ha="center")
            plt.text(
                self.t[self.t_noair_impact], 
                self.a_noair[self.t_noair_impact] - 3.5 * y_step, 
                f'{round(self.a_noair[self.t_noair_impact], 2)}$m/s^2$', 
                color="red", fontsize=12, ha="center")

        plt.plot(
            self.t, self.a, 
            color="purple", label="With Air Resistance", zorder=3)
        plt.plot(
            self.t, self.a_noair, 
            color="green", label="No Air Resistance", zorder=2)

        plt.scatter(
            self.t[self.t_impact], self.a[self.t_impact], 
            color="blue", zorder=4, label="Impact Point (With Air)")

        plt.scatter(
            self.t[self.t_noair_impact], self.a_noair[self.t_noair_impact], 
            color="red", zorder=3, label="Impact Point (No Air)")

        plt.grid(which="both", linestyle="-", color="gray", linewidth=1)
        plt.minorticks_on()
        plt.grid(which="minor", linestyle=":", color="black", linewidth=0.5)

        plt.title(
            f'Acceleration over Time for {self.name} '
            f'({self.positive_direction.capitalize()} Positive)')
        plt.legend()
        plt.xlabel(r'Time ($s$)')
        plt.ylabel(r'Acceleration $(m/s^2)$')
        plt.show()

    def run_simulation(self):
        """
        Run the full simulation.
        """
        self.calculate()
        self.find_impact_times()
        self.display_results()
        self.plot_height()
        self.plot_velocity()
        self.plot_acceleration()

# Run the simulation
if __name__ == "__main__":
    sim_1 = FallingSphere(
        "Sphere 1", 
        distance_from_ground = 50, end_time = 13.0, mass = 0.1, C = 0.05)
    sim_1.run_simulation()

    sim_2 = FallingSphere(
        "Sphere 2",
        distance_from_ground = 50, end_time = 13.0, mass = 0.1, C = 0.05, 
        positive_direction = "downwards")
    sim_2.run_simulation()


