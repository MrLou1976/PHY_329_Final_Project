from freegs import machine
from freegs.equilibrium import Equilibrium
from freegs.coil import Coil
from freegs.machine import Circuit
from freegs.gradshafranov import GSsparse
from freegs.gradshafranov import Greens
from freegs.jtor import ConstrainPaxisIp
from freegs import control
from freegs import picard
from freegs.critical import find_critical, find_separatrix
import gym
from gym import spaces
import numpy as np

class TokamakEnv(gym.Env):
    """
    Class to train an RL model to find the ideal coil current values to sustain a specific plasma current

    NOTE: The code does not yet work as intended as the current values determined by the RL model do not seem to have much of an effect
          but rather the psi value calculated from the control seems to heavily influence the plasma current solution. We are still unsure
          why this is the case and are troubleshooting how to incorporate psi calculations into our model.

    Args:
        target_current (float): desired plasma current to solve for

    Attributes:
        tokamak: a tokamak machine (currently set to TCV) from FreeGS
        eq: Equilibrium object from FreeGS specifying equilibrium space
        _R: Radial positions
        _Z: Vertical positions
        _psi: magnetic flux
        action_space: RL action space for poloidal field coils
        observation_space: observations to determine RL outcome
    """
    def __init__(self, target_current):
        super(TokamakEnv, self).__init__()
        self.tokamak = machine.TCV()
        self.eq = Equilibrium(self.tokamak, Rmin=0.1, Rmax=2.0, Zmin=-2.0, Zmax=2.0, nx=65, ny=65)
        self._R = self.eq.R
        self._Z = self.eq.Z
        self._psi = self.eq.psi()
        self.remove_controls()
        
        self.target_current = target_current

        # Action space: Adjust currents in poloidal field coils
        self.action_space = spaces.Box(low=-1000, high=1000, shape=(len(self.tokamak.coils),), dtype=np.float32)
        
        # Observation space: Plasma current
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        
    def seed(self, seed=None):
        """
        Create a random seed for the training model to use

        Args:
            seed (int): determines randomness 

        Returns:
            None
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def remove_controls(self):
        """
        Remove control for all coils so the control.constrain function does not affect current values
        """
        for coil in self.tokamak.coils:
            if type(coil[-1]) == Coil:
                coil[-1].control = False
            else: # Handles coil circuits
                coil[-1].control = False
                for c in coil[-1].coils:
                    c[1].control = False
    
    def reset(self):
        """
        Reset to initial conditions
        """
        self.eq = Equilibrium(self.tokamak, Rmin=0.1, Rmax=2.0, Zmin=-2.0, Zmax=2.0, nx=1+2**5, ny=1+2**5)
        return self._get_obs()
    
    def step(self, action):
        """
        Performs a step to adjust currents

        Args:
            action (np.ndarray): a numpy array filled with current values to apply to the coils

        Returns:
            self._get_obs (np.ndarray): calls the _get_obs function to return observations
            reward (float): calculated reward per step
            done (bool): indicates whether training should continue or stop
        """
        # Apply action (adjust coil currents)
        for (name, coil), current in zip(self.tokamak.coils, action):
            if hasattr(coil, "setCurrent"):  # Ensure the object has setCurrent method
                coil.setCurrent(current)
            elif hasattr(coil, "current"):  # For cases like 'Circuit'
                coil.current = current
            else:
                raise AttributeError(f"Cannot set current for coil '{name}'.")


        # Specify equilibrium constraints with emphasis on plasma pressure
        profiles = ConstrainPaxisIp(self.eq, 3e3, self.target_current, 0.4)
        
        # Sample points for X-Points and Isoflux (will add user input later)
        self._xpoints = [(0.7, -1.1), (0.7, 1.1)]
        self._isoflux = [(0.7, -1.1, 1.45, 0.0), (0.7, 1.1, 1.45, 0.0)]

        # Solve equilibrium constraints and determine magnetic flux psi
        constrain = control.constrain(xpoints=self._xpoints, gamma=1e-12, isoflux=self._isoflux)

        # Apply constraints to the equilibrium object
        constrain(self.eq)

        # Try to solve the Grad-Shafranov equation and move on if the solution does not converge
        try:
            picard.solve(self.eq, profiles, constrain)
        except:
            print("Picard solve failed to converge. Moving on...")

        # Calculate reward and check if done
        current_error = abs(self.eq.plasmaCurrent() - self.target_current)
        reward = - (current_error)
        done = current_error < 1e-3

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        """
        Get observed data

        Returns:
            np.ndarray: observed data (currently only plasma current)
        """
        return np.array([self.eq.plasmaCurrent()])