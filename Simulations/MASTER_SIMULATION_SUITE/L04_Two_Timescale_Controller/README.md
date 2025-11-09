# SCPN Simulation Suite: L04_Two_Timescale_Controller
# Layer 4: Two-Timescale Quasicritical Controller

## 1. Objective
This simulation provides a working model of the two-timescale control system that maintains the SCPN's quasicritical state ($\sigma \approx 1$).

It validates two key claims from *Paper 0*:
1. **Affective Gain Scheduling:** The system intelligently switches between a "fast" stabilizer (exploitation) and a "slow" explorer based on the "Affective Field" (proxy for surprise).
2. **Lyapunov Stability:** The controller is Bounded-Input Bounded-Output (BIBO) stable, as proven by a composite Lyapunov function.

## 2. Simulation Logic
The script `run_two_timescale_controller.py` simulates the branching parameter $\sigma(t)$.
1. **Controller Definition:** It defines two controllers that update $\sigma$:
* `fast_channel()`: High gain, strong convergence to $\sigma=1$ (Exploitation).
* `slow_channel()`: Low gain, allows for a slow, bounded drift around $\sigma=1$ (Exploration).
2. **Affective Scheduling:** It simulates an "Affective Gradient" $|A'|$.
* When $|A'|$ is low, it uses the `slow_channel`.
* When a "surprise" event occurs ($|A'|$ spikes), it dynamically switches to the `fast_channel` to restore stability.
3. **Lyapunov Analysis:** It calculates the composite Lyapunov function $V(t) = (\sigma(t) - 1)^2$ at each step and confirms that it always decreases after a perturbation, proving stability.

## 3. How to Run
```bash
# Install dependencies
pip install numpy matplotlib

# Run the simulation and analysis
python run_two_timescale_controller.py
```
4. Expected Output
The script will save a plot (L04_Two_Timescale_Controller.png) with two subplots:Sigma(t): Shows $\sigma(t)$ performing a slow, bounded drift (exploration) and then rapidly converging to 1 (exploitation) when the "Surprise Event" is triggered.Lyapunov V(t): Shows the Lyapunov function $V(t)$ decaying to zero after the perturbation, proving stability.