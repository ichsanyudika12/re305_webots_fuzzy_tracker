# Fuzzy Object Following e-puck Robot (Webots)

This project is a final semester assignment for the Control Systems course. 
It demonstrates a **fuzzy logic controller** applied to the e-puck robot in Webots. 
The robot can detect a colored object using its camera and avoid obstacles using 
proximity sensors. The fuzzy logic controller computes the robot’s linear and 
angular speed in real time.

---

## Requirements

- Webots R2023 or newer
- e-puck robot model (built-in Webots)
- Python 3.8 or newer
- OpenCV 4.x
- NumPy
- scikit-fuzzy

Install Python dependencies:

```bash
pip install numpy opencv-python scikit-fuzzy
