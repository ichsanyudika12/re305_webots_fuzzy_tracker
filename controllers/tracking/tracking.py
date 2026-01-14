from controller import Robot, Camera, DistanceSensor
import numpy as np
import cv2
import skfuzzy as fuzz
from skfuzzy import control

# === ROBOT PARAMETERS ===
r = 0.0205
b = 0.052
max_speed = 6.28
timer = 0
stop_dist = 100.0 

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice('camera')
camera.enable(timestep)

ps = [robot.getDevice(f'ps{i}') for i in range(8)]
for s in ps:
    s.enable(timestep)

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
for m in [left_motor, right_motor]:
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

# === FUZZY LOGIC ===
# input
error_pos = control.Antecedent(np.arange(-1.1, 1.1, 0.1), 'PositionError')
left_sensor = control.Antecedent(np.arange(0, 1001, 10), 'LeftSensor')
right_sensor = control.Antecedent(np.arange(0, 1001, 10), 'RightSensor')

# output
speed = control.Consequent(np.arange(0, 0.13, 0.01), 'speed')
turn_rate = control.Consequent(np.arange(-2.5, 2.6, 0.1), 'turn_rate')

error_pos['LEFT']   = fuzz.trimf(error_pos.universe, [-1.1, -1.1, 0])
error_pos['CENTER'] = fuzz.trimf(error_pos.universe, [-0.4, 0, 0.4])
error_pos['RIGHT']  = fuzz.trimf(error_pos.universe, [0, 1.1, 1.1])

left_sensor['SAFE']   = fuzz.trapmf(left_sensor.universe, [0, 0, 40, 80]) 
left_sensor['DANGER'] = fuzz.trapmf(left_sensor.universe, [50, 200, 1000, 1000])

right_sensor['SAFE']   = fuzz.trapmf(right_sensor.universe, [0, 0, 40, 80]) 
right_sensor['DANGER'] = fuzz.trapmf(right_sensor.universe, [50, 200, 1000, 1000])

speed['SLOW'] = fuzz.trimf(speed.universe, [0, 0.02, 0.07])
speed['FAST'] = fuzz.trimf(speed.universe, [0.04, 0.10, 0.12])

turn_rate['TURN_RIGHT'] = fuzz.trimf(turn_rate.universe, [-2.5, -2.5, -0.2])
turn_rate['FORWARD']    = fuzz.trimf(turn_rate.universe, [-0.5, 0, 0.5])
turn_rate['TURN_LEFT']  = fuzz.trimf(turn_rate.universe, [0.2, 2.5, 2.5])

rules = [
    control.Rule(left_sensor['DANGER'], (turn_rate['TURN_RIGHT'], speed['SLOW'])),
    control.Rule(right_sensor['DANGER'], (turn_rate['TURN_LEFT'], speed['SLOW'])),
    control.Rule(left_sensor['DANGER'] & right_sensor['DANGER'], (turn_rate['FORWARD'], speed['SLOW'])),
    control.Rule(left_sensor['SAFE'] & right_sensor['SAFE'] & error_pos['CENTER'], (speed['FAST'], turn_rate['FORWARD'])),
    control.Rule(left_sensor['SAFE'] & right_sensor['SAFE'] & error_pos['LEFT'],   (speed['SLOW'], turn_rate['TURN_LEFT'])),
    control.Rule(left_sensor['SAFE'] & right_sensor['SAFE'] & error_pos['RIGHT'],  (speed['SLOW'], turn_rate['TURN_RIGHT']))
]

fuzzy_sim = control.ControlSystemSimulation(control.ControlSystem(rules))

# === OPENCV WINDOWS ===
cv2.namedWindow("View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("View", 480, 240)

cv2.namedWindow("Mask View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mask View", 480, 240)

# === MAIN LOOP ===
while robot.step(timestep) != -1:
    sensor_values = [s.getValue() for s in ps]
    front_distance = max(sensor_values[0], sensor_values[7])
    
    val_left_sensor = np.clip(max(sensor_values[5], sensor_values[6], sensor_values[7]), 0, 1000)
    val_right_sensor = np.clip(max(sensor_values[1], sensor_values[2], sensor_values[0]), 0, 1000)

    width, height = camera.getWidth(), camera.getHeight()
    img = np.frombuffer(camera.getImage(), np.uint8).reshape((height, width, 4))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(img_hsv, np.array([0, 150, 150]), np.array([80, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    linear_speed = 0.0
    angular_speed = 0.0

    if contours and max(cv2.contourArea(c) for c in contours) > 80:
        timer = 0
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        object_pos = ((x + w // 2) - width / 2) / (width / 2)
        
        if front_distance >= stop_dist:
            linear_speed = 0.0
            angular_speed = 0.0
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)
        else:
            fuzzy_sim.input['LeftSensor'] = val_left_sensor
            fuzzy_sim.input['RightSensor'] = val_right_sensor
            fuzzy_sim.input['PositionError'] = object_pos
            fuzzy_sim.compute()
            linear_speed = fuzzy_sim.output['speed']
            angular_speed = fuzzy_sim.output['turn_rate']
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    else:
        timer += 1
        if val_right_sensor > 100 or val_left_sensor > 100:
            fuzzy_sim.input['LeftSensor'] = val_left_sensor
            fuzzy_sim.input['RightSensor'] = val_right_sensor
            fuzzy_sim.input['PositionError'] = 0.0
            fuzzy_sim.compute()
            linear_speed = fuzzy_sim.output['speed']
            angular_speed = fuzzy_sim.output['turn_rate']
        else:
            linear_speed = 0.0
            angular_speed = 1.0

    left_speed = (linear_speed - (angular_speed * b) / 2) / r
    right_speed = (linear_speed + (angular_speed * b) / 2) / r

    print(
        f"[INFO ]\n"
        f"  Front distance      : {front_distance:6.1f}\n"
        f"  Linear speed        : {linear_speed:4.2f} m/s\n"
        f"  Angular speed       : {angular_speed:5.2f} rad/s\n"
        f"  Left motor          : {left_speed:+5.2f} rad/s\n"
        f"  Right motor         : {right_speed:+5.2f} rad/s\n"
    )

    left_motor.setVelocity(np.clip(left_speed, -max_speed, max_speed))
    right_motor.setVelocity(np.clip(right_speed, -max_speed, max_speed))

    cv2.imshow("View", cv2.resize(img_bgr, (480, 240)))
    cv2.imshow("Mask View", cv2.resize(mask, (480, 240)))
    
    if cv2.waitKey(1) == 27: 
        break

cv2.destroyAllWindows()