from robot_hat import Pin, ADC, PWM, Servo, fileDB
from robot_hat import Grayscale_Module, Ultrasonic, utils
import time
import os
from .logging_setup import init_logger
from .sensors import SafeUltrasonic, SafeGrayscale
from .poller import SensorPoller
from .servo_control import ComprehensiveServoController


def constrain(x, min_val, max_val):
    '''
    Constrains value to be within a range.
    '''
    return max(min_val, min(max_val, x))

class Picarx(object):
    CONFIG = '/opt/picar-x/picar-x.conf'

    DEFAULT_LINE_REF = [1000, 1000, 1000]
    DEFAULT_CLIFF_REF = [500, 500, 500]

    DIR_MIN = -30
    DIR_MAX = 30
    CAM_PAN_MIN = -90
    CAM_PAN_MAX = 90
    CAM_TILT_MIN = -35
    CAM_TILT_MAX = 65

    PERIOD = 4095
    PRESCALER = 10
    TIMEOUT = 0.02

    # servo_pins: camera_pan_servo, camera_tilt_servo, direction_servo
    # motor_pins: left_swicth, right_swicth, left_pwm, right_pwm
    # grayscale_pins: 3 adc channels
    # ultrasonic_pins: trig, echo2
    # config: path of config file
    def __init__(self, servo_pins: list = ['P0', 'P1', 'P2'], motor_pins: list = ['D4', 'D5', 'P13', 'P12'], grayscale_pins: list = ['A0', 'A1', 'A2'], ultrasonic_pins: list = ['D2', 'D3'], config: str = CONFIG):
        
        # logger
        self.log = init_logger("picarx")

        # boot timestamp for early rate-limiting
        self._boot_ts = time.time()

        # reset robot_hat with guard
        try:
            utils.reset_mcu()
            time.sleep(0.2)
        except Exception as e:
            # Continue but log, allows running in dev environments without hardware
            self.log.warning(f"MCU reset failed or not available: {e}")

        # --------- config_flie ---------
        try:
            self.config_flie = fileDB(config, 777, os.getlogin())
        except Exception:
            # fallback to user config under home dir when /opt path not writable
            home_conf = os.path.join(os.path.expanduser("~"), ".picarx.conf")
            self.config_flie = fileDB(home_conf, 777, os.getlogin())
            self.log.info(f"Using fallback config at {home_conf}")

        # --------- servos init ---------
        self.cam_pan = Servo(servo_pins[0])
        self.cam_tilt = Servo(servo_pins[1])
        self.dir_servo_pin = Servo(servo_pins[2])
        
        # Initialize COMPREHENSIVE servo controller with advanced anti-jitter protection
        self.servo_controller = ComprehensiveServoController(self)
        
        # Legacy support - track last sent angles for compatibility
        self._last_pan_angle_sent = None
        self._last_tilt_angle_sent = None
        self._last_dir_angle_sent = None
        
        # get calibration values
        self.dir_cali_val = float(self.config_flie.get("picarx_dir_servo", default_value=0))
        self.cam_pan_cali_val = float(self.config_flie.get("picarx_cam_pan_servo", default_value=0))
        self.cam_tilt_cali_val = float(self.config_flie.get("picarx_cam_tilt_servo", default_value=0))
        
        # set servos to init angle using the comprehensive controller
        self.servo_controller.force_servo_position("cam_pan", self.cam_pan_cali_val)
        self.servo_controller.force_servo_position("cam_tilt", self.cam_tilt_cali_val) 
        self.servo_controller.force_servo_position("dir_servo", self.dir_cali_val)
        
        # Update legacy trackers
        self._last_pan_angle_sent = float(self.cam_pan_cali_val)
        self._last_tilt_angle_sent = float(self.cam_tilt_cali_val)
        self._last_dir_angle_sent = float(self.dir_cali_val)

        # --------- motors init ---------
        self.left_rear_dir_pin = Pin(motor_pins[0])
        self.right_rear_dir_pin = Pin(motor_pins[1])
        self.left_rear_pwm_pin = PWM(motor_pins[2])
        self.right_rear_pwm_pin = PWM(motor_pins[3])
        self.motor_direction_pins = [self.left_rear_dir_pin, self.right_rear_dir_pin]
        self.motor_speed_pins = [self.left_rear_pwm_pin, self.right_rear_pwm_pin]
        # Track last applied PWM duty (%) for gentle braking
        self._last_pwm_duty = [0, 0]
        # Track last commanded direction (-1, 1)
        self._last_dir = [1, -1]
        # get calibration values
        self.cali_dir_value = self.config_flie.get("picarx_dir_motor", default_value="[1, 1]")
        self.cali_dir_value = [int(i.strip()) for i in self.cali_dir_value.strip().strip("[]").split(",")]
        self.cali_speed_value = [0, 0]
        self.dir_current_angle = 0
        # init pwm
        for pin in self.motor_speed_pins:
            try:
                pin.period(self.PERIOD)
                pin.prescaler(self.PRESCALER)
            except Exception as e:
                self.log.warning(f"Motor PWM init failed: {e}")

        # --------- grayscale module init ---------
        # Use safe wrappers for sensors
        a0, a1, a2 = grayscale_pins
        self.grayscale = SafeGrayscale(a0, a1, a2)
        # get reference
        self.line_reference = self.config_flie.get("line_reference", default_value=str(self.DEFAULT_LINE_REF))
        self.line_reference = [float(i) for i in self.line_reference.strip().strip('[]').split(',')]
        self.cliff_reference = self.config_flie.get("cliff_reference", default_value=str(self.DEFAULT_CLIFF_REF))
        self.cliff_reference = [float(i) for i in self.cliff_reference.strip().strip('[]').split(',')]
        # transfer reference
        try:
            self.grayscale.reference(self.line_reference)
        except Exception as e:
            self.log.warning(f"Set grayscale reference failed: {e}")

        # --------- ultrasonic init ---------
        trig, echo = ultrasonic_pins
        self.ultrasonic = SafeUltrasonic(trig, echo)

        # --------- background sensor poller ---------
        try:
            self._poller = SensorPoller(
                ultrasonic_read=self.ultrasonic.read,
                grayscale_read=self.grayscale.read,
            )
            self._poller.start()
        except Exception as e:
            self._poller = None
            self.log.warning(f"SensorPoller init failed: {e}")
        
    def set_motor_speed(self, motor, speed):
        ''' set motor speed
        
        param motor: motor index, 1 means left motor, 2 means right motor
        type motor: int
        param speed: speed
        type speed: int      
        '''
        speed = constrain(speed, -100, 100)
        motor -= 1
        # Determine desired direction based on sign before abs
        if speed >= 0:
            direction = 1 * self.cali_dir_value[motor]
        else:
            direction = -1 * self.cali_dir_value[motor]
        # Map to PWM duty and apply calibration
        mag = abs(speed)
        pwm = 0 if mag == 0 else (int(mag / 2) + 50)
        pwm = max(0, pwm - self.cali_speed_value[motor])
        try:
            if pwm == 0:
                # Avoid toggling direction lines when stopping to prevent jerk
                self.motor_speed_pins[motor].pulse_width_percent(0)
                self._last_pwm_duty[motor] = 0
                # keep last direction unchanged
            else:
                if direction < 0:
                    self.motor_direction_pins[motor].high()
                else:
                    self.motor_direction_pins[motor].low()
                self.motor_speed_pins[motor].pulse_width_percent(pwm)
                # Record last applied state for gentle braking
                self._last_pwm_duty[motor] = int(pwm)
                self._last_dir[motor] = -1 if direction < 0 else 1

        except Exception as e:
            self.log.error(f"set_motor_speed failed (motor={motor+1}, speed={speed}): {e}")

    def motor_speed_calibration(self, value):
        self.cali_speed_value = value
        if value < 0:
            self.cali_speed_value[0] = 0
            self.cali_speed_value[1] = abs(self.cali_speed_value)
        else:
            self.cali_speed_value[0] = abs(self.cali_speed_value)
            self.cali_speed_value[1] = 0
    def motor_direction_calibrate(self, motor, value):
        ''' set motor direction calibration value
        
        param motor: motor index, 1 means left motor, 2 means right motor
        type motor: int
        param value: speed
        type value: int
        '''      
        motor -= 1
        if value == 1:
            self.cali_dir_value[motor] = 1
        elif value == -1:
            self.cali_dir_value[motor] = -1
        self.config_flie.set("picarx_dir_motor", self.cali_dir_value)

    def dir_servo_calibrate(self, value):
        self.dir_cali_val = value
        self.config_flie.set("picarx_dir_servo", "%s"%value)
        self.dir_servo_pin.angle(value)

    def set_dir_servo_angle(self, value):
        """Set steering servo with comprehensive anti-jitter protection."""
        try:
            self.dir_current_angle = constrain(value, self.DIR_MIN, self.DIR_MAX)
            target = self.dir_current_angle + self.dir_cali_val
            
            # Use lower priority for steering since it needs to be more responsive
            success = self.servo_controller.set_servo_angle("dir_servo", target, priority=-1, source="set_dir_servo_angle")
            
            if success:
                self._last_dir_angle_sent = float(target)
                self.log.debug(f"Steering command accepted: {value:.1f}° -> {target:.1f}°")
            else:
                self.log.debug(f"Steering command rejected by anti-jitter system")
                
        except Exception as e:
            self.log.warning(f"set_dir_servo_angle failed: {e}")

    def cam_pan_servo_calibrate(self, value):
        self.cam_pan_cali_val = value
        self.config_flie.set("picarx_cam_pan_servo", "%s"%value)
        # Use force command for calibration
        self.servo_controller.force_servo_position("cam_pan", value)
        self._last_pan_angle_sent = float(value)

    def cam_tilt_servo_calibrate(self, value):
        self.cam_tilt_cali_val = value
        self.config_flie.set("picarx_cam_tilt_servo", "%s"%value)
        # Use force command for calibration
        self.servo_controller.force_servo_position("cam_tilt", value)
        self._last_tilt_angle_sent = float(value)

    def set_cam_pan_angle(self, value):
        """Set camera pan angle with COMPREHENSIVE anti-jitter protection."""
        try:
            target = constrain(value, self.CAM_PAN_MIN, self.CAM_PAN_MAX)
            success = self.servo_controller.set_servo_angle("cam_pan", target, source="set_cam_pan_angle")
            
            if success:
                # Update calibrated output for legacy compatibility
                out = -1 * (target + -1 * self.cam_pan_cali_val)
                self._last_pan_angle_sent = float(out)
                self.log.debug(f"Pan command accepted: {target:.1f}° -> {out:.1f}°")
            else:
                self.log.debug(f"Pan command rejected by anti-jitter system")
                
        except Exception as e:
            self.log.warning(f"set_cam_pan_angle failed: {e}")

    def set_cam_tilt_angle(self,value):
        """Set camera tilt angle with COMPREHENSIVE anti-jitter protection."""
        try:
            target = constrain(value, self.CAM_TILT_MIN, self.CAM_TILT_MAX)
            success = self.servo_controller.set_servo_angle("cam_tilt", target, source="set_cam_tilt_angle")
            
            if success:
                # Update calibrated output for legacy compatibility
                out = -1 * (target + -1 * self.cam_tilt_cali_val)
                self._last_tilt_angle_sent = float(out)
                self.log.debug(f"Tilt command accepted: {target:.1f}° -> {out:.1f}°")
            else:
                self.log.debug(f"Tilt command rejected by anti-jitter system")
            
        except Exception as e:
            self.log.warning(f"set_cam_tilt_angle failed: {e}")

    def get_servo_status(self):
        """Get comprehensive servo status from the servo controller"""
        return self.servo_controller.get_servo_status()
    
    def emergency_stop_servos(self):
        """Emergency stop all servo operations"""
        self.servo_controller.emergency_stop_all_servos()
    
    def resume_servos(self):
        """Resume servo operations after emergency stop"""
        self.servo_controller.resume_servo_operations()
    
    @property
    def cam_pan_angle(self):
        """Get current camera pan angle"""
        return self.servo_controller.servo_states.get("cam_pan", {}).get("current_angle", 0.0)
    
    @property  
    def cam_tilt_angle(self):
        """Get current camera tilt angle"""
        return self.servo_controller.servo_states.get("cam_tilt", {}).get("current_angle", 0.0)

    def set_power(self, speed):
        self.set_motor_speed(1, speed)
        self.set_motor_speed(2, speed)

    def backward(self, speed):
        current_angle = self.dir_current_angle
        if current_angle != 0:
            abs_current_angle = abs(current_angle)
            if abs_current_angle > self.DIR_MAX:
                abs_current_angle = self.DIR_MAX
            power_scale = (100 - abs_current_angle) / 100.0 
            if (current_angle / abs_current_angle) > 0:
                self.set_motor_speed(1, -1*speed)
                self.set_motor_speed(2, speed * power_scale)
            else:
                self.set_motor_speed(1, -1*speed * power_scale)
                self.set_motor_speed(2, speed )
        else:
            self.set_motor_speed(1, -1*speed)
            self.set_motor_speed(2, speed)  

    def forward(self, speed):
        current_angle = self.dir_current_angle
        if current_angle != 0:
            abs_current_angle = abs(current_angle)
            if abs_current_angle > self.DIR_MAX:
                abs_current_angle = self.DIR_MAX
            power_scale = (100 - abs_current_angle) / 100.0
            if (current_angle / abs_current_angle) > 0:
                self.set_motor_speed(1, 1*speed * power_scale)
                self.set_motor_speed(2, -speed) 
            else:
                self.set_motor_speed(1, speed)
                self.set_motor_speed(2, -1*speed * power_scale)
        else:
            self.set_motor_speed(1, speed)
            self.set_motor_speed(2, -1*speed)                  

    def stop(self):
        '''
        Execute twice to make sure it stops
        '''
        for _ in range(2):
            try:
                self.motor_speed_pins[0].pulse_width_percent(0)
                self.motor_speed_pins[1].pulse_width_percent(0)
            except Exception as e:
                self.log.warning(f"stop failed to set PWM to 0: {e}")
            time.sleep(0.002)

    def slow_stop(self, duration_s: float = 0.6, steps: int = 12):
        """Gradually ramp motor PWM down to zero for a smoother stop.

        - Uses last applied PWM duty tracked from set_motor_speed.
        - Does not toggle direction pins; only reduces PWM.
        """
        try:
            steps = max(2, int(steps))
            duration_s = max(0.1, float(duration_s))
            d0_l, d0_r = self._last_pwm_duty
            dt = duration_s / steps
            for k in range(steps, -1, -1):
                ratio = k / float(steps)
                dl = int(d0_l * ratio)
                dr = int(d0_r * ratio)
                try:
                    self.motor_speed_pins[0].pulse_width_percent(dl)
                    self.motor_speed_pins[1].pulse_width_percent(dr)
                except Exception:
                    # Fall back to immediate stop if PWM write fails
                    try:
                        self.motor_speed_pins[0].pulse_width_percent(0)
                        self.motor_speed_pins[1].pulse_width_percent(0)
                    except Exception:
                        pass
                    break
                time.sleep(dt)
        finally:
            # ensure fully stopped
            try:
                self.motor_speed_pins[0].pulse_width_percent(0)
                self.motor_speed_pins[1].pulse_width_percent(0)
            except Exception:
                pass

    def get_distance(self):
        # Prefer fast, cached reading
        if getattr(self, "_poller", None):
            v = self._poller.get_distance()
            if v is not None:
                return v
        # Fallback to direct read
        try:
            return self.ultrasonic.read()
        except Exception as e:
            self.log.warning(f"get_distance error: {e}")
            return None

    def set_grayscale_reference(self, value):
        if isinstance(value, list) and len(value) == 3:
            self.line_reference = value
            self.grayscale.reference(self.line_reference)
            self.config_flie.set("line_reference", self.line_reference)
        else:
            raise ValueError("grayscale reference must be a 1*3 list")

    def get_grayscale_data(self):
        # Prefer fast, cached reading
        if getattr(self, "_poller", None):
            values = self._poller.get_grayscale()
            if values is None:
                values = self.grayscale.read()
        else:
            values = self.grayscale.read()
        return list.copy(values) if values is not None else [0.0, 0.0, 0.0]

    def get_line_status(self,gm_val_list):
        return self.grayscale.read_status(gm_val_list)

    def set_line_reference(self, value):
        self.set_grayscale_reference(value)

    def get_cliff_status(self,gm_val_list):
        for i in range(0,3):
            if gm_val_list[i]<=self.cliff_reference[i]:
                return True
        return False

    def set_cliff_reference(self, value):
        if isinstance(value, list) and len(value) == 3:
            self.cliff_reference = value
            self.config_flie.set("cliff_reference", self.cliff_reference)
        else:
            raise ValueError("grayscale reference must be a 1*3 list")

    def reset(self):
        try:
            self.stop()
        finally:
            try:
                self.set_dir_servo_angle(0)
                self.set_cam_tilt_angle(0)
                self.set_cam_pan_angle(0)
            except Exception as e:
                self.log.warning(f"reset servo angles failed: {e}")
            # stop poller last
            try:
                if getattr(self, "_poller", None):
                    self._poller.stop()
            except Exception:
                pass

if __name__ == "__main__":
    px = Picarx()
    px.forward(50)
    time.sleep(1)
    px.stop()
