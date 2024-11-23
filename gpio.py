import RPi.GPIO as GPIO

# use P1 header pin numbering convention
GPIO.setmode(GPIO.BOARD)

GPIO.setup(12, GPIO.OUT)
GPIO.output(12, GPIO.HIGH)

sleep(4)
GPIO.output(12, GPIO.LOW)
