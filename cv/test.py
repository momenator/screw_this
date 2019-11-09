# python test.py --image images/example_01.png --width 0.955
import argparse
import processing

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

measurements = processing.get_measurements(args["image"], args["width"], is_display=True)
print(measurements)
