#!/usr/bin/env python3
"""Regenerate Python UI modules from Qt Designer .ui files."""

import subprocess


ui_files = [
	("./ui/main_window.ui", "./ui/main_window_ui.py"),
	("./ui/pages/home.ui", "./ui/pages/home_ui.py"),
	("./ui/pages/ImageLabeling.ui", "./ui/pages/ImageLabeling_ui.py"),
	("./ui/pages/SPMImageProcess.ui", "./ui/pages/SPMImageProcess_ui.py"),
	("./ui/pages/computerVision.ui", "./ui/pages/computerVision_ui.py"),
	("./ui/popUps/labelTextInput.ui", "./ui/popUps/labelTextInput_ui.py"),
	("./ui/popUps/imageProcessing.ui", "./ui/popUps/imageProcessing_ui.py"),
	("./ui/popUps/automaticAnnotation.ui", "./ui/popUps/automaticAnnotation_ui.py"),
	("./ui/popUps/predictions.ui", "./ui/popUps/predictions_ui.py"),
    ("./ui/popUps/setupML.ui", "./ui/popUps/setupML_ui.py")
]

for input_file, output_file in ui_files:
	subprocess.run(
		["pyuic6.exe", input_file, "-o", output_file],
		check=True,
	)





