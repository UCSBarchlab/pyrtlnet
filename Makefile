requirements.txt: requirements.in FORCE
	pip install --upgrade pip-tools
	pip-compile --upgrade requirements.in

FORCE:
