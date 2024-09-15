You can either install the dependencies globally, or create a virtual environment. But in either case, you will need:

pip install opencv-python
pip install easyocr

There may be a depdenecy conflict. If so, this can be solved:

pip uninstall opencv-python-headless
pip uninstall opencv-python
pip install opencv-python

Run with:

python main.py