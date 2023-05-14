# ECE697MLFinalProject-CropYieldPrediction
This is a repository was created by a group of students at UMass Amherst to create a sample machine learning model to predict crop yields from an existing dataset of weather data.


# Info
--------

General Info
- Run on Windows
- Using Python 3.10.7 & venv

Link To Final Report
- [<span style="font-size:x-large;">Final Report</span>](https://docs.google.com/document/d/1-6q0iwsEXat8JwnOvRDJ6vi3KlRc1Vl3PnygjGronbQ/edit)

# Setup
--------

0. Setup GPU Acceleration (***Not Required***) Go to [here](https://pytorch.org/get-started/locally/), and follow instructions for relative python & CUDA+cuDNN installs 

1. Open a command prompt in the same folder the git is in

2. Use this command to create a local virtual environment `python -m venv ./`

3. Activate the virtual environment in the command prompt with `./Scripts/activate` on Linux, or `.\Scripts\activate.bat` on Windows (or .ps1 if using powershell)

4. Run `python -m pip install --upgrade pip` to upgrade pip

5. Import either GPU enabled pytorch`python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuxxx` (replace xxx with relative CUDA version used, use website in instruction 0 for further documentation)

6. Or if using CPU use `python -m pip install torch torchvision torchaudio` and set `useCUDA` at the top of `main.py` to `False`

7. Run `python -m pip install pandas scikit-learn`



