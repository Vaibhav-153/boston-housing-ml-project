# 🏠 Boston Housing Price Prediction - ML Project

This is an **end-to-end Machine Learning project** to predict **Boston housing prices** using regression models. The project uses the Boston Housing dataset with 13 important features influencing home prices.

---

## 📁 Project Structure

```
boston-price-predictor/
├── static/                # CSS & static assets
├── templates/             # HTML templates (home.html)
├── app.py                 # Flask backend
├── lr_model.pkl           # Trained Linear Regression model
├── scaling.pkl            # Standard Scaler object
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🔧 Software & Tools Requirements

| Tool     | Purpose                      | Link                                                    |
| -------- | ---------------------------- | ------------------------------------------------------- |
| GitHub   | Version Control & Hosting    | [github.com](https://github.com)                        |
| VS Code  | Code Editor                  | [code.visualstudio.com](https://code.visualstudio.com/) |
| Heroku   | App Deployment               | [heroku.com](https://heroku.com)                        |
| Git CLI  | Git Command Line             | [git-scm.com](https://git-scm.com/downloads)            |
| Anaconda | Python + Environment Manager | [anaconda.com](https://www.anaconda.com/)               |

---

## 🛠️ Setup Instructions (Using Conda)

### ✅ Step 1: Create & Activate Conda Environment

```bash
# Create a new environment with Python 3.12
conda create -p venv python=3.12 -y

# Activate the environment
conda activate ./venv
```

---

### ✅ Step 2: Install Project Dependencies

```bash
# Install required libraries
pip install -r requirements.txt
```

---

## 🌐 Git & GitHub Commands

### ✅ Git Configuration

```bash
# Set your name and email (configure Git)
git config --global user.name "Vaibhav Admane"
git config --global user.email "your-email@example.com"

# Check Git configuration
git config --global user.name
git config --global user.email
```

### ✅ Git Workflow

```bash
# Check current changes
git status

# Add a specific file
git add README.md

# Add all files
git add .

# Commit changes
git commit -m "Initial commit"

# View latest commit
git log -1

# Push changes to GitHub
git push origin main
```

---

## 🚀 Run the Project

```bash
# Run Flask app
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000/
```

---

> 📘 Tip: Learn more from the [Atlassian Git Tutorial](https://www.atlassian.com/git/tutorials)

