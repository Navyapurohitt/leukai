# LeukAI — Federated Leukaemia Detection Web App

A Flask web application that runs a federated-trained lightweight CNN to classify
blood smear images as Leukaemia (ALL) or Healthy (HEM).

---

## Project Structure

```
leukemia-app/
├── app.py                  ← Flask backend (prediction API)
├── templates/
│   └── index.html          ← Frontend website
├── model/
│   ├── model.pth           ← ⚠ YOU ADD THIS (from Colab training)
│   └── class_names.json    ← ⚠ YOU ADD THIS (from Colab training)
├── requirements.txt
├── Procfile                ← For Render deployment
└── train_on_colab.ipynb    ← Training notebook
```

---

## Step 1 — Train the Model on Google Colab

1. Open `train_on_colab.ipynb` in Google Colab
2. Set runtime to **GPU (T4)**
3. Upload your `kaggle.json` API key when prompted
4. Run all cells — training takes ~20–30 minutes
5. At the end, two files will auto-download:
   - `model.pth`
   - `class_names.json`
6. Place both files inside the `model/` folder of this project

---

## Step 2 — Test Locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

---

## Step 3 — Deploy to Render (Free)

1. Push this folder to a **GitHub repository**
   ```bash
   git init
   git add .
   git commit -m "initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/leukemia-app.git
   git push -u origin main
   ```

2. Go to **https://render.com** → Sign up (free) → New → Web Service

3. Connect your GitHub repo

4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Instance Type:** Free

5. Click **Deploy** — your site will be live at `https://your-app-name.onrender.com`

> Note: The free Render tier sleeps after 15 min of inactivity. For a demo/presentation,
> open the site 1–2 minutes before showing it so it wakes up.

---

## Important Notes

- The app works in **demo mode** even without `model.pth` — it returns simulated predictions
- Once `model.pth` is added, predictions are real
- This is for **research/academic demonstration only** — not a clinical tool
- Model: Lightweight CNN, ~340 KB, trained with FedAvg across 5 simulated clients
