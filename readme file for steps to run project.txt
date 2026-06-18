Steps to run the Hybrid Fake News Detection Project

🚀 VS Code Execution Steps

Step 1: Open Project in VS Code
1. Open VS Code
2. Go to File → Open Folder (or press `Ctrl+K Ctrl+O`)
3. Select the folder containing your project files (`app.py`, `style.css`, etc.)


Step 2: Create Virtual Environment
Open the Terminal in VS Code (`Ctrl + `` ` ``) and run:

Windows:
bash
python -m venv venv
venv\Scripts\activate

macOS/Linux:
bash
python3 -m venv venv
source venv/bin/activate

You should see `(venv)` at the start of your terminal prompt.


Step 3: Select Python Interpreter
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type: "Python: Select Interpreter"
3. Choose the one with `venv` in the path (e.g., `.\venv\Scripts\python.exe`)

Step 4: Install Dependencies
In the VS Code terminal (with venv activated):

bash
pip install streamlit torch transformers requests pillow pytesseract


### **Step 5: Install Tesseract OCR**
**Windows:**
1. Download: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to: `C:\Program Files\Tesseract-OCR\`
3. Verify path in `app.py` is correct (it already is in your code)

### **Step 6: Get NewsAPI Key**
1. Visit: https://newsapi.org/register
2. Sign up (use Gmail `+` trick if needed: `yourname+newsapi@gmail.com`)
3. Copy your API key
4. Paste into `app.py` where it says:
   ```python
   API_KEY = "your_key_here"
   ```

### **Step 7: Check Model Folder**
Ensure you have a folder named `distilbert_model/` in your project root containing:
- `config.json`
- `pytorch_model.bin` (or `model.safetensors`)
- `tokenizer_config.json`
- `vocab.txt`

---

### **Step 8: Run the Application**
In VS Code terminal:
```bash
streamlit run app.py
```

**Output will show:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

**Press `Ctrl+Click` on the URL** or open browser manually to `http://localhost:8501`

---

### **Step 9: Stop the App**
Press `Ctrl+C` in the VS Code terminal.

Quick Checklist Before Running

- [ ] `venv` created and activated
- [ ] Python interpreter set to `venv`
- [ ] `pip install` completed without errors
- [ ] Tesseract installed and path correct in `app.py`
- [ ] NewsAPI key added to `app.py`
- [ ] `distilbert_model/` folder exists with model files
- [ ] Terminal shows `(venv)` before commands

**Once all checked → Run `streamlit run app.py`