import os
import random
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
import gspread

app = Flask(__name__)

# Base directories
BASE_DIRS = {'real': 'real', 'synth': 'synth'}

def load_dataset():
    data = []
    # Process both real and synth folders
    for domain, folder in BASE_DIRS.items():
        csv_path = os.path.join(folder, 'captions.csv')
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            filename = row['filename']
            # LLM Models exactly as they appear in your CSV
            models = ['hybrid_gemma3-4b', 'hybrid_qwen3-vl-8b', 'text_qwen3-4b', 'vision_gemma3-4b', 'vision_qwen3-vl-8b']
            
            captions = [{'model': m, 'text': row[m]} for m in models if pd.notna(row[m])]
            random.shuffle(captions) # Randomize caption order per image
            
            data.append({
                'id': f"{domain}_{filename}",
                'domain': domain,
                'filename': filename,
                'image_url': f"/image/{domain}/images/{filename}",
                'mask_url': f"/image/{domain}/masks/{filename}",
                'captions': captions
            })
    return data

DATASET = load_dataset()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/trials')
def get_trials():
    # Serve a random subset of images to prevent fatigue
    sample_size = min(5, len(DATASET))
    sampled_data = random.sample(DATASET, sample_size)
    return jsonify(sampled_data)

@app.route('/image/<domain>/<type>/<filename>')
def serve_image(domain, type, filename):
    filepath = os.path.join(BASE_DIRS[domain], type, filename)
    return send_file(filepath)

@app.route('/api/submit', methods=['POST'])
def submit():
    results = request.json
    
    # 1. Locate the credentials file (Render will inject this path)
    # Defaults to local 'google_credentials.json' for your local testing
    cred_path = os.environ.get('GOOGLE_CREDENTIALS_PATH', 'google_credentials.json')
    
    try:
        # 2. Authenticate with Google
        gc = gspread.service_account(filename=cred_path)
        
        # 3. Open the sheet using the ID from your environment variables
        sheet_id = os.environ.get('SPREADSHEET_ID')
        sh = gc.open_by_key(sheet_id)
        worksheet = sh.sheet1
        
        # 4. Format data as a list of lists for gspread
        rows_to_insert = []
        for res in results:
            row = [
                res.get('filename', ''),
                res.get('actual_domain', ''),
                res.get('predicted_origin', ''),
                res.get('realism_score', ''),
                res.get('mask_alignment_score', ''),
                res.get('score_hybrid_gemma3-4b', ''),
                res.get('score_hybrid_qwen3-vl-8b', ''),
                res.get('score_text_qwen3-4b', ''),
                res.get('score_vision_gemma3-4b', ''),
                res.get('score_vision_qwen3-vl-8b', '')
            ]
            rows_to_insert.append(row)
            
        # 5. Append to Google Sheets
        worksheet.append_rows(rows_to_insert)
        
        return jsonify({"status": "success"})
        
    except Exception as e:
        print(f"Error saving to Google Sheets: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)