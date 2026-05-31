import os
import random
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

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
    results_df = pd.json_normalize(results)
    
    output_file = 'evaluation_results.csv'
    # Append to CSV if it exists, otherwise create new
    if os.path.exists(output_file):
        results_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(output_file, index=False)
        
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)