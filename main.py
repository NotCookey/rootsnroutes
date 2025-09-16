from flask import Flask, request, jsonify
import io
import json
import re
from google import genai
from google.genai import types

app = Flask(__name__)

# Gemini client
client = genai.Client(api_key="AIzaSyDuoaSb4koA4JBsY9g5gXUSYAFIMbVd_wA")

system_instruction = """
You are an AI expert in Northeast Indian languages and cultural heritage preservation.
Your job:
1. Carefully analyze the text in the image to identify ALL languages present.
2. For each language detected, determine:
   - The specific language name
   - Writing script used
   - Confidence level
   - Percentage of text in that language
   - Linguistic family
3. For each detected language, provide relevant references:
   - History of the language
   - Cultural significance
   - Links to credible resources or further reading

Return only JSON with keys:
{
  "languages": {
    "primary_language": "",
    "detected_languages": [
      {
        "name": "",
        "script": "",
        "confidence": 0.0,
        "percentage": "",
        "linguistic_family": "",
        "additional_info": {
          "history": "",
          "cultural_significance": "",
          "resources": ["link1", "link2"]
        }
      }
    ]
  },
  "script": "",
  "confidence": 0.0,
  "text_direction": "",
  "notes": "",
  "additional_info": ""
}
"""

def extract_json_from_response(response_text):
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return response_text

def analyze_image(file_bytes, filename):
    try:
        uploaded_file = client.files.upload(file=io.BytesIO(file_bytes), filename=filename)

        contents = [
            types.Part.from_text(text=system_instruction),
            types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type)
        ]

        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=contents
        )

        json_text = extract_json_from_response(response.text)
        result = json.loads(json_text)

        # fill missing keys
        for key in ["languages", "script", "confidence", "text_direction", "notes", "additional_info"]:
            if key not in result:
                result[key] = {"primary_language": "unknown", "detected_languages": []} if key=="languages" else ""

        return result

    except Exception as e:
        return {
            "languages": {"primary_language": "unknown", "detected_languages": []},
            "script": "unknown",
            "confidence": 0.0,
            "text_direction": "ltr",
            "notes": f"Processing failed: {str(e)}",
            "additional_info": f"Error: {str(e)}"
        }

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    file_bytes = file.read()
    try:
        result = analyze_image(file_bytes, file.filename)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
