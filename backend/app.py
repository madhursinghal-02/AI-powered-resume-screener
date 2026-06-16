from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from flask_cors import CORS
import os
import sqlite3
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import uuid
import traceback

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ---------------- Configuration ----------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- Initialize Services ----------------
from services.resume_parser import ResumeParser
from services.gemini_service import GeminiService
from services.bias_detection import BiasDetector
from services.database import DatabaseManager
from services.email_service import EmailService

resume_parser = ResumeParser()
ai_service = GeminiService()
bias_detector = BiasDetector()
db_manager = DatabaseManager()
email_service = EmailService()

# ---------------- Helper Functions ----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- Health Check ----------------
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# ---------------- Upload Resume ----------------
@app.route('/api/upload', methods=['POST'])
def upload_resume():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        job_description = request.form.get('job_description', '')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF and DOCX allowed'}), 400
        
        # Save file with unique name
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], unique_filename))
        file.save(file_path)
        print(f"[UPLOAD] Saved file to: {file_path}")  # Debug log
        
        # Parse resume
        resume_text = resume_parser.extract_text(file_path)
        
        # AI Analysis
        analysis = ai_service.analyze_resume(resume_text, job_description)
        
        # Bias Detection
        bias_analysis = bias_detector.analyze_bias(resume_text)
        blind_resume = bias_detector.create_blind_resume(resume_text)
        
        # Save to database
        candidate_id = db_manager.save_candidate({
            'filename': filename,
            'file_path': file_path,
            'resume_text': resume_text,
            'job_description': job_description,
            'analysis': analysis,
            'bias_analysis': bias_analysis,
            'blind_resume': blind_resume,
            'upload_date': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'candidate_id': candidate_id,
            'analysis': analysis,
            'bias_analysis': bias_analysis
        })
        
    except Exception as e:
        print(f"[ERROR] Upload Resume: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ---------------- List Uploaded Files (Debugging) ----------------
@app.route('/uploads', methods=['GET'])
def list_uploads():
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        files = [os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], f)) for f in files]
        return jsonify({'uploaded_files': files})
    except Exception as e:
        print(f"[ERROR] List Uploads: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ---------------- Candidate Management ----------------
@app.route('/api/candidates', methods=['GET'])
def get_candidates():
    try:
        candidates = db_manager.get_all_candidates()
        return jsonify({'candidates': candidates})
    except Exception as e:
        print(f"[ERROR] Get Candidates: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/candidates/<int:candidate_id>', methods=['GET'])
def get_candidate(candidate_id):
    try:
        candidate = db_manager.get_candidate(candidate_id)
        if not candidate:
            return jsonify({'error': 'Candidate not found'}), 404
        return jsonify({'candidate': candidate})
    except Exception as e:
        print(f"[ERROR] Get Candidate {candidate_id}: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ---------------- Bias & Blind Resume ----------------
@app.route('/api/bias-analysis/<int:candidate_id>', methods=['GET'])
def get_bias_analysis(candidate_id):
    try:
        candidate = db_manager.get_candidate(candidate_id)
        if not candidate:
            return jsonify({'error': 'Candidate not found'}), 404
        return jsonify({
            'bias_analysis': candidate.get('bias_analysis', {}),
            'recommendations': bias_detector.get_recommendations(candidate.get('bias_analysis', {}))
        })
    except Exception as e:
        print(f"[ERROR] Bias Analysis {candidate_id}: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/blind-resume/<int:candidate_id>', methods=['GET'])
def get_blind_resume(candidate_id):
    try:
        candidate = db_manager.get_candidate(candidate_id)
        if not candidate:
            return jsonify({'error': 'Candidate not found'}), 404
        return jsonify({'blind_resume': candidate.get('blind_resume', '')})
    except Exception as e:
        print(f"[ERROR] Blind Resume {candidate_id}: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ---------------- Candidate Chat ----------------
@app.route('/api/chat', methods=['POST'])
def candidate_chat():
    try:
        data = request.get_json()
        candidate_id = data.get('candidate_id')
        message = data.get('message')
        
        if not candidate_id or not message:
            return jsonify({'error': 'Missing candidate_id or message'}), 400
        
        candidate = db_manager.get_candidate(candidate_id)
        if not candidate:
            return jsonify({'error': 'Candidate not found'}), 404
        
        response = ai_service.chat_about_candidate(candidate, message)
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"[ERROR] Candidate Chat: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ---------------- HR Assistant Chat ----------------
@app.route('/api/hr-chat', methods=['POST'])
def hr_chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        candidates = db_manager.get_all_candidates()
        response = ai_service.hr_assistant_chat(candidates, message)
        db_manager.save_hr_chat_message(message, response)
        
        return jsonify({'response': response, 'timestamp': datetime.now().isoformat()})
        
    except Exception as e:
        print(f"[ERROR] HR Chat: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to process chat message'}), 500

# ---------------- Other Endpoints ----------------
# Fair Screening Toggle
@app.route('/api/fair-screening/toggle', methods=['POST'])
def toggle_fair_screening():
    try:
        data = request.get_json()
        enabled = data.get('enabled', False)
        return jsonify({
            'fair_screening_enabled': enabled,
            'message': f"Fair screening {'enabled' if enabled else 'disabled'}"
        })
    except Exception as e:
        print(f"[ERROR] Fair Screening Toggle: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Statistics
@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    try:
        stats = db_manager.get_statistics()
        return jsonify({'statistics': stats})
    except Exception as e:
        print(f"[ERROR] Get Statistics: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Collaboration Emails
@app.route('/api/collaboration/invite', methods=['POST'])
def send_team_invitation():
    try:
        data = request.get_json()
        to_email = data.get('email')
        inviter_name = data.get('inviter_name', 'Team Member')
        team_name = data.get('team_name', 'Resume Screener AI Team')
        
        if not to_email:
            return jsonify({'error': 'Email address is required'}), 400
        
        result = email_service.send_team_invitation(to_email, inviter_name, team_name)
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Send Team Invitation: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/collaboration/notify', methods=['POST'])
def send_notification():
    try:
        data = request.get_json()
        to_email = data.get('email')
        subject = data.get('subject', 'Team Notification')
        message = data.get('message', '')
        sender_name = data.get('sender_name', 'Team Member')
        
        if not to_email or not message:
            return jsonify({'error': 'Email and message are required'}), 400
        
        result = email_service.send_notification_email(to_email, subject, message, sender_name)
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Send Notification: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Interview Questions
@app.route('/api/interview-questions/<int:candidate_id>', methods=['GET'])
def get_interview_questions(candidate_id):
    try:
        candidate = db_manager.get_candidate(candidate_id)
        if not candidate:
            return jsonify({'error': 'Candidate not found'}), 404
        
        questions = ai_service.generate_interview_questions(candidate, candidate.get('job_description', ''))
        return jsonify(questions)
    except Exception as e:
        print(f"[ERROR] Interview Questions {candidate_id}: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Compare Candidates
@app.route('/api/compare-candidates', methods=['POST'])
def compare_candidates_endpoint():
    try:
        data = request.get_json()
        candidate_ids = data.get('candidate_ids', [])
        job_description = data.get('job_description', '')
        
        candidates = [db_manager.get_candidate(cid) for cid in candidate_ids if db_manager.get_candidate(cid)]
        
        if not candidates:
            return jsonify({'error': 'No valid candidates found'}), 400
            
        comparison = ai_service.compare_candidates(candidates, job_description)
        return jsonify({'comparison': comparison})
    except Exception as e:
        print(f"[ERROR] Compare Candidates: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ---------------- Run App ----------------
if __name__ == '__main__':
    db_manager.init_db()
    print(f"[INFO] Flask app running. Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    app.run(debug=True, host='0.0.0.0', port=5000)