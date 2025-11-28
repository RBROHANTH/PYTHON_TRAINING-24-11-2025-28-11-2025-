from flask import Flask, request, jsonify, render_template, send_file
import sqlite3, io, csv
from pathlib import Path

BASE = Path(__file__).parent
DB = BASE / "students.db"

app = Flask(__name__)

# --- DB ---
def get_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    if not DB.exists():
        conn = get_conn()
        cur = conn.cursor()
        cur.execute('''
        CREATE TABLE students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            email TEXT,
            final_score REAL
        )
        ''')
        conn.commit()
        conn.close()

init_db()

# --- API ---
@app.route('/api/add_student', methods=['POST'])
def add_student():
    data = request.json
    if not data or not data.get('name'):
        return jsonify({'error':'name required'}), 400
    name = data['name']
    age = data.get('age', 0)
    email = data.get('email','')
    final_score = data.get('final_score', 0.0)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO students (name, age, email, final_score) VALUES (?,?,?,?)',
                (name, age, email, final_score))
    conn.commit()
    nid = cur.lastrowid
    conn.close()
    return jsonify({'message':'student added','id':nid})

@app.route('/api/students', methods=['GET'])
def list_students():
    conn = get_conn()
    rows = conn.execute('SELECT * FROM students').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/insights', methods=['GET'])
def insights():
    # simple ML-like insight: student at-risk if score <50
    conn = get_conn()
    rows = conn.execute('SELECT final_score FROM students').fetchall()
    conn.close()
    total = len(rows)
    at_risk = sum(1 for r in rows if float(r['final_score']) < 50)
    pct_at_risk = round(at_risk / total * 100, 2) if total else 0
    return jsonify({'total_students': total, 'at_risk': at_risk, 'pct_at_risk': pct_at_risk})

@app.route('/reports/download_csv', methods=['GET'])
def download_csv():
    conn = get_conn()
    rows = conn.execute('SELECT * FROM students').fetchall()
    conn.close()
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['id','name','age','email','final_score'])
    for r in rows:
        cw.writerow([r['id'], r['name'], r['age'], r['email'], r['final_score']])
    mem = io.BytesIO()
    mem.write(si.getvalue().encode('utf-8'))
    mem.seek(0)
    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='students_report.csv')

# --- Frontend dashboard ---
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
