from flask import Flask, render_template_string
import sqlite3
import os

app = Flask(__name__)

DB_PATH = "db.sqlite"  # ensure this file is in the same folder as this script

# HTML template with dynamic table rendering
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Database Viewer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 40px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background-color: #f4f4f4; }
        tr:nth-child(even) { background-color: #f9f9f9; }
    </style>
</head>
<body>
    <h1>Database Viewer</h1>
    {% for table_name, columns, rows in tables %}
        <h2>{{ table_name }}</h2>
        <table>
            <tr>
                {% for col in columns %}
                    <th>{{ col }}</th>
                {% endfor %}
            </tr>
            {% for row in rows %}
                <tr>
                    {% for col in row %}
                        <td>{{ col }}</td>
                    {% endfor %}
                </tr>
            {% endfor %}
        </table>
    {% endfor %}
</body>
</html>
"""


def fetch_table_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    tables = []
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = cursor.fetchall()

    for (table_name,) in all_tables:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]

        cursor.execute(f"SELECT * FROM {table_name} LIMIT 100")  # limit for performance
        rows = cursor.fetchall()

        tables.append((table_name, columns, rows))

    conn.close()
    return tables


@app.route("/")
def index():
    tables = fetch_table_data()
    return render_template_string(TEMPLATE, tables=tables)


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"Database file {DB_PATH} not found!")
    else:
        app.run(host="0.0.0.0", port=5000, debug=True)
