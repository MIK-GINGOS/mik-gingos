"""mik-gingos: a one-file Flask app styled in the Swiss (International) style.

Install:          pip install flask
Run the app:      python3 main.py         then open http://127.0.0.1:5000
Run the tests:    python3 main.py --test
"""

import sys
import unittest

from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)


def greet(name: str = "world") -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}!"


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MIK-GINGOS</title>
<style>
  :root {
    --ink: #111;
    --paper: #fff;
    --red: #e30613;
    --rule: #111;
    --muted: #6b6b6b;
    --unit: 8px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { font-size: 16px; }
  body {
    background: var(--paper);
    color: var(--ink);
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    line-height: 1.35;
    -webkit-font-smoothing: antialiased;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(12, minmax(0, 1fr));
    column-gap: calc(var(--unit) * 3);
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 calc(var(--unit) * 5);
  }
  header {
    border-bottom: 2px solid var(--rule);
    padding: calc(var(--unit) * 3) 0;
  }
  header .mark {
    grid-column: 1 / 2;
    width: calc(var(--unit) * 4);
    height: calc(var(--unit) * 4);
    background: var(--red);
  }
  header nav {
    grid-column: 7 / 13;
    display: flex;
    gap: calc(var(--unit) * 4);
    align-items: center;
    font-size: 0.875rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  header nav a { color: var(--ink); text-decoration: none; }
  header nav a:hover { color: var(--red); }

  .hero { padding: calc(var(--unit) * 12) 0 calc(var(--unit) * 10); }
  .hero .index {
    grid-column: 1 / 3;
    font-size: 0.875rem;
    font-weight: 700;
    color: var(--red);
    padding-top: calc(var(--unit) * 2);
  }
  .hero h1 {
    grid-column: 3 / 13;
    font-size: clamp(3rem, 11vw, 9rem);
    font-weight: 700;
    line-height: 0.9;
    letter-spacing: -0.04em;
  }
  .hero h1 span { color: var(--red); }

  section { border-top: 1px solid var(--rule); padding: calc(var(--unit) * 6) 0; }
  section .label {
    grid-column: 1 / 4;
    font-size: 0.875rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  section .label b { color: var(--red); margin-right: var(--unit); }
  section .body { grid-column: 4 / 11; font-size: 1.5rem; max-width: 34ch; }

  form { grid-column: 4 / 13; display: flex; flex-wrap: wrap; gap: calc(var(--unit) * 2); }
  input[type=text] {
    flex: 1 1 240px;
    font: inherit;
    font-size: 1.5rem;
    border: 0;
    border-bottom: 2px solid var(--ink);
    padding: var(--unit) 0;
    background: transparent;
    color: var(--ink);
    border-radius: 0;
  }
  input[type=text]:focus { outline: none; border-color: var(--red); }
  button {
    font: inherit;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    background: var(--ink);
    color: var(--paper);
    border: 0;
    padding: calc(var(--unit) * 2) calc(var(--unit) * 4);
    cursor: pointer;
  }
  button:hover { background: var(--red); }
  .result {
    grid-column: 4 / 13;
    margin-top: calc(var(--unit) * 4);
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 700;
    letter-spacing: -0.02em;
  }

  footer {
    border-top: 2px solid var(--rule);
    padding: calc(var(--unit) * 3) 0 calc(var(--unit) * 6);
    font-size: 0.875rem;
    color: var(--muted);
  }
  footer p { grid-column: 1 / 7; }
  footer p + p { grid-column: 7 / 13; }

  @media (max-width: 720px) {
    .grid { grid-template-columns: minmax(0, 1fr); padding: 0 16px; }
    header nav, .hero .index, .hero h1, section .label, section .body,
    form, .result, footer p, footer p + p { grid-column: 1 / -1; }
    header nav { margin-top: calc(var(--unit) * 2); gap: calc(var(--unit) * 3); }
    .hero { padding: calc(var(--unit) * 6) 0; }
    section .label { margin-bottom: calc(var(--unit) * 2); }
  }
</style>
</head>
<body>
  <header>
    <div class="grid">
      <div class="mark" aria-hidden="true"></div>
      <nav>
        <a href="#about">About</a>
        <a href="#greet">Greet</a>
        <a href="/api/greet">API</a>
      </nav>
    </div>
  </header>

  <main>
    <div class="hero">
      <div class="grid">
        <div class="index">01</div>
        <h1>MIK&#8209;<br>GINGOS<span>.</span></h1>
      </div>
    </div>

    <section id="about">
      <div class="grid">
        <div class="label"><b>02</b>About</div>
        <p class="body">A small Python and Flask project. Clean grid, one typeface, one colour.</p>
      </div>
    </section>

    <section id="greet">
      <div class="grid">
        <div class="label"><b>03</b>Greet</div>
        <form method="get" action="/#greet">
          <input type="text" name="name" placeholder="Your name"
                 value="{{ name if name != 'world' else '' }}" aria-label="Your name">
          <button type="submit">Say hello</button>
        </form>
        <p class="result">{{ message }}</p>
      </div>
    </section>
  </main>

  <footer>
    <div class="grid">
      <p>MIK-GINGOS</p>
      <p>Built with Python &amp; Flask</p>
    </div>
  </footer>
</body>
</html>
"""


@app.route("/")
def index():
    name = request.args.get("name", "").strip() or "world"
    return render_template_string(PAGE, name=name, message=greet(name))


@app.route("/api/greet")
def api_greet():
    name = request.args.get("name", "").strip() or "world"
    return jsonify(message=greet(name))


class AppTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_greet(self):
        self.assertEqual(greet(), "Hello, world!")
        self.assertEqual(greet("MIK-GINGOS"), "Hello, MIK-GINGOS!")

    def test_index(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Hello, world!", res.data)

    def test_index_with_name(self):
        res = self.client.get("/?name=Ada")
        self.assertIn(b"Hello, Ada!", res.data)

    def test_name_is_escaped(self):
        res = self.client.get("/?name=<script>")
        self.assertNotIn(b"<script>", res.data)
        self.assertIn(b"&lt;script&gt;", res.data)

    def test_api(self):
        res = self.client.get("/api/greet?name=Ada")
        self.assertEqual(res.get_json(), {"message": "Hello, Ada!"})


if __name__ == "__main__":
    if "--test" in sys.argv:
        unittest.main(argv=[sys.argv[0], "-v"])
    else:
        app.run(debug=True)
