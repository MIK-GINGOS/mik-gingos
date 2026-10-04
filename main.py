"""mik-gingos: a one-file Flask app styled in a Swiss + brutalist mix.

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
  /* Swiss: 12-column grid, Helvetica, flush-left type, red accent.
     Brutalism: thick black borders, hard shadows, raw monospace, exposed structure. */
  :root {
    --ink: #000;
    --paper: #f2f0eb;
    --white: #fff;
    --red: #ff2a00;
    --line: 4px;
    --unit: 8px;
    --sans: "Helvetica Neue", Helvetica, Arial, sans-serif;
    --mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--paper);
    color: var(--ink);
    font-family: var(--sans);
    line-height: 1.3;
  }
  .page {
    max-width: 1280px;
    margin: calc(var(--unit) * 3) auto;
    border: var(--line) solid var(--ink);
    background: var(--white);
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(12, minmax(0, 1fr));
  }
  .cell { border-right: var(--line) solid var(--ink); padding: calc(var(--unit) * 3); }
  .cell:last-child { border-right: 0; }
  .row { border-bottom: var(--line) solid var(--ink); }
  .mono {
    font-family: var(--mono);
    font-size: 0.8125rem;
    text-transform: uppercase;
    letter-spacing: 0.02em;
  }

  /* header */
  .brand {
    grid-column: span 3;
    background: var(--red);
    font-weight: 700;
    font-size: 1.25rem;
    letter-spacing: -0.02em;
  }
  .meta { grid-column: span 3; }
  nav.cell { grid-column: span 6; display: flex; padding: 0; }
  nav a {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: calc(var(--unit) * 3);
    border-right: var(--line) solid var(--ink);
    color: var(--ink);
    text-decoration: none;
    font-weight: 700;
    text-transform: uppercase;
  }
  nav a:last-child { border-right: 0; }
  nav a:hover { background: var(--ink); color: var(--white); }

  /* hero */
  .index {
    grid-column: span 2;
    font-family: var(--mono);
    font-size: 3rem;
    font-weight: 700;
    color: var(--red);
  }
  .hero h1 {
    grid-column: span 10;
    font-size: clamp(3.5rem, 13vw, 11rem);
    font-weight: 700;
    line-height: 0.82;
    letter-spacing: -0.06em;
    text-transform: uppercase;
    padding-top: calc(var(--unit) * 6);
    padding-bottom: calc(var(--unit) * 4);
    word-break: break-word;
  }
  .hero h1 span { color: var(--red); }

  /* sections */
  .label { grid-column: span 3; display: flex; flex-direction: column; gap: var(--unit); }
  .label b { font-family: var(--mono); font-size: 2rem; color: var(--red); }
  .label strong { font-size: 1.5rem; text-transform: uppercase; letter-spacing: -0.02em; }
  .body { grid-column: span 9; font-size: clamp(1.25rem, 2.4vw, 2rem); font-weight: 500; }
  .body p { max-width: 30ch; }

  .stack { grid-column: span 9; display: flex; flex-direction: column; gap: calc(var(--unit) * 3); }
  form { display: flex; flex-wrap: wrap; gap: calc(var(--unit) * 2); }
  input[type=text] {
    flex: 1 1 260px;
    min-width: 0;
    font: inherit;
    font-size: 1.5rem;
    font-weight: 700;
    padding: calc(var(--unit) * 2);
    border: var(--line) solid var(--ink);
    border-radius: 0;
    background: var(--paper);
    color: var(--ink);
  }
  input[type=text]:focus { outline: none; background: var(--white); box-shadow: 6px 6px 0 var(--red); }
  button {
    font: inherit;
    font-size: 1.125rem;
    font-weight: 700;
    text-transform: uppercase;
    padding: calc(var(--unit) * 2) calc(var(--unit) * 4);
    border: var(--line) solid var(--ink);
    border-radius: 0;
    background: var(--red);
    color: var(--ink);
    box-shadow: 6px 6px 0 var(--ink);
    cursor: pointer;
  }
  button:hover { transform: translate(3px, 3px); box-shadow: 3px 3px 0 var(--ink); }
  button:active { transform: translate(6px, 6px); box-shadow: none; }
  .result {
    background: var(--ink);
    color: var(--white);
    padding: calc(var(--unit) * 3);
    font-size: clamp(2rem, 6vw, 4.5rem);
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.04em;
    word-break: break-word;
  }

  footer .cell { grid-column: span 6; }
  footer .cell:last-child { background: var(--ink); color: var(--white); }

  @media (max-width: 720px) {
    .page { margin: 0; border-left: 0; border-right: 0; }
    .grid { grid-template-columns: minmax(0, 1fr); }
    .grid > * { grid-column: 1 / -1 !important; }
    .cell { border-right: 0; border-bottom: var(--line) solid var(--ink); padding: 16px; }
    .cell:last-child { border-bottom: 0; }
    nav { flex-wrap: wrap; }
    .index { font-size: 2rem; }
    .hero h1 { padding-top: 16px; }
  }
</style>
</head>
<body>
<div class="page">
  <header class="grid row">
    <div class="cell brand">MIK&#8209;GINGOS</div>
    <div class="cell meta mono">Python / Flask<br>Est. 2026</div>
    <nav class="cell">
      <a href="#about">About</a>
      <a href="#greet">Greet</a>
      <a href="/api/greet">API</a>
    </nav>
  </header>

  <main>
    <div class="hero grid row">
      <div class="cell index">01</div>
      <h1 class="cell">MIK&#8209;<br>GINGOS<span>.</span></h1>
    </div>

    <section id="about" class="grid row">
      <div class="cell label"><b>02</b><strong>About</strong></div>
      <div class="cell body"><p>A small Python and Flask project. Raw structure, strict grid, one typeface, one colour.</p></div>
    </section>

    <section id="greet" class="grid row">
      <div class="cell label"><b>03</b><strong>Greet</strong></div>
      <div class="cell stack">
        <form method="get" action="/#greet">
          <input type="text" name="name" placeholder="YOUR NAME"
                 value="{{ name if name != 'world' else '' }}" aria-label="Your name">
          <button type="submit">Say hello &rarr;</button>
        </form>
        <p class="result">{{ message }}</p>
      </div>
    </section>
  </main>

  <footer class="grid">
    <div class="cell mono">&copy; MIK-GINGOS</div>
    <div class="cell mono">Built with Python &amp; Flask</div>
  </footer>
</div>
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
