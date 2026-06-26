import json


def _render_gui(schema):
    schema_json = json.dumps(schema)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Accelera Deployment</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
        sans-serif;
      color: #17202a;
      background: #f4f7f9;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: #f4f7f9;
    }}
    main {{
      width: min(1040px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0;
    }}
    header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      font-weight: 760;
      letter-spacing: 0;
    }}
    .status {{
      border: 1px solid #c9d7df;
      background: #ffffff;
      border-radius: 8px;
      padding: 8px 12px;
      color: #48606f;
      font-size: 14px;
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 420px);
      gap: 18px;
      align-items: start;
    }}
    section {{
      background: #ffffff;
      border: 1px solid #d8e3e9;
      border-radius: 8px;
      padding: 18px;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 17px;
      font-weight: 720;
      letter-spacing: 0;
    }}
    .fields {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    label {{
      display: grid;
      gap: 6px;
      color: #405463;
      font-size: 13px;
      font-weight: 620;
    }}
    input, select, button {{
      font: inherit;
      border-radius: 7px;
    }}
    input, select {{
      min-width: 0;
      border: 1px solid #b9c9d2;
      padding: 10px 11px;
      color: #17202a;
      background: #fff;
    }}
    input[type="checkbox"] {{
      width: 18px;
      height: 18px;
      margin: 0;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 16px;
    }}
    button {{
      border: 1px solid #174ea6;
      background: #1f6feb;
      color: #fff;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
    }}
    button.secondary {{
      border-color: #b9c9d2;
      background: #eef4f7;
      color: #23313b;
    }}
    .muted {{
      color: #6a7b86;
      font-size: 13px;
      line-height: 1.5;
    }}
    pre {{
      margin: 0;
      min-height: 280px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid #d8e3e9;
      border-radius: 8px;
      background: #0f1720;
      color: #e8f0f7;
      padding: 14px;
      font-size: 13px;
      line-height: 1.5;
    }}
    .hidden {{
      display: none;
    }}
    @media (max-width: 820px) {{
      header, .grid {{
        display: block;
      }}
      .status {{
        display: inline-block;
        margin-top: 12px;
      }}
      section {{
        margin-bottom: 14px;
      }}
      .fields {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Accelera Deployment</h1>
        <div class="muted">Run predictions against the deployed model.</div>
      </div>
      <div id="schema-status" class="status"></div>
    </header>

    <div class="grid">
      <div>
        <section id="manual-section">
          <h2>Single Prediction</h2>
          <form id="manual-form">
            <div id="fields" class="fields"></div>
            <div class="actions">
              <button type="submit">Predict</button>
              <button type="button" class="secondary" id="reset-form">Reset</button>
            </div>
          </form>
        </section>

        <section>
          <h2>CSV Prediction</h2>
          <form id="csv-form">
            <label>
              CSV file
              <input id="csv-file" type="file" accept=".csv,text/csv" required>
            </label>
            <div class="actions">
              <button type="submit">Upload CSV</button>
            </div>
          </form>
        </section>
      </div>

      <section>
        <h2>Result</h2>
        <pre id="result">Ready.</pre>
      </section>
    </div>
  </main>

  <script>
    const schema = {schema_json};
    const result = document.getElementById("result");
    const manualSection = document.getElementById("manual-section");
    const fields = document.getElementById("fields");
    const statusEl = document.getElementById("schema-status");

    function show(value) {{
      result.textContent = typeof value === "string"
        ? value
        : JSON.stringify(value, null, 2);
    }}

    function inputFor(feature) {{
      const wrapper = document.createElement("label");
      wrapper.textContent = feature.name;

      let input;
      if (Array.isArray(feature.allowed_values)) {{
        input = document.createElement("select");
        feature.allowed_values.forEach((value) => {{
          const option = document.createElement("option");
          option.value = value;
          option.textContent = value;
          input.appendChild(option);
        }});
      }} else {{
        input = document.createElement("input");
        input.type = feature.type === "integer" || feature.type === "number"
          ? "number"
          : feature.type === "boolean"
            ? "checkbox"
            : "text";
        if (feature.type === "integer") {{
          input.step = "1";
        }}
        if (feature.type === "number") {{
          input.step = "any";
        }}
        if (feature.min !== undefined) {{
          input.min = feature.min;
        }}
        if (feature.max !== undefined) {{
          input.max = feature.max;
        }}
      }}

      input.name = feature.name;
      input.dataset.type = feature.type || "number";
      input.required = input.dataset.type !== "boolean"
        && feature.required !== false;
      wrapper.appendChild(input);
      return wrapper;
    }}

    function valueFromInput(input) {{
      if (input.dataset.type === "boolean") {{
        return input.checked;
      }}
      if (input.dataset.type === "integer") {{
        return Number.parseInt(input.value, 10);
      }}
      if (input.dataset.type === "number") {{
        return Number.parseFloat(input.value);
      }}
      return input.value;
    }}

    if (!schema.enabled) {{
      manualSection.classList.add("hidden");
      statusEl.textContent = "No schema: CSV upload only";
    }} else {{
      statusEl.textContent = `${{schema.features.length}} schema fields`;
      schema.features.forEach((feature) => fields.appendChild(inputFor(feature)));
    }}

    document.getElementById("manual-form").addEventListener("submit", async (e) => {{
      e.preventDefault();
      const row = {{}};
      schema.features.forEach((feature) => {{
        const input = e.currentTarget.elements.namedItem(feature.name);
        row[feature.name] = valueFromInput(input);
      }});

      show("Predicting...");
      const response = await fetch("/predict", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{input: [row]}}),
      }});
      show(await response.json());
    }});

    document.getElementById("reset-form").addEventListener("click", () => {{
      document.getElementById("manual-form").reset();
      show("Ready.");
    }});

    document.getElementById("csv-form").addEventListener("submit", async (e) => {{
      e.preventDefault();
      const file = document.getElementById("csv-file").files[0];
      const formData = new FormData();
      formData.append("file", file);

      show("Uploading CSV...");
      const response = await fetch("/predict/csv", {{
        method: "POST",
        body: formData,
      }});
      show(await response.json());
    }});
  </script>
</body>
</html>"""
