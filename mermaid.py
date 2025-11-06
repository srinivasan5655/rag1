
# pip install playwright
# playwright install chromium


import asyncio
from playwright.async_api import async_playwright

async def render_mermaid_to_png(mermaid_code: str, output_path: str = "diagram.png"):
    html = f"""
    <html>
      <head>
        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
          mermaid.initialize({{ startOnLoad: true, theme: "dark" }});
        </script>
        <style>
          body {{ background: #1e1e1e; display: flex; justify-content: center; align-items: center; height: 100vh; }}
          .mermaid {{ width: 100%; color: white; }}
        </style>
      </head>
      <body>
        <div class="mermaid">
          {mermaid_code}
        </div>
      </body>
    </html>
    """

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html)
        # Wait for Mermaid to finish rendering
        await page.wait_for_selector(".mermaid svg", timeout=8000)
        element = await page.query_selector(".mermaid")
        await element.screenshot(path=output_path)
        await browser.close()
        print(f"✅ Diagram saved at {output_path}")


mermaid_code = """
graph TD
  A[Start] --> B{Is it working?}
  B -->|Yes| C[Ship it]
  B -->|No| D[Fix it]
  D --> B
"""

asyncio.run(render_mermaid_to_png(mermaid_code, "diagram.png"))
