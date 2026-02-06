import logging
import time
from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)

class ScreenCaptureService:
    """
    Service for generating screenshots from HTML/CSS using Playwright.
    """

    def generate_screenshot(self, html: str, css: str, width: int = 1920, height: int = 1080, clip: dict = None) -> bytes:
        """
        Generate a screenshot from HTML and CSS.

        Args:
            html (str): The HTML content.
            css (str): The CSS styles.
            width (int): Viewport width.
            height (int): Viewport height.
            clip (dict): Optional clipping region {x, y, width, height}.

        Returns:
            bytes: The PNG image data.
        """
        full_screen_css = """
        html, body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
        }
        .full-screen {
            width: 100vw;
            height: 100vh;
            object-fit: cover;
        }
        """

        full_html = f"""
        <html>
        <head>
            <link href="https://pro.fontawesome.com/releases/v6.0.0-beta1/css/all.css" rel="stylesheet">
            <link href="https://fonts.googleapis.com/css2?family=Material+Icons" rel="stylesheet">
        </head>
        <style>
            {full_screen_css}
            {css}
        </style>
        <body>
            <div class="full-screen">
            {html}
            </div>
        </body>
        </html>
        """

        try:
            with sync_playwright() as p:
                # Launch the browser
                # Note: 'chromium' is installed in the Dockerfile
                browser = p.chromium.launch(
                    args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu', '--font-render-hinting=none', '--force-color-profile=srgb'],
                    headless=True
                )
                
                page = browser.new_page(viewport={'width': width, 'height': height})
                
                # Set content
                page.set_content(full_html, wait_until='networkidle', timeout=60000)

                # Wait for images to load explicitly (similar to original logic)
                page.evaluate("""() => {
                    const images = document.querySelectorAll('img');
                    return Promise.all(Array.from(images).map(img => new Promise(resolve => {
                        if (img.complete) resolve();
                        img.onload = img.onerror = resolve;
                    })));
                }""")

                # Scroll to bottom if necessary (replicated from original logic)
                # But typically for a single screen capture, scrolling might not be needed if we capture the viewport
                # or full page. The original code had a scroll loop.
                # If we are just capturing a specific size, maybe not strictly needed unless content renders on scroll.
                # I will include a simplified scroll to bottom just in case.
                page.evaluate("""async () => {
                    await new Promise((resolve) => {
                        let totalHeight = 0;
                        const distance = 100;
                        const timer = setInterval(() => {
                            const scrollHeight = document.body.scrollHeight;
                            window.scrollBy(0, distance);
                            totalHeight += distance;

                            if(totalHeight >= scrollHeight){
                                clearInterval(timer);
                                resolve();
                            }
                        }, 100);
                    });
                }""")
                
                # Final network idle wait
                try:
                   page.wait_for_load_state("networkidle", timeout=5000)
                except:
                   logger.warning("Timeout waiting for final network idle, proceeding to capture.")

                screenshot_args = {'type': 'png'}
                if clip:
                    screenshot_args['clip'] = {
                        'x': clip['x'],
                        'y': clip['y'],
                        'width': clip['width'],
                        'height': clip['height']
                    }

                image_buffer = page.screenshot(**screenshot_args)
                
                browser.close()
                return image_buffer

        except Exception as e:
            logger.error(f"Error generating screenshot: {str(e)}", exc_info=True)
            raise e
