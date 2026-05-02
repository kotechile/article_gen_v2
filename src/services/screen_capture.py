import logging
import time
from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)

class ScreenCaptureService:
    """
    Service for generating screenshots from HTML/CSS using Playwright.
    """

    def _apply_background_canvas_sizing(self, page, element):
        return page.evaluate(
            """async (node) => {
                if (!node) return null;

                const computed = window.getComputedStyle(node);
                const backgroundImage = computed.backgroundImage || '';
                const urlMatch = backgroundImage.match(/url\\((["']?)(.*?)\\1\\)/);
                const imageUrl = urlMatch && urlMatch[2];
                if (!imageUrl) return null;

                const assetSize = await new Promise((resolve) => {
                    const img = new Image();
                    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
                    img.onerror = () => resolve(null);
                    img.src = imageUrl;
                });

                if (!assetSize || !assetSize.width || !assetSize.height) return null;

                const rect = node.getBoundingClientRect();
                const declaredWidth = parseFloat(node.style.width) || parseFloat(computed.width) || 0;
                const declaredHeight = parseFloat(node.style.height) || parseFloat(computed.height) || 0;
                const hasExplicitWidth = declaredWidth > 0;
                const hasExplicitHeight = declaredHeight > 0;

                let targetWidth = rect.width || declaredWidth || assetSize.width;
                let targetHeight = rect.height || declaredHeight || assetSize.height;

                if (!hasExplicitWidth && !hasExplicitHeight) {
                    targetWidth = assetSize.width;
                    targetHeight = assetSize.height;
                } else if (hasExplicitWidth && !hasExplicitHeight) {
                    targetWidth = declaredWidth;
                    targetHeight = declaredWidth * (assetSize.height / assetSize.width);
                } else if (!hasExplicitWidth && hasExplicitHeight) {
                    targetHeight = declaredHeight;
                    targetWidth = declaredHeight * (assetSize.width / assetSize.height);
                }

                if (
                    rect.width < 2 ||
                    rect.height < 2 ||
                    Math.abs(rect.width - targetWidth) > 1 ||
                    Math.abs(rect.height - targetHeight) > 1
                ) {
                    node.style.width = `${targetWidth}px`;
                    node.style.maxWidth = 'none';
                    node.style.height = `${targetHeight}px`;
                }

                return {
                    assetWidth: assetSize.width,
                    assetHeight: assetSize.height,
                    width: targetWidth,
                    height: targetHeight,
                };
            }""",
            element,
        )

    def generate_screenshot(self, html: str, css: str, width: int = 1920, height: int = 1080, clip: dict = None, root_selector: str = None, root_selectors: list[str] | None = None) -> bytes:
        """
        Generate a screenshot from HTML and CSS.

        Args:
            html (str): The HTML content.
            css (str): The CSS styles.
            width (int): Viewport width.
            height (int): Viewport height.
            clip (dict): Optional clipping region {x, y, width, height}.
            root_selector (str): Optional selector for element-based capture.
            root_selectors (list[str] | None): Optional ordered selector candidates.

        Returns:
            bytes: The PNG image data.
        """
        base_css = """
        html, body {
            margin: 0;
            padding: 0;
            background: transparent;
        }
        """

        full_html = f"""
        <html>
        <head>
            <link href="https://pro.fontawesome.com/releases/v6.0.0-beta1/css/all.css" rel="stylesheet">
            <link href="https://fonts.googleapis.com/css2?family=Material+Icons" rel="stylesheet">
        </head>
        <style>
            {base_css}
            {css}
        </style>
        <body>
            {html}
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

                # Wait for both <img> tags and CSS background images.
                page.evaluate("""async () => {
                    const images = document.querySelectorAll('img');
                    const imagePromises = Array.from(images).map(img => new Promise(resolve => {
                        if (img.complete) resolve();
                        img.onload = img.onerror = resolve;
                    }));

                    const backgroundUrls = new Set();
                    document.querySelectorAll('*').forEach((node) => {
                        const backgroundImage = window.getComputedStyle(node).backgroundImage;
                        const matches = backgroundImage.match(/url\\((["']?)(.*?)\\1\\)/g) || [];
                        matches.forEach((match) => {
                            const urlMatch = match.match(/url\\((["']?)(.*?)\\1\\)/);
                            const url = urlMatch && urlMatch[2];
                            if (url) backgroundUrls.add(url);
                        });
                    });

                    const backgroundPromises = Array.from(backgroundUrls).map(url => new Promise(resolve => {
                        const img = new Image();
                        img.onload = img.onerror = resolve;
                        img.src = url;
                    }));

                    await Promise.all([...imagePromises, ...backgroundPromises]);
                }""")
                
                # Final network idle wait
                try:
                   page.wait_for_load_state("networkidle", timeout=5000)
                except:
                   logger.warning("Timeout waiting for final network idle, proceeding to capture.")

                if clip:
                    image_buffer = page.screenshot(
                        type='png',
                        clip={
                            'x': clip['x'],
                            'y': clip['y'],
                            'width': clip['width'],
                            'height': clip['height']
                        }
                    )
                elif root_selector or root_selectors:
                    selectors = root_selectors or [root_selector]
                    element = None
                    matched_selector = None

                    for selector in selectors:
                        if not selector:
                            continue
                        element = page.query_selector(selector)
                        if element:
                            matched_selector = selector
                            break

                    if not element:
                        logger.warning("No root selector matched; falling back to viewport screenshot.")
                        image_buffer = page.screenshot(type='png')
                        browser.close()
                        return image_buffer

                    self._apply_background_canvas_sizing(page, element)
                    bounds = element.bounding_box()
                    if not bounds:
                        raise ValueError(f"Unable to measure root selector: {matched_selector}")

                    page.set_viewport_size({
                        'width': max(width, int(bounds['x'] + bounds['width'])),
                        'height': max(height, int(bounds['y'] + bounds['height']))
                    })
                    image_buffer = element.screenshot(type='png')
                else:
                    image_buffer = page.screenshot(type='png')
                
                browser.close()
                return image_buffer

        except Exception as e:
            logger.error(f"Error generating screenshot: {str(e)}", exc_info=True)
            raise e
