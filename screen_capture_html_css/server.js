const express = require('express');
const puppeteer = require('puppeteer');
const sharp = require('sharp');

const app = express(); // Define the Express app instance here

app.use(express.json());

let browser;

const initializeBrowser = async () => {
  if (!browser) {
    browser = await puppeteer.launch({
      args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu','--font-render-hinting=none','--force-color-profile=srgb' ],
      headless: true,
      defaultViewport: null,
    });
  }
};

app.post('/generate-image', async (req, res) => {
  const { html, css, width = 1920, height = 1080, clip } = req.body;

  console.log('Received request:', { html, css, width, height, clip });

  const fullScreenCSS = `
    html, body {
      margin: 0;
      padding: 0;
      width: 100%;
      min-height: 100%;
    }
    .full-screen {
      width: 100%;
      min-height: 100vh;
      position: relative;
    }
  `;

  const fullHtml = `
    <html>
      <head>
        <link href="https://pro.fontawesome.com/releases/v6.0.0-beta1/css/all.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Material+Icons" rel="stylesheet">        
      </head>
      <style>
        ${fullScreenCSS}
        ${css}
      </style>
      <body>
        <div class="full-screen">
          ${html}
        </div>
      </body>
    </html>
  `;

  try {
    await initializeBrowser();
    const page = await browser.newPage();

    await page.setViewport({
      width: parseInt(width),
      height: parseInt(height),
      deviceScaleFactor: 1,
    });

    // Set content and wait for network to be idle
    await page.setContent(fullHtml, { waitUntil: 'networkidle0', timeout: 60000 });

    // Wait for images to load
    await page.evaluate(() => {
      const images = document.querySelectorAll('img');
      return Promise.all(Array.from(images).map(img => new Promise(resolve => {
        if (img.complete) resolve();
      img.onload = img.onerror = resolve;
      })));
    });

    // Scroll to the bottom of the page if necessary
    await page.evaluate(() => {
      return new Promise((resolve) => {
        const intervalId = setInterval(() => {
          if (document.body.scrollTop + window.innerHeight >= document.body.offsetHeight) {
            clearInterval(intervalId);
            resolve();
          } else {
            window.scrollTo(0, document.body.scrollTop + 100);
          }
        }, 100);
      });
    });

    // Additional wait to ensure all resources are loaded
    await page.waitForNetworkIdle({ timeout: 60000 });

    const { fullPage = false } = req.body;
    const screenshotOptions = {
      type: 'png',
      fullPage: clip ? false : fullPage,
      clip: clip ? {
        x: Number(clip.x),
        y: Number(clip.y),
        width: Number(clip.width),
        height: Number(clip.height),
      } : undefined,
    };

    const imageBuffer = await page.screenshot(screenshotOptions);

    await page.close();

    res.writeHead(200, { 'Content-Type': 'image/png' });
    res.end(imageBuffer, 'binary');
  } catch (error) {
    console.error('Error generating image:', error);
    res.status(500).json({ error: 'Failed to generate image', details: error.message });
  }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server is running on port ${PORT}`);
});

